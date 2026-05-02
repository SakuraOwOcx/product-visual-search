import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.product_search.config import (  # noqa: E402
    FULL_CLASS_MAPPING_JSON,
    FULL_SPLIT_CSV,
    REPORT_DIR,
    RESNET18_FULL_BEST_CHECKPOINT,
    RESNET18_FULL_LAST_CHECKPOINT,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TRAIN_LOG = REPORT_DIR / "resnet18_full_training_log.csv"


class ProductDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), int(row["class_id"])


def build_transforms(image_size):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_model(num_classes):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def topk_accuracy(logits, labels, k):
    k = min(k, logits.shape[1])
    _, preds = logits.topk(k, dim=1)
    return preds.eq(labels.view(-1, 1)).any(dim=1).float().sum().item()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, max_batches=None, progress_label=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_samples, top1_hits, top5_hits = 0.0, 0, 0.0, 0.0
    amp_enabled = scaler is not None and device.type == "cuda"

    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    iterator = enumerate(loader, start=1)
    if progress_label:
        iterator = tqdm(iterator, total=total_batches, desc=progress_label, leave=True)

    for batch_idx, (images, labels) in iterator:
        if max_batches is not None and batch_idx > max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if is_train:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        top1_hits += topk_accuracy(logits.detach(), labels, 1)
        top5_hits += topk_accuracy(logits.detach(), labels, 5)

    return {
        "loss": total_loss / max(total_samples, 1),
        "top1": top1_hits / max(total_samples, 1),
        "top5": top5_hits / max(total_samples, 1),
        "samples": total_samples,
    }


def save_checkpoint(path, model, optimizer, epoch, best_val_top1, best_val_top5, mapping, args, train_config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_top1": best_val_top1,
            "best_val_top5": best_val_top5,
            "class_to_idx": mapping["class_to_idx"],
            "idx_to_class": mapping["idx_to_class"],
            "num_classes": mapping["num_classes"],
            "image_size": args.image_size,
            "model_name": "resnet18_full_dataset",
            "train_config": train_config,
            "is_smoke_test": bool(train_config["is_smoke_test"]),
        },
        path,
    )


def append_log(row):
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not TRAIN_LOG.exists()
    with TRAIN_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 on the full product dataset split.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save an extra epoch checkpoint every N epochs. Use 0 to disable.")
    args = parser.parse_args()

    if not FULL_SPLIT_CSV.exists() or not FULL_CLASS_MAPPING_JSON.exists():
        raise FileNotFoundError("Run scripts/prepare_full_dataset_split.py first.")
    with FULL_CLASS_MAPPING_JSON.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    split_df = pd.read_csv(FULL_SPLIT_CSV)
    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
    train_tf, eval_tf = build_transforms(args.image_size)
    train_loader = DataLoader(ProductDataset(train_df, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(ProductDataset(val_df, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(mapping["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 1
    best_val_top1 = -1.0
    best_val_top5 = -1.0
    if args.resume and RESNET18_FULL_LAST_CHECKPOINT.exists():
        ckpt = torch.load(RESNET18_FULL_LAST_CHECKPOINT, map_location=device, weights_only=False)
        if ckpt.get("is_smoke_test") or ckpt.get("train_config", {}).get("is_smoke_test"):
            raise RuntimeError(
                "Refusing to resume from a smoke-test checkpoint. "
                "Back it up and start formal training without --resume."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_top1 = float(ckpt.get("best_val_top1", -1.0))
        best_val_top5 = float(ckpt.get("best_val_top5", -1.0))
        print(f"resumed_from={RESNET18_FULL_LAST_CHECKPOINT}")

    train_config = vars(args).copy()
    train_config["is_smoke_test"] = args.max_train_batches is not None or args.max_val_batches is not None
    print(f"device={device}")
    print(f"num_classes={mapping['num_classes']}")
    print(f"train_size={len(train_df)}")
    print(f"val_size={len(val_df)}")
    print(f"smoke_test={train_config['is_smoke_test']}")

    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        try:
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scaler,
                args.max_train_batches,
                progress_label=f"resnet train epoch {epoch}/{args.epochs}",
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                None,
                None,
                args.max_val_batches,
                progress_label=f"resnet val epoch {epoch}/{args.epochs}",
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print("CUDA out of memory. Try lowering --batch-size, for example --batch-size 16.")
            raise

        epoch_time = time.perf_counter() - epoch_start
        is_best = val_metrics["top1"] > best_val_top1
        if is_best:
            best_val_top1 = val_metrics["top1"]
            best_val_top5 = val_metrics["top5"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(RESNET18_FULL_LAST_CHECKPOINT, model, optimizer, epoch, best_val_top1, best_val_top5, mapping, args, train_config)
        if is_best:
            save_checkpoint(RESNET18_FULL_BEST_CHECKPOINT, model, optimizer, epoch, best_val_top1, best_val_top5, mapping, args, train_config)
        if (
            args.checkpoint_every > 0
            and epoch % args.checkpoint_every == 0
            and not train_config["is_smoke_test"]
        ):
            periodic_checkpoint = RESNET18_FULL_LAST_CHECKPOINT.with_name(f"resnet18_full_epoch_{epoch:03d}.pth")
            save_checkpoint(periodic_checkpoint, model, optimizer, epoch, best_val_top1, best_val_top5, mapping, args, train_config)
            print(f"periodic_checkpoint={periodic_checkpoint}")

        log_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "epoch_time_seconds": epoch_time,
            "learning_rate": args.lr,
            "train_samples_seen": train_metrics["samples"],
            "val_samples_seen": val_metrics["samples"],
            "is_smoke_test": train_config["is_smoke_test"],
        }
        append_log(log_row)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.4f} "
            f"val_top5={val_metrics['top5']:.4f} epoch_time={epoch_time:.1f}s lr={args.lr}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"early_stopping_triggered=True patience={args.patience}")
            break

    print(f"last_checkpoint={RESNET18_FULL_LAST_CHECKPOINT}")
    print(f"best_checkpoint={RESNET18_FULL_BEST_CHECKPOINT}")
    print(f"training_log={TRAIN_LOG}")


if __name__ == "__main__":
    main()
