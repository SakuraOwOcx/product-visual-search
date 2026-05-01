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
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.product_search.config import (  # noqa: E402
    FULL_CLASS_MAPPING_JSON,
    FULL_SPLIT_CSV,
    REPORT_DIR,
    VIT_SUPERVISED_BEST_CHECKPOINT,
    VIT_SUPERVISED_LAST_CHECKPOINT,
)
from src.product_search.data_utils import load_full_split_dataframe  # noqa: E402
from src.product_search.path_utils import resolve_project_path, to_project_relative  # noqa: E402
from src.product_search.vit_engine import IMAGENET_MEAN, IMAGENET_STD, build_vit_b16  # noqa: E402


TRAIN_LOG = REPORT_DIR / "vit_supervised_training_log.csv"
SMOKE_TRAIN_LOG = REPORT_DIR / "vit_supervised_training_log_smoke.csv"


class ProductDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(resolve_project_path(row["image_path"])).convert("RGB")
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


def topk_accuracy(logits, labels, k):
    k = min(k, logits.shape[1])
    _, preds = logits.topk(k, dim=1)
    return preds.eq(labels.view(-1, 1)).any(dim=1).float().sum().item()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_samples, top1_hits, top5_hits = 0.0, 0, 0.0, 0.0
    amp_enabled = scaler is not None and device.type == "cuda"
    for images, labels in loader:
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


def append_log(row):
    log_path = SMOKE_TRAIN_LOG if row.get("is_smoke_test") else TRAIN_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, epoch, best_val_top1, best_val_top5, mapping, args, train_config):
    path = Path(path)
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
            "model_name": "vit_b16_supervised",
            "architecture": "Vision Transformer ViT-B/16",
            "embedding_type": "CLS token",
            "train_config": train_config,
            "is_smoke_test": bool(train_config["is_smoke_test"]),
        },
        path,
    )


def subset_df(df, max_samples, seed=42):
    if max_samples is None or len(df) <= max_samples:
        return df.reset_index(drop=True)
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune supervised ViT-B/16 on product articleType labels.")
    parser.add_argument("--split-csv", type=Path, default=FULL_SPLIT_CSV)
    parser.add_argument("--class-mapping", type=Path, default=FULL_CLASS_MAPPING_JSON)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-checkpoint", type=Path, default=VIT_SUPERVISED_BEST_CHECKPOINT)
    parser.add_argument("--last-checkpoint", type=Path, default=VIT_SUPERVISED_LAST_CHECKPOINT)
    parser.add_argument("--no-pretrained", action="store_true", help="Use random initialization; useful for offline smoke tests.")
    parser.add_argument("--strict-pretrained", action="store_true", help="Fail instead of falling back if pretrained weights are unavailable.")
    args = parser.parse_args()

    if not args.split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_csv}. Run prepare_full_dataset_split.py first.")
    if not args.class_mapping.exists():
        raise FileNotFoundError(f"Class mapping not found: {args.class_mapping}.")
    with args.class_mapping.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    split_df = load_full_split_dataframe(args.split_csv)
    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)

    if args.debug:
        args.max_train_samples = args.max_train_samples or 100
        args.max_val_samples = args.max_val_samples or 50
        args.epochs = min(args.epochs, 1)
    train_df = subset_df(train_df, args.max_train_samples)
    val_df = subset_df(val_df, args.max_val_samples)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_tf, eval_tf = build_transforms(args.image_size)
    train_loader = DataLoader(ProductDataset(train_df, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(ProductDataset(val_df, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_config = vars(args).copy()
    train_config["split_csv"] = to_project_relative(args.split_csv)
    train_config["class_mapping"] = to_project_relative(args.class_mapping)
    train_config["output_checkpoint"] = to_project_relative(args.output_checkpoint)
    train_config["last_checkpoint"] = to_project_relative(args.last_checkpoint)
    train_config["is_smoke_test"] = bool(args.debug or args.max_train_samples or args.max_val_samples)

    print(f"device={device}")
    print(f"model=vit_b16_supervised")
    print(f"num_classes={mapping['num_classes']}")
    print(f"train_size={len(train_df)}")
    print(f"val_size={len(val_df)}")
    print(f"smoke_test={train_config['is_smoke_test']}")
    print(f"pretrained={not args.no_pretrained}")

    model = build_vit_b16(
        num_classes=int(mapping["num_classes"]),
        pretrained=not args.no_pretrained,
        strict_pretrained=args.strict_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_top1, best_val_top5 = -1.0, -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        try:
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_metrics = run_epoch(model, val_loader, criterion, device)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print("CUDA out of memory. Try --batch-size 8 or --batch-size 4 for ViT-B/16.")
            raise
        scheduler.step()
        epoch_time = time.perf_counter() - start
        is_best = val_metrics["top1"] > best_val_top1
        if is_best:
            best_val_top1 = val_metrics["top1"]
            best_val_top5 = val_metrics["top5"]

        save_checkpoint(args.last_checkpoint, model, optimizer, epoch, best_val_top1, best_val_top5, mapping, args, train_config)
        if is_best:
            save_checkpoint(args.output_checkpoint, model, optimizer, epoch, best_val_top1, best_val_top5, mapping, args, train_config)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "epoch_time_seconds": epoch_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_samples_seen": train_metrics["samples"],
            "val_samples_seen": val_metrics["samples"],
            "is_smoke_test": train_config["is_smoke_test"],
        }
        append_log(row)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.4f} "
            f"val_top5={val_metrics['top5']:.4f} epoch_time={epoch_time:.1f}s"
        )

    print(f"best_checkpoint={to_project_relative(args.output_checkpoint)}")
    print(f"last_checkpoint={to_project_relative(args.last_checkpoint)}")
    print(f"training_log={to_project_relative(SMOKE_TRAIN_LOG if train_config['is_smoke_test'] else TRAIN_LOG)}")


if __name__ == "__main__":
    main()
