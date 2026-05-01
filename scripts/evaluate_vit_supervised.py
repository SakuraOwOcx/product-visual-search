import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.product_search.config import FULL_SPLIT_CSV, REPORT_DIR, VIT_SUPERVISED_BEST_CHECKPOINT  # noqa: E402
from src.product_search.data_utils import load_full_split_dataframe  # noqa: E402
from src.product_search.path_utils import resolve_project_path, to_project_relative  # noqa: E402
from src.product_search.vit_engine import get_vit_eval_transform, load_vit_supervised_model  # noqa: E402


METRICS_JSON = REPORT_DIR / "vit_supervised_classification_metrics.json"
SMOKE_METRICS_JSON = REPORT_DIR / "vit_supervised_classification_metrics_smoke.json"
SUMMARY_CSV = REPORT_DIR / "experiment_summary.csv"


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


def topk_hits(logits, labels, k):
    k = min(k, logits.shape[1])
    _, preds = logits.topk(k, dim=1)
    return preds.eq(labels.view(-1, 1)).any(dim=1)


def upsert_summary(metrics):
    row = {
        "experiment_mode": "full" if not metrics["is_smoke_test"] else "smoke",
        "model_name": "vit_b16_supervised",
        "num_classes": metrics["num_classes"],
        "samples_per_class": "all",
        "train_size": metrics["train_size"],
        "val_size": metrics["val_size"],
        "test_size": metrics["test_size"],
        "top1_acc": metrics["top1_acc"],
        "top5_acc": metrics["top5_acc"],
        "recall@1": np.nan,
        "recall@5": np.nan,
        "recall@10": np.nan,
        "training_time": np.nan,
        "embedding_extraction_time_seconds": np.nan,
        "embedding_dim": metrics["embedding_dim"],
        "checkpoint_source": metrics["checkpoint"],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        df = df[df["model_name"] != "vit_b16_supervised"]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SUMMARY_CSV, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate supervised ViT-B/16 classifier.")
    parser.add_argument("--split-csv", type=Path, default=FULL_SPLIT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=VIT_SUPERVISED_BEST_CHECKPOINT)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--update-summary", action="store_true")
    args = parser.parse_args()

    if not args.split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_csv}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"ViT checkpoint not found: {args.checkpoint}")

    model, transform, checkpoint = load_vit_supervised_model(args.checkpoint)
    device = next(model.parameters()).device
    split_df = load_full_split_dataframe(args.split_csv)
    eval_df = split_df[split_df["split"] == args.split].reset_index(drop=True)
    if args.max_samples is not None:
        eval_df = eval_df.head(args.max_samples).reset_index(drop=True)
    loader = DataLoader(ProductDataset(eval_df, transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss()
    total_loss, total_samples, top1, top5 = 0.0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            top1 += int(topk_hits(logits, labels, 1).sum().item())
            top5 += int(topk_hits(logits, labels, 5).sum().item())
    metrics = {
        "split": args.split,
        "loss": total_loss / max(total_samples, 1),
        "top1_acc": top1 / max(total_samples, 1),
        "top5_acc": top5 / max(total_samples, 1),
        "eval_size": int(len(eval_df)),
        "train_size": int((split_df["split"] == "train").sum()),
        "val_size": int((split_df["split"] == "val").sum()),
        "test_size": int((split_df["split"] == "test").sum()),
        "num_classes": int(checkpoint["num_classes"]),
        "embedding_dim": int(checkpoint.get("embedding_dim", 768)),
        "checkpoint": to_project_relative(args.checkpoint),
        "is_smoke_test": bool(checkpoint.get("is_smoke_test") or checkpoint.get("train_config", {}).get("is_smoke_test") or args.max_samples),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = SMOKE_METRICS_JSON if metrics["is_smoke_test"] else METRICS_JSON
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.update_summary and not metrics["is_smoke_test"]:
        upsert_summary(metrics)
    elif args.update_summary:
        print("summary_update_skipped=True reason=smoke_checkpoint_or_max_samples")
    print(json.dumps(metrics, indent=2))
    print(f"metrics_json={to_project_relative(metrics_path)}")


if __name__ == "__main__":
    main()
