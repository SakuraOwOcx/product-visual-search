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

from src.product_search.clip_supervised_engine import get_clip_train_eval_transforms, load_clip_articletype_model  # noqa: E402
from src.product_search.config import CLIP_ARTICLETYPE_BEST_CHECKPOINT, FULL_SPLIT_CSV, REPORT_DIR  # noqa: E402


METRICS_JSON = REPORT_DIR / "clip_articletype_test_metrics.json"
PER_CLASS_CSV = REPORT_DIR / "clip_articletype_per_class_accuracy.csv"
SUMMARY_CSV = REPORT_DIR / "experiment_summary.csv"


class ProductDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), int(row["class_id"]), str(row["articleType"])


def topk_hits(logits, labels, k):
    k = min(k, logits.shape[1])
    _, preds = logits.topk(k, dim=1)
    return preds.eq(labels.view(-1, 1)).any(dim=1)


def upsert_summary(metrics):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "experiment_mode": "full",
        "model_name": "clip_vit_b_32_articletype_lite",
        "num_classes": metrics["num_classes"],
        "samples_per_class": "all",
        "train_size": metrics["train_size"],
        "val_size": metrics["val_size"],
        "test_size": metrics["test_size"],
        "top1_acc": metrics["test_top1_acc"],
        "top5_acc": metrics["test_top5_acc"],
        "recall@1": np.nan,
        "recall@5": np.nan,
        "recall@10": np.nan,
        "training_time": np.nan,
        "embedding_extraction_time_seconds": np.nan,
        "embedding_dim": 512,
        "checkpoint_source": str(CLIP_ARTICLETYPE_BEST_CHECKPOINT),
    }
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        df = df[df["model_name"] != "clip_vit_b_32_articletype_lite"]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SUMMARY_CSV, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate lightweight fine-tuned CLIP classifier on test split.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not FULL_SPLIT_CSV.exists():
        raise FileNotFoundError("Run scripts/prepare_full_dataset_split.py first.")
    if not CLIP_ARTICLETYPE_BEST_CHECKPOINT.exists():
        raise FileNotFoundError("Run scripts/train_clip_articletype.py before evaluation.")

    model, _, checkpoint = load_clip_articletype_model()
    device = next(model.parameters()).device
    image_size = int(checkpoint.get("image_size", 224))
    _, transform = get_clip_train_eval_transforms(image_size=image_size)
    split_df = pd.read_csv(FULL_SPLIT_CSV)
    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)
    loader = DataLoader(ProductDataset(test_df, transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss()

    total_loss, total_samples, top1, top5 = 0.0, 0, 0, 0
    per_class_total, per_class_correct = {}, {}
    model.eval()
    with torch.no_grad():
        for images, labels, article_types in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            hits1 = topk_hits(logits, labels, 1)
            hits5 = topk_hits(logits, labels, 5)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            top1 += int(hits1.sum().item())
            top5 += int(hits5.sum().item())
            for cls, hit in zip(article_types, hits1.cpu().numpy().tolist()):
                per_class_total[cls] = per_class_total.get(cls, 0) + 1
                per_class_correct[cls] = per_class_correct.get(cls, 0) + int(hit)

    per_class_rows = [
        {
            "articleType": cls,
            "test_count": per_class_total[cls],
            "top1_correct": per_class_correct.get(cls, 0),
            "top1_accuracy": per_class_correct.get(cls, 0) / per_class_total[cls],
        }
        for cls in sorted(per_class_total)
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_class_rows).to_csv(PER_CLASS_CSV, index=False)

    metrics = {
        "test_loss": total_loss / max(total_samples, 1),
        "test_top1_acc": top1 / max(total_samples, 1),
        "test_top5_acc": top5 / max(total_samples, 1),
        "test_size": int(len(test_df)),
        "train_size": int((split_df["split"] == "train").sum()),
        "val_size": int((split_df["split"] == "val").sum()),
        "num_classes": int(checkpoint["num_classes"]),
        "checkpoint": str(CLIP_ARTICLETYPE_BEST_CHECKPOINT),
    }
    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    upsert_summary(metrics)

    print(json.dumps(metrics, indent=2))
    print(f"metrics_json={METRICS_JSON}")
    print(f"per_class_accuracy_csv={PER_CLASS_CSV}")
    print(f"summary_csv_updated={SUMMARY_CSV}")


if __name__ == "__main__":
    main()
