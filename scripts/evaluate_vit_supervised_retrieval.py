import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.product_search.config import FULL_SPLIT_CSV, REPORT_DIR, VIT_SUPERVISED_BEST_CHECKPOINT, VIT_SUPERVISED_INDEX_PATH  # noqa: E402
from src.product_search.data_utils import load_full_split_dataframe  # noqa: E402
from src.product_search.path_utils import resolve_project_path, to_project_relative  # noqa: E402
from src.product_search.vit_engine import extract_vit_features, load_vit_supervised_model  # noqa: E402


METRICS_JSON = REPORT_DIR / "vit_supervised_retrieval_metrics.json"
SMOKE_METRICS_JSON = REPORT_DIR / "vit_supervised_retrieval_metrics_smoke.json"
SUMMARY_CSV = REPORT_DIR / "experiment_summary.csv"


class ProductQueryDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(resolve_project_path(row["image_path"])).convert("RGB")
        return self.transform(image), int(row["class_id"])


def extract_query_embeddings(model, transform, query_df, batch_size, num_workers):
    device = next(model.parameters()).device
    loader = DataLoader(ProductQueryDataset(query_df, transform), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            features = extract_vit_features(model, images.to(device))
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu().numpy())
            labels.extend(batch_labels.numpy().tolist())
    return np.vstack(embeddings).astype(np.float32), labels


def recall_at_k_batched(query_embeddings, query_labels, gallery_embeddings, gallery_labels, k, batch_size=256):
    hits = 0
    top_k = min(k, gallery_embeddings.shape[0])
    gallery_labels = np.asarray(gallery_labels).astype(int)
    for start in range(0, len(query_embeddings), batch_size):
        batch = query_embeddings[start:start + batch_size]
        sims = batch @ gallery_embeddings.T
        top_indices = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
        for row_idx, indices in enumerate(top_indices):
            sorted_indices = indices[np.argsort(-sims[row_idx, indices])]
            if int(query_labels[start + row_idx]) in gallery_labels[sorted_indices]:
                hits += 1
    return hits / max(len(query_embeddings), 1)


def update_summary(metrics):
    row = {
        "experiment_mode": "full",
        "model_name": "vit_b16_supervised",
        "num_classes": metrics["num_classes"],
        "samples_per_class": "all",
        "train_size": metrics["gallery_size"],
        "val_size": np.nan,
        "test_size": metrics["query_size"],
        "top1_acc": np.nan,
        "top5_acc": np.nan,
        "recall@1": metrics["recall@1"],
        "recall@5": metrics["recall@5"],
        "recall@10": metrics["recall@10"],
        "training_time": np.nan,
        "embedding_extraction_time_seconds": np.nan,
        "embedding_dim": metrics["embedding_dim"],
        "checkpoint_source": metrics["checkpoint"],
    }
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        mask = df["model_name"] == "vit_b16_supervised"
        if mask.any():
            for key, value in row.items():
                if key not in df.columns:
                    df[key] = np.nan
                df.loc[mask, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SUMMARY_CSV, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate supervised ViT retrieval with cosine similarity.")
    parser.add_argument("--split-csv", type=Path, default=FULL_SPLIT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=VIT_SUPERVISED_BEST_CHECKPOINT)
    parser.add_argument("--index", type=Path, default=VIT_SUPERVISED_INDEX_PATH)
    parser.add_argument("--query-split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-query-images", type=int, default=None)
    parser.add_argument("--update-summary", action="store_true")
    args = parser.parse_args()

    if not args.split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_csv}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"ViT checkpoint not found: {args.checkpoint}")
    if not args.index.exists():
        raise FileNotFoundError(f"ViT index not found: {args.index}. Run build_vit_supervised_index.py first.")
    model, transform, checkpoint = load_vit_supervised_model(args.checkpoint)
    split_df = load_full_split_dataframe(args.split_csv)
    query_df = split_df[split_df["split"] == args.query_split].reset_index(drop=True)
    if args.max_query_images is not None:
        query_df = query_df.head(args.max_query_images).reset_index(drop=True)
    with np.load(args.index, allow_pickle=False) as data:
        gallery_embeddings = data["embeddings"].astype(np.float32)
        gallery_labels = data["labels"].astype(int).tolist() if "labels" in data.files else []
        index_scope = str(data["index_scope"]) if "index_scope" in data.files else ""
    query_embeddings, query_labels = extract_query_embeddings(model, transform, query_df, args.batch_size, args.num_workers)
    metrics = {
        "recall@1": recall_at_k_batched(query_embeddings, query_labels, gallery_embeddings, gallery_labels, 1),
        "recall@5": recall_at_k_batched(query_embeddings, query_labels, gallery_embeddings, gallery_labels, 5),
        "recall@10": recall_at_k_batched(query_embeddings, query_labels, gallery_embeddings, gallery_labels, 10),
        "query_size": int(len(query_df)),
        "gallery_size": int(gallery_embeddings.shape[0]),
        "num_classes": int(checkpoint["num_classes"]),
        "embedding_dim": int(gallery_embeddings.shape[1]),
        "checkpoint": to_project_relative(args.checkpoint),
        "index": to_project_relative(args.index),
        "index_scope": index_scope,
        "is_smoke_test": bool(checkpoint.get("is_smoke_test") or checkpoint.get("train_config", {}).get("is_smoke_test") or args.max_query_images or index_scope.startswith("smoke")),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = SMOKE_METRICS_JSON if metrics["is_smoke_test"] else METRICS_JSON
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.update_summary and not metrics["is_smoke_test"]:
        update_summary(metrics)
    elif args.update_summary:
        print("summary_update_skipped=True reason=smoke_checkpoint_index_or_query_limit")
    print(json.dumps(metrics, indent=2))
    print(f"metrics_json={to_project_relative(metrics_path)}")


if __name__ == "__main__":
    main()
