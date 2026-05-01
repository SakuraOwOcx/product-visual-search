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

from src.product_search.config import FULL_SPLIT_CSV, REPORT_DIR, RESNET18_FULL_INDEX_PATH  # noqa: E402
from src.product_search.data_utils import load_full_split_dataframe  # noqa: E402
from src.product_search.path_utils import resolve_project_path, to_project_relative  # noqa: E402
from src.product_search.resnet_engine import get_resnet_eval_transform, load_resnet18_full_model  # noqa: E402


METRICS_JSON = REPORT_DIR / "resnet18_full_retrieval_metrics.json"
SUMMARY_CSV = REPORT_DIR / "experiment_summary.csv"


class ProductPathDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(resolve_project_path(row["image_path"])).convert("RGB")
        return self.transform(image), str(row["articleType"])


def extract_query_embeddings(model, transform, query_df, batch_size, num_workers):
    device = next(model.parameters()).device
    loader = DataLoader(ProductPathDataset(query_df, transform), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            features = feature_extractor(images.to(device))
            features = torch.flatten(features, 1)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu().numpy())
            labels.extend(list(batch_labels))
    return np.vstack(embeddings).astype(np.float32), labels


def recall_at_k(query_embeddings, query_labels, gallery_embeddings, gallery_labels, k):
    sims = query_embeddings @ gallery_embeddings.T
    hits = 0
    top_k = min(k, gallery_embeddings.shape[0])
    for i, query_label in enumerate(query_labels):
        indices = np.argsort(-sims[i])[:top_k]
        retrieved = [gallery_labels[idx] for idx in indices]
        if query_label in retrieved:
            hits += 1
    return hits / max(len(query_labels), 1)


def update_summary(metrics, is_smoke):
    if is_smoke:
        print("summary_update_skipped=True reason=smoke_test")
        return
    row_updates = {
        "recall@1": metrics["recall@1"],
        "recall@5": metrics["recall@5"],
        "recall@10": metrics["recall@10"],
        "embedding_dim": metrics["embedding_dim"],
    }
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        mask = df["model_name"] == "resnet18_full_dataset"
        if mask.any():
            for key, value in row_updates.items():
                df.loc[mask, key] = value
        else:
            row = {
                "experiment_mode": "full",
                "model_name": "resnet18_full_dataset",
                "recall@1": metrics["recall@1"],
                "recall@5": metrics["recall@5"],
                "recall@10": metrics["recall@10"],
                "embedding_dim": metrics["embedding_dim"],
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([{"experiment_mode": "full", "model_name": "resnet18_full_dataset", **row_updates}])
    df.to_csv(SUMMARY_CSV, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 full retrieval with test queries.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-query-images", type=int, default=None)
    args = parser.parse_args()

    if not RESNET18_FULL_INDEX_PATH.exists():
        raise FileNotFoundError("Run scripts/build_resnet18_full_index.py first.")
    model, _, checkpoint = load_resnet18_full_model()
    transform = get_resnet_eval_transform(int(checkpoint.get("image_size", 224)))
    split_df = load_full_split_dataframe(FULL_SPLIT_CSV)
    query_df = split_df[split_df["split"] == "test"].reset_index(drop=True)
    if args.max_query_images is not None:
        query_df = query_df.head(args.max_query_images).reset_index(drop=True)

    index_data = np.load(RESNET18_FULL_INDEX_PATH, allow_pickle=False)
    gallery_embeddings = index_data["embeddings"].astype(np.float32)
    gallery_labels = index_data["articleTypes"].astype(str).tolist()
    query_embeddings, query_labels = extract_query_embeddings(model, transform, query_df, args.batch_size, args.num_workers)

    metrics = {
        "recall@1": recall_at_k(query_embeddings, query_labels, gallery_embeddings, gallery_labels, 1),
        "recall@5": recall_at_k(query_embeddings, query_labels, gallery_embeddings, gallery_labels, 5),
        "recall@10": recall_at_k(query_embeddings, query_labels, gallery_embeddings, gallery_labels, 10),
        "query_size": int(len(query_df)),
        "gallery_size": int(gallery_embeddings.shape[0]),
        "embedding_dim": int(gallery_embeddings.shape[1]),
        "index_path": to_project_relative(RESNET18_FULL_INDEX_PATH),
        "is_smoke_test": args.max_query_images is not None,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    update_summary(metrics, metrics["is_smoke_test"])
    print(json.dumps(metrics, indent=2))
    print(f"metrics_json={to_project_relative(METRICS_JSON)}")


if __name__ == "__main__":
    main()
