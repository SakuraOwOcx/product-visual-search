import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.product_search.clip_engine import CLIPImageDataset, load_clip_model  # noqa: E402
from src.product_search.config import BATCH_SIZE, CLIP_FULL_INDEX_PATH, DEVICE, NUM_WORKERS, REPORT_DIR  # noqa: E402
from src.product_search.data_utils import get_full_gallery_query_dataframes  # noqa: E402


METRICS_JSON = REPORT_DIR / "clip_full_retrieval_metrics.json"
SUMMARY_CSV = REPORT_DIR / "experiment_summary.csv"


def extract_query_embeddings(model, preprocess, query_df, batch_size=BATCH_SIZE, max_images=None):
    if max_images is not None:
        query_df = query_df.head(max_images).reset_index(drop=True)
    loader = DataLoader(
        CLIPImageDataset(query_df, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    embeddings, labels = [], []
    with torch.no_grad():
        for images, batch_labels, _ in loader:
            features = model.encode_image(images.to(DEVICE))
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu().numpy())
            labels.extend(batch_labels.numpy().tolist())
    return np.vstack(embeddings).astype(np.float32), labels


def recall_at_k_batched(query_embeddings, query_labels, gallery_embeddings, gallery_labels, k, batch_size=256):
    hits = 0
    top_k = min(k, gallery_embeddings.shape[0])
    gallery_labels = np.asarray(gallery_labels)
    for start in range(0, len(query_embeddings), batch_size):
        batch = query_embeddings[start:start + batch_size]
        sims = batch @ gallery_embeddings.T
        top_indices = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
        for row_idx, indices in enumerate(top_indices):
            sorted_indices = indices[np.argsort(-sims[row_idx, indices])]
            if int(query_labels[start + row_idx]) in gallery_labels[sorted_indices].astype(int):
                hits += 1
    return hits / max(len(query_embeddings), 1)


def upsert_summary(metrics, is_smoke):
    if is_smoke:
        print("summary_update_skipped=True reason=smoke_test")
        return
    row = {
        "experiment_mode": "full",
        "model_name": "clip_vit_b_32_openai_full",
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
        "training_time": 0.0,
        "embedding_extraction_time_seconds": metrics["embedding_extraction_time_seconds"],
        "embedding_dim": metrics["embedding_dim"],
        "checkpoint_source": metrics["checkpoint_source"],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        df = df[df["model_name"] != "clip_vit_b_32_openai_full"]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SUMMARY_CSV, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP full retrieval using shared full split.")
    parser.add_argument("--max-query-images", type=int, default=None, help="Optional smoke-test query limit.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if not CLIP_FULL_INDEX_PATH.exists():
        raise FileNotFoundError(f"CLIP full index not found: {CLIP_FULL_INDEX_PATH}")

    gallery_df, query_df, _ = get_full_gallery_query_dataframes()
    if args.max_query_images is not None:
        query_df = query_df.head(args.max_query_images).reset_index(drop=True)

    with np.load(CLIP_FULL_INDEX_PATH, allow_pickle=False) as data:
        gallery_embeddings = data["embeddings"].astype(np.float32)
        metadata = json.loads(str(data["metadata_json"]))
        checkpoint_source = str(data["checkpoint_source"]) if "checkpoint_source" in data.files else ""
    gallery_labels = [int(item["class_id"]) for item in metadata]

    start = time.perf_counter()
    model, preprocess = load_clip_model()
    query_embeddings, query_labels = extract_query_embeddings(
        model,
        preprocess,
        query_df,
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - start

    metrics = {
        "recall@1": recall_at_k_batched(query_embeddings, query_labels, gallery_embeddings, gallery_labels, 1),
        "recall@5": recall_at_k_batched(query_embeddings, query_labels, gallery_embeddings, gallery_labels, 5),
        "recall@10": recall_at_k_batched(query_embeddings, query_labels, gallery_embeddings, gallery_labels, 10),
        "query_size": int(len(query_df)),
        "gallery_size": int(gallery_embeddings.shape[0]),
        "num_classes": int(gallery_df["class_name"].nunique()),
        "embedding_dim": int(gallery_embeddings.shape[1]),
        "embedding_extraction_time_seconds": float(elapsed),
        "checkpoint_source": checkpoint_source,
        "index_path": str(CLIP_FULL_INDEX_PATH),
        "is_smoke_test": args.max_query_images is not None,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    upsert_summary(metrics, metrics["is_smoke_test"])
    print(json.dumps(metrics, indent=2))
    print(f"metrics_json={METRICS_JSON}")


if __name__ == "__main__":
    main()
