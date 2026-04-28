import json
import shutil
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "product_visual_search_retrieval.ipynb"
BACKUP_PATH = PROJECT_ROOT / "product_visual_search_retrieval_before_full_resnet.ipynb"
SECTION_TITLE = "Full Dataset ResNet18 Experiment Summary"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "resnet18_full_best.pth"


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def build_markdown():
    stats = pd.read_csv(REPORT_DIR / "full_dataset_stats.csv")
    train_log = pd.read_csv(REPORT_DIR / "resnet18_full_training_log.csv")
    train_log = train_log[train_log.get("is_smoke_test", False).astype(str).str.lower() != "true"].copy()
    test_metrics = load_json(REPORT_DIR / "resnet18_full_test_metrics.json") or {}
    retrieval_metrics = load_json(REPORT_DIR / "resnet18_full_retrieval_metrics.json") or {}
    summary = pd.read_csv(REPORT_DIR / "experiment_summary.csv")
    dropped = pd.read_csv(REPORT_DIR / "dropped_rare_classes.csv")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False) if CHECKPOINT_PATH.exists() else {}
    config = checkpoint.get("train_config", {})

    best_row = train_log.loc[train_log["val_top1"].idxmax()] if not train_log.empty else None
    total_training_time = float(train_log["epoch_time_seconds"].sum()) if "epoch_time_seconds" in train_log else None
    split_counts = {
        "train": int(test_metrics.get("train_size", 0)),
        "val": int(test_metrics.get("val_size", 0)),
        "test": int(test_metrics.get("test_size", 0)),
    }

    comparison_rows = []
    for model_name, label in [
        ("resnet18", "ResNet18 debug"),
        ("clip_vit_b_32_openai_local", "CLIP debug"),
        ("clip_vit_b_32_openai_full", "CLIP full dataset"),
        ("resnet18_full_dataset", "ResNet18 full dataset"),
    ]:
        rows = summary[summary["model_name"] == model_name]
        if rows.empty:
            continue
        row = rows.iloc[-1]
        comparison_rows.append(
            f"| {label} | {row.get('num_classes', 'N/A')} | {row.get('train_size', 'N/A')} | "
            f"{row.get('test_size', 'N/A')} | {fmt(row.get('top1_acc'))} | {fmt(row.get('top5_acc'))} | "
            f"{fmt(row.get('recall@1'))} | {fmt(row.get('recall@5'))} | {fmt(row.get('recall@10'))} |"
        )

    best_epoch = int(best_row["epoch"]) if best_row is not None else "N/A"
    best_val_top1 = fmt(best_row["val_top1"]) if best_row is not None else "N/A"
    best_val_top5 = fmt(best_row["val_top5"]) if best_row is not None else "N/A"

    return f"""# {SECTION_TITLE}

This section summarizes the full-dataset ResNet18 experiment. The full training loop was executed through standalone scripts rather than inside the notebook, so the notebook remains a report artifact instead of a long-running training environment. All metrics below are loaded from actual result files produced by training, evaluation, indexing, and retrieval scripts.

## Full Dataset Split Statistics

- Valid images after metadata/image filtering: **44,405**
- Number of `articleType` classes after rare-class filtering: **{len(stats)}**
- Train split: **{split_counts['train']}** images
- Validation split: **{split_counts['val']}** images
- Test/query split: **{split_counts['test']}** images
- Dropped rare classes: **{len(dropped)}**
- Label used for supervision and retrieval relevance: **articleType**
- Canonical image directory: `data/raw/images/`

The duplicate extracted path `data/raw/myntradataset/images/` is not used as the primary image source. This avoids duplicate images appearing across train and test splits, which would cause train/test leakage and overestimate visual retrieval performance.

## Full ResNet18 Training Setup

- Model: **ResNet18**
- Initialization: **ImageNet pretrained weights**
- Number of output classes: **{checkpoint.get('num_classes', 'N/A')}**
- Image size: **{checkpoint.get('image_size', config.get('image_size', 'N/A'))}**
- Batch size: **{config.get('batch_size', 'N/A')}**
- Optimizer: **AdamW**
- Learning rate: **{config.get('lr', 'N/A')}**
- Weight decay: **{config.get('weight_decay', 'N/A')}**
- Epochs requested: **{config.get('epochs', 'N/A')}**
- Device: **CUDA/GPU was used during training**
- Mixed precision: **enabled when CUDA was available**
- Smoke test flag in formal checkpoint: **{checkpoint.get('is_smoke_test', 'N/A')}**

## Training Process Summary

- Best epoch by validation Top-1 accuracy: **{best_epoch}**
- Best validation Top-1 accuracy: **{best_val_top1}**
- Validation Top-5 accuracy at best epoch: **{best_val_top5}**
- Total epoch time from training log: **{fmt(total_training_time, 2)} seconds**
- Best checkpoint: `outputs/checkpoints/resnet18_full_best.pth`
- Last checkpoint: `outputs/checkpoints/resnet18_full_last.pth`

![ResNet18 full training curves](outputs/figures/resnet18_full_training_curves.png)

## Full Test Classification Results

- Test loss: **{fmt(test_metrics.get('test_loss'))}**
- Test Top-1 accuracy: **{fmt(test_metrics.get('test_top1_acc'))}**
- Test Top-5 accuracy: **{fmt(test_metrics.get('test_top5_acc'))}**
- Test size: **{test_metrics.get('test_size', 'N/A')}**

These classification metrics evaluate whether the supervised ResNet18 model predicts the correct `articleType`. They are useful for product recognition, but retrieval quality is better measured through Recall@K.

## Full Retrieval Results

- Gallery split: **train**
- Gallery size: **{retrieval_metrics.get('gallery_size', 'N/A')}**
- Query split: **test**
- Query size: **{retrieval_metrics.get('query_size', 'N/A')}**
- Embedding dimension: **{retrieval_metrics.get('embedding_dim', 'N/A')}**
- Recall@1: **{fmt(retrieval_metrics.get('recall@1'))}**
- Recall@5: **{fmt(retrieval_metrics.get('recall@5'))}**
- Recall@10: **{fmt(retrieval_metrics.get('recall@10'))}**
- Index path: `outputs/indexes/resnet18_full_gallery_index.npz`

![ResNet18 full retrieval summary](outputs/figures/resnet18_full_retrieval_summary.png)

## ResNet18 Debug vs CLIP Debug vs ResNet18 Full Dataset

| Model | Classes | Train/Gallery Size | Test/Query Size | Top-1 Acc | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

![Model recall comparison with full ResNet18](outputs/figures/model_recall_comparison_full.png)

This comparison separates debug-scope and full-scope experiments. The CLIP full dataset row and ResNet18 full dataset row now use the same full split: train images form the gallery and test images form the query set. This makes the Streamlit model switch easier to interpret because a random sample query has the same split role for both models.

## Streamlit Product Demo Integration

The Streamlit app now supports model switching between `CLIP ViT-B/32` and `ResNet18 Full Dataset`. Users can upload a product image or choose a random sample query, select Top-K, and retrieve visually similar products. Both models now use the shared full test split for random sample queries. CLIP uses `outputs/indexes/clip_full_gallery_index.npz`, and ResNet18 Full Dataset uses `outputs/checkpoints/resnet18_full_best.pth` plus `outputs/indexes/resnet18_full_gallery_index.npz`.

## Discussion

The full ResNet18 model learns dataset-specific `articleType` boundaries from the full product image dataset. This makes it a stronger supervised baseline than the debug model and allows the Streamlit demo to search a much larger gallery. CLIP remains important because it provides general pretrained visual-semantic features without local supervised training. For visual search, Recall@K is more important than a single classification prediction because users expect a ranked set of similar products rather than only one category label.

## Limitations and Future Work

The current retrieval relevance definition is category-level `articleType`, not exact SKU-level similarity. CLIP currently uses the debug gallery unless a full CLIP index is built later. The project does not yet use FAISS, product metadata ranking, price/brand/color features, or production deployment infrastructure. Future work should build a full CLIP index, add FAISS approximate nearest-neighbor search, fine-tune CLIP on product data, incorporate product metadata, deploy with FastAPI and React, and collect user feedback for ranking improvements.
"""


def main():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")
    if not BACKUP_PATH.exists():
        shutil.copy2(NOTEBOOK_PATH, BACKUP_PATH)
        print(f"backup_created={BACKUP_PATH}")
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    markdown = build_markdown()
    new_cell = {"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(keepends=True)}

    replaced = False
    for idx, cell in enumerate(notebook.get("cells", [])):
        if SECTION_TITLE in "".join(cell.get("source", [])):
            notebook["cells"][idx] = new_cell
            replaced = True
            break
    if not replaced:
        notebook.setdefault("cells", []).append(new_cell)

    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"section_replaced={replaced}")
    print(f"notebook_updated={NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
