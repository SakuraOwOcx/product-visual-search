# Product Visual Search and Retrieval

This course project builds a product visual search pipeline using supervised ResNet18 embeddings and frozen CLIP ViT-B/32 embeddings. The final notebook is `product_visual_search_retrieval.ipynb`.

## Project Goal

The goal is to simulate a product visual search system. A query product image is converted into an image embedding, compared with gallery product embeddings using cosine similarity, and used to retrieve Top-K visually similar products.

## Dataset

Dataset used during the project:

- Kaggle Fashion Product Images Small
- Expected local image folder: `data/raw/images/`
- Expected metadata file: `data/raw/styles.csv`
- Label used for experiments: `articleType`

The project prioritizes `data/raw/images/` to avoid duplicate images from multiple extracted folders and reduce train/test leakage risk.

## Existing Experimental Results

The current debug experiment uses 20 product categories and at most 50 images per category.

| Model | Training Strategy | Recall@1 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|
| ResNet18 | Supervised training on `articleType` | 0.8792 | 0.9463 | 0.9732 |
| CLIP ViT-B/32 | Frozen pretrained image encoder | 0.8792 | 0.9664 | 0.9866 |

The summary CSV is stored at `outputs/reports/experiment_summary.csv`.

## Current Features

- End-to-end notebook report for product visual search and retrieval.
- ResNet18 debug supervised classification and embedding retrieval.
- CLIP ViT-B/32 frozen embedding retrieval using a local checkpoint.
- Full-dataset ResNet18 training, checkpointing, classification evaluation, gallery indexing, and retrieval evaluation.
- Streamlit product demo with model switching between CLIP and ResNet18 Full Dataset.
- Cached embedding indexes for faster interactive search.

## Full Dataset ResNet18 Results

The full split uses `data/raw/images/` as the canonical image source.

| Item | Value |
|---|---:|
| Valid images | 44,405 |
| articleType classes | 132 |
| Train size | 31,070 |
| Validation size | 6,666 |
| Test size | 6,669 |
| Test Top-1 accuracy | 0.8851 |
| Test Top-5 accuracy | 0.9892 |
| Retrieval Recall@1 | 0.8991 |
| Retrieval Recall@5 | 0.9351 |
| Retrieval Recall@10 | 0.9466 |
| Full gallery size | 31,070 |

## CLIP Full Split Retrieval

CLIP has also been synchronized to the same full train/test split used by ResNet18 Full Dataset:

| Item | Value |
|---|---:|
| Gallery split | train |
| Gallery size | 31,070 |
| Query split | test |
| Query size | 6,669 |
| Retrieval Recall@1 | 0.8283 |
| Retrieval Recall@5 | 0.9541 |
| Retrieval Recall@10 | 0.9724 |

The full CLIP gallery index is:

```text
outputs/indexes/clip_full_gallery_index.npz
```

The formal full-dataset checkpoint is:

```text
outputs/checkpoints/resnet18_full_best.pth
```

The formal full-dataset gallery index is:

```text
outputs/indexes/resnet18_full_gallery_index.npz
```

## Product Demo

The Streamlit app provides an interactive visual search demo:

- Upload a product image.
- Or randomly select a query image from the debug query set.
- Choose Top-K retrieval size: 5, 10, or 20.
- Search the cached debug gallery index.
- View retrieved product images, `articleType`, similarity score, and image id.
- Optionally view the ResNet18 vs CLIP comparison table and Recall@K chart.

The app does not train models, does not run the notebook, does not download new models, and does not process the full 44,441-image dataset by default.

## Product Demo Overview

`app.py` supports model switching between:

- `CLIP ViT-B/32`: uses the local CLIP checkpoint and the debug gallery index.
- `ResNet18 Full Dataset`: uses a full-dataset supervised ResNet18 checkpoint and full train-gallery index after they are created.

If the ResNet18 full checkpoint or index does not exist, the app shows a clear message and keeps CLIP available. The app never trains ResNet18 from the browser.

## CLIP Local Checkpoint

The Streamlit demo uses the local CLIP checkpoint by default:

```text
models/huggingface/timm_vit_base_patch32_clip_224_openai/open_clip_model.safetensors
```

If this file exists, the demo does not need to download the model again.

## Gallery Index Cache

The app uses a cached debug gallery embedding index:

```text
outputs/indexes/clip_gallery_debug_index.npz
```

If the cache exists, the app loads it directly. If it does not exist, use the sidebar button `Build / refresh debug CLIP index`. This builds only the debug gallery index, not the full dataset.

## How to Run the Streamlit Demo

Use the project Python environment:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

If Streamlit is missing:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" -m pip install streamlit
```

## Full ResNet18 Training Pipeline

The ResNet18 full-dataset pipeline is intentionally implemented as standalone scripts, not notebook cells. This avoids freezing the notebook and makes checkpoint/resume safer.

1. Prepare the full split:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" .\scripts\prepare_full_dataset_split.py
```

2. Train ResNet18 on the full split:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" .\scripts\train_resnet18_full.py --epochs 10
```

3. Resume interrupted training:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" .\scripts\train_resnet18_full.py --resume
```

4. Evaluate classification performance:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" .\scripts\evaluate_resnet18_full.py
```

5. Build the ResNet18 full gallery index:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" .\scripts\build_resnet18_full_index.py
```

6. Evaluate ResNet18 retrieval:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" .\scripts\evaluate_resnet18_full_retrieval.py
```

7. Start the Streamlit app:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

8. Generate the full report:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" .\scripts\generate_full_report_docx.py
```

## Smoke Tests

Before full training, use small smoke tests:

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" .\scripts\train_resnet18_full.py --epochs 1 --batch-size 16 --max-train-batches 5 --max-val-batches 2 --num-workers 0
```

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" .\scripts\build_resnet18_full_index.py --max-images 64
```

```powershell
& "C:\Users\16611\AppData\Local\Programs\Python\Python313\python.exe" .\scripts\evaluate_resnet18_full_retrieval.py --max-query-images 64
```

Smoke-test indexes are marked as `smoke train gallery`, so the Streamlit app will not present them as completed full-dataset indexes.

## Model Switching in Streamlit

The CLIP mode uses:

```text
outputs/indexes/clip_gallery_debug_index.npz
```

The ResNet18 Full Dataset mode uses:

```text
outputs/checkpoints/resnet18_full_best.pth
outputs/indexes/resnet18_full_gallery_index.npz
```

The Streamlit app now uses the shared full split for both CLIP and ResNet18 Full Dataset. Random sample queries come from the full test split, while both models search train-gallery indexes. This prevents a query image from being treated as test for one model and gallery for the other model.

## File Structure

```text
product_visual_search_retrieval.ipynb
app.py
README.md
scripts/
  prepare_full_dataset_split.py
  train_resnet18_full.py
  evaluate_resnet18_full.py
  build_resnet18_full_index.py
  evaluate_resnet18_full_retrieval.py
  generate_full_result_figures.py
  generate_full_report_docx.py
  update_notebook_full_resnet_summary.py
src/product_search/
  config.py
  data_utils.py
  clip_engine.py
  resnet_engine.py
  search_engine.py
  index_utils.py
outputs/checkpoints/
outputs/indexes/
outputs/reports/
outputs/figures/
```

## Safe Run Instructions

- Do not use Run All for full training.
- Do not run full training inside the notebook.
- Full ResNet18 training may take tens of minutes to several hours depending on GPU, disk speed, and batch size.
- If CUDA runs out of memory, rerun with a smaller batch size such as `--batch-size 16`.
- CLIP is not trained or fine-tuned by this project.
- The notebook should only read completed result files and summarize them.
- `experiment_summary.csv` keeps existing debug results and is updated only by evaluation scripts.

## Troubleshooting

- If `num_workers` causes issues on Windows, pass `--num-workers 0`.
- If ResNet18 mode is unavailable in the app, first check that `resnet18_full_best.pth` and `resnet18_full_gallery_index.npz` exist.
- If the app says the ResNet18 index is a smoke index, run `build_resnet18_full_index.py` without `--max-images` after full training.
- If torchvision tries to download ImageNet weights and network is unavailable, rerun after ensuring the pretrained ResNet18 weights are cached or internet access is available.

## Useful Scripts

- `clip_smoke_test.py`: validates CLIP loading and embedding extraction on at most 16 images.
- `clip_debug_retrieval.py`: runs CLIP debug retrieval on the debug subset only.
- `download_clip_weights.py`: downloads the CLIP checkpoint into the local model directory.
- `check_clip_local_cache.py`: checks whether local Hugging Face checkpoint files exist.
- `generate_report_docx.py`: generates the course report from existing CSV and figures.

## Safety Notes

Do not use Run All unless intentionally rerunning the full notebook. For the product demo, use `app.py` and the cached debug index. The Streamlit app does not modify `outputs/reports/experiment_summary.csv`.
