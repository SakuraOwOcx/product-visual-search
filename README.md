# Product Visual Search and Retrieval

This course project builds a product visual search system for fashion products. A query image is encoded into an embedding, compared with gallery product embeddings, and used to retrieve the Top-K most similar products.

The final submission compares three model routes:

- `ResNet18 supervised`: a self-trained CNN baseline using `articleType` labels.
- `Frozen CLIP ViT-B/32`: a pretrained CLIP image encoder used without task-specific training.
- `Fine-tuned CLIP ViT-B/32`: a lightweight supervised adaptation of CLIP using `articleType` labels.

The full Colab pipeline is:

```text
colab_all_models_full_pipeline.ipynb
```

## Dataset

Dataset used during the project:

- Kaggle Fashion Product Images Small
- Source: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
- Expected image folder: `data/raw/images/`
- Expected metadata file: `data/raw/styles.csv`
- Main label: `articleType`

The dataset is not included in the GitHub repository because the image zip is too large for normal GitHub tracking. Download it from Kaggle or use the local assignment asset:

```text
submission_assets/archives/fashion-product-images-small.zip
```

## Project Pipeline

The project follows this workflow:

1. Prepare a clean full dataset split from `styles.csv` and image files.
2. Train ResNet18 on `articleType` classification.
3. Build a ResNet18 train-gallery embedding index.
4. Evaluate ResNet18 retrieval on the test split.
5. Build a frozen CLIP train-gallery embedding index.
6. Evaluate frozen CLIP retrieval on the test split.
7. Fine-tune CLIP's image encoder with `articleType` supervision.
8. Build a fine-tuned CLIP train-gallery index.
9. Compare all models using classification and retrieval metrics.

## Final Dataset Split

| Split | Images |
|---|---:|
| Train | 31,070 |
| Validation | 6,666 |
| Test | 6,669 |

The final experiments use `132` valid `articleType` classes.

## Final Experimental Results

| Model | Classification Top-1 | Classification Top-5 | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| ResNet18 supervised | 0.8947 | 0.9880 | 0.9016 | 0.9393 | 0.9498 |
| Frozen CLIP ViT-B/32 | N/A | N/A | 0.8289 | 0.9543 | 0.9726 |
| Fine-tuned CLIP ViT-B/32 | 0.8667 | 0.9847 | 0.8613 | 0.9412 | 0.9574 |

Summary:

- ResNet18 has the best Recall@1, so it is strongest for first-result precision.
- Frozen CLIP has the best Recall@5 and Recall@10, so it is strongest for broad top-k recall.
- Fine-tuned CLIP improves Recall@1 over frozen CLIP, showing that lightweight domain adaptation helps CLIP rank the best match higher.

Detailed result notes are available in:

```text
MODEL_RESULTS_SUMMARY.md
outputs/presentation_docs/MODEL_RESULTS_SUMMARY.md
outputs/presentation_docs/PROJECT_FRAMEWORK.md
```

## Model Design

### ResNet18 supervised

ResNet18 is trained with cross-entropy loss on `articleType`. After training, the feature vector before the final classification layer is used as the retrieval embedding.

This route represents a traditional supervised CNN approach and satisfies the requirement of training a model on the project dataset.

### Frozen CLIP ViT-B/32

Frozen CLIP uses pretrained OpenAI CLIP ViT-B/32 weights. The model is not trained on this dataset. It is used as a strong pretrained retrieval baseline.

The local CLIP checkpoint path expected by the scripts is:

```text
models/huggingface/timm_vit_base_patch32_clip_224_openai/open_clip_model.safetensors
```

### Fine-tuned CLIP ViT-B/32

Fine-tuned CLIP starts from the same OpenAI CLIP ViT-B/32 pretrained weights. The project freezes most CLIP parameters and trains only a small part of the visual encoder plus a classification head using `articleType` supervision.

This route tests whether lightweight supervised adaptation can improve CLIP on the fashion product domain.

## Key Files

```text
app.py
product_visual_search_retrieval.ipynb
colab_all_models_full_pipeline.ipynb
scripts/
src/product_search/
outputs/reports/
outputs/figures/
outputs/presentation_docs/
```

Important result files:

```text
outputs/reports/experiment_summary.csv
outputs/reports/resnet18_full_test_metrics.json
outputs/reports/resnet18_full_retrieval_metrics.json
outputs/reports/clip_full_retrieval_metrics.json
outputs/reports/clip_articletype_test_metrics.json
outputs/reports/clip_articletype_retrieval_metrics.json
```

## Run the Full Colab Pipeline

Upload these files to Google Drive:

```text
MyDrive/product_visual_search_v2/project.zip
MyDrive/product_visual_search_v2/fashion-product-images-small.zip
```

Then open:

```text
colab_all_models_full_pipeline.ipynb
```

Use a GPU runtime:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

The notebook prepares the dataset, runs all three model routes, evaluates retrieval, and backs up final outputs to Google Drive.

## Run Locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

Prepare the full split:

```powershell
python scripts/prepare_full_dataset_split.py
```

Train and evaluate ResNet18:

```powershell
python scripts/train_resnet18_full.py --epochs 20
python scripts/evaluate_resnet18_full.py
python scripts/build_resnet18_full_index.py
python scripts/evaluate_resnet18_full_retrieval.py
```

Evaluate frozen CLIP:

```powershell
python scripts/build_clip_full_index.py
python scripts/evaluate_clip_full_retrieval.py
```

Train and evaluate fine-tuned CLIP:

```powershell
python scripts/train_clip_articletype.py --epochs 20 --unfreeze-visual-blocks 1
python scripts/evaluate_clip_articletype.py
python scripts/build_clip_articletype_index.py
python scripts/evaluate_clip_articletype_retrieval.py
```

## Streamlit Demo

The Streamlit app provides an interactive product visual search demo:

```powershell
streamlit run app.py
```

The app supports model switching where the required checkpoints and indexes exist locally.

## Assignment Assets

Large assignment files are kept locally under:

```text
submission_assets/
```

This includes:

- `project.zip`
- `fashion-product-images-small.zip`
- final output zip files
- Colab result PDF

These files are intentionally ignored by Git because they are too large for normal GitHub tracking.

## Limitations

- Retrieval correctness is mainly based on matching `articleType`, so the evaluation is category-level rather than fine-grained visual similarity.
- The dataset is long-tailed; some product categories have very few samples.
- Fine-tuned CLIP improves Recall@1 over frozen CLIP but does not outperform ResNet18 on Recall@1.
- Frozen CLIP keeps stronger Recall@5 and Recall@10, suggesting a tradeoff between domain-specific discrimination and broad semantic coverage.

