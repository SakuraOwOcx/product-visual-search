# Product Visual Search and Retrieval Using ResNet18 and CLIP

Course Project Report  
Date: April 26, 2026  
Author: Student Name

## Abstract

This project implements an end-to-end product visual search pipeline. It includes a supervised ResNet18 debug baseline, a frozen CLIP ViT-B/32 retrieval model, a full-dataset ResNet18 training pipeline, synchronized full-split CLIP retrieval, and a Streamlit product demo. Retrieval is evaluated with Recall@K because product search returns ranked candidate products rather than only one class label.

The full ResNet18 model reaches test Top-1 accuracy of 0.8851 and test Top-5 accuracy of 0.9892. In full retrieval, ResNet18 achieves Recall@1 / Recall@5 / Recall@10 of 0.8991 / 0.9351 / 0.9466. The full CLIP index uses the same full train/test split, allowing the demo to compare both models without query/gallery mismatch.

## Introduction

Product visual search allows users to find visually similar products from images instead of keywords. This is important for e-commerce, social shopping, and content platforms because users often know what a product looks like before they know its exact name. A useful system should return a ranked set of visually similar products, not only a category label.

This project therefore combines classification and retrieval. Classification validates whether a model understands product categories. Retrieval tests whether the learned representation can support search and recommendation. The main technical idea is to convert every product image into an embedding vector, compare embeddings with cosine similarity, and return the Top-K nearest gallery products.

## Dataset and Preprocessing

- Valid full dataset images: 44405
- Number of articleType classes: 132
- Train / val / test: 31070 / 6666 / 6669
- Dropped rare classes: 10
- Main image source: `data/raw/images/`

The project avoids the duplicate extracted folder under `data/raw/myntradataset/images/` to reduce train/test leakage. Rare classes with fewer than three valid images are removed before splitting. Train images are used as the gallery; test images are used as query images.

## Methodology

ResNet18 is used as a supervised baseline trained on articleType labels. Its final classification layer is replaced for the dataset classes, and the penultimate 512-dimensional feature vector is used as the retrieval embedding.

CLIP is used as a frozen pretrained visual-semantic embedding model. It is not fine-tuned on the local dataset. This provides a useful comparison between a dataset-specific supervised model and a general pretrained foundation model.

Both methods convert images into embeddings and use cosine similarity for Top-K product retrieval. Recall@K measures whether a same-articleType item appears in the first K retrieved products.

## Full ResNet18 Results

- Best epoch: 10
- Best val Top-1: 0.8843
- Best val Top-5: 0.9880
- Test loss: 0.4571
- Test Top-1: 0.8851
- Test Top-5: 0.9892
- Full Recall@1: 0.8991
- Full Recall@5: 0.9351
- Full Recall@10: 0.9466
- Full gallery size: 31070

## Comparison

| Model | Classes | Train/Gallery | Test/Query | Top-1 | Top-5 | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 debug | 20 | 725 | 149 | 0.8792 | 0.9866 | 0.8792 | 0.9463 | 0.9732 |
| CLIP debug | 20 | 725 | 149 | N/A | N/A | 0.8792 | 0.9664 | 0.9866 |
| CLIP full dataset | 132 | 31070 | 6669 | N/A | N/A | 0.8283 | 0.9541 | 0.9724 |
| ResNet18 full dataset | 132 | 31070 | 6669 | 0.8851 | 0.9892 | 0.8991 | 0.9351 | 0.9466 |

## Product Demo

The Streamlit demo supports image upload, random query selection, Top-K retrieval, and model switching between CLIP ViT-B/32 and ResNet18 Full Dataset. Both models now use the same full split in the app: test images are sampled as queries and train images form the gallery. CLIP uses the local checkpoint and full CLIP train-gallery index. ResNet18 uses the full supervised checkpoint and full train-gallery index.

This synchronized split is important for a fair demo. Previously, a random query could be a test image for one model but a gallery image for another model. The current design avoids that confusion and makes model switching easier to explain.

## Discussion

ResNet18 is strong at dataset-specific articleType recognition because it is trained directly on the target labels. CLIP is strong at Top-K recall because its pretrained representation captures broader visual-semantic relationships. In a product search business scenario, CLIP can serve as a strong recall model, while ResNet18 can serve as a lightweight supervised baseline or a category-aware model.

## Business Interpretation

The system can support image-based product discovery, similar item recommendation, merchant listing support, and duplicate detection. In a production system, the embedding model would likely act as the first-stage retrieval model, while a ranking layer would combine visual similarity with product metadata such as price, brand, color, inventory, and user preferences.

## Limitations and Future Work

The current relevance metric is articleType-level rather than exact SKU-level similarity. The system does not yet use FAISS, metadata ranking, or a production backend. Future work should add approximate nearest-neighbor indexing, fine-tune CLIP, include product metadata ranking, evaluate SKU-level retrieval when labels are available, and deploy a production web service.
