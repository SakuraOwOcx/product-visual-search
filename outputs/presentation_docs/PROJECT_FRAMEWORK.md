# Product Visual Search Project Framework

## 1. Project Motivation

This project builds a product visual search system for fashion e-commerce. The goal is to allow users to search for visually similar products by image, rather than relying only on text keywords.

In a real shopping scenario, users may not know the exact product name or category. They may only have a reference image, such as a pair of shoes, a handbag, or a shirt. A visual retrieval system can convert the query image into a feature vector and then search for the most similar products in a gallery.

The project is therefore framed as an image retrieval task:

- Input: a query product image.
- Gallery: a collection of product images.
- Output: the top-k most similar product images.

The main evaluation target is whether retrieved products have the same `articleType` as the query image.

## 2. Dataset and Task Definition

The project uses the Kaggle `fashion-product-images-small` dataset. The dataset contains product images and metadata stored in `styles.csv`.

Important metadata fields include:

- `id`
- `gender`
- `masterCategory`
- `subCategory`
- `articleType`
- `baseColour`
- `season`
- `usage`
- `productDisplayName`

For this project, `articleType` is used as the main supervised label because it provides a practical category-level definition of product similarity. For example, products with the same `articleType`, such as `Tshirts`, `Watches`, or `Handbags`, are treated as relevant retrieval matches.

The processed full dataset split contains:

| Split | Number of images |
|---|---:|
| Train | 31,070 |
| Validation | 6,666 |
| Test | 6,669 |

There are `132` valid `articleType` classes after filtering rare classes and invalid images.

## 3. Overall Pipeline

The system follows a standard visual retrieval pipeline:

1. Data preparation
   - Load `styles.csv`.
   - Match metadata rows with image files.
   - Filter invalid images and very rare classes.
   - Create train, validation, and test splits.

2. Model training or feature extraction
   - Train supervised visual models on `articleType`.
   - Use pretrained CLIP as a frozen feature extractor.
   - Fine-tune CLIP using the product dataset.

3. Embedding index construction
   - Encode gallery images into fixed-length feature vectors.
   - Normalize embeddings.
   - Store embeddings and metadata in index files.

4. Retrieval evaluation
   - Encode test images as query embeddings.
   - Compute similarity between query and gallery embeddings.
   - Evaluate Recall@1, Recall@5, and Recall@10.

5. Result comparison
   - Compare supervised ResNet18, frozen CLIP, and fine-tuned CLIP.
   - Analyze tradeoffs between first-hit accuracy and broad top-k recall.

## 4. Model Design

The project compares three model routes.

### 4.1 ResNet18 Supervised Baseline

The first model is a supervised ResNet18 trained on `articleType` classification.

Reason for using ResNet18:

- It is a classic CNN architecture.
- It is lightweight enough for course-level experiments.
- It provides a strong supervised baseline.
- It is directly connected to the course content on CNNs and supervised deep learning.

Training design:

- Input: product image.
- Target: `articleType` class.
- Loss: cross-entropy loss.
- Backbone: ImageNet-pretrained ResNet18.
- Output head: a classification layer for 132 classes.

After training, the final classification layer is not used directly for retrieval. Instead, the feature vector before the classifier is extracted as the product embedding. Similar products are retrieved by comparing these embeddings.

This model represents the traditional supervised visual learning approach.

### 4.2 Frozen CLIP ViT-B/32

The second model is frozen CLIP ViT-B/32.

Reason for using CLIP:

- CLIP is pretrained on large-scale image-text pairs.
- It provides strong general visual-semantic representations.
- It can be used without additional training.
- It serves as a strong pretrained baseline.

In this route, CLIP is not fine-tuned. The image encoder is used directly to extract embeddings for product images.

This model tests whether a large pretrained vision-language model can perform well on fashion product retrieval without task-specific training.

### 4.3 Fine-tuned CLIP with articleType Supervision

The third model fine-tunes the CLIP image encoder using `articleType` supervision.

Reason for this design:

- Frozen CLIP has strong general representation ability, but it is not specifically optimized for the fashion product dataset.
- Fully training a large model from scratch is expensive.
- Lightweight fine-tuning can adapt the pretrained model to the product domain while keeping the training cost manageable.

Training design:

- Initial model: OpenAI CLIP ViT-B/32 pretrained weights.
- Input: product image.
- Target: `articleType`.
- Loss: cross-entropy loss.
- Trainable parameters: classification head and a small part of the visual encoder.
- Most CLIP parameters are frozen.

This route is designed to answer the question:

Can supervised domain adaptation improve CLIP's retrieval performance on fashion products?

## 5. Why These Three Models

The three-model design gives the project a clear experimental structure.

ResNet18 answers:

How well can a traditional supervised CNN learn product visual features from this dataset?

Frozen CLIP answers:

How strong is a large pretrained model without any task-specific training?

Fine-tuned CLIP answers:

Can lightweight training adapt a pretrained model to the fashion product domain?

This comparison is useful because the models represent three different levels of training:

| Model | Training style | Role |
|---|---|---|
| ResNet18 | Supervised training on product labels | Main self-trained baseline |
| Frozen CLIP | No training on this dataset | Pretrained baseline |
| Fine-tuned CLIP | Lightweight supervised adaptation | Domain-adapted pretrained model |

This design also fits the course requirement that the project should include model training, because both ResNet18 and fine-tuned CLIP involve training on the project dataset.

## 6. Evaluation Metrics

The project uses both classification and retrieval metrics.

Classification metrics:

- Top-1 accuracy
- Top-5 accuracy

These metrics evaluate whether the trained classifier can correctly predict the product `articleType`.

Retrieval metrics:

- Recall@1
- Recall@5
- Recall@10

These metrics evaluate whether at least one product with the same `articleType` appears in the top-k retrieved results.

Recall@1 focuses on first-rank precision. Recall@5 and Recall@10 evaluate whether the model can retrieve relevant products within a broader candidate set.

## 7. Implementation Structure

The project is organized into reusable scripts and modules.

Main modules:

- `src/product_search/data_utils.py`: data loading and split preparation.
- `src/product_search/resnet_engine.py`: ResNet model loading and feature extraction.
- `src/product_search/clip_engine.py`: frozen CLIP loading and feature extraction.
- `src/product_search/clip_supervised_engine.py`: fine-tuned CLIP model wrapper.
- `src/product_search/index_utils.py`: index construction and loading utilities.
- `src/product_search/search_engine.py`: retrieval logic.

Main scripts:

- `scripts/prepare_full_dataset_split.py`
- `scripts/train_resnet18_full.py`
- `scripts/evaluate_resnet18_full.py`
- `scripts/build_resnet18_full_index.py`
- `scripts/evaluate_resnet18_full_retrieval.py`
- `scripts/build_clip_full_index.py`
- `scripts/evaluate_clip_full_retrieval.py`
- `scripts/train_clip_articletype.py`
- `scripts/evaluate_clip_articletype.py`
- `scripts/build_clip_articletype_index.py`
- `scripts/evaluate_clip_articletype_retrieval.py`

The full Colab pipeline is stored in:

- `colab_all_models_full_pipeline.ipynb`

## 8. Final Model Comparison Logic

The comparison is not just about selecting one winner. Each model shows a different strength.

ResNet18 is strongest at Recall@1, meaning it is best at returning the correct class as the first result.

Frozen CLIP is strongest at Recall@5 and Recall@10, meaning it has better broad semantic recall.

Fine-tuned CLIP improves over frozen CLIP at Recall@1, showing that supervised fine-tuning helps adapt CLIP to the fashion product domain. However, it slightly reduces Recall@5 and Recall@10, suggesting a tradeoff between domain-specific discrimination and general semantic coverage.

## 9. Limitations

The project has several limitations:

- Retrieval correctness is mainly defined by `articleType`, so the evaluation is category-level rather than fine-grained visual similarity.
- Some product classes have many samples, while others are long-tail classes with very few images.
- Fine-tuned CLIP improves Recall@1 over frozen CLIP but does not outperform ResNet18 on the first-rank retrieval metric.
- The system is evaluated offline and does not include real user click feedback.
- The current retrieval index uses direct embedding similarity rather than a more scalable approximate nearest neighbor system.

## 10. Possible Future Work

Future improvements could include:

- Use multi-attribute retrieval labels, such as `articleType + color + gender + usage`.
- Add class-balanced sampling or long-tail reweighting.
- Try metric learning losses such as contrastive loss or triplet loss.
- Add text-query or image-text hybrid retrieval using CLIP.
- Use approximate nearest neighbor indexing for larger product catalogs.
- Add reranking based on metadata consistency.

## 11. Presentation Takeaway

The project successfully builds a complete visual product search system. It includes data preparation, supervised model training, pretrained model comparison, lightweight CLIP fine-tuning, embedding index construction, and retrieval evaluation.

The main conclusion is:

ResNet18 gives the best first-rank retrieval accuracy, frozen CLIP gives the best broad top-k recall, and fine-tuned CLIP demonstrates that lightweight domain adaptation can improve pretrained CLIP's first-rank retrieval performance on fashion products.

