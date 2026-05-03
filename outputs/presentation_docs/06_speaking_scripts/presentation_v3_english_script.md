# Presentation V3 English Speech Script

Estimated length: about 10 minutes  
Structure: 5 parts  
Suggested matching deck: `01_html_exports/presentation_v3.html`

## Part 1. Problem Setting, Task Definition, and Dataset

Slides: 1-5 in `presentation_v3.html`

Good morning everyone. Today I am presenting our course project on Product Visual Search. The purpose of this project is to build a fashion retrieval system that allows users to search for visually similar products using an image rather than only using text.

This is useful in real e-commerce settings because users often do not know the exact keyword they need. They may only have a screenshot or a product photo. So the retrieval problem here is very practical: the input is a query image, the system searches a gallery of catalog images, and the output is a ranked top-k list of visually similar products.

To make the task measurable, we define relevance mainly by `articleType`. In other words, if the query is a T-shirt, then other T-shirts are treated as relevant matches. This is a category-level definition of similarity, which is practical and stable for a course project, even though it is not yet fine-grained enough to capture every visual style detail.

We used the Kaggle fashion-product-images-small dataset. After cleaning invalid images and filtering rare classes, the final split contained 31,070 training images, 6,666 validation images, and 6,669 test images, across 132 valid `articleType` classes.

## Part 2. Pipeline, Implementation, and Experiment Completeness

Slides: 6-8 in `presentation_v3.html`

The full pipeline has four main stages. First, we load `styles.csv`, match metadata with image files, filter bad examples, and create the train, validation, and test split. Second, we either train a model or extract features from a pretrained model. Third, we encode gallery images into normalized embeddings and save the index. Fourth, we encode test queries, compare them with the gallery, and evaluate retrieval performance.

In implementation terms, the project is organized into reusable engines and pipeline scripts. For example, we have separate modules for data utilities, the ResNet engine, the frozen CLIP engine, the fine-tuned CLIP engine, indexing utilities, and retrieval logic. This structure makes the project easier to reuse and easier to explain.

It is also important that all three experiment routes completed the full loop. The trained models completed training and classification evaluation, all routes completed index construction and retrieval evaluation, and the final output package contained the expected metric files and checkpoints. So the project is not only conceptually complete, but also experimentally complete.

## Part 3. Why We Compared Three Routes and How We Evaluated Them

Slides: 9-13 in `presentation_v3.html`

The project compares three model routes because they represent three different levels of training.

The first route is ResNet18, a classic supervised CNN baseline. It is light enough for course experiments, directly connected to supervised visual learning, and strong enough to serve as a serious baseline.

The second route is frozen CLIP ViT-B/32. This route asks whether large-scale image-text pretraining already gives strong retrieval features without any task-specific training on our fashion dataset.

The third route is fine-tuned CLIP. Here, we adapt only a small part of the visual encoder plus a classification head. The goal is to test whether lightweight supervised domain adaptation can improve retrieval performance while keeping training cost manageable.

To evaluate these routes, we use both classification metrics and retrieval metrics. For classification, we report Top-1 and Top-5 accuracy for the trained models. For retrieval, we report Recall at 1, Recall at 5, and Recall at 10. Recall at 1 is especially important because it tells us whether the first returned item already matches the query's `articleType`. Recall at 5 and Recall at 10 tell us whether the model can place at least one relevant item inside a broader candidate set.

## Part 4. Main Results and What They Mean

Slides: 14-18 in `presentation_v3.html`

Now let me summarize the main quantitative findings.

ResNet18 achieved 89.47 percent Top-1 classification accuracy, 98.80 percent Top-5 classification accuracy, and 90.16 percent Recall at 1. Frozen CLIP achieved 82.89 percent Recall at 1, 95.43 percent Recall at 5, and 97.26 percent Recall at 10. Fine-tuned CLIP achieved 86.67 percent Top-1 classification accuracy, 98.47 percent Top-5 classification accuracy, and 86.13 percent Recall at 1, with 94.12 percent Recall at 5 and 95.74 percent Recall at 10.

These results show a very clear tradeoff. ResNet18 is the best model when we care most about the first returned result. Frozen CLIP is the best model when we care more about broad top-k semantic recall. Fine-tuned CLIP improves first-hit retrieval compared with frozen CLIP, but slightly reduces Recall at 5 and Recall at 10.

This means that fine-tuning makes CLIP more domain-specific. In fact, its Recall at 1 improves by 3.24 percentage points over frozen CLIP, from 82.89 percent to 86.13 percent. However, Recall at 5 drops by 1.31 points and Recall at 10 drops by 1.52 points. So the adaptation helps the model rank the best category match earlier, but it slightly weakens broader semantic coverage.

We also examined the ResNet18 training curves. The final epoch showed very low train loss and noticeably higher validation loss, which suggests some overfitting. Even so, the final test and retrieval metrics remained strong. And finally, the qualitative retrieval example shows that a T-shirt query returns other T-shirts with high similarity scores, which makes the system behavior easy to interpret.

## Part 5. Limitations, Future Work, and Final Takeaway

Slides: 19-20 in `presentation_v3.html`

There are still several limitations in the current system. The first is that relevance is mainly defined by `articleType`, so the evaluation is category-level rather than fine-grained style-level retrieval. The second is that the dataset is long-tailed, so some classes have much fewer training examples. The third is that the system is evaluated offline, without user feedback signals. And the fourth is that the current index uses direct embedding similarity instead of a scalable approximate nearest neighbor backend.

These limitations also suggest clear future work. We could add richer labels that combine product type, color, gender, and usage. We could test retrieval-oriented metric learning losses such as contrastive loss or triplet loss. We could also extend the system toward hybrid text-and-image retrieval using CLIP, and we could make the retrieval system more scalable with approximate nearest neighbor search and metadata reranking.

To conclude, this project delivers both a working visual search pipeline and a clear empirical comparison. ResNet18 gives the strongest first-rank retrieval accuracy. Frozen CLIP gives the strongest broad top-k recall. Fine-tuned CLIP demonstrates that lightweight supervised domain adaptation can improve pretrained first-hit retrieval performance. So the final answer is not that one model is universally best. The best choice depends on whether we value precise first-hit ranking or broader semantic coverage in the candidate set.
