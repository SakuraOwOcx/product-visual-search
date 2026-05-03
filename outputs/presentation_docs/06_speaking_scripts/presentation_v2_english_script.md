# Presentation V2 English Speech Script

Estimated length: about 9 to 10 minutes  
Structure: 5 parts  
Suggested matching deck: `01_html_exports/presentation_v2.html`

## Part 1. Introduction and Motivation

Slides: 1-2 in `presentation_v2.html`

Good morning everyone. Today I will present our course project, Product Visual Search. The main goal of this project is to build a retrieval system for fashion e-commerce, where a user can search for visually similar products using an image instead of text.

This problem is important because in real shopping scenarios, users often do not know the exact product name, category, or keyword. What they may have is only a reference image, such as a T-shirt, a handbag, or a pair of shoes. A visual search system should therefore be able to take that image, convert it into a meaningful feature representation, compare it with a gallery of product images, and return the most similar items.

So the core idea of our project is very simple: the input is a query image, the system searches a gallery, and the output is a ranked top-k list of visually similar products. That is the practical motivation behind the entire pipeline.

## Part 2. Dataset and Overall Pipeline

Slides: 3-4 in `presentation_v2.html`

For this project, we used the Kaggle fashion-product-images-small dataset. The metadata is stored in `styles.csv`, and it includes information such as product ID, gender, master category, subcategory, article type, color, season, usage, and display name.

Among these fields, we selected `articleType` as the main supervised label. We made this choice because it gives us a practical category-level definition of retrieval relevance. For example, if the query image is a T-shirt, then other T-shirts are treated as relevant matches.

After cleaning the dataset, filtering invalid images, and removing very rare classes, our final split contained 31,070 training images, 6,666 validation images, and 6,669 test images, across 132 valid articleType classes.

The overall pipeline has four steps. First, we prepare the data and create the split. Second, we either train a model or extract features from a pretrained model. Third, we build the embedding index for the gallery images. Fourth, we evaluate retrieval by encoding test queries and measuring Recall at different top-k levels.

## Part 3. Model Design and Main Quantitative Results

Slides: 5-6 in `presentation_v2.html`

We compared three model routes.

The first route is a supervised ResNet18 baseline. It is ImageNet-pretrained, then trained on the 132-way `articleType` classification task. For retrieval, we use the feature vector before the classifier as the embedding.

The second route is frozen CLIP ViT-B/32. In this case, we do not fine-tune the model at all. We directly use the pretrained CLIP image encoder as a feature extractor and test whether a large pretrained model can already perform well on fashion retrieval.

The third route is fine-tuned CLIP. This model starts from the same CLIP pretrained weights, but we train a classification head and a small part of the visual encoder using `articleType` supervision. The goal is to test lightweight domain adaptation.

Now let me highlight the main results. ResNet18 achieved 89.47 percent Top-1 classification accuracy, 98.80 percent Top-5 classification accuracy, and 90.16 percent Recall at 1. Frozen CLIP achieved 82.89 percent Recall at 1, 95.43 percent Recall at 5, and 97.26 percent Recall at 10. Fine-tuned CLIP achieved 86.67 percent Top-1 classification accuracy, 98.47 percent Top-5 classification accuracy, and 86.13 percent Recall at 1.

## Part 4. Interpretation, Training Evidence, and Qualitative Example

Slides: 7-9 in `presentation_v2.html`

The most important conclusion is that there is no single universal winner. The answer depends on what ranking behavior we care about.

If we care most about the first returned result, then ResNet18 is the strongest model, because it has the best Recall at 1. That means it is best at returning the correct product type at the very top of the ranking.

If we care more about broader candidate coverage, then frozen CLIP is the strongest model, because it gives the best Recall at 5 and Recall at 10. This suggests that large-scale visual-semantic pretraining helps CLIP maintain broad semantic recall.

Fine-tuned CLIP sits between the other two. Compared with frozen CLIP, it improves Recall at 1 from 82.89 percent to 86.13 percent, which shows that supervised adaptation helps CLIP become more domain-specific. However, its Recall at 5 and Recall at 10 are slightly lower than frozen CLIP, which suggests a tradeoff between sharper category discrimination and broader semantic coverage.

We also looked at the training evidence for ResNet18. The final epoch reported very low train loss but higher validation loss, which suggests some overfitting. Even so, the final test and retrieval performance remained strong. Finally, our qualitative retrieval example shows that when the query is a T-shirt, the top returned items are also T-shirts with high similarity scores, which gives intuitive support to the quantitative results.

## Part 5. Limitations, Future Work, and Conclusion

Slides: 10-11 in `presentation_v2.html`

This project still has several limitations. First, retrieval correctness is mainly defined by `articleType`, so the evaluation is category-level rather than fine-grained visual similarity. Second, the dataset is long-tailed, so some classes have many more samples than others. Third, the current system is evaluated offline, without real user click feedback. Finally, the retrieval index is based on direct embedding similarity and does not yet use a more scalable approximate nearest neighbor method.

There are several clear directions for future work. We could use richer labels, such as product type combined with color, gender, and usage. We could also try retrieval-oriented metric learning losses, such as contrastive loss or triplet loss. Another extension would be hybrid text-and-image retrieval using CLIP. And for a larger catalog, we could add approximate nearest neighbor search and reranking with metadata.

To conclude, our project successfully builds a complete visual product search pipeline. It includes data preparation, model training, pretrained model comparison, lightweight CLIP adaptation, embedding index construction, and retrieval evaluation. The key takeaway is that ResNet18 gives the best first-rank retrieval accuracy, frozen CLIP gives the best broad top-k recall, and fine-tuned CLIP shows that lightweight domain adaptation can improve pretrained first-hit retrieval performance.
