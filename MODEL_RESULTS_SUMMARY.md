# Product Visual Search Model Results Summary

This summary is based on the final Colab output package in:

`D:\Codex\product-visual-search\ouputs_v3`

The final results are stored across these files:

- `outputs_all_models_final-20260502T065506Z-3-001.zip`
- `outputs_all_models_final-20260502T065506Z-3-002.zip`
- `colab_all_models_full_pipeline.ipynb - Colab.pdf`

Google Drive split the final `outputs_all_models_final` folder into two zip files. The result set is complete when both zip files are kept together.

## Completion Check

All three model routes were successfully executed:

| Model route | Training | Classification evaluation | Retrieval index | Retrieval evaluation | Checkpoint |
|---|---:|---:|---:|---:|---:|
| ResNet18 supervised | Yes | Yes | Yes | Yes | Yes |
| Frozen CLIP ViT-B/32 | Not applicable | Not applicable | Yes | Yes | Uses pretrained CLIP |
| Fine-tuned CLIP ViT-B/32 | Yes | Yes | Yes | Yes | Yes |

The Colab notebook also reported that the key files all existed:

- `experiment_summary.csv`
- `resnet18_full_test_metrics.json`
- `resnet18_full_retrieval_metrics.json`
- `clip_full_retrieval_metrics.json`
- `clip_articletype_test_metrics.json`
- `clip_articletype_retrieval_metrics.json`
- `resnet18_full_best.pth`
- `clip_articletype_best.pth`

## Dataset Split

The full dataset split used in this run contains:

| Split | Images |
|---|---:|
| Train | 31,070 |
| Validation | 6,666 |
| Test | 6,669 |

The classification task contains `132` `articleType` classes. Retrieval uses the train split as the gallery and the test split as queries.

## Main Results

| Model | Classification Top-1 | Classification Top-5 | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| ResNet18 supervised | 0.8947 | 0.9880 | 0.9016 | 0.9393 | 0.9498 |
| Frozen CLIP ViT-B/32 | N/A | N/A | 0.8289 | 0.9543 | 0.9726 |
| Fine-tuned CLIP ViT-B/32 | 0.8667 | 0.9847 | 0.8613 | 0.9412 | 0.9574 |

## Model-by-Model Analysis

### ResNet18 Supervised

The ResNet18 model was trained with `articleType` supervision on the full training set. It achieved the strongest first-rank retrieval performance:

- Classification Top-1: `89.47%`
- Classification Top-5: `98.80%`
- Recall@1: `90.16%`
- Recall@5: `93.93%`
- Recall@10: `94.98%`

This indicates that the supervised ResNet18 features are highly effective for precise category-level retrieval. If the search system prioritizes returning the correct class as the first result, ResNet18 is the strongest model among the three.

The final ResNet18 training epoch reported:

- Train loss: `0.0220`
- Validation loss: `0.5149`
- Validation Top-1: `0.8918`
- Validation Top-5: `0.9889`

The low train loss and higher validation loss suggest some overfitting after longer training, but the final test and retrieval scores are still strong.

### Frozen CLIP ViT-B/32

Frozen CLIP uses the OpenAI CLIP ViT-B/32 pretrained weights without task-specific training. It does not produce classification accuracy because it is evaluated as an embedding-based retrieval model.

Frozen CLIP achieved:

- Recall@1: `82.89%`
- Recall@5: `95.43%`
- Recall@10: `97.26%`

Compared with ResNet18, frozen CLIP is weaker at Recall@1 but stronger at Recall@5 and Recall@10. This suggests that CLIP is less precise at ranking the single best match first, but it has better broad semantic recall in the top candidate set.

### Fine-tuned CLIP ViT-B/32

The fine-tuned CLIP model starts from the same OpenAI CLIP ViT-B/32 pretrained weights, then adapts the visual encoder with `articleType` supervision. The training is lightweight: only a small part of the visual encoder and the classification head are trainable.

Fine-tuned CLIP achieved:

- Classification Top-1: `86.67%`
- Classification Top-5: `98.47%`
- Recall@1: `86.13%`
- Recall@5: `94.12%`
- Recall@10: `95.74%`

Compared with frozen CLIP, fine-tuning improved Recall@1 from `82.89%` to `86.13%`. This shows that supervised adaptation on the fashion product dataset made CLIP more domain-specific and improved its ability to rank the best match first.

However, Recall@5 and Recall@10 decreased compared with frozen CLIP. This suggests a tradeoff: fine-tuning makes CLIP more discriminative for `articleType`, but slightly reduces the broader semantic coverage that frozen CLIP provides.

The final fine-tuned CLIP training epoch reported:

- Train loss: `0.1838`
- Validation loss: `0.5708`
- Validation Top-1: `0.8615`
- Validation Top-5: `0.9842`

The best checkpoint was saved as `clip_articletype_best.pth`.

## Overall Comparison

The three models show different strengths:

| Use case | Best model |
|---|---|
| Best first result | ResNet18 supervised |
| Best broad Top-5 / Top-10 recall | Frozen CLIP |
| Best evidence of domain adaptation through training | Fine-tuned CLIP |

The ResNet18 model performs best for precise category-level retrieval, especially Recall@1. Frozen CLIP performs best when the evaluation allows a wider candidate set, showing the benefit of large-scale visual-language pretraining. Fine-tuned CLIP improves the first-rank accuracy of frozen CLIP and demonstrates that lightweight supervised adaptation can make a pretrained model more suitable for the fashion product domain.

## Notes on Output Completeness

The final output is complete, but it is split across two zip files:

- `outputs_all_models_final-20260502T065506Z-3-001.zip` contains most reports, figures, CLIP fine-tuned checkpoints, CLIP indexes, and metric JSON files.
- `outputs_all_models_final-20260502T065506Z-3-002.zip` contains the remaining large files, including ResNet checkpoints and the ResNet gallery index.

Both zip files should be kept. The small `outputs_after_resnet` package is an intermediate backup after the ResNet stage, not the final all-model output.

One minor caveat: `resnet18_full_training_log.csv` contains previous ResNet training records as well as this latest 20-epoch run. The final JSON metrics and checkpoint files are the reliable source for the final reported ResNet results.

