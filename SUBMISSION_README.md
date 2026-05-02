# Submission Package Guide

This folder is a cleaned submission copy of the project. The original working folder remains at:

`D:\Codex\product-visual-search`

## Main Project Files

- `README.md`: project overview and setup instructions.
- `app.py`: Streamlit demo app.
- `src/`: reusable project code.
- `scripts/`: data preparation, training, indexing, and evaluation scripts.
- `product_visual_search_retrieval.ipynb`: original project notebook.
- `colab_all_models_full_pipeline.ipynb`: runnable Colab pipeline for the full model comparison.
- `MODEL_RESULTS_SUMMARY.md`: concise summary of the latest model results.

## Latest Lightweight Results

The latest lightweight result files are stored under:

`outputs/`

Important files include:

- `outputs/reports/experiment_summary.csv`
- `outputs/reports/resnet18_full_test_metrics.json`
- `outputs/reports/resnet18_full_retrieval_metrics.json`
- `outputs/reports/clip_full_retrieval_metrics.json`
- `outputs/reports/clip_articletype_test_metrics.json`
- `outputs/reports/clip_articletype_retrieval_metrics.json`
- `outputs/reports/clip_articletype_training_log.csv`
- `outputs/reports/resnet18_full_training_log.csv`
- `outputs/splits/full_class_mapping.json`
- `outputs/splits/full_dataset_split.csv`

## Assignment Assets

Large files that may be needed for assignment submission are stored under:

`submission_assets/`

This includes:

- `submission_assets/archives/project.zip`
- `submission_assets/archives/fashion-product-images-small.zip`
- `submission_assets/final_outputs_zips/outputs_all_models_final-20260502T065506Z-3-001.zip`
- `submission_assets/final_outputs_zips/outputs_all_models_final-20260502T065506Z-3-002.zip`
- `submission_assets/colab_all_models_full_pipeline_result.pdf`

The final output was split into two zip files by Google Drive. Keep both zip files together.

The most complete saved execution evidence is `submission_assets/colab_all_models_full_pipeline_result.pdf` plus the two final output zip files. The notebook in the repository should be treated as the reproducible pipeline file.

## GitHub Note

The large zip files, dataset, checkpoints, and indexes are intentionally ignored by `.gitignore`. They are kept locally for assignment submission, but should not be pushed to GitHub.

For GitHub, the useful tracked content should be the source code, notebooks, lightweight reports, and result summaries.
