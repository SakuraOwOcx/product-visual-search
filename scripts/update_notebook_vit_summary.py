import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "product_visual_search_retrieval.ipynb"
SECTION_TITLE = "Supervised Vision Transformer Baseline"


MARKDOWN = f"""# {SECTION_TITLE}

This section adds a third baseline to the product visual search project: a supervised Vision Transformer, or ViT-B/16. The goal is to make the model comparison more precise.

The project now contains three conceptually different baselines:

| Model | Architecture | Training strategy | Pretraining source | Fine-tuned locally? | Embedding type |
|---|---|---|---|---|---|
| ResNet18 supervised | CNN | Supervised articleType classification | ImageNet | Yes | Penultimate CNN feature |
| ViT-B/16 supervised | Vision Transformer | Supervised articleType classification | ImageNet | Yes | CLS token feature |
| CLIP ViT-B/32 frozen | Vision Transformer | Image-text contrastive pretraining | Large-scale image-text pairs | No | CLIP image embedding |

This design separates two important questions. ResNet18 supervised vs ViT supervised mainly compares CNN and Transformer architecture under the same local supervision. ViT supervised vs CLIP frozen mainly compares local supervised fine-tuning with large-scale contrastive image-text pretraining. ResNet18 supervised vs CLIP frozen is still useful as a practical baseline comparison, but it mixes architecture, training objective, and pretraining data scale, so it should not be used to claim that Transformer architecture alone is better than CNN architecture.

## How to Run the ViT Baseline

The ViT pipeline is implemented as standalone scripts rather than a long-running notebook cell:

```powershell
python scripts/train_vit_supervised.py --epochs 3 --batch-size 8 --num-workers 0
python scripts/evaluate_vit_supervised.py --update-summary
python scripts/build_vit_supervised_index.py
python scripts/evaluate_vit_supervised_retrieval.py --update-summary
```

For a quick smoke test:

```powershell
python scripts/train_vit_supervised.py --debug --epochs 1 --batch-size 4 --num-workers 0 --no-pretrained
python scripts/evaluate_vit_supervised.py --max-samples 20
python scripts/build_vit_supervised_index.py --max-images 32
python scripts/evaluate_vit_supervised_retrieval.py --max-query-images 16
```

## Result Loading Logic

The notebook does not fabricate ViT results. If the result files exist, they can be loaded from:

- `outputs/reports/vit_supervised_classification_metrics.json`
- `outputs/reports/vit_supervised_retrieval_metrics.json`
- `outputs/reports/experiment_summary.csv`

If full ViT training has not been run yet, this section should be interpreted as the implementation plan and code entry point for the supervised Transformer baseline.
"""


CODE = r"""from pathlib import Path
import json
import pandas as pd

vit_cls_path = Path("outputs/reports/vit_supervised_classification_metrics.json")
vit_ret_path = Path("outputs/reports/vit_supervised_retrieval_metrics.json")
summary_path = Path("outputs/reports/experiment_summary.csv")

if vit_cls_path.exists():
    print("ViT classification metrics:")
    print(json.dumps(json.loads(vit_cls_path.read_text()), indent=2))
else:
    print("ViT classification metrics are not available yet. Run scripts/evaluate_vit_supervised.py first.")

if vit_ret_path.exists():
    print("ViT retrieval metrics:")
    print(json.dumps(json.loads(vit_ret_path.read_text()), indent=2))
else:
    print("ViT retrieval metrics are not available yet. Run scripts/evaluate_vit_supervised_retrieval.py first.")

if summary_path.exists():
    summary_df = pd.read_csv(summary_path)
    display_cols = ["model_name", "num_classes", "train_size", "test_size", "top1_acc", "top5_acc", "recall@1", "recall@5", "recall@10"]
    available_cols = [col for col in display_cols if col in summary_df.columns]
    display(summary_df[available_cols])
"""


def main():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    new_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": MARKDOWN.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(keepends=True)},
    ]

    cells = notebook.get("cells", [])
    for idx, cell in enumerate(cells):
        if SECTION_TITLE in "".join(cell.get("source", [])):
            notebook["cells"] = cells[:idx] + new_cells + cells[idx + 2 :]
            break
    else:
        notebook.setdefault("cells", []).extend(new_cells)

    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"notebook_updated={NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
