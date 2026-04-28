import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TRAIN_LOG = REPORT_DIR / "resnet18_full_training_log.csv"
RETRIEVAL_JSON = REPORT_DIR / "resnet18_full_retrieval_metrics.json"
SUMMARY_CSV = REPORT_DIR / "experiment_summary.csv"


def save_training_curves():
    if not TRAIN_LOG.exists():
        print(f"missing_training_log={TRAIN_LOG}")
        return None
    df = pd.read_csv(TRAIN_LOG)
    df = df[df.get("is_smoke_test", False).astype(str).str.lower() != "true"].copy()
    if df.empty:
        print("missing_formal_training_rows=True")
        return None
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "resnet18_full_training_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("ResNet18 Full Dataset Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["val_top1"], marker="o", label="Val Top-1")
    axes[1].plot(df["epoch"], df["val_top5"], marker="o", label="Val Top-5")
    axes[1].set_title("ResNet18 Full Dataset Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"training_curves={out}")
    return out


def save_retrieval_summary():
    if not RETRIEVAL_JSON.exists():
        print(f"missing_retrieval_metrics={RETRIEVAL_JSON}")
        return None
    metrics = json.loads(RETRIEVAL_JSON.read_text(encoding="utf-8"))
    out = FIGURE_DIR / "resnet18_full_retrieval_summary.png"
    labels = ["Recall@1", "Recall@5", "Recall@10"]
    values = [metrics["recall@1"], metrics["recall@5"], metrics["recall@10"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#2E86AB", "#52B788", "#F4A261"])
    ax.set_title("ResNet18 Full Dataset Retrieval Recall")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"retrieval_summary={out}")
    return out


def save_model_recall_comparison_full():
    if not SUMMARY_CSV.exists():
        print(f"missing_summary_csv={SUMMARY_CSV}")
        return None
    df = pd.read_csv(SUMMARY_CSV)
    wanted = ["resnet18", "clip_vit_b_32_openai_local", "clip_vit_b_32_openai_full", "resnet18_full_dataset"]
    labels = {
        "resnet18": "ResNet18 debug",
        "clip_vit_b_32_openai_local": "CLIP debug",
        "clip_vit_b_32_openai_full": "CLIP full",
        "resnet18_full_dataset": "ResNet18 full",
    }
    rows = df[df["model_name"].isin(wanted)].copy()
    if rows.empty:
        print("missing_comparison_rows=True")
        return None
    rows["display"] = rows["model_name"].map(labels)
    rows = rows.set_index("display").loc[[labels[m] for m in wanted if labels[m] in rows.set_index("display").index]]
    metrics = ["recall@1", "recall@5", "recall@10"]
    out = FIGURE_DIR / "model_recall_comparison_full.png"
    ax = rows[metrics].astype(float).plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_title("Visual Retrieval Recall Comparison\n(Debug rows use debug gallery; full rows use the shared full train gallery)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"model_recall_comparison_full={out}")
    return out


def main():
    save_training_curves()
    save_retrieval_summary()
    save_model_recall_comparison_full()


if __name__ == "__main__":
    main()
