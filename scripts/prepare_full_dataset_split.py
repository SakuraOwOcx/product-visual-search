import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import sys


ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGE_DIR = DATA_RAW_DIR / "images"
STYLES_CSV = DATA_RAW_DIR / "styles.csv"
SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
SPLIT_CSV = SPLIT_DIR / "full_dataset_split.csv"
CLASS_MAPPING_JSON = SPLIT_DIR / "full_class_mapping.json"
STATS_CSV = REPORT_DIR / "full_dataset_stats.csv"
DROPPED_CLASSES_CSV = REPORT_DIR / "dropped_rare_classes.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

from src.product_search.path_utils import to_project_relative  # noqa: E402


def is_readable_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def assign_splits(df, seed=SEED):
    rng = np.random.default_rng(seed)
    split_series = pd.Series(index=df.index, dtype="object")
    for _, group in df.groupby("articleType"):
        indices = group.index.to_list()
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(round(n * 0.70)))
        n_val = max(1, int(round(n * 0.15)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        assignments = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        for idx, split in zip(indices, assignments):
            split_series.loc[idx] = split
    return split_series


def main():
    parser = argparse.ArgumentParser(description="Prepare full Fashion Product Images split using data/raw/images only.")
    parser.add_argument("--skip-image-verify", action="store_true", help="Skip PIL verification for faster split creation.")
    args = parser.parse_args()

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing image directory: {IMAGE_DIR}")
    if not STYLES_CSV.exists():
        raise FileNotFoundError(f"Missing styles metadata: {STYLES_CSV}")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    styles = pd.read_csv(STYLES_CSV, on_bad_lines="skip")
    if "id" not in styles.columns or "articleType" not in styles.columns:
        raise ValueError("styles.csv must contain id and articleType columns.")
    styles = styles[["id", "articleType"]].copy()
    styles["image_id"] = styles["id"].astype(str)
    styles["articleType"] = styles["articleType"].astype("string")
    styles = styles.dropna(subset=["articleType"])
    styles = styles[styles["articleType"].str.len() > 0]

    image_map = {
        p.stem: p
        for p in IMAGE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    rows = []
    unreadable = 0
    missing = 0
    for row in styles.itertuples(index=False):
        image_path = image_map.get(row.image_id)
        if image_path is None:
            missing += 1
            continue
        if not args.skip_image_verify and not is_readable_image(image_path):
            unreadable += 1
            continue
        rows.append(
            {
                "image_id": row.image_id,
                "image_path": to_project_relative(image_path),
                "articleType": str(row.articleType),
            }
        )

    df = pd.DataFrame(rows)
    counts = df["articleType"].value_counts().sort_values(ascending=False)
    dropped = counts[counts < 3].reset_index()
    dropped.columns = ["articleType", "image_count"]
    dropped.to_csv(DROPPED_CLASSES_CSV, index=False)
    keep_classes = counts[counts >= 3].index
    df = df[df["articleType"].isin(keep_classes)].reset_index(drop=True)

    class_names = sorted(df["articleType"].unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {str(idx): name for name, idx in class_to_idx.items()}
    df["class_id"] = df["articleType"].map(class_to_idx).astype(int)
    df["split"] = assign_splits(df)

    stats = df["articleType"].value_counts().rename_axis("articleType").reset_index(name="image_count")
    stats.to_csv(STATS_CSV, index=False)
    df[["image_id", "image_path", "articleType", "class_id", "split"]].to_csv(SPLIT_CSV, index=False)
    with CLASS_MAPPING_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "num_classes": len(class_names),
                "label_name": "articleType",
            },
            f,
            indent=2,
        )

    split_counts = df["split"].value_counts().to_dict()
    print(f"valid_image_count={len(df)}")
    print(f"num_classes={len(class_names)}")
    print(f"train_size={split_counts.get('train', 0)}")
    print(f"val_size={split_counts.get('val', 0)}")
    print(f"test_size={split_counts.get('test', 0)}")
    print(f"dropped_rare_classes_count={len(dropped)}")
    print(f"missing_image_metadata_rows={missing}")
    print(f"unreadable_images={unreadable}")
    print("images_per_class_top20=")
    print(stats.head(20).to_string(index=False))
    print(f"min_images_per_class={int(stats['image_count'].min())}")
    print(f"max_images_per_class={int(stats['image_count'].max())}")
    print(f"median_images_per_class={float(stats['image_count'].median()):.1f}")
    print(f"split_csv={SPLIT_CSV}")
    print(f"class_mapping_json={CLASS_MAPPING_JSON}")
    print(f"stats_csv={STATS_CSV}")
    print(f"dropped_rare_classes_csv={DROPPED_CLASSES_CSV}")


if __name__ == "__main__":
    main()
