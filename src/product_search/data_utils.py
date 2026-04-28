from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CANONICAL_IMAGE_DIR,
    DEBUG_MAX_IMAGES_PER_CLASS,
    DEBUG_NUM_CLASSES,
    FULL_SPLIT_CSV,
    IMAGE_EXTENSIONS,
    LABEL_COLUMN_PRIORITY,
    SEED,
    STYLES_CSV,
)


def normalize_split_name(name):
    name = str(name).lower()
    if name in {"train", "training"}:
        return "train"
    if name in {"val", "valid", "validation"}:
        return "val"
    if name in {"test", "testing"}:
        return "test"
    return None


def infer_split_from_path(path):
    for part in Path(path).parts:
        split = normalize_split_name(part)
        if split is not None:
            return split
    return None


def assign_splits_per_class(df, seed=SEED):
    rng = np.random.default_rng(seed)
    split_series = pd.Series(index=df.index, dtype="object")
    for _, group in df.groupby("class_name"):
        indices = group.index.to_list()
        rng.shuffle(indices)
        n = len(indices)
        if n == 1:
            assignments = ["train"]
        elif n == 2:
            assignments = ["train", "test"]
        else:
            n_train = max(1, int(round(n * 0.70)))
            n_val = max(1, int(round(n * 0.15)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            n_test = n - n_train - n_val
            assignments = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        for idx, split in zip(indices, assignments):
            split_series.loc[idx] = split
    return split_series


def load_metadata_map(styles_csv=STYLES_CSV):
    if not styles_csv.exists():
        raise FileNotFoundError(f"Missing metadata file: {styles_csv}")
    styles_df = pd.read_csv(styles_csv, on_bad_lines="skip")
    label_column = next((col for col in LABEL_COLUMN_PRIORITY if col in styles_df.columns), None)
    if label_column is None:
        raise ValueError(f"None of {LABEL_COLUMN_PRIORITY} exists in {styles_csv}")
    styles_df = styles_df.copy()
    styles_df["image_id"] = styles_df["id"].astype(str)
    styles_df["metadata_class_name"] = styles_df[label_column].fillna("unknown").astype(str)
    return styles_df.set_index("image_id")["metadata_class_name"].to_dict(), label_column


def scan_image_index():
    if not CANONICAL_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Missing image folder: {CANONICAL_IMAGE_DIR}. Put extracted product images under data/raw/images/."
        )

    metadata_map, label_column = load_metadata_map()
    image_paths = sorted(
        p for p in CANONICAL_IMAGE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No product images found under {CANONICAL_IMAGE_DIR}")

    rows = []
    for image_path in image_paths:
        image_id = image_path.stem
        class_name = metadata_map.get(image_id, "unknown")
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "class_name": class_name,
                "split": infer_split_from_path(image_path),
            }
        )
    df = pd.DataFrame(rows)
    if df["split"].isna().any():
        df["split"] = df["split"].fillna(assign_splits_per_class(df))

    class_names = sorted(df["class_name"].unique())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    df["class_id"] = df["class_name"].map(class_to_id).astype(int)
    return df, label_column


def build_debug_subset(num_classes=DEBUG_NUM_CLASSES, max_images_per_class=DEBUG_MAX_IMAGES_PER_CLASS):
    image_index, label_column = scan_image_index()
    selected_classes = image_index["class_name"].value_counts().head(num_classes).index.tolist()
    sampled_parts = []
    for class_name in selected_classes:
        class_df = image_index[image_index["class_name"] == class_name]
        sampled_parts.append(
            class_df.sample(n=min(len(class_df), max_images_per_class), random_state=SEED)
        )
    debug_df = pd.concat(sampled_parts, ignore_index=True)
    debug_class_names = sorted(debug_df["class_name"].unique())
    debug_class_to_id = {name: idx for idx, name in enumerate(debug_class_names)}
    debug_df["class_id"] = debug_df["class_name"].map(debug_class_to_id).astype(int)
    id_to_class = {idx: name for name, idx in debug_class_to_id.items()}
    return debug_df[["image_id", "image_path", "class_id", "class_name", "split"]], id_to_class, label_column


def get_gallery_query_dataframes():
    debug_df, id_to_class, label_column = build_debug_subset()
    gallery_df = debug_df[debug_df["split"] == "train"].reset_index(drop=True)
    val_df = debug_df[debug_df["split"] == "val"].reset_index(drop=True)
    test_df = debug_df[debug_df["split"] == "test"].reset_index(drop=True)
    query_df = test_df if len(test_df) > 0 else val_df
    return gallery_df, query_df, id_to_class, label_column


def get_full_gallery_query_dataframes():
    if not FULL_SPLIT_CSV.exists():
        raise FileNotFoundError(f"Full split CSV not found: {FULL_SPLIT_CSV}")
    split_df = pd.read_csv(FULL_SPLIT_CSV)
    required = {"image_id", "image_path", "articleType", "class_id", "split"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Full split CSV is missing columns: {sorted(missing)}")
    full_df = split_df.copy()
    full_df["class_name"] = full_df["articleType"].astype(str)
    gallery_df = full_df[full_df["split"] == "train"].reset_index(drop=True)
    query_df = full_df[full_df["split"] == "test"].reset_index(drop=True)
    return gallery_df, query_df, "articleType"
