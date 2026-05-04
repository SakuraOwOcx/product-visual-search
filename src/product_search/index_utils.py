import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clip_engine import extract_gallery_embeddings, load_clip_model
from .config import CLIP_FULL_INDEX_PATH, INDEX_PATH, LOCAL_CLIP_CHECKPOINT, RESNET18_FULL_INDEX_PATH
from .data_utils import get_full_gallery_query_dataframes, get_gallery_query_dataframes, relocate_image_paths


def index_exists(index_path=INDEX_PATH):
    return Path(index_path).exists()


def save_gallery_index(embeddings, gallery_df, index_path=INDEX_PATH, model_name="clip_vit_b_32_openai_local", index_scope="debug gallery"):
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    df = gallery_df.copy()
    if "class_name" not in df.columns and "articleType" in df.columns:
        df["class_name"] = df["articleType"]
    metadata = df[["image_id", "image_path", "class_id", "class_name"]].to_dict("records")
    np.savez_compressed(
        index_path,
        embeddings=embeddings.astype(np.float32),
        metadata_json=json.dumps(metadata),
        checkpoint_source=str(LOCAL_CLIP_CHECKPOINT),
        model_name=np.array(model_name),
        index_scope=np.array(index_scope),
        embedding_dim=np.array(embeddings.shape[1]),
    )
    return index_path


def load_gallery_index(index_path=INDEX_PATH):
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Gallery index not found: {index_path}")
    data = np.load(index_path, allow_pickle=False)
    embeddings = data["embeddings"]
    metadata = json.loads(str(data["metadata_json"]))
    return embeddings, metadata


def load_visual_search_index(index_path):
    index_path = Path(index_path)
    if not index_path.exists():
        return None
    data = np.load(index_path, allow_pickle=False)
    embeddings = data["embeddings"]

    if "metadata_json" in data.files:
        metadata = json.loads(str(data["metadata_json"]))
        metadata = relocate_image_paths(pd.DataFrame(metadata), keep_relative=False).to_dict("records")
        model_name = str(data["model_name"]) if "model_name" in data.files else "clip"
        index_scope = str(data["index_scope"]) if "index_scope" in data.files else "debug gallery"
    else:
        image_paths = data["image_paths"].astype(str).tolist()
        article_types = data["articleTypes"].astype(str).tolist()
        image_ids = data["image_ids"].astype(str).tolist()
        splits = data["split"].astype(str).tolist() if "split" in data.files else ["train"] * len(image_paths)
        metadata = [
            {
                "image_id": image_id,
                "image_path": image_path,
                "class_name": article_type,
                "articleType": article_type,
                "split": split,
            }
            for image_id, image_path, article_type, split in zip(image_ids, image_paths, article_types, splits)
        ]
        metadata = relocate_image_paths(pd.DataFrame(metadata), keep_relative=False).to_dict("records")
        model_name = str(data["model_name"]) if "model_name" in data.files else "resnet18"
        index_scope = str(data["index_scope"]) if "index_scope" in data.files else "full train gallery"

    return {
        "embeddings": embeddings,
        "metadata": metadata,
        "model_name": model_name,
        "index_scope": index_scope,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "gallery_size": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
        "index_path": str(index_path),
    }


def load_clip_debug_index():
    return load_visual_search_index(INDEX_PATH)


def load_clip_full_index():
    return load_visual_search_index(CLIP_FULL_INDEX_PATH)


def load_resnet18_full_index():
    return load_visual_search_index(RESNET18_FULL_INDEX_PATH)


def build_gallery_index(max_images=None, index_path=INDEX_PATH):
    gallery_df, query_df, id_to_class, label_column = get_gallery_query_dataframes()
    if max_images is not None:
        gallery_df = gallery_df.head(max_images).reset_index(drop=True)
    model, preprocess = load_clip_model()
    embeddings, _, _ = extract_gallery_embeddings(model, preprocess, gallery_df)
    saved_path = save_gallery_index(embeddings, gallery_df, index_path=index_path)
    summary = {
        "index_path": str(saved_path),
        "gallery_size": int(len(gallery_df)),
        "embedding_shape": tuple(embeddings.shape),
        "query_size": int(len(query_df)),
        "num_classes": int(gallery_df["class_name"].nunique()),
        "label_column": label_column,
    }
    return summary


def build_clip_full_gallery_index(max_images=None, index_path=CLIP_FULL_INDEX_PATH):
    gallery_df, query_df, label_column = get_full_gallery_query_dataframes()
    if max_images is not None:
        gallery_df = gallery_df.head(max_images).reset_index(drop=True)
    model, preprocess = load_clip_model()
    embeddings, _, _ = extract_gallery_embeddings(model, preprocess, gallery_df)
    index_scope = "smoke full train gallery" if max_images is not None else "full train gallery"
    saved_path = save_gallery_index(
        embeddings,
        gallery_df,
        index_path=index_path,
        model_name="clip_vit_b_32_openai_full",
        index_scope=index_scope,
    )
    return {
        "index_path": str(saved_path),
        "gallery_size": int(len(gallery_df)),
        "embedding_shape": tuple(embeddings.shape),
        "query_size": int(len(query_df)),
        "num_classes": int(gallery_df["class_name"].nunique()),
        "label_column": label_column,
        "index_scope": index_scope,
    }


def load_experiment_summary(summary_csv):
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_csv)
