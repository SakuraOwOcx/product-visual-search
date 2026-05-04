from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _candidate_project_roots():
    candidates = [PROJECT_ROOT]
    sibling_original = PROJECT_ROOT.parent / "product-visual-search"
    if sibling_original != PROJECT_ROOT:
        candidates.append(sibling_original)
    return candidates


def _first_existing_dir(relative_parts):
    for root in _candidate_project_roots():
        candidate = root.joinpath(*relative_parts)
        if candidate.exists():
            return candidate
    return PROJECT_ROOT.joinpath(*relative_parts)


DATA_RAW_DIR = _first_existing_dir(["data", "raw"])
CANONICAL_IMAGE_DIR = DATA_RAW_DIR / "images"
STYLES_CSV = DATA_RAW_DIR / "styles.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
INDEX_DIR = OUTPUT_DIR / "indexes"
INDEX_PATH = INDEX_DIR / "clip_gallery_debug_index.npz"
CLIP_FULL_INDEX_PATH = INDEX_DIR / "clip_full_gallery_index.npz"
RESNET18_FULL_INDEX_PATH = INDEX_DIR / "resnet18_full_gallery_index.npz"
CLIP_ARTICLETYPE_INDEX_PATH = INDEX_DIR / "clip_articletype_full_gallery_index.npz"

SPLIT_DIR = OUTPUT_DIR / "splits"
FULL_SPLIT_CSV = SPLIT_DIR / "full_dataset_split.csv"
FULL_CLASS_MAPPING_JSON = SPLIT_DIR / "full_class_mapping.json"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESNET18_FULL_LAST_CHECKPOINT = CHECKPOINT_DIR / "resnet18_full_last.pth"
RESNET18_FULL_BEST_CHECKPOINT = CHECKPOINT_DIR / "resnet18_full_best.pth"
CLIP_ARTICLETYPE_LAST_CHECKPOINT = CHECKPOINT_DIR / "clip_articletype_last.pth"
CLIP_ARTICLETYPE_BEST_CHECKPOINT = CHECKPOINT_DIR / "clip_articletype_best.pth"

LOCAL_CLIP_CHECKPOINT = _first_existing_dir(
    ["models", "huggingface", "timm_vit_base_patch32_clip_224_openai"]
) / "open_clip_model.safetensors"

DEBUG_MODE = True
DEBUG_NUM_CLASSES = 20
DEBUG_MAX_IMAGES_PER_CLASS = 50
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

MODEL_NAME = "ViT-B-32"
MODEL_DISPLAY_NAME = "CLIP ViT-B/32"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_COLUMN_PRIORITY = ["articleType", "subCategory", "masterCategory", "gender"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
