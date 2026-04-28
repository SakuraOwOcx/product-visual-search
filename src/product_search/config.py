from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
CANONICAL_IMAGE_DIR = DATA_RAW_DIR / "images"
STYLES_CSV = DATA_RAW_DIR / "styles.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
INDEX_DIR = OUTPUT_DIR / "indexes"
INDEX_PATH = INDEX_DIR / "clip_gallery_debug_index.npz"
CLIP_FULL_INDEX_PATH = INDEX_DIR / "clip_full_gallery_index.npz"
RESNET18_FULL_INDEX_PATH = INDEX_DIR / "resnet18_full_gallery_index.npz"

SPLIT_DIR = OUTPUT_DIR / "splits"
FULL_SPLIT_CSV = SPLIT_DIR / "full_dataset_split.csv"
FULL_CLASS_MAPPING_JSON = SPLIT_DIR / "full_class_mapping.json"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESNET18_FULL_LAST_CHECKPOINT = CHECKPOINT_DIR / "resnet18_full_last.pth"
RESNET18_FULL_BEST_CHECKPOINT = CHECKPOINT_DIR / "resnet18_full_best.pth"

LOCAL_CLIP_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "huggingface"
    / "timm_vit_base_patch32_clip_224_openai"
    / "open_clip_model.safetensors"
)

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
