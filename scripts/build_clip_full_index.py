import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.product_search.config import CLIP_FULL_INDEX_PATH, LOCAL_CLIP_CHECKPOINT  # noqa: E402
from src.product_search.index_utils import build_clip_full_gallery_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build CLIP full train-gallery index using the shared full split.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    if not LOCAL_CLIP_CHECKPOINT.exists():
        raise FileNotFoundError(f"Local CLIP checkpoint not found: {LOCAL_CLIP_CHECKPOINT}")

    summary = build_clip_full_gallery_index(max_images=args.max_images, index_path=CLIP_FULL_INDEX_PATH)
    print(f"checkpoint_source={LOCAL_CLIP_CHECKPOINT}")
    print(f"download_new_model=False")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
