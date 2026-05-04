from pathlib import Path

from .config import CANONICAL_IMAGE_DIR, PROJECT_ROOT


COLAB_PROJECT_PREFIX = "/content/product_visual_search/"
COLAB_IMAGE_PREFIX = "/content/product_visual_search/data/raw/images/"


def to_project_relative(path):
    path_text = str(path).replace("\\", "/")
    marker = "Visual Search Project/"
    if marker in path_text:
        return path_text.split(marker, 1)[1]
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path_text


def resolve_project_path(path):
    path_text = str(path).replace("\\", "/")
    direct_path = Path(path_text)
    if direct_path.exists():
        return direct_path

    if path_text.startswith(COLAB_IMAGE_PREFIX):
        candidate = CANONICAL_IMAGE_DIR / Path(path_text).name
        if candidate.exists():
            return candidate

    if path_text.startswith(COLAB_PROJECT_PREFIX):
        rel = path_text[len(COLAB_PROJECT_PREFIX):]
        candidate = PROJECT_ROOT / Path(rel)
        if candidate.exists():
            return candidate

    if direct_path.is_absolute():
        return direct_path
    return PROJECT_ROOT / direct_path


def path_exists(path):
    return resolve_project_path(path).exists()
