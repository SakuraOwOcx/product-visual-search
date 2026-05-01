from pathlib import Path

from .config import PROJECT_ROOT


def to_project_relative(path):
    path_text = str(path).replace("\\", "/")
    marker = "Visual Search Project/"
    if marker in path_text:
        return path_text.split(marker, 1)[1]
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def path_exists(path):
    return resolve_project_path(path).exists()
