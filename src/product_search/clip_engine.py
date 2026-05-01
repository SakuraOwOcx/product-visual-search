from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

import open_clip

from .config import BATCH_SIZE, DEVICE, LOCAL_CLIP_CHECKPOINT, MODEL_NAME, NUM_WORKERS
from .path_utils import resolve_project_path, to_project_relative


ImageFile.LOAD_TRUNCATED_IMAGES = True


class CLIPImageDataset(Dataset):
    def __init__(self, dataframe, preprocess):
        self.df = dataframe.reset_index(drop=True).copy()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = resolve_project_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image), int(row["class_id"]), to_project_relative(image_path)


def load_clip_model(checkpoint_path=LOCAL_CLIP_CHECKPOINT):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Local CLIP checkpoint not found: {checkpoint_path}. "
            "Run the existing download script first, or place the checkpoint at this path."
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(checkpoint_path),
        device=DEVICE,
    )
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess


def encode_pil_image(model, preprocess, image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy()[0]


def extract_gallery_embeddings(model, preprocess, gallery_df, batch_size=BATCH_SIZE, max_images=None):
    if max_images is not None:
        gallery_df = gallery_df.head(max_images).reset_index(drop=True)
    loader = DataLoader(
        CLIPImageDataset(gallery_df, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    embeddings, labels, paths = [], [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels, batch_paths in loader:
            features = model.encode_image(images.to(DEVICE))
            features = nn.functional.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu().numpy())
            labels.extend(batch_labels.numpy().tolist())
            paths.extend(list(batch_paths))
    if not embeddings:
        return np.empty((0, 512), dtype=np.float32), [], []
    return np.vstack(embeddings), labels, paths
