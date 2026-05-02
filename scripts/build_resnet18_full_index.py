import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.product_search.config import FULL_SPLIT_CSV, RESNET18_FULL_INDEX_PATH  # noqa: E402
from src.product_search.resnet_engine import get_resnet_eval_transform, load_resnet18_full_model  # noqa: E402


class ProductPathDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), str(row["image_id"]), str(row["image_path"]), str(row["articleType"]), str(row["split"])


def main():
    parser = argparse.ArgumentParser(description="Build ResNet18 full gallery embedding index.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--include-val-as-gallery", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    if not FULL_SPLIT_CSV.exists():
        raise FileNotFoundError("Run scripts/prepare_full_dataset_split.py first.")
    model, _, checkpoint = load_resnet18_full_model()
    device = next(model.parameters()).device
    transform = get_resnet_eval_transform(int(checkpoint.get("image_size", 224)))

    split_df = pd.read_csv(FULL_SPLIT_CSV)
    gallery_splits = ["train", "val"] if args.include_val_as_gallery else ["train"]
    gallery_df = split_df[split_df["split"].isin(gallery_splits)].reset_index(drop=True)
    if args.max_images is not None:
        gallery_df = gallery_df.head(args.max_images).reset_index(drop=True)

    loader = DataLoader(ProductPathDataset(gallery_df, transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    embeddings, image_ids, image_paths, article_types, splits = [], [], [], [], []
    with torch.no_grad():
        for images, batch_ids, batch_paths, batch_types, batch_splits in loader:
            features = feature_extractor(images.to(device))
            features = torch.flatten(features, 1)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu().numpy())
            image_ids.extend(list(batch_ids))
            image_paths.extend(list(batch_paths))
            article_types.extend(list(batch_types))
            splits.extend(list(batch_splits))

    embeddings = np.vstack(embeddings).astype(np.float32)
    RESNET18_FULL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index_scope = "smoke train gallery" if args.max_images is not None else (
        "full train+val gallery" if args.include_val_as_gallery else "full train gallery"
    )
    np.savez_compressed(
        RESNET18_FULL_INDEX_PATH,
        embeddings=embeddings,
        image_paths=np.array(image_paths),
        articleTypes=np.array(article_types),
        image_ids=np.array(image_ids),
        split=np.array(splits),
        model_name=np.array("resnet18_full_dataset"),
        embedding_dim=np.array(embeddings.shape[1]),
        index_scope=np.array(index_scope),
    )
    print(f"gallery_size={len(gallery_df)}")
    print(f"embedding_shape={embeddings.shape}")
    print(f"index_scope={index_scope}")
    print(f"output_path={RESNET18_FULL_INDEX_PATH}")


if __name__ == "__main__":
    main()
