import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.product_search.config import FULL_SPLIT_CSV, VIT_SUPERVISED_BEST_CHECKPOINT, VIT_SUPERVISED_INDEX_PATH  # noqa: E402
from src.product_search.data_utils import load_full_split_dataframe  # noqa: E402
from src.product_search.path_utils import resolve_project_path, to_project_relative  # noqa: E402
from src.product_search.vit_engine import extract_vit_features, load_vit_supervised_model  # noqa: E402


class ProductPathDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = resolve_project_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), str(row["image_id"]), to_project_relative(image_path), str(row["articleType"]), int(row["class_id"]), str(row["split"])


def main():
    parser = argparse.ArgumentParser(description="Build supervised ViT gallery embedding index.")
    parser.add_argument("--split-csv", type=Path, default=FULL_SPLIT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=VIT_SUPERVISED_BEST_CHECKPOINT)
    parser.add_argument("--output-index", type=Path, default=VIT_SUPERVISED_INDEX_PATH)
    parser.add_argument("--gallery-split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    if not args.split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_csv}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"ViT checkpoint not found: {args.checkpoint}")
    model, transform, checkpoint = load_vit_supervised_model(args.checkpoint)
    device = next(model.parameters()).device
    split_df = load_full_split_dataframe(args.split_csv)
    gallery_df = split_df[split_df["split"] == args.gallery_split].reset_index(drop=True)
    if args.max_images is not None:
        gallery_df = gallery_df.head(args.max_images).reset_index(drop=True)
    loader = DataLoader(ProductPathDataset(gallery_df, transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    embeddings, image_ids, image_paths, article_types, labels, splits = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for images, batch_ids, batch_paths, batch_types, batch_labels, batch_splits in loader:
            features = extract_vit_features(model, images.to(device))
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu().numpy())
            image_ids.extend(list(batch_ids))
            image_paths.extend(list(batch_paths))
            article_types.extend(list(batch_types))
            labels.extend([int(x) for x in batch_labels])
            splits.extend(list(batch_splits))
    embeddings = np.vstack(embeddings).astype(np.float32)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    is_smoke = bool(checkpoint.get("is_smoke_test") or checkpoint.get("train_config", {}).get("is_smoke_test") or args.max_images)
    index_scope = "smoke train gallery" if is_smoke else f"{args.gallery_split} gallery"
    np.savez_compressed(
        args.output_index,
        embeddings=embeddings,
        image_paths=np.array(image_paths),
        articleTypes=np.array(article_types),
        image_ids=np.array(image_ids),
        labels=np.array(labels),
        split=np.array(splits),
        model_name=np.array("vit_b16_supervised"),
        embedding_dim=np.array(embeddings.shape[1]),
        index_scope=np.array(index_scope),
        created_at=np.array(datetime.now(timezone.utc).isoformat()),
        checkpoint_source=np.array(to_project_relative(args.checkpoint)),
    )
    print(f"gallery_size={len(gallery_df)}")
    print(f"embedding_shape={embeddings.shape}")
    print(f"index_scope={index_scope}")
    print(f"output_index={to_project_relative(args.output_index)}")


if __name__ == "__main__":
    main()
