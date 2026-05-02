import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.product_search.clip_supervised_engine import get_clip_train_eval_transforms, load_clip_articletype_model  # noqa: E402
from src.product_search.config import CLIP_ARTICLETYPE_INDEX_PATH, FULL_SPLIT_CSV  # noqa: E402


class ProductPathDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        metadata = {
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "articleType": str(row["articleType"]),
            "class_name": str(row["articleType"]),
            "class_id": int(row["class_id"]),
            "split": str(row["split"]),
        }
        for key in ("gender", "masterCategory", "subCategory", "baseColour", "season", "year", "usage", "productDisplayName"):
            if key in row.index:
                metadata[key] = None if pd.isna(row[key]) else row[key]
        return self.transform(image), metadata


def main():
    parser = argparse.ArgumentParser(description="Build lightweight fine-tuned CLIP full gallery embedding index.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--include-val-as-gallery", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    if not FULL_SPLIT_CSV.exists():
        raise FileNotFoundError("Run scripts/prepare_full_dataset_split.py first.")
    model, _, checkpoint = load_clip_articletype_model()
    image_size = int(checkpoint.get("image_size", 224))
    _, transform = get_clip_train_eval_transforms(image_size=image_size)

    split_df = pd.read_csv(FULL_SPLIT_CSV)
    gallery_splits = ["train", "val"] if args.include_val_as_gallery else ["train"]
    gallery_df = split_df[split_df["split"].isin(gallery_splits)].reset_index(drop=True)
    if args.max_images is not None:
        gallery_df = gallery_df.head(args.max_images).reset_index(drop=True)

    loader = DataLoader(ProductPathDataset(gallery_df, transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    embeddings, metadata_rows = [], []
    model.eval()
    with np.errstate(all="ignore"):
        import torch

        with torch.no_grad():
            for images, metadata_batch in loader:
                features = model.encode_image(images.to(next(model.parameters()).device), normalize=True)
                embeddings.append(features.cpu().numpy())
                batch_size = len(metadata_batch["image_id"])
                for idx in range(batch_size):
                    item = {key: metadata_batch[key][idx] for key in metadata_batch}
                    if "class_id" in item:
                        item["class_id"] = int(item["class_id"])
                    metadata_rows.append(item)

    embeddings = np.vstack(embeddings).astype(np.float32)
    CLIP_ARTICLETYPE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index_scope = "smoke train gallery" if args.max_images is not None else (
        "full train+val gallery" if args.include_val_as_gallery else "full train gallery"
    )
    np.savez_compressed(
        CLIP_ARTICLETYPE_INDEX_PATH,
        embeddings=embeddings,
        metadata_json=json.dumps(metadata_rows),
        model_name=np.array("clip_vit_b_32_articletype_lite"),
        embedding_dim=np.array(embeddings.shape[1]),
        index_scope=np.array(index_scope),
        checkpoint_source=np.array(str(checkpoint.get("base_clip_checkpoint", ""))),
    )
    print(f"gallery_size={len(gallery_df)}")
    print(f"embedding_shape={embeddings.shape}")
    print(f"index_scope={index_scope}")
    print(f"output_path={CLIP_ARTICLETYPE_INDEX_PATH}")


if __name__ == "__main__":
    main()
