from pathlib import Path

import open_clip
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .config import (
    CLIP_ARTICLETYPE_BEST_CHECKPOINT,
    DEVICE,
    LOCAL_CLIP_CHECKPOINT,
    MODEL_NAME,
)


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_clip_train_eval_transforms(image_size=224):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    return train_tf, eval_tf


def resolve_clip_pretrained_source(use_pretrained=True, checkpoint_path=LOCAL_CLIP_CHECKPOINT):
    if not use_pretrained:
        return None
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        return str(checkpoint_path)
    return "openai"


class CLIPArticleTypeClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.num_classes = int(num_classes)
        self.embedding_dim = int(getattr(self.clip_model.visual, "output_dim", 512))
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, images):
        features = self.clip_model.encode_image(images, normalize=False)
        return self.classifier(features)

    def encode_image(self, images, normalize=True):
        features = self.clip_model.encode_image(images, normalize=False)
        if normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features


def build_clip_articletype_model(num_classes, use_pretrained=True, checkpoint_path=LOCAL_CLIP_CHECKPOINT):
    pretrained_source = resolve_clip_pretrained_source(use_pretrained=use_pretrained, checkpoint_path=checkpoint_path)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=pretrained_source,
        device=DEVICE,
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()
    return CLIPArticleTypeClassifier(clip_model, num_classes=num_classes).to(DEVICE)


def configure_lightweight_finetune(model, unfreeze_visual_blocks=2):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    for name, param in model.clip_model.named_parameters():
        if name == "visual.proj" or name.startswith("visual.ln_post"):
            param.requires_grad = True

    total_blocks = len(model.clip_model.visual.transformer.resblocks)
    start_idx = max(0, total_blocks - int(unfreeze_visual_blocks))
    for block_idx in range(start_idx, total_blocks):
        prefix = f"visual.transformer.resblocks.{block_idx}."
        for name, param in model.clip_model.named_parameters():
            if name.startswith(prefix):
                param.requires_grad = True

    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_count = sum(param.numel() for param in model.parameters())
    return {
        "trainable_params": int(trainable_count),
        "total_params": int(total_count),
        "trainable_ratio": float(trainable_count / max(total_count, 1)),
        "unfreeze_visual_blocks": int(unfreeze_visual_blocks),
    }


def get_trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def load_clip_articletype_model(checkpoint_path=CLIP_ARTICLETYPE_BEST_CHECKPOINT):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "CLIP articleType checkpoint is not available yet. "
            f"Expected checkpoint: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    num_classes = int(checkpoint.get("num_classes", 0))
    if num_classes <= 0:
        raise ValueError("Checkpoint is missing a valid num_classes value.")
    model = build_clip_articletype_model(
        num_classes=num_classes,
        use_pretrained=False,
        checkpoint_path=checkpoint.get("base_clip_checkpoint", LOCAL_CLIP_CHECKPOINT),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    image_size = int(checkpoint.get("image_size", 224))
    _, eval_tf = get_clip_train_eval_transforms(image_size=image_size)
    return model, eval_tf, checkpoint


def encode_clip_articletype_pil_image(model, transform, image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(image_tensor, normalize=True)
    return features.cpu().numpy()[0]
