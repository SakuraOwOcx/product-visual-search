from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .config import DEVICE, VIT_SUPERVISED_BEST_CHECKPOINT


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_vit_eval_transform(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_vit_b16(num_classes, pretrained=True, strict_pretrained=False):
    weights = None
    if pretrained:
        try:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        except Exception as exc:
            if strict_pretrained:
                raise RuntimeError(f"Could not load torchvision ViT weights metadata: {exc}") from exc
            print(f"warning=vit_pretrained_weights_unavailable_falling_back_to_random reason={exc}")
    try:
        model = models.vit_b_16(weights=weights)
    except Exception as exc:
        if strict_pretrained:
            raise RuntimeError(f"Could not instantiate pretrained ViT-B/16: {exc}") from exc
        print(f"warning=vit_pretrained_init_failed_falling_back_to_random reason={exc}")
        model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def extract_vit_features(model, images):
    # torchvision VisionTransformer stores the CLS token output before classification in encoder output[:, 0].
    x = model._process_input(images)
    n = x.shape[0]
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = model.encoder(x)
    return x[:, 0]


def load_vit_supervised_model(checkpoint_path=VIT_SUPERVISED_BEST_CHECKPOINT):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "ViT supervised checkpoint is not available yet. "
            f"Expected checkpoint: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    num_classes = int(checkpoint.get("num_classes", 0))
    if num_classes <= 0:
        raise ValueError("ViT checkpoint is missing a valid num_classes value.")
    model = build_vit_b16(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    image_size = int(checkpoint.get("image_size", 224))
    transform = get_vit_eval_transform(image_size=image_size)
    return model, transform, checkpoint


def encode_vit_pil_image(model, transform, image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = extract_vit_features(model, image_tensor)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy()[0]
