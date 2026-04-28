import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .config import DEVICE, FULL_CLASS_MAPPING_JSON, RESNET18_FULL_BEST_CHECKPOINT


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_resnet_eval_transform(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ResNet18EmbeddingModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def encode_image(self, x):
        layers = list(self.backbone.children())[:-1]
        feature_extractor = nn.Sequential(*layers).to(x.device)
        features = feature_extractor(x)
        return torch.flatten(features, 1)


def build_resnet18(num_classes, pretrained=True):
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_class_mapping(mapping_path=FULL_CLASS_MAPPING_JSON):
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Class mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_resnet18_full_model(checkpoint_path=RESNET18_FULL_BEST_CHECKPOINT):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "ResNet18 full model is not available yet. "
            f"Expected checkpoint: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    num_classes = int(checkpoint.get("num_classes", 0))
    if num_classes <= 0:
        raise ValueError("Checkpoint is missing a valid num_classes value.")
    model = build_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    image_size = int(checkpoint.get("image_size", 224))
    transform = get_resnet_eval_transform(image_size=image_size)
    return model, transform, checkpoint


def encode_resnet_pil_image(model, transform, image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        layers = list(model.children())[:-1]
        feature_extractor = nn.Sequential(*layers).to(DEVICE)
        features = feature_extractor(image_tensor)
        features = torch.flatten(features, 1)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy()[0]
