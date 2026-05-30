from __future__ import annotations

import torch.nn as nn
from torchvision import models

DEFAULT_ARCHITECTURE = "densenet121"


def build_classifier(
    num_classes: int,
    architecture: str = DEFAULT_ARCHITECTURE,
    pretrained: bool = False,
) -> tuple[nn.Module, str]:
    arch = architecture.lower()

    if arch == "densenet121":
        if pretrained:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model, arch

    if arch == "resnet18":
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model, arch

    raise ValueError(f"Unsupported architecture: {architecture}")
