"""
model.py
---

Model which is used for task2.
"""

import torch.nn as nn
import torchvision.models as models


def get_model(num_classes, device):
    """Load ResNet50 with pretrained weights and finetune all layers."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Finetune all layers - replace the final classification layer
    # TODO All layers? Or only last layer??
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = model.to(device)
    return model
