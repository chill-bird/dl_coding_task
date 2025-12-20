"""
model.py
---

Models which are used throughout task 2.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_model(num_classes, device):
    """Load ResNet50 with pretrained weights and finetune by replacing last layer."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Finetune by replacing final classification layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = model.to(device)
    return model


def load_model_from_checkpoint(model_path, num_classes, device):
    """Load a trained model from checkpoint."""

    model = models.resnet50(weights=None)  # No pretrained weights
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Load checkpoint
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model
