"""
model.py
---

Late Fusion model that is used in task 3.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50LateFusion(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50LateFusion, self).__init__()

        # Branch A (3 channels)
        self.branch_a = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove classifier
        self.branch_a.fc = nn.Identity()

        # Branch B (3 channels)
        self.branch_b = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.branch_b.fc = nn.Identity()

        # Fusion classifier layer
        self.fusion_classifier = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        # x: (Batch, 6, 64, 64)
        x_a = x[:, :3, :, :]  # Channels 0-2
        x_b = x[:, 3:6, :, :]  # Channels 3-5

        # extract features
        f_a = self.branch_a(x_a)
        f_b = self.branch_b(x_b)

        # Combine features
        combined = torch.cat((f_a, f_b), dim=1)  # Shape: (Batch, 4096)

        return self.fusion_classifier(combined)


# # Test model
# model = ResNet50LateFusion(num_classes=6)
# print(model.eval())


def get_model(num_classes, device):
    """Load ResNet50 with pretrained weights and finetune by replacing last layer."""
    model = ResNet50LateFusion(num_classes)

    model = model.to(device)
    return model


def load_model_from_checkpoint(model_path, num_classes, device):
    """Load a trained model from checkpoint."""
    model = ResNet50LateFusion(num_classes)
    # 2. Load the state dictionary
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)
    )
    # 3. Move the model to the specified device
    model = model.to(device)
    model.eval()

    return model
