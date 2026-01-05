"""
_ranking_check.py
---

Performs ranking check for top- and bottom scoring images for three classes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import matplotlib.pyplot as plt
import numpy as np


NUM_CLASSES = 3


def find_top_bottom_images(
    model: nn.Module,
    test_dataset: Dataset,
    test_loader: DataLoader,
    index_to_class: dict[int, str],
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> tuple[dict, torch.Tensor]:
    """Find top-5 and bottom-5 scoring images for each class."""
    model.eval()

    # Store logits, predictions, and indices for all test samples
    all_logits = []
    all_true_labels = []

    with torch.no_grad():
        idx = 0
        for batch in test_loader:
            images = batch["image"].to(device)
            logits = model(images)
            all_logits.append(logits.detach().cpu())
            all_true_labels.append(batch["label"])
            idx += len(batch["image"])

    all_logits = torch.cat(all_logits)
    all_true_labels = torch.cat(all_true_labels)

    # For each class, find top-5 and bottom-5 predictions
    results = {}

    # Select classes to visualize (first 3 classes)
    assert (
        len(index_to_class) > num_classes
    ), f"Index to class map must contain at least {num_classes} entries for ranking check."
    classes_to_analyze = list(range(num_classes))

    for class_idx in classes_to_analyze:
        class_name = index_to_class[class_idx]

        # Get scores for this class from logits
        class_scores = all_logits[:, class_idx]

        # Get indices sorted by score
        sorted_indices = torch.argsort(class_scores, descending=True)

        # Top-5 and bottom-5 indices
        top_5_indices = sorted_indices[:5].numpy()
        bottom_5_indices = sorted_indices[-5:].numpy()

        results[class_name] = {
            "top_5": [
                test_dataset.samples[idx]["file"].name for idx in top_5_indices.tolist()
            ],
            "bottom_5": [
                test_dataset.samples[idx]["file"].name
                for idx in bottom_5_indices.tolist()
            ],
            "top_5_scores": class_scores[top_5_indices].numpy().tolist(),
            "bottom_5_scores": class_scores[bottom_5_indices].numpy().tolist(),
        }

    return results, all_logits


def ranking_check(model, test_dataset, test_loader, index_to_class, device, output_dir):
    """Ranking check for 3 classes."""

    results_ranking, _ = find_top_bottom_images(
        model, test_dataset, test_loader, index_to_class, device, NUM_CLASSES
    )

    # Save ranking results
    ranking_json_path = output_dir / "top_bottom_images.json"
    with open(ranking_json_path, "w") as f:
        json.dump(results_ranking, f, indent=4)
    print(f"\nRanking results saved to {ranking_json_path}")
