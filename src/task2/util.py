"""
util.py
---

Provides utility functions for task2.
"""

import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn



def parse_class_index(class_index_file: Path) -> pd.DataFrame:
    """Parses class to index file as pd.DataFrame"""

    assert class_index_file.is_file(), f"Class index file not found at {class_index_file}"
    df = pd.read_csv(class_index_file)
    return df


def class_to_index_map(class_index_file: Path) -> dict[str, int]:
    """Returns dict of class (key), index (value)"""
    df = parse_class_index(class_index_file)
    return dict(zip(df["class"], df["index"]))


def index_to_class_map(class_index_file: Path) -> dict[str, int]:
    """Returns dict of index (key), class (value)"""
    df = parse_class_index(class_index_file)
    return dict(zip(df["index"], df["class"]))

def find_top_bottom_images(
    model: nn.Module,
    test_dataset,
    test_loader,
    index_to_class: dict[int, str],
    device: torch.device,
    num_classes: int = 3,
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
            "top_5": top_5_indices.tolist(),
            "bottom_5": bottom_5_indices.tolist(),
            "top_5_scores": class_scores[top_5_indices].numpy().tolist(),
            "bottom_5_scores": class_scores[bottom_5_indices].numpy().tolist(),
        }

    return results, all_logits