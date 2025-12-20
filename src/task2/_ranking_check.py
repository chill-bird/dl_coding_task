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
            "top_5": top_5_indices.tolist(),
            "bottom_5": bottom_5_indices.tolist(),
            "top_5_scores": class_scores[top_5_indices].numpy().tolist(),
            "bottom_5_scores": class_scores[bottom_5_indices].numpy().tolist(),
        }

    return results, all_logits


def visualize_top_bottom_images(
    results, test_dataset, index_to_class, output_dir, num_classes=NUM_CLASSES
):
    """Visualize top-5 and bottom-5 images for selected classes."""
    classes_to_analyze = list(range(min(num_classes, len(index_to_class))))

    for class_idx in classes_to_analyze:
        class_name = index_to_class[class_idx]
        if class_name not in results:
            continue

        top_5_idx = results[class_name]["top_5"]
        bottom_5_idx = results[class_name]["bottom_5"]

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f"Top-5 (left) and Bottom-5 (right) Images for {class_name}", fontsize=14)

        # Top-5 images
        for i, idx in enumerate(top_5_idx):
            sample = test_dataset[idx]
            img = sample["image"].numpy().transpose(1, 2, 0)

            # Denormalize for visualization
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)

            ax = axes[0, i]
            ax.imshow(img)
            score = results[class_name]["top_5_scores"][i]
            filename = test_dataset.samples[idx]["file"].name
            ax.set_title(f"{filename}\n({score:.3f})", fontsize=8)
            ax.axis("off")

        # Bottom-5 images
        for i, idx in enumerate(bottom_5_idx):
            sample = test_dataset[idx]
            img = sample["image"].numpy().transpose(1, 2, 0)

            # Denormalize for visualization
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)

            ax = axes[1, i]
            ax.imshow(img)
            score = results[class_name]["bottom_5_scores"][i]
            filename = test_dataset.samples[idx]["file"].name
            ax.set_title(f"{filename}\n({score:.3f})", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        plot_path = output_dir / f"top_bottom_images_{class_name}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Top/Bottom images saved to {plot_path}")


def ranking_check(model, test_dataset, test_loader, index_to_class, device, output_dir):
    """Ranking check for 3 classes."""

    results_ranking, _ = find_top_bottom_images(
        model, test_dataset, test_loader, index_to_class, device, NUM_CLASSES
    )
    visualize_top_bottom_images(
        results_ranking, test_dataset, index_to_class, output_dir, NUM_CLASSES
    )

    # Save ranking results
    ranking_json_path = output_dir / "top_bottom_images.json"
    with open(ranking_json_path, "w") as f:
        json.dump(results_ranking, f, indent=4)
    print(f"\nRanking results saved to {ranking_json_path}")
