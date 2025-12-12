"""
plots.py
---

Provides methods to plot data for task2.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_history(
    history: dict, index_to_class: dict, augmentation_name: str, output_dir: Path
):
    """Plot accuracy and TPR per class over epochs."""
    num_classes = len(index_to_class)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Training History - {augmentation_name} Augmentation", fontsize=16)

    # Overall Accuracy
    axes[0, 0].plot(history["train_accuracy"], label="Train", marker="o")
    axes[0, 0].plot(history["val_accuracy"], label="Val", marker="s")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Overall Accuracy per Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(history["train_loss"], label="Train", marker="o")
    axes[0, 1].plot(history["val_loss"], label="Val", marker="s")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Loss per Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Train TPR per class
    train_tpr_array = np.array(history["train_tpr"])
    for class_idx in range(num_classes):
        axes[1, 0].plot(
            train_tpr_array[:, class_idx], label=index_to_class[class_idx], marker="o", markersize=3
        )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("TPR")
    axes[1, 0].set_title("Train TPR per Class per Epoch")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True)

    # Val TPR per class
    val_tpr_array = np.array(history["val_tpr"])
    for class_idx in range(num_classes):
        axes[1, 1].plot(
            val_tpr_array[:, class_idx], label=index_to_class[class_idx], marker="s", markersize=3
        )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("TPR")
    axes[1, 1].set_title("Validation TPR per Class per Epoch")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True)

    plt.tight_layout()
    plot_path = output_dir / f"training_history_{augmentation_name}.png"
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")
