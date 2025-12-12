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


def visualize_top_bottom_images(results, test_dataset, index_to_class, output_dir, num_classes=3):
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
            ax.set_title(f"Top {i+1}\n({score:.3f})", fontsize=9)
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
            ax.set_title(f"Bottom {i+1}\n({score:.3f})", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        plot_path = output_dir / f"top_bottom_images_{class_name}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Top/Bottom images saved to {plot_path}")
