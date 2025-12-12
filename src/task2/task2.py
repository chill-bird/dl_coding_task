"""
Task 2 - Classification with RGB channels

Perform a train-val-test split of the data which depends on a seed parameter, and use a manual
seed so that you and us can reproduce the same split your experiments. Use at least 2500
images for training, 1000 for validation and 2000 for testing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from src.task2.util import index_to_class_map
from src.task2.data_loader import dataloaders
from src.task2.plot import plot_training_history
from src.util.seed import set_seed
from src.util.paths import results_parent_dir


def get_augmentations() -> dict:
    """Define two different data augmentation settings."""

    # Augmentation 1: 1 augmentation (only random horizontal flip + normalization)
    augmentation_1 = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Augmentation 2: 2 augmentation with multiple transformations
    augmentation_2 = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Validation/Test augmentation (only normalization)
    val_augmentation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return {
        "mild": augmentation_1,
        "advanced": augmentation_2,
        "val": val_augmentation,
    }


def get_model(num_classes, device):
    """Load ResNet50 with pretrained weights and finetune all layers."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Finetune all layers - replace the final classification layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = model.to(device)
    return model


def calculate_metrics(outputs, targets):
    """Calculate accuracy and per-class TPR (True Positive Rate)."""
    _, predicted = torch.max(outputs, 1)

    # Overall accuracy
    accuracy = (predicted == targets).sum().item() / targets.size(0)

    # Per-class TPR
    num_classes = outputs.size(1)
    tpr_per_class = []

    for class_idx in range(num_classes):
        true_positives = ((predicted == class_idx) & (targets == class_idx)).sum().item()
        total_positives = (targets == class_idx).sum().item()
        tpr = true_positives / total_positives if total_positives > 0 else 0.0
        tpr_per_class.append(tpr)

    return accuracy, tpr_per_class


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for batch in tqdm(train_loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_outputs.append(outputs.detach().cpu())
        all_targets.append(labels.detach().cpu())

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    accuracy, tpr_per_class = calculate_metrics(all_outputs, all_targets)

    return total_loss / len(train_loader), accuracy, tpr_per_class


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(labels.detach().cpu())

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    accuracy, tpr_per_class = calculate_metrics(all_outputs, all_targets)

    return total_loss / len(val_loader), accuracy, tpr_per_class


def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=0.001):
    """Train model with early stopping based on validation accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_tpr": [],
        "val_tpr": [],
    }

    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc, train_tpr = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_tpr = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["train_tpr"].append(train_tpr)
        history["val_tpr"].append(val_tpr)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val TPR per class: {[f'{tpr:.4f}' for tpr in val_tpr]}")

        scheduler.step(val_acc)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✓ Best model updated (Val Acc: {best_val_accuracy:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def test_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(labels.detach().cpu())

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    accuracy, tpr_per_class = calculate_metrics(all_outputs, all_targets)

    return accuracy, tpr_per_class, all_outputs, all_targets


def find_top_bottom_images(model, test_dataset, test_loader, index_to_class, device, num_classes=3):
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
    classes_to_analyze = list(range(min(num_classes, len(index_to_class))))

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


def main(
    dataset_dir: Path,
    img_format: str,
    class_index_file_name: str,
    split_files: list[str],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
):

    print("=" * 60 + "\nTASK 2 - Training model\n" + "=" * 60 + "\n")

    # Set seed
    set_seed(seed)

    # Create output directory
    output_dir = Path(
        results_parent_dir() / "results" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)  # Raise error if already exists
    print(f"\nOutput directory: {output_dir}\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Get augmentations
    augmentation_dict = get_augmentations()

    # Load class mapping
    class_index_file = Path(dataset_dir / class_index_file_name).resolve()
    index_to_class = index_to_class_map(class_index_file)
    num_classes = len(index_to_class)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {index_to_class}\n")

    # Train models with different augmentations
    all_results = {}
    best_model_path = None
    best_val_accuracy = 0.0
    best_augmentation = None

    for augmentation_name in ["mild", "advanced"]:
        print(f"\n{'='*60}")
        print(f"Training with {augmentation_name.upper()} augmentation")
        print(f"{'='*60}\n")

        # Create dataloaders
        train_loader, val_loader, test_loader, test_dataset = dataloaders(
            dataset_dir=dataset_dir,
            split_files=split_files,
            img_format=img_format,
            aug_dict=augmentation_dict,
            aug_name=augmentation_name,
            batch_size=batch_size,
        )

        print(f"Train set size: {len(train_loader.dataset)}")
        print(f"Val set size: {len(val_loader.dataset)}")
        print(f"Test set size: {len(test_loader.dataset)}\n")

        # Create and train model
        model = get_model(num_classes, device)
        print("Model created: ResNet50\n")

        trained_model, history = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=epochs,
            device=device,
            learning_rate=learning_rate,
        )

        # Plot training history
        plot_training_history(history, index_to_class, augmentation_name, output_dir)

        # Test on test set
        test_accuracy, test_tpr, test_logits, test_targets = test_model(
            trained_model, test_loader, device
        )

        print(f"TEST RESULTS - {augmentation_name.upper()} augmentation")
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print("Test TPR per class:")
        for class_idx, tpr in enumerate(test_tpr):
            print(f"  {index_to_class[class_idx]}: {tpr:.4f}")

        # Save results
        results = {
            "augmentation": augmentation_name,
            "test_accuracy": float(test_accuracy),
            "test_tpr": [float(t) for t in test_tpr],
            "val_accuracy": [float(a) for a in history["val_accuracy"]],
            "val_tpr": [[float(t) for t in tpr_list] for tpr_list in history["val_tpr"]],
        }
        all_results[augmentation_name] = results

        # Save model and logits if best so far
        if max(history["val_accuracy"]) > best_val_accuracy:
            best_val_accuracy = max(history["val_accuracy"])
            best_augmentation = augmentation_name
            best_model_path = output_dir / f"best_model_{augmentation_name}.pt"
            torch.save(trained_model.state_dict(), best_model_path)
            best_logits = test_logits
            best_model = trained_model
            best_test_dataset = test_dataset
            print(f"\n✓ New best model saved: {best_model_path}\n")

    # Save all results to JSON
    results_json_path = output_dir / "training_results.json"
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {results_json_path}")

    # Save test logits for best model
    logits_path = output_dir / "test_logits_best_model.npy"
    np.save(logits_path, best_logits.numpy())
    print(f"Test logits saved to {logits_path}")

    # Find and visualize top-5 and bottom-5 images
    print(f"\n{'='*60}")
    print("Finding top-5 and bottom-5 images for selected classes")
    print(f"{'='*60}\n")

    results_ranking, _ = find_top_bottom_images(
        best_model, best_test_dataset, test_loader, index_to_class, device, num_classes=3
    )
    visualize_top_bottom_images(
        results_ranking, best_test_dataset, index_to_class, output_dir, num_classes=3
    )

    # Save ranking results
    ranking_json_path = output_dir / "top_bottom_images.json"
    with open(ranking_json_path, "w") as f:
        json.dump(results_ranking, f, indent=4)
    print(f"\nRanking results saved to {ranking_json_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Best augmentation: {best_augmentation}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"Results directory: {output_dir}")

    return output_dir
