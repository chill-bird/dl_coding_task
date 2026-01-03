"""
Task 3 - Classification with TIF channels

Fine-tunes image model to classify data from eurosat_dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from src.constants import (
    BATCH_SIZE,
    BEST_MODEL_FILENAME,
    TIF_DATASET_DIR_NAME,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    SPLIT_FILES,
    LOGITS_TEST_SET_FILE,
)
from src.task3._data_loader import dataloaders
from src.task3._model import get_model
from src.task3._plot import plot_training_history
from src.task3._classname_index_mapping import index_to_class_map
from src.util.paths import results_parent_dir, root_path
from src.util.run_config import get_dat_dir_args
from src.util.seed import set_seed
from src.task3._normalize_multi_channel import NormalizeMultiChannel

# Data augmentations for model selection
AUGMENTATIONS = {
    "mild": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            
            NormalizeMultiChannel()
        ]
    ),
    "advanced": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            
            NormalizeMultiChannel()   
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            NormalizeMultiChannel()
        ]
    ),
}


def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, list[float]]:
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


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, list[float]]:
    """Train for one epoch."""

    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for batch in tqdm(train_loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()  # Prevent accumulating gradients of previous batches
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Sum over losses of each item in a batch
        all_outputs.append(outputs.detach().cpu())
        all_targets.append(labels.detach().cpu())

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    accuracy, tpr_per_class = calculate_metrics(all_outputs, all_targets)

    return total_loss / len(train_loader), accuracy, tpr_per_class


def validate(
    model: nn.Module, val_loader, criterion: nn.Module, device: torch.device
) -> tuple[float, float, list[float]]:
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


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    device: torch.device,
    learning_rate: float,
) -> tuple[nn.Module, dict[str, list]]:
    """Train model with early stopping based on validation accuracy."""

    criterion = nn.CrossEntropyLoss()  # for multi-class
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )  # Adjust learning rate

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


def test_model(
    model: nn.Module, test_loader, device: torch.device
) -> tuple[float, list[float], torch.Tensor, torch.Tensor]:
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


def fine_tune(
    dataset_dir: Path,
    img_format: str,
    split_files: list[str],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[Path, Path]:
    """
    Fine-tunes model for EuroSat image classification.
    Two models, varying in augmentation mode, are trained.
    Selects model based on validation accuracy.

    Args:
        dataset_dir: Directory containing dataset (sub directory of dat_dir)
        img_format: Image file extension of dataset
        split_files: Names of split files containing train, test, val sets
        learning_rate: Learning rate for training
        epochs: Maximum epochs during training
        batch_size: Batch size
        seed: Seed for RNG

    Returns tuple of
        - Path to best model
        - Path to rest of output files
    """

    print("=" * 60 + "\nTASK 3 - Training model\n" + "=" * 60 + "\n")

    # Set seed
    set_seed(seed)

    # Create output directory
    output_dir = Path(
        results_parent_dir() / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)  # Raise error if already exists
    print(f"\nOutput directory: {output_dir}\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load class mapping
    index_to_class = index_to_class_map(dataset_dir)
    num_classes = len(index_to_class)
    print(f"Classes: {index_to_class}\n")
    print(f"Number of classes: {num_classes}")

    # Train models with different augmentations
    all_results = {}
    best_model_path = None
    best_val_accuracy = 0.0
    best_augmentation = None

    for augmentation_name in ["mild", "advanced"]:
        print(f"\n{'='*30}")
        print(f"Training with {augmentation_name.upper()} augmentation")
        print(f"{'='*30}\n")

        # Create dataloaders
        train_loader, val_loader, test_loader, test_dataset = dataloaders(
            dataset_dir=dataset_dir,
            split_files=split_files,
            img_format=img_format,
            aug_dict=AUGMENTATIONS,
            aug_name=augmentation_name,
            batch_size=batch_size,
        )

        print(f"Train set size: {len(train_loader.dataset)}")
        print(f"Val set size: {len(val_loader.dataset)}")
        print(f"Test set size: {len(test_loader.dataset)}\n")

        # Create and train model based on validation accuracy
        model = get_model(num_classes, device)
        print(f"Model: {model.__class__.__name__}\n")

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

        # Test on test set (hold-out set)
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
            print(f"\n✓ New best model saved: {best_model_path}\n")

    # Save overall best model once again for follow-up scripts
    overall_best_model_path = output_dir / BEST_MODEL_FILENAME
    torch.save(best_model.state_dict(), overall_best_model_path)

    # Save all results to JSON
    results_json_path = output_dir / "training_results.json"
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {results_json_path}")

    # Save test logits for best model (Task 3.2)
    logits_path = output_dir / LOGITS_TEST_SET_FILE
    np.save(logits_path, best_logits.numpy())
    print(f"Test logits saved to {logits_path}")

    print(f"\n{'='*30}")
    print("SUMMARY")
    print(f"{'='*30}")
    print(f"Best augmentation: {best_augmentation}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"Results directory: {output_dir}")

    print("\n[✓] TASK 3 - Fine-tuning completed")

    return best_model_path, output_dir


def run():
    """Runs fine tuning script."""

    img_format = ".tif"

    # Dataset parent directory (dat_dir) containing zip files
    dat_dir = get_dat_dir_args()
    dataset_dir = dat_dir / TIF_DATASET_DIR_NAME

    print(f"Settings:\nROOT DIR: {root_path()}\nDAT DIR: {dat_dir}\nIMG_EXT: {img_format}\n")

    # Run Task 3: Fine-tune model
    fine_tune(
        dataset_dir=dataset_dir,
        img_format=img_format,
        split_files=SPLIT_FILES,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )


if __name__ == "__main__":
    run()
