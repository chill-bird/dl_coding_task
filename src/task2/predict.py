"""
predict.py
---

Prediction script for using a saved EuroSAT model.
Loads a trained model and makes predictions on test data.
"""

import torch
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

from src.constants import (
    BATCH_SIZE,
    BEST_MODEL_FILENAME,
    PREDICTIONS_DIR_NAME,
    CLASS_INDEX_FILE,
    DATASET_DIR_NAME,
    EPOCHS,
    IMG_FORMAT,
    LEARNING_RATE,
    SEED,
    SPLIT_FILES,
)
from src.task2.fine_tune import AUGMENTATIONS
from src.task2._data_loader import test_dataloader
from src.task2.util import class_to_index_map, index_to_class_map
from src.task2._model import load_model_from_checkpoint
from src.util.paths import root_path, results_parent_dir
from src.util.run_config import get_dat_dir_args


def predict(model, dataloader, device):
    """Make predictions on a dataset."""
    all_logits = []
    all_predictions = []
    all_targets = []

    print("\nMaking predictions on test set...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            all_logits.append(logits.detach().cpu())
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    return all_logits, all_predictions, all_targets


def calculate_accuracy_per_class(predictions, targets, index_to_class):
    """Calculate per-class accuracy."""
    num_classes = len(index_to_class)
    accuracies = {}

    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        if class_mask.sum() > 0:
            class_accuracy = (
                predictions[class_mask] == targets[class_mask]
            ).sum().item() / class_mask.sum().item()
            accuracies[index_to_class[class_idx]] = float(class_accuracy)

    return accuracies


def predict_on_test_set(
    dataset_dir: Path,
    img_format: str,
    class_index_file_name: str,
    split_files: list[str],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
):

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class mapping
    class_index_file = Path(dataset_dir / class_index_file_name).resolve()
    assert class_index_file.is_file(), f"Class index file not found at {class_index_file}"
    index_to_class = index_to_class_map(class_index_file)
    class_to_index = class_to_index_map(class_index_file)
    num_classes = len(index_to_class)
    print(f"Classes: {index_to_class}\n")
    print(f"Number of classes: {num_classes}")

    # Setup output directory
    output_dir = Path(results_parent_dir() / PREDICTIONS_DIR_NAME).resolve()
    output_dir.mkdir(parents=False, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    # Load model
    model_path = Path(results_parent_dir() / BEST_MODEL_FILENAME).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"\nLoading model from {model_path}")
    model = load_model_from_checkpoint(model_path, num_classes, device)
    print("Model loaded successfully")

    # Dataloader
    dataloader = test_dataloader(
        dataset_dir=dataset_dir,
        split_files=split_files,
        class_to_index_map=class_to_index,
        img_format=img_format,
        aug_dict=AUGMENTATIONS,
        batch_size=batch_size,
    )

    # Make predictions
    logits, predictions, targets = predict(model, dataloader, device)

    # Calculate metrics
    overall_accuracy = (predictions == targets).sum().item() / len(targets)
    per_class_accuracy = calculate_accuracy_per_class(predictions, targets, index_to_class)

    # Print results
    print(f"\n{'='*30}")
    print("PREDICTION RESULTS ON TEST SET")
    print(f"{'='*30}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print("\nPer-class Accuracy:")
    for class_name, accuracy in per_class_accuracy.items():
        print(f"  {class_name}: {accuracy:.4f}")

    # Save results
    results = {
        "split": "test",
        "overall_accuracy": float(overall_accuracy),
        "per_class_accuracy": per_class_accuracy,
        "model_path": str(model_path),
    }

    results_path = output_dir / "predictions_test_set.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")

    # Save logits
    logits_path = output_dir / "logits_test_set.npy"
    np.save(logits_path, logits.numpy())
    print(f"Logits saved to {logits_path}")

    # Save predictions
    predictions_path = output_dir / "predictions_test_set.npy"
    np.save(predictions_path, predictions.numpy())
    print(f"Predictions saved to {predictions_path}")

    print(f"\nOutput directory: {output_dir}")


def run():
    """Runs fine tuning script."""

    # Dataset parent directory (dat_dir) containing zip files
    dat_dir = get_dat_dir_args()
    dataset_dir = dat_dir / DATASET_DIR_NAME

    print(f"Settings:\nROOT DIR:{root_path()}\nDAT_DIR:  {dat_dir}\nIMG_EXT:  {IMG_FORMAT}\n")

    # Run Task 2: Fine-tune model
    predict_on_test_set(
        dataset_dir=dataset_dir,
        img_format=IMG_FORMAT,
        class_index_file_name=CLASS_INDEX_FILE,
        split_files=SPLIT_FILES,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )


if __name__ == "__main__":
    run()
