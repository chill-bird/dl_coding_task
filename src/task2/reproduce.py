"""
reproduce.py
---

Loads best model from most recent train run and
- checks for reproducibility.
- performs visual check for top/bottom predictions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import json

from src.constants import (
    BATCH_SIZE,
    BEST_MODEL_FILENAME,
    REPRODUCE_OUTPUT_DIR_NAME,
    DATASET_DIR_NAME,
    IMG_FORMAT,
    LOGITS_TEST_SET_FILE,
    REPRODUCED_LOGITS_TEST_SET_FILE,
    SPLIT_FILES,
)
from src.task2.fine_tune import AUGMENTATIONS, test_model
from src.task2._data_loader import dataloaders
from src.task2._ranking_check import ranking_check
from src.task2._classname_index_mapping import index_to_class_map
from src.task2._model import load_model_from_checkpoint
from src.util.paths import find_most_recent_train_results_dir, root_path
from src.util.run_config import get_dat_dir_args


def equals_saved_logits(logits: torch.Tensor, saved_logits_path: Path) -> bool:
    """Compares a logits tensor to the logits saved previously."""
    saved_logits = np.load(saved_logits_path)
    logits_np = logits.cpu().numpy()
    return np.allclose(logits_np, saved_logits)


def predict_on_test_set(
    model: nn.Module,
    model_path: Path,
    test_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
):

    # Test on test set
    print("[3] Predict on test set")
    test_accuracy, test_tpr, test_logits, test_targets = test_model(model, test_loader, device)

    # Save results
    results = {
        "split": "test",
        "overall_accuracy": float(test_accuracy),
        "per_class_accuracy": test_tpr,
        "model_path": str(model_path),
    }

    results_path = output_dir / "predictions_test_set.json"
    print(f"[4] Saving result overview saved to {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save logits
    logits_path = output_dir / REPRODUCED_LOGITS_TEST_SET_FILE
    print(f"[5] Saving logits to {logits_path}")
    np.save(logits_path, test_logits.numpy())

    return test_logits


def run():
    """
    Runs reproduce script.
    Loads model from checkpoint and compare predictions of model on test set.
    Saves top-/bottom-5 predictions in three classes as images

    Results are saved in REPRODUCE_OUTPUT_DIR_NAME directory.
    """

    # Dataset parent directory (dat_dir) containing zip files
    dat_dir = get_dat_dir_args()
    dataset_dir = dat_dir / DATASET_DIR_NAME
    results_dir = find_most_recent_train_results_dir()
    model_path = results_dir / BEST_MODEL_FILENAME
    previous_predictions_path = results_dir / LOGITS_TEST_SET_FILE
    predictions_output_dir = find_most_recent_train_results_dir() / REPRODUCE_OUTPUT_DIR_NAME
    predictions_output_dir.mkdir(parents=False, exist_ok=True)

    print(f"Settings:\nROOT DIR: {root_path()}\nDAT DIR: {dat_dir}\nIMG_EXT: {IMG_FORMAT}\n")
    print(f"\nOutput directory: {predictions_output_dir}\n")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class mapping
    index_to_class = index_to_class_map(dataset_dir)
    num_classes = len(index_to_class)
    print(f"Classes: {index_to_class}\n")
    print(f"Number of classes: {num_classes}")

    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    print(f"[1] Loading model checkpoint from {model_path}")
    model = load_model_from_checkpoint(model_path, num_classes, device)

    # Create test loader
    print("[2] Creating test data loader")
    _, _, test_loader, test_dataset = dataloaders(
        dataset_dir=dataset_dir,
        split_files=SPLIT_FILES,
        img_format=IMG_FORMAT,
        aug_dict=AUGMENTATIONS,
        aug_name="val",
        batch_size=BATCH_SIZE,
    )

    # Load checkpoint and predict on test set
    test_logits = predict_on_test_set(
        model=model,
        model_path=model_path,
        test_loader=test_loader,
        output_dir=predictions_output_dir,
        device=device,
    )

    # Compare to previously saved logits
    if equals_saved_logits(test_logits, previous_predictions_path):
        print("\n[✓] Logits were reproduced successfully")
    else:
        print("ERROR. Logits do not match.")

    print("[6] Performing ranking check")
    ranking_check(
        model=model,
        test_dataset=test_dataset,
        test_loader=test_loader,
        index_to_class=index_to_class,
        device=device,
        output_dir=predictions_output_dir,
    )

    print("\n[✓] TASK 2 - Validation completed")


if __name__ == "__main__":
    run()
