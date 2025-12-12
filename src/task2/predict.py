"""
predict.py
---

Prediction script for using a saved EuroSAT model.
Loads a trained model and makes predictions on test data.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
import json
from tqdm import tqdm

from src.task2.eurosat_dataset import EuroSatDataset
from src.task2.util import index_to_class_map
from src.util.paths import dataset_path, parse_dat_dir, root_path
from src.util.constants import DATASETS, TEST_FILE, VAL_FILE, TRAIN_FILE, CLASS_INDEX_FILE


def load_model(model_path, num_classes, device):
    """Load a trained model from checkpoint."""
    model = models.resnet50(weights=None)  # No pretrained weights
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load checkpoint
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def predict_on_dataset(model, dataloader, device, dataset_name=""):
    """Make predictions on a dataset."""
    all_logits = []
    all_predictions = []
    all_targets = []
    
    print(f"\nMaking predictions on {dataset_name} set...")
    
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
            class_accuracy = (predictions[class_mask] == targets[class_mask]).sum().item() / class_mask.sum().item()
            accuracies[index_to_class[class_idx]] = float(class_accuracy)
    
    return accuracies


def main():
    parser = argparse.ArgumentParser(description="Make predictions using a saved EuroSAT model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved model checkpoint (.pt file)"
    )
    parser.add_argument(
        "-d", "--data_dir",
        type=str,
        default=None,
        help="Path to data directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to predict on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for predictions"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save predictions and logits"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    dat_dir = parse_dat_dir(args.data_dir)
    dataset_dir = dataset_path(DATASETS, "euro_sat_rgb", dat_dir)
    class_index_file = Path(dataset_dir / CLASS_INDEX_FILE).resolve()
    
    # Get class mapping
    index_to_class = index_to_class_map(class_index_file)
    num_classes = len(index_to_class)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        root = root_path()
        output_dir = root / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"\nLoading model from {model_path}")
    model = load_model(model_path, num_classes, device)
    print("Model loaded successfully")
    
    # Select split file
    split_file_map = {
        "train": TRAIN_FILE,
        "val": VAL_FILE,
        "test": TEST_FILE,
    }
    split_file = split_file_map[args.split]
    
    # Create dataset and dataloader
    dataset = EuroSatDataset(
        root_dir=dataset_dir,
        split_file=split_file,
        class_index_file=CLASS_INDEX_FILE,
        img_format=".jpg",
        transform=None,  # Will add transform in dataset
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Make predictions
    logits, predictions, targets = predict_on_dataset(model, dataloader, device, args.split)
    
    # Calculate metrics
    overall_accuracy = (predictions == targets).sum().item() / len(targets)
    per_class_accuracy = calculate_accuracy_per_class(predictions, targets, index_to_class)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS ON {args.split.upper()} SET")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"\nPer-class Accuracy:")
    for class_name, accuracy in per_class_accuracy.items():
        print(f"  {class_name}: {accuracy:.4f}")
    
    # Save results
    results = {
        "split": args.split,
        "overall_accuracy": float(overall_accuracy),
        "per_class_accuracy": per_class_accuracy,
        "model_path": str(model_path),
    }
    
    results_path = output_dir / f"predictions_{args.split}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")
    
    # Save logits
    logits_path = output_dir / f"logits_{args.split}.npy"
    np.save(logits_path, logits.numpy())
    print(f"Logits saved to {logits_path}")
    
    # Save predictions
    predictions_path = output_dir / f"predictions_{args.split}.npy"
    np.save(predictions_path, predictions.numpy())
    print(f"Predictions saved to {predictions_path}")
    
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
