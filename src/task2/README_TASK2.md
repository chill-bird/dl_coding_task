"""
TASK 2 - EuroSAT Classification with Data Augmentation
========================================================

This directory contains the implementation for Task 2: Classification with RGB channels.

## Overview

This task performs a comprehensive training pipeline on the EuroSAT RGB dataset:

1. **Data Handling**: Loads EuroSAT RGB images using custom dataset class
2. **Training**: ResNet50 model with two different data augmentation strategies
3. **Evaluation**: Tracks accuracy and TPR per class over epochs
4. **Model Selection**: Selects best model based on validation performance
5. **Analysis**: Finds top-5 and bottom-5 predictions per class
6. **Reporting**: Generates comprehensive results and visualizations

## Files

- `task2.py` - Main training script
  - Implements training pipeline with 2 augmentation strategies
  - Generates plots for each augmentation setting
  - Saves best model and test logits
  
- `predict.py` - Prediction script
  - Load a saved model and make predictions on any split (train/val/test)
  - Save predictions and logits
  - Report per-class accuracy
  
- `generate_report.py` - Report generation script
  - Creates markdown report from experiment results
  - Summarizes all metrics and visualizations
  
- `eurosat_dataset.py` - Custom PyTorch Dataset
  - Handles EuroSAT image loading
  - Applies transformations
  
- `util.py` - Utility functions
  - Class index mapping

## Data Requirements

The following data structure is expected:

```
dat/
├── EuroSAT_RGB/
│   ├── class_index.csv          # Class to index mapping
│   ├── trainfile.csv            # Training split
│   ├── valfile.csv              # Validation split
│   ├── testfile.csv             # Test split
│   └── [class folders]/          # Images organized by class
│       ├── AnnualCrop/
│       ├── Forest/
│       ├── HerbaceousVegetation/
│       ├── Highway/
│       ├── Industrial/
│       ├── Pasture/
│       ├── PermanentCrop/
│       ├── Residential/
│       ├── River/
│       └── SeaLake/
```

## Usage

### Run Training

Basic usage with default data directory:
```bash
cd /path/to/dl_coding_task
python -m src.task2.task2
```

With custom data directory:
```bash
python -m src.task2.task2 -d /path/to/data/directory
```

With custom hyperparameters:
```bash
python -m src.task2.task2 -d /path/to/data -e 100 -b 64 -lr 0.0001
```

#### Arguments

- `-d, --data_dir`: Path to data directory (default: project dat/ folder)
- `--seed`: Random seed for reproducibility (default: 42)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)

### Make Predictions

Using a saved model:
```bash
python -m src.task2.predict /path/to/model.pt --split test
```

With custom data directory:
```bash
python -m src.task2.predict /path/to/model.pt -d /path/to/data --split test
```

#### Arguments

- `model_path`: Path to saved model checkpoint (required)
- `-d, --data_dir`: Path to data directory
- `--split`: Which split to predict on (train/val/test, default: test)
- `--batch_size`: Batch size for predictions (default: 32)
- `--output_dir`: Directory to save predictions

### Generate Report

From experiment results directory:
```bash
python -m src.task2.generate_report /path/to/results
```

With custom output file:
```bash
python -m src.task2.generate_report /path/to/results -o /path/to/report.md
```

## Output Structure

Training results are saved in: `results/task2_YYYYMMDD_HHMMSS/`

```
results/task2_YYYYMMDD_HHMMSS/
├── training_history_mild.png           # Accuracy/TPR plots for mild augmentation
├── training_history_strong.png         # Accuracy/TPR plots for strong augmentation
├── top_bottom_images_AnnualCrop.png    # Top-5 and bottom-5 images
├── top_bottom_images_Forest.png
├── top_bottom_images_HerbaceousVegetation.png
├── best_model_mild.pt                  # Saved best model
├── best_model_strong.pt
├── test_logits_best_model.npy          # Logits for test set
├── training_results.json               # Complete metrics
├── top_bottom_images.json              # Top-5/bottom-5 metadata
└── REPORT.md                           # Generated report
```

## Key Features

### 1. Data Augmentation Strategies

**Mild Augmentation:**
- Random horizontal flip (p=0.5)
- ImageNet normalization

**Strong Augmentation:**
- Random horizontal & vertical flips
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformations
- Gaussian blur
- ImageNet normalization

### 2. Metrics Tracked

For each augmentation strategy:
- **Per Epoch**: Overall accuracy, loss, TPR per class
- **Test Set**: Overall accuracy, TPR per class
- **Logits**: Raw model outputs for test set

### 3. Model Selection

Model selection is based on:
- Best validation accuracy across all epochs
- Automatically restores best model before test evaluation

### 4. Early Stopping

Training includes early stopping with:
- Patience: 10 epochs without improvement
- Learning rate reduction on plateau

### 5. Analysis

For each class, identifies:
- Top-5 highest confidence predictions
- Bottom-5 lowest confidence predictions
- Visualizes images with confidence scores

## Deliverables

The deliverables include:

1. **Training Plots**: One graph per augmentation strategy showing:
   - Overall accuracy per epoch
   - Loss per epoch
   - TPR per class per epoch

2. **Test Results**: Single set of results for selected model showing:
   - Overall accuracy on test set
   - TPR per class on test set

3. **Logits**: Saved model outputs (logits) for test data

4. **Top-5 Bottom-5 Analysis**: For 3 classes, shows:
   - 5 most confident predictions (images + scores)
   - 5 least confident predictions (images + scores)
   - Visualized as grids

5. **Report**: Comprehensive markdown report with:
   - Experimental setup
   - Results summary
   - Per-class performance
   - Analysis and conclusions

## Model Architecture

- **Base**: ResNet50 with ImageNet pretrained weights
- **Finetuning**: All layers are finetuned
- **Final Layer**: Linear layer with num_classes outputs
- **Optimizer**: Adam
- **Loss**: Cross-Entropy Loss
- **Scheduler**: ReduceLROnPlateau

## GPU Support

The script automatically uses GPU if available via CUDA.
Falls back to CPU if GPU is not available.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Requirements

See `requirements.txt` in project root:
- torch
- torchvision
- numpy
- pandas
- scikit-learn
- scikit-image
- matplotlib
- seaborn
- tqdm

## Example Workflow

```bash
# 1. Run training (generates results directory)
python -m src.task2.task2 -d /path/to/data --epochs 50

# 2. Results are saved to results/task2_YYYYMMDD_HHMMSS/

# 3. Generate report
python -m src.task2.generate_report results/task2_YYYYMMDD_HHMMSS/

# 4. Make predictions on new data (optional)
python -m src.task2.predict results/task2_YYYYMMDD_HHMMSS/best_model_strong.pt \
    -d /path/to/data --split test

# 5. View results in results/task2_YYYYMMDD_HHMMSS/
# - REPORT.md - Main report
# - training_history_*.png - Training curves
# - top_bottom_images_*.png - Analysis visualizations
```

## Reproducibility

All results are reproducible via:
- Manual seed setting for Python, NumPy, and PyTorch
- Deterministic data split via split files (trainfile.csv, valfile.csv, testfile.csv)
- Default seed: 42 (can be overridden with --seed)

## Notes

- First run may take longer due to data loading and model initialization
- GPU usage significantly accelerates training (50+ fps vs 10+ fps on CPU)
- Memory usage: ~6GB GPU / ~8GB RAM (batch_size=32)
- Training on full dataset typically takes 30-60 minutes (50 epochs)

"""
