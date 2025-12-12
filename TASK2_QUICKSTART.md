# Task 2: EuroSAT Classification - Quick Start Guide

## Summary

You now have a complete training pipeline for Task 2. Here's what's implemented:

### ✅ Deliverables

1. **Two Data Augmentation Strategies**
   - ✅ Mild: Random horizontal flip only
   - ✅ Strong: Multiple augmentations (rotation, color jitter, affine, blur, flips)

2. **Training Pipeline**
   - ✅ ResNet50 with pretrained weights
   - ✅ All layers finetuned
   - ✅ Early stopping and learning rate scheduling
   - ✅ Validation accuracy tracking per epoch

3. **Evaluation Metrics**
   - ✅ Overall accuracy per epoch
   - ✅ TPR (True Positive Rate) per class per epoch
   - ✅ Per-class accuracy on test set

4. **Visualizations**
   - ✅ Training curves (accuracy/loss/TPR per epoch) - one graph per augmentation
   - ✅ Top-5 and bottom-5 images per class with confidence scores

5. **Model Artifacts**
   - ✅ Best model saved after training
   - ✅ Test logits saved for selected model
   - ✅ Separate prediction script for inference

6. **Report Generation**
   - ✅ Comprehensive markdown report
   - ✅ Experimental setup documentation
   - ✅ Results summary with comparison

## Quick Start

### 1. Run Training

```bash
cd /home/elisa/OneDrive/Studium/DeepLearning/dl_coding_task
python -m src.task2.task2 -d ./dat
```

This will:
- Train with mild augmentation
- Train with strong augmentation
- Generate training plots for each
- Test on test set
- Save best model and logits
- Create top-5/bottom-5 visualizations
- Save results to `results/task2_YYYYMMDD_HHMMSS/`

### 2. Generate Report

```bash
python -m src.task2.generate_report results/task2_YYYYMMDD_HHMMSS/
```

This creates `REPORT.md` in the results directory.

### 3. Use Saved Model for Predictions

```bash
python -m src.task2.predict results/task2_YYYYMMDD_HHMMSS/best_model_strong.pt \
    -d ./dat --split test
```

## File Structure

```
src/task2/
├── task2.py                  ← Main training script
├── predict.py               ← Prediction script
├── generate_report.py       ← Report generation
├── eurosat_dataset.py       ← Dataset class
├── util.py                  ← Utilities
└── README_TASK2.md          ← Full documentation
```

## Key Features

| Feature | Status | Details |
|---------|--------|---------|
| Data Augmentation (Mild) | ✅ | Horizontal flip + normalization |
| Data Augmentation (Strong) | ✅ | 7 transformations including rotation, color jitter, blur |
| Training with Early Stopping | ✅ | Patience=10, LR scheduling |
| Per-Epoch Metrics | ✅ | Accuracy + TPR per class + loss |
| Model Selection | ✅ | Based on best validation accuracy |
| Test Evaluation | ✅ | Accuracy + per-class TPR |
| Logits Saving | ✅ | Saved as numpy array |
| Top-5/Bottom-5 Analysis | ✅ | For 3 classes, with visualizations |
| Inference Script | ✅ | Load and predict with saved model |
| Report Generation | ✅ | Markdown with full results |

## Output Files

After training completes, you'll find in `results/task2_YYYYMMDD_HHMMSS/`:

**Visualizations:**
- `training_history_mild.png` - Plots for mild augmentation
- `training_history_strong.png` - Plots for strong augmentation
- `top_bottom_images_AnnualCrop.png` - Top/bottom images
- `top_bottom_images_Forest.png`
- `top_bottom_images_HerbaceousVegetation.png` (or other classes)

**Data:**
- `best_model_mild.pt` - Model weights
- `best_model_strong.pt` - Model weights
- `test_logits_best_model.npy` - Raw model outputs
- `training_results.json` - All metrics
- `top_bottom_images.json` - Metadata for visualizations

**Report:**
- `REPORT.md` - Full results report

## Hyperparameter Customization

Adjust training parameters as needed:

```bash
# Longer training with smaller learning rate
python -m src.task2.task2 -d ./dat --epochs 100 --learning_rate 0.0001

# Larger batch size (needs more memory)
python -m src.task2.task2 -d ./dat --batch_size 64

# Custom seed for reproducibility
python -m src.task2.task2 -d ./dat --seed 123
```

## Expected Behavior

1. **First Epoch**: Usually slower (~30-60s), rest faster (~10-20s per epoch) on GPU
2. **Convergence**: Validation accuracy typically improves over first 10-15 epochs
3. **Early Stopping**: Training stops if validation accuracy plateaus for 10 epochs
4. **Best Model**: Automatically loaded before test evaluation
5. **Total Time**: ~30-60 minutes for 50 epochs on modern GPU

## Troubleshooting

**Out of Memory Error**
- Reduce `--batch_size` (e.g., 16 instead of 32)
- Reduce `--epochs` to test with fewer epochs

**Slow Training**
- Check if GPU is being used: NVIDIA-SMI or `torch.cuda.is_available()`
- Increase batch size if memory allows

**Missing Data**
- Ensure data directory contains EuroSAT_RGB folder
- Check that split files exist: trainfile.csv, valfile.csv, testfile.csv

## Documentation

For complete details, see `README_TASK2.md` in this directory.
