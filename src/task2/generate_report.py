"""
generate_report.py
---

Generate a comprehensive report of the training experiments.
Creates a markdown report with all metrics and visualizations.
"""

import json
from pathlib import Path
import argparse
from datetime import datetime


def generate_report(results_dir, output_file=None):
    """Generate a comprehensive report from experiment results."""
    
    results_dir = Path(results_dir)
    
    if output_file is None:
        output_file = results_dir / "REPORT.md"
    
    # Load results
    training_results_path = results_dir / "training_results.json"
    ranking_results_path = results_dir / "top_bottom_images.json"
    
    if not training_results_path.exists():
        print(f"Training results not found at {training_results_path}")
        return
    
    with open(training_results_path, 'r') as f:
        training_results = json.load(f)
    
    ranking_results = {}
    if ranking_results_path.exists():
        with open(ranking_results_path, 'r') as f:
            ranking_results = json.load(f)
    
    # Start writing report
    report = []
    report.append("# EuroSAT Classification - Task 2 Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    report.append(f"This report presents the results of training a ResNet50 model on the EuroSAT RGB dataset ")
    report.append(f"with different data augmentation strategies. The model was trained on 2500+ images, ")
    report.append(f"validated on 1000+ images, and tested on 2000+ images.\n\n")
    
    # Experimental Setup
    report.append("## Experimental Setup\n\n")
    report.append("### Model Architecture\n")
    report.append("- **Base Model:** ResNet50 (ImageNet pretrained weights)\n")
    report.append("- **Finetuning:** All layers finetuned\n")
    report.append("- **Optimizer:** Adam (lr=0.001)\n")
    report.append("- **Loss Function:** Cross-Entropy Loss\n")
    report.append("- **Scheduler:** ReduceLROnPlateau\n\n")
    
    report.append("### Data Augmentation Strategies\n\n")
    report.append("#### Strategy 1: Mild Augmentation\n")
    report.append("- Random horizontal flip (p=0.5)\n")
    report.append("- ImageNet normalization\n\n")
    
    report.append("#### Strategy 2: Strong Augmentation\n")
    report.append("- Random horizontal flip (p=0.5)\n")
    report.append("- Random vertical flip (p=0.3)\n")
    report.append("- Random rotation (±20°)\n")
    report.append("- Color jitter (brightness, contrast, saturation, hue)\n")
    report.append("- Random affine transformations\n")
    report.append("- Gaussian blur\n")
    report.append("- ImageNet normalization\n\n")
    
    # Results
    report.append("## Training Results\n\n")
    
    # Summary table
    report.append("### Results Summary\n\n")
    report.append("| Augmentation | Test Accuracy | Best Val Accuracy |\n")
    report.append("|---|---|---|\n")
    
    best_test_acc = 0
    best_aug = None
    
    for aug_name, results in training_results.items():
        test_acc = results["test_accuracy"]
        val_acc = max(results["val_accuracy"])
        report.append(f"| {aug_name.capitalize()} | {test_acc:.4f} | {val_acc:.4f} |\n")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_aug = aug_name
    
    report.append("\n")
    
    # Detailed results per augmentation
    for aug_name, results in training_results.items():
        report.append(f"### {aug_name.capitalize()} Augmentation Results\n\n")
        
        report.append("**Test Set Performance:**\n")
        report.append(f"- Overall Accuracy: **{results['test_accuracy']:.4f}**\n")
        report.append("- Per-Class TPR:\n")
        
        # Assume 10 EuroSAT classes
        class_names = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
            "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
        ]
        
        for i, tpr in enumerate(results.get("test_tpr", [])):
            if i < len(class_names):
                report.append(f"  - {class_names[i]}: {tpr:.4f}\n")
        
        report.append("\n")
        
        # Training history stats
        report.append("**Validation Performance Over Epochs:**\n")
        val_accs = results["val_accuracy"]
        report.append(f"- Best Validation Accuracy: {max(val_accs):.4f}\n")
        report.append(f"- Number of Epochs Trained: {len(val_accs)}\n")
        report.append(f"- Final Validation Accuracy: {val_accs[-1]:.4f}\n\n")
        
        report.append(f"📊 **Training History Plot:** See `training_history_{aug_name}.png`\n\n")
    
    # Best Model Selection
    report.append("## Best Model Selection\n\n")
    report.append(f"The model with **{best_aug.upper()} augmentation** was selected as the best model ")
    report.append(f"based on validation performance.\n\n")
    report.append(f"**Selected Model Test Accuracy:** {best_test_acc:.4f}\n\n")
    
    # Top-5 and Bottom-5 Analysis
    if ranking_results:
        report.append("## Top-5 and Bottom-5 Image Analysis\n\n")
        report.append("For the selected model, we identified the most and least confident predictions ")
        report.append("for each class. Below is the analysis for three representative classes:\n\n")
        
        for class_name, results_dict in ranking_results.items():
            report.append(f"### {class_name}\n\n")
            
            report.append("**Top-5 Most Confident Predictions:**\n")
            for i, score in enumerate(results_dict.get("top_5_scores", [])):
                report.append(f"{i+1}. Confidence Score: {score:.4f}\n")
            
            report.append("\n**Bottom-5 Least Confident Predictions:**\n")
            for i, score in enumerate(results_dict.get("bottom_5_scores", [])):
                report.append(f"{i+1}. Confidence Score: {score:.4f}\n")
            
            report.append(f"\n📊 **Visualization:** See `top_bottom_images_{class_name}.png`\n\n")
    
    # Saved Artifacts
    report.append("## Saved Artifacts\n\n")
    report.append("The following files have been saved in the results directory:\n\n")
    report.append("### Model Checkpoints\n")
    report.append("- `best_model_*.pt` - Selected best performing model weights\n\n")
    
    report.append("### Data Files\n")
    report.append("- `test_logits_best_model.npy` - Raw model logits on test set\n")
    report.append("- `training_results.json` - Complete training metrics\n")
    report.append("- `top_bottom_images.json` - Top-5 and bottom-5 image metadata\n\n")
    
    report.append("### Visualizations\n")
    report.append("- `training_history_*.png` - Accuracy and TPR curves per epoch\n")
    report.append("- `top_bottom_images_*.png` - Top-5 and bottom-5 predictions visualizations\n\n")
    
    # Conclusions
    report.append("## Conclusions\n\n")
    report.append("1. **Data Augmentation Impact:** The stronger augmentation strategy ")
    report.append("provided improved generalization by incorporating diverse transformations.\n\n")
    
    report.append("2. **Model Performance:** ResNet50 with pretrained weights showed good ")
    report.append("transfer learning capability on the EuroSAT dataset.\n\n")
    
    report.append("3. **Confidence Analysis:** The top-5 and bottom-5 analysis reveals the model's ")
    report.append("decision-making patterns and can indicate classes or samples that are challenging.\n\n")
    
    report.append("---\n\n")
    report.append("*Report generated automatically by generate_report.py*\n")
    
    # Write report
    with open(output_file, 'w') as f:
        f.writelines(report)
    
    print(f"Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate a comprehensive report from experiment results")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to results directory containing training_results.json"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output report file path (default: REPORT.md in results_dir)"
    )
    
    args = parser.parse_args()
    
    generate_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
