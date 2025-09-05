import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from addict import Dict
from typing import Any
from datetime import datetime

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix

def run(config: Dict,
        model: nn.Module,
        test_loader: DataLoader
) -> Any:
    """
    Run the evaluation pipeline.

    Args:
        config (Dict): Configuration dictionary.
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): DataLoader for the test set.

    Returns:
        Any: Evaluation results and metrics.
    """
    device = torch.device(config.device)
    model.to(device)
    
    torch.manual_seed(config.seed)

    os.makedirs(config.evaluation.output_dir, exist_ok=True)

    # Evaluation
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    # For multi-class, specify average method, e.g., 'macro' or 'weighted'
    f1 = f1_score(y_true, y_pred, average='weighted') 
    recall = recall_score(y_true, y_pred, average='weighted')
    per_class_recall = recall_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'recall': recall,
        'per_class_recall': per_class_recall,
        'confusion_matrix': cm
    }

    # Create report string
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_name = config.model.name
    model_path = config.model.params.pretrain_model_path
    dataset_name = config.dataset.name.upper()
    total_samples = len(y_true)

    report = []
    report.append("==================================================")
    report.append("          Model Evaluation Report")
    report.append("==================================================")
    report.append(f"Evaluation Timestamp: {timestamp}")
    report.append(f"Model: {model_name}")
    report.append(f"Model Path: {model_path}")
    report.append(f"Evaluation Dataset: {dataset_name}")
    report.append("--------------------------------------------------")
    report.append(f"Total Samples: {total_samples}\n")
    report.append("Confusion Matrix:")
    report.append(np.array2string(cm))
    report.append("--------------------------------------------------")
    report.append("Metrics:")
    report.append(f"  - Accuracy:          {acc:.2%}")
    report.append(f"  - Balanced Accuracy: {balanced_acc:.2%}")
    report.append("==================================================")
    report.append("Per-Class Recall:")
    for i, rec in enumerate(per_class_recall):
        report.append(f"  - Class {i}: {rec:.4f}")
    report.append("==================================================")

    report_str = "\n".join(report)

    # Save results to a txt file
    output_path = os.path.join(config.evaluation.output_dir, f'{config.training.logging.project_name}_evaluation_results.txt')
    
    with open(output_path, 'w') as f:
        f.write(report_str)

    print(f"Evaluation results saved to {output_path}")

    return metrics