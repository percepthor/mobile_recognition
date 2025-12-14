"""Metrics computation for model evaluation."""
import logging
from typing import List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels (indices)
        y_pred: Predicted labels (indices)
        class_names: List of class names

    Returns:
        Dictionary with metrics
    """
    logger.info("Computing evaluation metrics")

    # Overall accuracy
    accuracy_top1 = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0
    )

    # Macro-averaged F1
    macro_f1 = np.mean(f1)

    # Create per-class breakdown
    per_class_metrics = {}
    for idx, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1_score': float(f1[idx]),
            'support': int(support[idx])
        }

    metrics = {
        'accuracy_top1': float(accuracy_top1),
        'macro_f1': float(macro_f1),
        'num_test_samples': int(len(y_true)),
        'num_classes': len(class_names),
        'per_class_metrics': per_class_metrics
    }

    # Log summary
    logger.info(f"Top-1 Accuracy: {accuracy_top1:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")

    # Print classification report for detailed view
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
    logger.info(f"\nClassification Report:\n{report}")

    return metrics


def compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute confusion-based metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with confusion metrics
    """
    correct = np.sum(y_true == y_pred)
    incorrect = np.sum(y_true != y_pred)
    total = len(y_true)

    return {
        'correct_predictions': int(correct),
        'incorrect_predictions': int(incorrect),
        'total_predictions': int(total),
        'error_rate': float(incorrect / total) if total > 0 else 0.0
    }
