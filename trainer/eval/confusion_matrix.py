"""Confusion matrix generation and visualization."""
import logging
import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


logger = logging.getLogger(__name__)


def create_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str
) -> np.ndarray:
    """
    Create and save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save outputs

    Returns:
        Confusion matrix as numpy array
    """
    logger.info("Creating confusion matrix")

    # Compute confusion matrix
    cm = sklearn_confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    # Save as CSV
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{name}" for name in class_names],
        columns=[f"Pred_{name}" for name in class_names]
    )
    csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(csv_path)
    logger.info(f"Confusion matrix CSV saved: {csv_path}")

    # Create visualization
    plt.figure(figsize=(max(10, len(class_names)), max(8, len(class_names) * 0.8)))

    # Use seaborn for better visualization
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )

    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save figure
    png_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix plot saved: {png_path}")

    return cm


def create_normalized_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str
) -> np.ndarray:
    """
    Create normalized confusion matrix (percentages).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save outputs

    Returns:
        Normalized confusion matrix
    """
    logger.info("Creating normalized confusion matrix")

    # Compute confusion matrix
    cm = sklearn_confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    # Create visualization
    plt.figure(figsize=(max(10, len(class_names)), max(8, len(class_names) * 0.8)))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )

    plt.title('Normalized Confusion Matrix (Row %)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save figure
    png_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Normalized confusion matrix plot saved: {png_path}")

    return cm_normalized
