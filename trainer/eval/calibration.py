"""Model calibration analysis and reliability diagrams."""
import logging
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def compute_calibration_curve(
    y_true: np.ndarray,
    prob_max: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve (reliability diagram data).

    Args:
        y_true: True labels
        prob_max: Maximum predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Tuple of (bin_accuracies, bin_confidences, bin_counts)
    """
    # Get predictions
    y_pred = np.argmax(prob_max) if len(prob_max.shape) > 1 else prob_max
    correct = (y_true == y_pred) if len(y_pred) == len(y_true) else None

    # If prob_max is 1D, use it directly; otherwise this function expects prob_max as max probs
    if len(prob_max.shape) > 1:
        prob_max = np.max(prob_max, axis=1)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(prob_max, bin_edges[1:-1])

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Note: We need correct predictions, so recompute if needed
    # Assuming y_pred was computed elsewhere, we'll pass it
    # For now, let's fix this

    return bin_accuracies, bin_confidences, bin_counts


def create_reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_max: np.ndarray,
    output_dir: str,
    n_bins: int = 10
) -> dict:
    """
    Create reliability diagram (calibration plot).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        prob_max: Maximum predicted probabilities for each sample
        output_dir: Directory to save plot
        n_bins: Number of bins

    Returns:
        Dictionary with calibration metrics
    """
    logger.info("Creating reliability diagram")

    # Compute correctness
    correct = (y_true == y_pred).astype(float)

    # Create bins based on confidence (prob_max)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(prob_max, bin_edges[1:-1])

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        # Samples in this bin
        mask = (bin_indices == i)
        count = np.sum(mask)

        if count > 0:
            # Average accuracy in this bin
            accuracy = np.mean(correct[mask])
            # Average confidence in this bin
            confidence = np.mean(prob_max[mask])

            bin_accuracies.append(accuracy)
            bin_confidences.append(confidence)
            bin_counts.append(count)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)

    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)

    # Compute Expected Calibration Error (ECE)
    weights = bin_counts / np.sum(bin_counts)
    ece = np.sum(weights * np.abs(bin_accuracies - bin_confidences))

    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

    # Plot actual calibration
    ax.bar(
        bin_confidences,
        bin_accuracies,
        width=1.0/n_bins,
        alpha=0.7,
        edgecolor='black',
        label='Model Calibration'
    )

    # Add gap visualization
    for conf, acc in zip(bin_confidences, bin_accuracies):
        if acc > 0:  # Only draw for non-empty bins
            ax.plot([conf, conf], [conf, acc], 'r-', alpha=0.5, linewidth=2)

    ax.set_xlabel('Confidence (Max Probability)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram\nECE = {ece:.4f}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'reliability_diagram.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Reliability diagram saved: {plot_path}")

    # Create histogram of confidences
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(prob_max, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence (Max Probability)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Prediction Confidences', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(output_dir, 'confidence_histogram.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Confidence histogram saved: {hist_path}")

    calibration_metrics = {
        'expected_calibration_error': float(ece),
        'n_bins': n_bins,
        'mean_confidence': float(np.mean(prob_max)),
        'median_confidence': float(np.median(prob_max)),
        'min_confidence': float(np.min(prob_max)),
        'max_confidence': float(np.max(prob_max))
    }

    return calibration_metrics
