"""Confidence threshold recommendation using validation set."""
import logging
import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    all_probs: np.ndarray,
    target_accept_accuracy: float = 0.95,
    min_coverage: float = 0.60,
    penalty_incorrect: float = 3.0,
    grid_step: float = 0.01
) -> Tuple[float, dict]:
    """
    Find optimal confidence threshold using validation set.

    Algorithm (from requirements):
    1. Try to find threshold that satisfies:
       - accept_accuracy >= target_accept_accuracy
       - coverage >= min_coverage
       If found, choose one with maximum coverage (tie: minimum threshold)

    2. If no feasible threshold found:
       - Maximize utility = correct_accepted_rate - penalty * incorrect_accepted_rate
       - Tie: choose minimum threshold

    Args:
        y_true: True labels [N]
        all_probs: All class probabilities [N, num_classes]
        target_accept_accuracy: Target accuracy for accepted samples
        min_coverage: Minimum required coverage
        penalty_incorrect: Penalty for accepting incorrect predictions
        grid_step: Step size for threshold grid

    Returns:
        Tuple of (recommended_threshold, metrics_dict)
    """
    logger.info("Finding optimal confidence threshold")
    logger.info(f"Target accept accuracy: {target_accept_accuracy}")
    logger.info(f"Min coverage: {min_coverage}")
    logger.info(f"Penalty for incorrect: {penalty_incorrect}")

    # Get max probabilities and predictions
    prob_max = np.max(all_probs, axis=1)
    y_pred = np.argmax(all_probs, axis=1)
    correct = (y_pred == y_true).astype(float)

    # Create threshold grid
    thresholds = np.arange(0.0, 1.0, grid_step)

    # Store metrics for each threshold
    results = []

    for t in thresholds:
        # Compute metrics at this threshold
        accepted = (prob_max >= t).astype(float)
        num_accepted = np.sum(accepted)

        if num_accepted == 0:
            # No samples accepted
            coverage = 0.0
            accept_accuracy = 0.0
            correct_accepted_rate = 0.0
            incorrect_accepted_rate = 0.0
        else:
            coverage = num_accepted / len(y_true)

            # Accept accuracy: accuracy on accepted samples
            correct_and_accepted = correct * accepted
            accept_accuracy = np.sum(correct_and_accepted) / num_accepted

            # Utility components
            correct_accepted_rate = np.sum(correct_and_accepted) / len(y_true)
            incorrect_accepted_rate = np.sum((1 - correct) * accepted) / len(y_true)

        utility = correct_accepted_rate - penalty_incorrect * incorrect_accepted_rate

        results.append({
            'threshold': t,
            'coverage': coverage,
            'accept_accuracy': accept_accuracy,
            'correct_accepted_rate': correct_accepted_rate,
            'incorrect_accepted_rate': incorrect_accepted_rate,
            'utility': utility
        })

    results_df = pd.DataFrame(results)

    # Strategy 1: Find feasible thresholds
    feasible = results_df[
        (results_df['accept_accuracy'] >= target_accept_accuracy) &
        (results_df['coverage'] >= min_coverage)
    ]

    if len(feasible) > 0:
        # Choose maximum coverage, tie-break with minimum threshold
        best = feasible.sort_values(
            by=['coverage', 'threshold'],
            ascending=[False, True]
        ).iloc[0]

        selection_method = "constraint_feasible"
        logger.info(f"Found {len(feasible)} feasible thresholds")

    else:
        # Strategy 2: Maximize utility
        logger.info("No feasible threshold found, using utility maximization")

        best = results_df.sort_values(
            by=['utility', 'threshold'],
            ascending=[False, True]
        ).iloc[0]

        selection_method = "utility_maximization"

    recommended_threshold = float(best['threshold'])

    logger.info(f"Recommended threshold: {recommended_threshold:.3f}")
    logger.info(f"  Coverage: {best['coverage']:.3f}")
    logger.info(f"  Accept accuracy: {best['accept_accuracy']:.3f}")
    logger.info(f"  Selection method: {selection_method}")

    metrics = {
        'recommended_threshold': recommended_threshold,
        'calibration_notes': (
            f"Selected using validation set with grid step {grid_step}. "
            f"Method: {selection_method}. "
            f"{'Constraints satisfied: accept_accuracy >= ' + str(target_accept_accuracy) + ' and coverage >= ' + str(min_coverage) if selection_method == 'constraint_feasible' else 'No feasible threshold found, maximized utility with penalty ' + str(penalty_incorrect) + ' for incorrect predictions.'}"
        ),
        'target_accept_accuracy': target_accept_accuracy,
        'min_coverage': min_coverage,
        'achieved_accept_accuracy': float(best['accept_accuracy']),
        'achieved_coverage': float(best['coverage']),
        'selection_method': selection_method,
        'grid_step': grid_step,
        'utility': float(best['utility'])
    }

    return recommended_threshold, metrics, results_df


def save_threshold_analysis(
    results_df: pd.DataFrame,
    threshold_metrics: dict,
    output_dir: str
) -> None:
    """
    Save threshold analysis artifacts.

    Args:
        results_df: DataFrame with threshold sweep results
        threshold_metrics: Metrics dictionary
        output_dir: Output directory
    """
    from trainer.utils.io import save_json

    # Save threshold recommendation JSON
    save_json(
        threshold_metrics,
        os.path.join(output_dir, 'threshold_recommendation.json')
    )
    logger.info("Threshold recommendation saved")

    # Save threshold curve CSV
    results_df.to_csv(
        os.path.join(output_dir, 'threshold_curve.csv'),
        index=False
    )
    logger.info("Threshold curve CSV saved")

    # Create threshold curve plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    recommended_t = threshold_metrics['recommended_threshold']

    # Plot 1: Coverage vs Threshold
    ax = axes[0, 0]
    ax.plot(results_df['threshold'], results_df['coverage'], 'b-', linewidth=2)
    ax.axvline(recommended_t, color='r', linestyle='--', label=f'Recommended: {recommended_t:.3f}')
    ax.axhline(threshold_metrics['min_coverage'], color='g', linestyle=':', label=f'Min coverage: {threshold_metrics["min_coverage"]:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Accept Accuracy vs Threshold
    ax = axes[0, 1]
    ax.plot(results_df['threshold'], results_df['accept_accuracy'], 'b-', linewidth=2)
    ax.axvline(recommended_t, color='r', linestyle='--', label=f'Recommended: {recommended_t:.3f}')
    ax.axhline(threshold_metrics['target_accept_accuracy'], color='g', linestyle=':', label=f'Target: {threshold_metrics["target_accept_accuracy"]:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accept Accuracy')
    ax.set_title('Accept Accuracy vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Utility vs Threshold
    ax = axes[1, 0]
    ax.plot(results_df['threshold'], results_df['utility'], 'b-', linewidth=2)
    ax.axvline(recommended_t, color='r', linestyle='--', label=f'Recommended: {recommended_t:.3f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Utility')
    ax.set_title('Utility vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Coverage vs Accept Accuracy
    ax = axes[1, 1]
    scatter = ax.scatter(
        results_df['coverage'],
        results_df['accept_accuracy'],
        c=results_df['threshold'],
        cmap='viridis',
        s=20
    )
    recommended_row = results_df[results_df['threshold'] == recommended_t].iloc[0]
    ax.scatter(
        [recommended_row['coverage']],
        [recommended_row['accept_accuracy']],
        color='r',
        s=200,
        marker='*',
        label='Recommended',
        edgecolors='black',
        linewidths=1.5
    )
    ax.axhline(threshold_metrics['target_accept_accuracy'], color='g', linestyle=':', alpha=0.5)
    ax.axvline(threshold_metrics['min_coverage'], color='g', linestyle=':', alpha=0.5)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Accept Accuracy')
    ax.set_title('Accept Accuracy vs Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Threshold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("Threshold analysis plots saved")
