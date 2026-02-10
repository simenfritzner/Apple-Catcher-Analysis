"""Evaluation metrics for EEG classification."""

from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Union[np.ndarray, None] = None,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional, for AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "n_samples": len(y_true),
    }

    # Multi-class or binary metrics
    n_classes = len(np.unique(y_true))
    
    if n_classes == 2:
        # Binary classification
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:  # 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics["confusion_matrix"] = cm.tolist()
        
        # AUC if probabilities provided
        if y_proba is not None:
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]  # Probability of positive class
            metrics["auc"] = roc_auc_score(y_true, y_proba)
    
    else:
        # Multi-class
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def aggregate_subject_results(
    results: Dict[str, Dict[str, float]],
    metric_name: str = "accuracy",
) -> Dict[str, float]:
    """Aggregate results across subjects.

    Args:
        results: Dictionary mapping subject_id -> metrics
        metric_name: Name of metric to aggregate

    Returns:
        Dictionary with mean, std, min, max of the metric
    """
    values = [r[metric_name] for r in results.values() if metric_name in r]
    
    if not values:
        return {}
    
    return {
        f"{metric_name}_mean": np.mean(values),
        f"{metric_name}_std": np.std(values),
        f"{metric_name}_min": np.min(values),
        f"{metric_name}_max": np.max(values),
        "n_subjects": len(values),
    }


def print_results_summary(
    results: Dict[str, Dict[str, float]],
    metric_name: str = "accuracy",
) -> None:
    """Print a formatted summary of results.

    Args:
        results: Dictionary mapping subject_id -> metrics
        metric_name: Name of metric to display
    """
    print(f"\n{'='*60}")
    print(f"Results Summary - {metric_name}")
    print(f"{'='*60}")
    
    for subject_id, metrics in sorted(results.items()):
        if metric_name in metrics:
            value = metrics[metric_name]
            print(f"{subject_id}: {value:.2%}" if value < 1.1 else f"{subject_id}: {value:.4f}")
    
    print(f"{'='*60}")
    
    # Aggregate stats
    agg = aggregate_subject_results(results, metric_name)
    if agg:
        print(f"Mean:  {agg[f'{metric_name}_mean']:.2%}")
        print(f"Std:   {agg[f'{metric_name}_std']:.2%}")
        print(f"Range: {agg[f'{metric_name}_min']:.2%} - {agg[f'{metric_name}_max']:.2%}")
    print(f"{'='*60}\n")
