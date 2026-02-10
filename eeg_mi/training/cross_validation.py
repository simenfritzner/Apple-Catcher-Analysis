"""Cross-validation utilities for EEG classification.

Provides LOSO (Leave-One-Subject-Out) and nested CV functionality.
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import logging

logger = logging.getLogger(__name__)


def loso_cross_validation(
    subjects_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    trainer: Any,
    train_size: int = 100,
    test_start: int = 100,
    test_end: int = 200,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Perform Leave-One-Subject-Out cross-validation.

    Args:
        subjects_data: Dict mapping subject_id -> (data, labels)
        trainer: Trainer instance (SklearnTrainer or EEGNetTrainer)
        train_size: Number of trials per training subject
        test_start: Start index for test set
        test_end: End index for test set
        verbose: Print progress

    Returns:
        Dictionary mapping subject_id -> evaluation metrics
    """
    from eeg_mi.data.loaders import prepare_loso_fold

    results = {}

    for test_subject_id in subjects_data.keys():
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing on subject: {test_subject_id}")
            logger.info(f"{'='*60}")

        # Prepare fold
        train_data, train_labels, test_data, test_labels = prepare_loso_fold(
            subjects_data,
            test_subject_id,
            train_size=train_size,
            test_eval_start=test_start,
            test_eval_end=test_end,
        )

        # NOTE: Normalization should be applied here if needed
        # Use normalize_train_test_split() to prevent data leakage
        # Example:
        #   from eeg_mi.data.normalization import normalize_train_test_split
        #   train_data, test_data = normalize_train_test_split(train_data, test_data, method='zscore_subject')

        # Train
        model = trainer.train(train_data, train_labels)

        # Evaluate
        metrics = trainer.evaluate(model, test_data, test_labels)
        results[test_subject_id] = metrics

        if verbose:
            logger.info(f"Results: {metrics}")

    return results


def nested_loso_cv(
    subjects_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    trainer_factory: Callable,
    param_grid: Dict[str, List[Any]],
    train_size: int = 100,
    test_start: int = 100,
    test_end: int = 200,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Nested LOSO cross-validation with hyperparameter search.

    Args:
        subjects_data: Dict mapping subject_id -> (data, labels)
        trainer_factory: Function that creates trainer given params
        param_grid: Dictionary of hyperparameters to search
        train_size: Number of trials per training subject
        test_start: Start index for test set
        test_end: End index for test set
        verbose: Print progress

    Returns:
        Dictionary with results and best params per subject
    """
    from eeg_mi.data.loaders import prepare_loso_fold
    from itertools import product

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    results = {}

    for test_subject_id in subjects_data.keys():
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing on subject: {test_subject_id}")
            logger.info(f"{'='*60}")

        # Prepare outer fold
        train_data, train_labels, test_data, test_labels = prepare_loso_fold(
            subjects_data,
            test_subject_id,
            train_size=train_size,
            test_eval_start=test_start,
            test_eval_end=test_end,
        )

        # Inner CV for hyperparameter search (simple validation split)
        val_size = len(train_data) // 5  # 20% for validation
        indices = np.random.permutation(len(train_data))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_data_inner = train_data[train_indices]
        train_labels_inner = train_labels[train_indices]
        val_data_inner = train_data[val_indices]
        val_labels_inner = train_labels[val_indices]

        # Search hyperparameters
        best_val_acc = 0
        best_params = None

        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))
            
            # Create trainer with these params
            trainer = trainer_factory(**params)
            
            # Train on inner train set
            model = trainer.train(train_data_inner, train_labels_inner)
            
            # Evaluate on validation set
            val_metrics = trainer.evaluate(model, val_data_inner, val_labels_inner)
            val_acc = val_metrics["accuracy"]

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params

        if verbose:
            logger.info(f"Best params: {best_params} (val_acc: {best_val_acc:.2%})")

        # Retrain on full training set with best params
        best_trainer = trainer_factory(**best_params)
        final_model = best_trainer.train(train_data, train_labels)

        # Evaluate on test set
        test_metrics = best_trainer.evaluate(final_model, test_data, test_labels)

        results[test_subject_id] = {
            "test_metrics": test_metrics,
            "best_params": best_params,
            "val_accuracy": best_val_acc,
        }

        if verbose:
            logger.info(f"Test accuracy: {test_metrics['accuracy']:.2%}")

    return results
