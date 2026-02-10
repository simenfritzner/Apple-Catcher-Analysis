"""Nested LOSO cross-validation for hyperparameter tuning.

Implements proper nested CV for LOSO:
- Outer loop: Leave-one-subject-out (test subject)
- Inner loop: 5-fold cross-validation among N-1 training subjects (validation)
- Select best hyperparameters on inner loop
- Retrain with best params and evaluate on outer test subject
"""

import logging
from itertools import product
from typing import Any, Callable, Dict, List

import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def nested_loso_cv(
    subjects_data: Dict[str, tuple],
    trainer_factory: Callable,
    param_grid: Dict[str, List[Any]],
    outer_test_subject: str,
    verbose: bool = True,
    config: Dict[str, Any] = None,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Run nested LOSO cross-validation with hyperparameter tuning.

    Outer loop: Leave out outer_test_subject
    Inner loop: Among remaining N-1 subjects, do n_folds-fold CV for validation

    Args:
        subjects_data: Dict mapping subject_id -> (data, labels)
        trainer_factory: Function that creates trainer given hyperparameters
        param_grid: Dictionary of hyperparameter lists to search
        outer_test_subject: Subject ID to hold out for final testing
        verbose: Print progress
        config: Configuration dictionary
        n_folds: Number of folds for inner cross-validation (default: 5)

    Returns:
        Dictionary with best params, validation scores, and final test results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Nested LOSO CV - Outer Test Subject: {outer_test_subject}")
    logger.info(f"{'='*80}")
    
    # Separate outer test subject
    outer_test_data, outer_test_labels = subjects_data[outer_test_subject]
    
    # Remaining subjects for training (N-1 subjects)
    training_subjects = {
        sid: (data, labels)
        for sid, (data, labels) in subjects_data.items()
        if sid != outer_test_subject
    }

    logger.info(f"Outer test: {outer_test_subject}")
    logger.info(f"Training pool: {len(training_subjects)} subjects (before subject selection)")

    # Apply subject selection ONCE at outer level (if enabled in config)
    # This filters the training pool before HP search begins
    if config is not None:
        use_subject_selection = config.get('training', {}).get('use_subject_selection', False)

        if use_subject_selection:
            from eeg_mi.training.subject_selection import (
                load_subject_rankings,
                filter_subjects_by_rankings
            )
            from pathlib import Path

            rankings_dir = Path(config.get('training', {}).get('subject_rankings_dir', 'results/subject_rankings'))
            rankings_file = rankings_dir / f"rankings_{outer_test_subject}.json"

            logger.info(f"Subject selection enabled - loading rankings from: {rankings_file}")

            if rankings_file.exists():
                # Load pre-computed rankings for the outer test subject
                subject_rankings = load_subject_rankings(rankings_file)

                # Filter subjects based on rankings
                selection_method = config.get('training', {}).get('selection_method', 'adaptive')
                selection_threshold = config.get('training', {}).get('selection_threshold', 0.55)
                selection_top_k = config.get('training', {}).get('selection_top_k', 9)
                selection_min_subjects = config.get('training', {}).get('selection_min_subjects', 5)

                training_subjects = filter_subjects_by_rankings(
                    source_subjects=training_subjects,
                    rankings=subject_rankings,
                    selection_method=selection_method,
                    threshold=selection_threshold,
                    top_k=selection_top_k,
                    min_subjects=selection_min_subjects,
                )

                selected_subject_ids = sorted(training_subjects.keys())
                logger.info(f"✓ Selected {len(selected_subject_ids)} subjects for outer training pool: {selected_subject_ids}")
            else:
                logger.warning(f"Rankings file not found: {rankings_file}")
                logger.warning("Using all source subjects (no subject selection)")
        else:
            logger.info("Subject selection disabled - using all source subjects")

    logger.info(f"Final training pool: {len(training_subjects)} subjects")

    # Safety check: Ensure we have enough subjects for nested CV
    min_required_subjects = 2  # Need at least 2 subjects for k-fold CV
    if len(training_subjects) < min_required_subjects:
        raise ValueError(
            f"Not enough subjects for nested CV! "
            f"Training pool has {len(training_subjects)} subjects, need at least {min_required_subjects}. "
            f"Consider: (1) Disabling subject selection, or (2) Increasing selection_min_subjects parameter."
        )

    # Adjust n_folds if we have fewer subjects than folds
    actual_n_folds = min(n_folds, len(training_subjects))
    if actual_n_folds < n_folds:
        logger.warning(f"Reducing n_folds from {n_folds} to {actual_n_folds} due to limited subjects")
        n_folds = actual_n_folds

    # Generate all hyperparameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    logger.info(f"\nHyperparameter search space: {len(param_combinations)} combinations")
    for name in param_names:
        logger.info(f"  {name}: {param_grid[name]}")

    # Inner loop: Validate each hyperparameter combination
    best_params = None
    best_val_score = -np.inf
    all_val_scores = {}

    total_hp_combos = len(param_combinations)
    total_inner_folds = n_folds

    # Convert training_subjects to a list for indexing
    training_subject_ids = list(training_subjects.keys())

    for hp_idx, param_combo in enumerate(param_combinations, 1):
        params = dict(zip(param_names, param_combo))

        # Merge hyperparameter combo with full config (config has priority for non-HP settings)
        if config is not None:
            training_config = config.get('training', {})
            # Add config settings that aren't in the HP grid
            for key, value in training_config.items():
                if key not in params:
                    params[key] = value

        logger.info(f"\n{'='*80}")
        logger.info(f"HP COMBINATION {hp_idx}/{total_hp_combos}")
        logger.info(f"Parameters: {params}")
        logger.info(f"{'='*80}")

        # Inner K-Fold CV: Split training subjects into k folds
        inner_val_scores = []
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for inner_idx, (train_indices, val_indices) in enumerate(kfold.split(training_subject_ids), 1):
            # Get subject IDs for this fold
            inner_train_subject_ids = [training_subject_ids[i] for i in train_indices]
            inner_val_subject_ids = [training_subject_ids[i] for i in val_indices]

            logger.info(f"\n--- Inner Fold {inner_idx}/{total_inner_folds}: "
                       f"Train subjects = {len(inner_train_subject_ids)}, "
                       f"Val subjects = {len(inner_val_subject_ids)} ---")

            # Create inner train subjects dictionary
            inner_train_subjects = {
                sid: training_subjects[sid]
                for sid in inner_train_subject_ids
            }

            # Pool validation subjects' calibration and test data separately
            # This ensures proper calibration/test split for each validation subject
            inner_val_cal_size = params.get('target_cal_size', 50)
            inner_val_cal_data_list = []
            inner_val_cal_labels_list = []
            inner_val_test_data_list = []
            inner_val_test_labels_list = []

            for sid in inner_val_subject_ids:
                data, labels = training_subjects[sid]
                # Calibration: first inner_val_cal_size trials
                inner_val_cal_data_list.append(data[:inner_val_cal_size])
                inner_val_cal_labels_list.append(labels[:inner_val_cal_size])
                # Test: trials 100-200
                inner_val_test_data_list.append(data[100:200])
                inner_val_test_labels_list.append(labels[100:200])

            # Concatenate calibration and test data from all validation subjects
            inner_val_cal_data = np.concatenate(inner_val_cal_data_list, axis=0)
            inner_val_cal_labels = np.concatenate(inner_val_cal_labels_list, axis=0)
            inner_val_test_data = np.concatenate(inner_val_test_data_list, axis=0)
            inner_val_test_labels = np.concatenate(inner_val_test_labels_list, axis=0)

            # Safety check: Ensure we have at least one training subject
            if len(inner_train_subjects) == 0:
                logger.warning(f"⊘ No training subjects available for inner fold {inner_idx}, skipping...")
                inner_val_scores.append(0.0)
                continue

            # Get normalization method from config
            normalization_method = config.get('preprocessing', {}).get('normalization', 'none') if config else 'none'

            # Validate normalization method
            if normalization_method not in ['none', 'zscore_subject', 'zscore_zero_shot', 'zscore_calibration']:
                raise ValueError(
                    f"Unknown normalization method: {normalization_method}. "
                    f"Use 'none', 'zscore_subject', 'zscore_zero_shot', or 'zscore_calibration'"
                )

            # Pool inner training subjects
            # For DANN: also get domain labels (subject IDs)
            # Each training subject is normalized independently (prevents cross-subject leakage)
            inner_train_data, inner_train_labels, inner_train_domains = pool_subjects(
                inner_train_subjects,
                train_size=params.get('train_size', 100),
                return_subject_ids=True,  # Get domain labels for DANN
                normalization=normalization_method if normalization_method != 'none' else 'none'
            )

            # Normalize validation subjects
            # In nested CV, we ALWAYS use calibration statistics for validation
            # (we have calibration data available from the validation subjects)
            if normalization_method not in ['none']:
                from eeg_mi.data.normalization import normalize_calibration_test_split
                inner_val_cal_data, inner_val_test_data, _, _ = normalize_calibration_test_split(
                    cal_data=inner_val_cal_data,
                    test_data=inner_val_test_data,
                    method='zscore_subject'  # Always use calibration stats in nested CV
                )

            # Log data shapes for debugging
            logger.info(f"Training data shape: {inner_train_data.shape}, {len(inner_train_subjects)} subjects × {params.get('train_size', 100)} trials")
            logger.info(f"Calibration data shape: {inner_val_cal_data.shape} ({len(inner_val_subject_ids)} val subjects)")
            logger.info(f"Validation test shape: {inner_val_test_data.shape}")

            # Create trainer with these params
            trainer = trainer_factory(**params)

            # Determine if supervised or unsupervised mode
            use_supervised = params.get('supervised', True)  # Default: supervised with labels

            # Get training strategy params
            two_stage = params.get('two_stage', True)
            pretrain_epochs = params.get('pretrain_epochs', None)
            finetune_epochs = params.get('finetune_epochs', None)
            freeze_features = params.get('freeze_features', False)

            # Train (handle both DANN and CNN trainers)
            try:
                # Check if trainer expects 'cal_data' (CNN) or 'target_data' (DANN)
                import inspect
                sig = inspect.signature(trainer.train)

                if 'cal_data' in sig.parameters:
                    # CNN trainer: uses train_data + cal_data
                    model = trainer.train(
                        train_data=inner_train_data,
                        train_labels=inner_train_labels,
                        cal_data=inner_val_cal_data,
                        cal_labels=inner_val_cal_labels,
                        val_data=inner_val_test_data,
                        val_labels=inner_val_test_labels,
                        two_stage=two_stage,
                        pretrain_epochs=pretrain_epochs,
                        finetune_epochs=finetune_epochs,
                        freeze_features=freeze_features,
                    )
                else:
                    # DANN trainer: uses source_data + target_data + source_domains
                    model = trainer.train(
                        source_data=inner_train_data,
                        source_labels=inner_train_labels,
                        source_domains=inner_train_domains,  # Domain labels for multi-domain DANN
                        target_data=inner_val_cal_data,
                        target_labels=inner_val_cal_labels if use_supervised else None,
                        val_data=inner_val_test_data,
                        val_labels=inner_val_test_labels,
                        two_stage=two_stage,
                        pretrain_epochs=pretrain_epochs,
                        finetune_epochs=finetune_epochs,
                        freeze_features=freeze_features,
                    )

                # Evaluate on inner validation
                val_metrics = trainer.evaluate(model, inner_val_test_data, inner_val_test_labels)
                val_acc = val_metrics['accuracy']
                inner_val_scores.append(val_acc)
                logger.info(f"✓ Fold {inner_idx} completed: Val acc = {val_acc:.2%}")

            except Exception as e:
                import traceback
                logger.warning(f"✗ Failed on inner fold {inner_idx}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                inner_val_scores.append(0.0)

        # Average validation score across inner folds
        mean_val_score = np.mean(inner_val_scores)
        std_val_score = np.std(inner_val_scores)

        all_val_scores[str(params)] = {
            'mean': mean_val_score,
            'std': std_val_score,
            'scores': inner_val_scores,
        }

        logger.info(f"\n{'─'*80}")
        logger.info(f"HP {hp_idx}/{total_hp_combos} Summary: {mean_val_score:.2%} ± {std_val_score:.2%}")
        logger.info(f"{'─'*80}")

        # Update best
        if mean_val_score > best_val_score:
            best_val_score = mean_val_score
            best_params = params
            logger.info(f"🏆 NEW BEST! Val acc: {best_val_score:.2%}")
            logger.info(f"Best params: {best_params}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Best hyperparameters (val acc: {best_val_score:.2%}):")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"{'='*80}")
    
    # Retrain on all N-1 subjects with best params
    logger.info(f"\nRetraining on all {len(training_subjects)} subjects with best params...")

    # Get normalization method from config
    normalization_method = config.get('preprocessing', {}).get('normalization', 'none') if config else 'none'

    # Pool training subjects with normalization
    final_train_data, final_train_labels = pool_subjects(
        training_subjects,
        train_size=best_params.get('train_size', 100),
        normalization=normalization_method  # Normalize each subject independently
    )

    # Outer test subject calibration
    outer_cal_size = best_params.get('target_cal_size', 50)
    outer_cal_data = outer_test_data[:outer_cal_size]
    outer_cal_labels = outer_test_labels[:outer_cal_size]
    outer_test_data_final = outer_test_data[100:200]
    outer_test_labels_final = outer_test_labels[100:200]

    # Normalize outer test subject
    # In nested CV, we ALWAYS use calibration statistics for the outer test subject
    # (we have calibration data available from the test subject)
    if normalization_method not in ['none']:
        from eeg_mi.data.normalization import normalize_calibration_test_split
        logger.info(f"Normalizing outer test subject using calibration statistics (no leakage)...")
        outer_cal_data, outer_test_data_final, _, _ = normalize_calibration_test_split(
            cal_data=outer_cal_data,
            test_data=outer_test_data_final,
            method='zscore_subject'  # Always use calibration stats in nested CV
        )

    # Determine if supervised or unsupervised mode
    use_supervised = best_params.get('supervised', True)  # Default: supervised

    # Get training strategy params
    two_stage = best_params.get('two_stage', True)
    pretrain_epochs = best_params.get('pretrain_epochs', None)
    finetune_epochs = best_params.get('finetune_epochs', None)
    freeze_features = best_params.get('freeze_features', False)

    # Final training (handle both DANN and CNN trainers)
    final_trainer = trainer_factory(**best_params)

    import inspect
    sig = inspect.signature(final_trainer.train)

    if 'cal_data' in sig.parameters:
        # CNN trainer
        final_model = final_trainer.train(
            train_data=final_train_data,
            train_labels=final_train_labels,
            cal_data=outer_cal_data,
            cal_labels=outer_cal_labels,
            val_data=outer_test_data_final,
            val_labels=outer_test_labels_final,
            two_stage=two_stage,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
            freeze_features=freeze_features,
        )
    else:
        # DANN trainer
        final_model = final_trainer.train(
            source_data=final_train_data,
            source_labels=final_train_labels,
            target_data=outer_cal_data,
            target_labels=outer_cal_labels if use_supervised else None,
            val_data=outer_test_data_final,
            val_labels=outer_test_labels_final,
            two_stage=two_stage,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
            freeze_features=freeze_features,
        )
    
    # Final evaluation on outer test set
    final_metrics = final_trainer.evaluate(
        final_model,
        outer_test_data_final,
        outer_test_labels_final,
    )
    
    logger.info(f"\nFinal test accuracy: {final_metrics['accuracy']:.2%}")

    # Build results dict
    results = {
        'test_subject': outer_test_subject,
        'best_params': best_params,
        'best_val_score': float(best_val_score),
        'all_val_scores': all_val_scores,
        'test_accuracy': float(final_metrics['accuracy']),
        'test_metrics': final_metrics,
        'model': final_model,  # Include trained model for saving
    }

    # Add config if provided (for reproducibility)
    if config is not None:
        results['config'] = config

    return results


def pool_subjects(
    subjects_dict: Dict[str, tuple],
    train_size: int = 100,
    return_subject_ids: bool = False,
    normalization: str = 'none',
) -> tuple:
    """Pool multiple subjects into single source domain.

    Args:
        subjects_dict: Dict mapping subject_id -> (data, labels)
        train_size: Number of trials per subject
        return_subject_ids: If True, also return domain labels (all set to 0 for binary DANN)
        normalization: Normalization method ('zscore_subject', 'none')
                      If not 'none', each subject is normalized independently before pooling

    Returns:
        If return_subject_ids=False: (pooled_data, pooled_labels)
        If return_subject_ids=True: (pooled_data, pooled_labels, domain_labels)
                                    where domain_labels are all 0 (source domain)
    """
    from eeg_mi.data.normalization import normalize_subject_zscore

    data_list = []
    labels_list = []
    domain_labels_list = []

    for subject_id, (data, labels) in subjects_dict.items():
        subject_data = data[:train_size]

        # Normalize each subject independently (prevents cross-subject leakage)
        if normalization != 'none':
            subject_data, _, _ = normalize_subject_zscore(subject_data)

        data_list.append(subject_data)
        labels_list.append(labels[:train_size])

        if return_subject_ids:
            # Create binary domain labels: all source subjects get label 0
            # (target will get label 1 in the trainer)
            domain_labels = np.zeros(min(train_size, len(data)), dtype=np.int64)
            domain_labels_list.append(domain_labels)

    pooled_data = np.concatenate(data_list, axis=0)
    pooled_labels = np.concatenate(labels_list, axis=0)

    if return_subject_ids:
        pooled_domains = np.concatenate(domain_labels_list, axis=0)
        return pooled_data, pooled_labels, pooled_domains
    else:
        return pooled_data, pooled_labels
