"""Normalization utilities for EEG data.

Implements various normalization strategies for cross-subject BCI experiments.

Normalization Methods
--------------------
For LOSO cross-validation, choose ONE of these methods in your config:

1. 'none': No normalization

2. 'zscore_subject': Per-subject z-score normalization (DEFAULT)
   - SOURCE subjects: Each normalized independently using their own statistics
   - TEST subject: Uses SOURCE statistics (same as zscore_zero_shot)
   - Prevents cross-subject data leakage
   - Use this for most experiments unless you need calibration

3. 'zscore_zero_shot': Zero-shot transfer normalization (explicit)
   - Same as zscore_subject, but more explicit
   - Test subject normalized using SOURCE subject statistics only
   - Realistic for deployment scenarios with no calibration
   - Use for single-stage models (no calibration phase)

4. 'zscore_calibration': Calibration-based normalization
   - SOURCE subjects: Each normalized independently
   - TEST subject: Normalized using CALIBRATION trial statistics
   - Requires calibration data from test subject (target_cal_size > 0)
   - Use for two-stage models (pretrain + finetune with calibration)

5. 'zscore_global': Global z-score across all subjects
   - Single mean/std computed from all training subjects
   - Applied to all subjects (including test)
   - Can introduce leakage if not used carefully
   - Rarely used
"""

import numpy as np
from typing import Dict, Tuple, Optional


def normalize_subject_zscore(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply subject-specific z-score normalization.

    Normalizes across all trials for this subject: (X - mean) / std

    Args:
        data: EEG data (n_trials, n_channels, n_timepoints)
        mean: Pre-computed mean (n_channels, n_timepoints). If None, compute from data
        std: Pre-computed std (n_channels, n_timepoints). If None, compute from data

    Returns:
        normalized_data: Z-score normalized data
        mean: Mean used for normalization (for applying to test data)
        std: Std used for normalization (for applying to test data)
    """
    if mean is None:
        mean = np.mean(data, axis=0, keepdims=False)  # (n_channels, n_timepoints)
    if std is None:
        std = np.std(data, axis=0, keepdims=False)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)

    # Broadcast and normalize
    normalized = (data - mean) / std

    return normalized, mean, std


def normalize_subjects_dict(
    subjects_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    method: str = "zscore_subject",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Apply normalization to all subjects in a dictionary.

    Args:
        subjects_data: Dict mapping subject_id -> (data, labels)
        method: Normalization method. Options:
            - "zscore_subject": Per-subject z-score (recommended)
            - "zscore_global": Global z-score across all subjects
            - "none": No normalization

    Returns:
        Dictionary with normalized data, same structure as input
    """
    if method == "none":
        return subjects_data

    if method == "zscore_subject":
        # Per-subject normalization
        normalized = {}
        for subject_id, (data, labels) in subjects_data.items():
            norm_data, _, _ = normalize_subject_zscore(data)
            normalized[subject_id] = (norm_data, labels)
        return normalized

    elif method == "zscore_global":
        # Compute global mean/std across all subjects
        all_data = np.concatenate([data for data, _ in subjects_data.values()], axis=0)
        global_mean = np.mean(all_data, axis=0, keepdims=False)
        global_std = np.std(all_data, axis=0, keepdims=False)
        global_std = np.where(global_std == 0, 1.0, global_std)

        # Apply to each subject
        normalized = {}
        for subject_id, (data, labels) in subjects_data.items():
            norm_data, _, _ = normalize_subject_zscore(data, mean=global_mean, std=global_std)
            normalized[subject_id] = (norm_data, labels)
        return normalized

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_train_test_split(
    train_data: np.ndarray,
    test_data: np.ndarray,
    method: str = "zscore_subject",
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize train/test split (prevents data leakage).

    Computes normalization parameters from training data ONLY,
    then applies to both train and test.

    Args:
        train_data: Training data (n_trials, n_channels, n_timepoints)
        test_data: Test data (n_trials, n_channels, n_timepoints)
        method: Normalization method ("zscore_subject" or "none")

    Returns:
        normalized_train: Normalized training data
        normalized_test: Normalized test data (using train stats)
    """
    if method == "none":
        return train_data, test_data

    if method == "zscore_subject":
        # Compute stats from training data only
        train_normalized, mean, std = normalize_subject_zscore(train_data)

        # Apply same stats to test data
        test_normalized, _, _ = normalize_subject_zscore(test_data, mean=mean, std=std)

        return train_normalized, test_normalized

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_calibration_test_split(
    cal_data: np.ndarray,
    test_data: np.ndarray,
    method: str = "zscore_subject",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize calibration/test split for target subject (prevents data leakage).

    In transfer learning scenarios with calibration:
    - Calibration data: labeled trials from target subject for fine-tuning
    - Test data: unlabeled trials from target subject for evaluation

    IMPORTANT: Test data must be normalized using calibration statistics,
    since in real scenarios you wouldn't have access to test trial statistics.

    Args:
        cal_data: Calibration data from target subject (n_cal_trials, n_channels, n_timepoints)
        test_data: Test data from target subject (n_test_trials, n_channels, n_timepoints)
        method: Normalization method ("zscore_subject" or "none")

    Returns:
        normalized_cal: Normalized calibration data
        normalized_test: Normalized test data (using calibration stats)
        mean: Mean used for normalization
        std: Std used for normalization
    """
    if method == "none":
        return cal_data, test_data, None, None

    if method == "zscore_subject":
        # Compute stats from calibration data only
        cal_normalized, mean, std = normalize_subject_zscore(cal_data)

        # Apply same stats to test data (CRITICAL: no leakage from test data)
        test_normalized, _, _ = normalize_subject_zscore(test_data, mean=mean, std=std)

        return cal_normalized, test_normalized, mean, std

    else:
        raise ValueError(f"Unknown normalization method: {method}")
