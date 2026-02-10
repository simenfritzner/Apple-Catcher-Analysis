"""Data loading utilities for EEG motor imagery experiments.

Consolidates loading functions from:
- src/classification/dataManager/ea_loader.py
- src/classification/dataManager/baseline_loader.py

Provides unified data loading with optional preprocessing pipelines (EA, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import mne
import numpy as np


logger = logging.getLogger(__name__)


def load_subject_data_npz(
    subject_dir: Union[str, Path],
    max_trials: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed NPZ data for a single subject.

    Args:
        subject_dir: Path to subject directory containing trial_XXX.npz files
        max_trials: Maximum number of trials to load

    Returns:
        data: EEG data (n_trials, n_channels, n_timepoints)
        labels: Label array (n_trials,)

    Raises:
        IOError: If no NPZ files found or no valid data loaded
    """
    subject_dir = Path(subject_dir)
    trial_files = sorted(list(subject_dir.glob("trial_*.npz")))

    if not trial_files:
        raise IOError(f"No NPZ trial files found in {subject_dir}")

    # Limit to max_trials
    trial_files = trial_files[:max_trials]

    data_list = []
    labels_list = []

    for trial_file in trial_files:
        try:
            trial_data = np.load(trial_file)
            data_list.append(trial_data['data'])
            labels_list.append(trial_data['label'])
        except Exception as e:
            logger.warning(f"Could not load {trial_file}: {e}")

    if not data_list:
        raise IOError(f"No valid NPZ data loaded from {subject_dir}")

    data = np.stack(data_list, axis=0)  # (n_trials, n_channels, n_timepoints)
    labels = np.array(labels_list)  # (n_trials,)

    # WORKAROUND: Fix shape if data was preprocessed with old buggy MEMD code
    # Expected shape: (n_trials, n_channels=32, n_timepoints)
    # Buggy shape: (n_trials, n_timepoints, n_channels=32)
    if data.ndim == 3:
        n_trials, dim1, dim2 = data.shape
        # If dim2 is 32 (typical n_channels) and dim1 is much larger (timepoints)
        # then we have the wrong shape and need to transpose
        if dim2 == 32 and dim1 > 100:
            logger.warning(
                f"Detected wrong data shape {data.shape}. "
                f"Transposing to fix MEMD preprocessing bug. "
                f"Shape (n_trials, n_timepoints={dim1}, n_channels={dim2}) -> "
                f"(n_trials, n_channels={dim2}, n_timepoints={dim1})"
            )
            data = data.transpose(0, 2, 1)  # (n_trials, n_timepoints, n_channels) -> (n_trials, n_channels, n_timepoints)

    return data, labels


def load_subject_data(
    subject_dir: Union[str, Path],
    max_trials: int = 200,
    tmin: float = 0.5,
    tmax: float = 2.5,
    return_epochs: bool = False,
    l_freq: Union[float, None] = None,
    h_freq: Union[float, None] = None,
    detrend: Union[int, None] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[mne.Epochs, np.ndarray]]:
    """Load all epoch data for a single subject.

    Args:
        subject_dir: Path to subject directory containing *_epo.fif files OR trial_*.npz files
        max_trials: Maximum number of trials to load
        tmin: Start time for epoch cropping (seconds)
        tmax: End time for epoch cropping (seconds)
        return_epochs: If True, return MNE Epochs object instead of numpy array
        l_freq: Low frequency cutoff for bandpass filter (Hz). None to skip filtering
        h_freq: High frequency cutoff for bandpass filter (Hz). None to skip filtering
        detrend: Detrending order (None=no detrend, 0=constant/DC, 1=linear). Applied before filtering.

    Returns:
        data: EEG data (n_trials, n_channels, n_timepoints) or MNE Epochs
        labels: Label array (n_trials,)

    Raises:
        IOError: If no epoch files found or no valid data loaded
    """
    subject_dir = Path(subject_dir)

    # Check if this is preprocessed NPZ data
    if list(subject_dir.glob("trial_*.npz")):
        if return_epochs:
            raise ValueError("return_epochs=True not supported for NPZ data")
        if l_freq is not None or h_freq is not None:
            logger.warning("Filtering parameters ignored for preprocessed NPZ data")
        if detrend is not None:
            logger.warning("Detrending parameters ignored for preprocessed NPZ data")
        if tmin != 0.5 or tmax != 2.5:
            logger.warning("Time cropping parameters ignored for preprocessed NPZ data")
        return load_subject_data_npz(subject_dir, max_trials)

    # Otherwise, load from FIF files
    epoch_files = sorted(list(subject_dir.glob("*_epo.fif")))

    if not epoch_files:
        raise IOError(f"No epoch files found in {subject_dir}")

    all_epochs = []
    all_labels = []

    for epoch_file in epoch_files:
        try:
            epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)

            # Crop to motor imagery period
            epochs = epochs.crop(tmin=tmin, tmax=tmax, verbose=False)

            # Apply detrending if specified (before filtering)
            if detrend is not None:
                epochs = epochs.apply_function(
                    lambda x: mne.filter.detrend(x, order=detrend, axis=-1),
                    picks='eeg',
                    channel_wise=True,
                    verbose=False
                )

            # Apply bandpass filter if specified
            if l_freq is not None or h_freq is not None:
                epochs = epochs.filter(l_freq, h_freq, verbose=False)

            # Get labels
            labels = epochs.events[:, -1]

            all_epochs.append(epochs)
            all_labels.append(labels)

        except Exception as e:
            logger.warning(f"Could not load {epoch_file}: {e}")

    if not all_epochs:
        raise IOError(f"No valid data loaded from {subject_dir}")

    # Concatenate all sessions
    epochs_concat = mne.concatenate_epochs(all_epochs, verbose=False)
    labels_concat = np.concatenate(all_labels, axis=0)

    # Limit to max_trials
    if len(epochs_concat) > max_trials:
        epochs_concat = epochs_concat[:max_trials]
        labels_concat = labels_concat[:max_trials]

    if return_epochs:
        return epochs_concat, labels_concat
    else:
        data = epochs_concat.get_data()  # (n_epochs, n_channels, n_timepoints)
        return data, labels_concat


def load_all_subjects(
    data_path: Union[str, Path],
    subject_ids: Union[List[str], None] = None,
    max_trials: int = 200,
    tmin: float = 0.5,
    tmax: float = 2.5,
    return_epochs: bool = False,
    l_freq: Union[float, None] = None,
    h_freq: Union[float, None] = None,
    detrend: Union[int, None] = None,
    normalization: str = "none",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load data for multiple subjects.

    Args:
        data_path: Path to data directory containing subject subdirectories
        subject_ids: List of subject IDs to load (e.g., ['s01', 's02']).
                    If None, auto-detects all subjects starting with 's'
        max_trials: Maximum trials per subject
        tmin: Start time for epoch cropping (seconds)
        tmax: End time for epoch cropping (seconds)
        return_epochs: If True, return MNE Epochs objects instead of numpy arrays
        l_freq: Low frequency cutoff for bandpass filter (Hz). None to skip filtering
        h_freq: High frequency cutoff for bandpass filter (Hz). None to skip filtering
        detrend: Detrending order (None=no detrend, 0=constant/DC, 1=linear). Applied before filtering.
        normalization: Normalization method ("zscore_subject", "zscore_global", "none")

    Returns:
        Dictionary mapping subject_id -> (data, labels)

    Raises:
        IOError: If no subjects loaded successfully
    """
    data_path = Path(data_path)

    if subject_ids is None:
        # Auto-detect subjects
        subject_dirs = sorted(
            [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("s")]
        )
        subject_ids = [d.name for d in subject_dirs]
    else:
        subject_dirs = [data_path / sid for sid in subject_ids]

    subjects_data = {}

    filter_info = ""
    if l_freq is not None or h_freq is not None:
        filter_info = f" (filtering: {l_freq or 'None'}-{h_freq or 'None'} Hz)"

    logger.info(f"Loading data for {len(subject_ids)} subjects{filter_info}...")

    for subject_id, subject_dir in zip(subject_ids, subject_dirs):
        try:
            data, labels = load_subject_data(
                subject_dir,
                max_trials=max_trials,
                tmin=tmin,
                tmax=tmax,
                return_epochs=return_epochs,
                l_freq=l_freq,
                h_freq=h_freq,
                detrend=detrend,
            )
            subjects_data[subject_id] = (data, labels)

            shape_info = (
                f"{len(data)} trials"
                if return_epochs
                else f"{data.shape[0]} trials, {data.shape[1]} channels, {data.shape[2]} timepoints"
            )
            logger.info(f"  {subject_id}: {shape_info}")

        except Exception as e:
            logger.warning(f"Could not load {subject_id}: {e}")

    if not subjects_data:
        raise IOError("No subjects loaded successfully")

    # NOTE: Normalization should NOT be applied here to prevent data leakage!
    # For LOSO validation:
    #   - Training subjects should be normalized using their own statistics
    #   - Test subject should be normalized AFTER splitting into calibration/test
    #   - Normalization must happen AFTER the train/test split in the calling code
    #
    # The 'normalization' parameter is kept for backward compatibility but ignored.
    if normalization != "none":
        logger.warning(
            f"Normalization '{normalization}' specified in load_all_subjects() is IGNORED. "
            f"Normalization must be applied AFTER train/test split to prevent data leakage. "
            f"Apply normalization in your training code using normalize_train_test_split() "
            f"or normalize_calibration_test_split()."
        )

    return subjects_data


def prepare_loso_fold(
    subjects_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_subject_id: str,
    train_size: int = 100,
    test_cal_size: int = 100,
    test_eval_start: int = 100,
    test_eval_end: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for one Leave-One-Subject-Out (LOSO) fold.

    Args:
        subjects_data: Dictionary mapping subject_id -> (data, labels)
        test_subject_id: ID of test subject (left out)
        train_size: Number of trials per training subject to use
        test_cal_size: Number of trials from test subject for calibration
                      (not used in this basic version, reserved for EA)
        test_eval_start: Start index for test evaluation set
        test_eval_end: End index for test evaluation set

    Returns:
        train_data: Training data (n_train_trials, n_channels, n_timepoints)
        train_labels: Training labels (n_train_trials,)
        test_data: Test data (n_test_trials, n_channels, n_timepoints)
        test_labels: Test labels (n_test_trials,)

    Note:
        This basic version does not apply preprocessing pipelines.
        Use with pipelines.py (EA, etc.) for advanced preprocessing.
    """
    train_data_list = []
    train_labels_list = []

    # Process training subjects
    for subject_id, (data, labels) in subjects_data.items():
        if subject_id == test_subject_id:
            continue

        # Use first train_size trials for training
        train_data_list.append(data[:train_size])
        train_labels_list.append(labels[:train_size])

    # Concatenate all training data
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    # Process test subject
    test_data_full, test_labels_full = subjects_data[test_subject_id]

    # Use evaluation set (e.g., trials 100-200)
    test_data = test_data_full[test_eval_start:test_eval_end]
    test_labels = test_labels_full[test_eval_start:test_eval_end]

    return train_data, train_labels, test_data, test_labels


def split_calibration_test(
    data: np.ndarray,
    labels: np.ndarray,
    cal_size: int,
    test_start: int = None,
    test_end: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into calibration and test sets.

    Args:
        data: Full data array (n_trials, ...)
        labels: Full label array (n_trials,)
        cal_size: Number of trials for calibration set
        test_start: Start index for test set. If None, uses cal_size
        test_end: End index for test set. If None, uses all remaining trials

    Returns:
        cal_data: Calibration data
        cal_labels: Calibration labels
        test_data: Test data
        test_labels: Test labels
    """
    cal_data = data[:cal_size]
    cal_labels = labels[:cal_size]

    if test_start is None:
        test_start = cal_size
    if test_end is None:
        test_end = len(data)

    test_data = data[test_start:test_end]
    test_labels = labels[test_start:test_end]

    return cal_data, cal_labels, test_data, test_labels
