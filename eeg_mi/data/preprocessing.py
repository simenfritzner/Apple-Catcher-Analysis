"""EEG preprocessing and feature extraction.

This module consolidates preprocessing functions from:
- src/game/preprocessing.py
- src/classification/dataManager/preprocessing/preprocessing.py
- src/classification/train_pca_lda.py

Provides source reconstruction, feature extraction, and basic preprocessing utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Union

import mne
import numpy as np


# Default frequency bands for feature extraction (alpha and mu)
DEFAULT_F_BANDS = ((7, 11), (9, 13))


def create_inverse_operator(
    info: mne.Info,
    spacing: str = "oct5",
    subjects_dir: Union[str, Path, None] = None,
) -> mne.minimum_norm.InverseOperator:
    """Create an inverse operator for source reconstruction using fsaverage template.

    Args:
        info: MNE Info object containing channel information
        spacing: Source space spacing ('oct5' or 'oct6')
        subjects_dir: Path to MNE subjects directory. If None, uses MNE config or sample data.

    Returns:
        Inverse operator for source reconstruction

    Note:
        Uses fsaverage template with sLORETA method.
        The oct5 spacing is faster, oct6 is more accurate but slower.
    """
    import time
    import logging

    logger = logging.getLogger(__name__)

    if subjects_dir is None:
        subjects_dir = mne.get_config("SUBJECTS_DIR")
        if subjects_dir is None:
            # Fall back to sample data subjects directory
            subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
    subjects_dir = Path(subjects_dir)

    # Ensure fsaverage is available (with retry for race conditions)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            mne.datasets.fetch_fsaverage(subjects_dir=str(subjects_dir), verbose=False)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to fetch fsaverage (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5)  # Wait before retry
            else:
                raise

    # Setup source space
    src = mne.setup_source_space(
        "fsaverage",
        spacing=spacing,
        subjects_dir=str(subjects_dir),
        add_dist=False,
        verbose=False,
    )

    # Load BEM solution with retry logic (file may be corrupted by concurrent writes)
    bem_path = subjects_dir / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            bem = mne.read_bem_solution(str(bem_path), verbose=False)
            break
        except (ValueError, RuntimeError, OSError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"BEM file corrupted or incomplete (attempt {attempt+1}/{max_retries}): {e}")
                # Remove corrupted file and re-download
                if bem_path.exists():
                    try:
                        bem_path.unlink()
                        logger.info("Removed corrupted BEM file, re-downloading...")
                    except:
                        pass
                time.sleep(5)
                # Re-fetch fsaverage to download BEM again
                mne.datasets.fetch_fsaverage(subjects_dir=str(subjects_dir), verbose=False)
            else:
                raise RuntimeError(
                    f"Failed to load BEM solution after {max_retries} attempts. "
                    f"The file may be corrupted. Try deleting: {bem_path}"
                ) from e

    # Create forward solution
    fwd = mne.make_forward_solution(
        info,
        trans="fsaverage",  # Use fsaverage transformation
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        verbose=False,
    )

    # Create noise covariance (ad-hoc diagonal)
    noise_cov = mne.make_ad_hoc_cov(info, verbose=False)

    # Create inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        info,
        fwd,
        noise_cov,
        loose=0.2,
        depth=0.8,
        verbose=False,
    )

    return inverse_operator


def source_reconstruction(
    epochs: mne.Epochs,
    inverse_operator: mne.minimum_norm.InverseOperator,
    method: str = "sLORETA",
    snr: float = 3.0,
    lambda2: Union[float, None] = None,
    pick_ori: str = "normal",
) -> List[mne.SourceEstimate]:
    """Apply inverse solution to epochs for source reconstruction.

    Args:
        epochs: MNE Epochs object
        inverse_operator: Inverse operator from create_inverse_operator()
        method: Inverse method ('sLORETA', 'dSPM', 'MNE', etc.)
        snr: Signal-to-noise ratio for regularization
        lambda2: Regularization parameter. If None, computed from snr
        pick_ori: Orientation constraint ('normal', 'vector', etc.)

    Returns:
        List of source estimates (one per epoch)
    """
    if lambda2 is None:
        lambda2 = 1.0 / snr**2

    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=pick_ori,
        verbose=False,
    )

    return stcs


def extract_features(
    epochs: mne.Epochs,
    inverse_operator: mne.minimum_norm.InverseOperator,
    frequencies: Union[Iterable[Tuple[int, int]], None] = None,
    tmin: float = -0.5,
    tmax: float = 1.0,
    decimation_factor: int = 1,
    method: str = "sLORETA",
    snr: float = 3.0,
) -> np.ndarray:
    """Extract bandpower features in source space.

    Computes log-bandpower features by:
    1. Filtering epochs in each frequency band
    2. Applying source reconstruction (sLORETA)
    3. Computing mean squared source activity
    4. Concatenating features across frequency bands

    Args:
        epochs: MNE Epochs object
        inverse_operator: Inverse operator from create_inverse_operator()
        frequencies: Iterable of (low, high) frequency band tuples.
                    If None, uses DEFAULT_F_BANDS (alpha/mu: 7-11, 9-13 Hz)
        tmin: Start time for epoch crop (seconds)
        tmax: End time for epoch crop (seconds)
        decimation_factor: Decimation factor for downsampling
        method: Inverse solution method
        snr: Signal-to-noise ratio

    Returns:
        Feature matrix of shape (n_epochs, n_sources * n_bands)
        where n_sources depends on the source space used
    """
    if frequencies is None:
        frequencies = DEFAULT_F_BANDS

    X_parts = []
    for f_low, f_high in frequencies:
        # Filter, decimate, and crop for this frequency band
        ep_filt = (
            epochs.copy()
            .filter(f_low, f_high, verbose=False)
            .decimate(decimation_factor, verbose=False)
            .crop(tmin=tmin, tmax=tmax, verbose=False)
        )

        # Apply source reconstruction
        stcs = source_reconstruction(
            ep_filt, inverse_operator, method=method, snr=snr
        )

        # Compute mean squared activity across time for each source
        X_band = np.array([np.mean(np.abs(stc.data) ** 2, axis=1) for stc in stcs])
        X_parts.append(X_band)

    # Concatenate features from all frequency bands
    return np.concatenate(X_parts, axis=1)


def filter_epochs(
    epochs: mne.Epochs,
    l_freq: Union[float, None] = None,
    h_freq: Union[float, None] = None,
    notch_freq: Union[float, None] = None,
) -> mne.Epochs:
    """Apply bandpass and notch filtering to epochs.

    Args:
        epochs: MNE Epochs object
        l_freq: Low frequency cutoff (Hz). None for high-pass only
        h_freq: High frequency cutoff (Hz). None for low-pass only
        notch_freq: Notch filter frequency (Hz). None to skip notch filtering

    Returns:
        Filtered epochs (modified in-place and returned)
    """
    if l_freq is not None or h_freq is not None:
        epochs.filter(l_freq, h_freq, verbose=False)

    if notch_freq is not None:
        epochs.notch_filter(notch_freq, trans_bandwidth=2, verbose=False)

    return epochs


def apply_reference(
    epochs: mne.Epochs,
    ref_type: str = "average",
) -> mne.Epochs:
    """Apply EEG reference to epochs.

    Args:
        epochs: MNE Epochs object
        ref_type: Reference type ('average', 'REST', or channel name)

    Returns:
        Re-referenced epochs (modified in-place and returned)
    """
    epochs.set_eeg_reference(ref_type, projection=True, verbose=False)
    return epochs


# Motor-relevant ROIs for motor imagery tasks
DEFAULT_MOTOR_ROIS = [
    'precentral-lh',        # Left primary motor cortex (M1)
    'precentral-rh',        # Right primary motor cortex (M1)
    'postcentral-lh',       # Left primary somatosensory cortex (S1)
    'postcentral-rh',       # Right primary somatosensory cortex (S1)
    'paracentral-lh',       # Left paracentral lobule (supplementary motor)
    'paracentral-rh',       # Right paracentral lobule
    'superiorfrontal-lh',   # Left superior frontal (includes SMA)
    'superiorfrontal-rh',   # Right superior frontal
    'supramarginal-lh',     # Left supramarginal gyrus (sensorimotor integration)
    'supramarginal-rh',     # Right supramarginal gyrus
]


def load_fsaverage_labels(
    subjects_dir: Union[str, Path, None] = None,
    parc: str = "aparc",
) -> List[mne.Label]:
    """Load fsaverage anatomical labels (ROIs) from an atlas.

    Args:
        subjects_dir: Path to MNE subjects directory. If None, uses MNE config or sample data.
        parc: Parcellation to use ('aparc' for Desikan-Killiany, 'aparc.a2009s' for Destrieux)

    Returns:
        List of MNE Label objects for all ROIs in the atlas

    Note:
        'aparc' (Desikan-Killiany) provides ~34 regions per hemisphere
        'aparc.a2009s' (Destrieux) provides ~74 regions per hemisphere
    """
    if subjects_dir is None:
        subjects_dir = mne.get_config("SUBJECTS_DIR")
        if subjects_dir is None:
            subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
    subjects_dir = Path(subjects_dir)

    # Ensure fsaverage is available
    mne.datasets.fetch_fsaverage(subjects_dir=str(subjects_dir), verbose=False)

    # Read labels
    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc=parc,
        subjects_dir=str(subjects_dir),
        verbose=False,
    )

    return labels


def extract_roi_timeseries(
    stcs: List[mne.SourceEstimate],
    roi_names: Union[List[str], None] = None,
    subjects_dir: Union[str, Path, None] = None,
    mode: str = "mean",
    parc: str = "aparc",
    spacing: str = "oct5",
) -> np.ndarray:
    """Extract ROI time-series from source estimates.

    Args:
        stcs: List of source estimates (one per epoch)
        roi_names: List of ROI names to extract. If None, uses DEFAULT_MOTOR_ROIS.
                  ROI names should match the atlas (e.g., 'precentral-lh', 'precentral-rh')
        subjects_dir: Path to MNE subjects directory
        mode: Aggregation mode ('mean', 'max', 'pca_flip')
              - 'mean': Average activity across vertices in ROI
              - 'max': Maximum activity across vertices
              - 'pca_flip': First PCA component with sign flip for consistency
        parc: Parcellation to use ('aparc' for Desikan-Killiany)
        spacing: Source space spacing ('oct5' or 'oct6'). Must match spacing used for stcs.

    Returns:
        ROI time-series array of shape (n_epochs, n_rois, n_timepoints)

    Raises:
        ValueError: If ROI name not found in atlas

    Note:
        This function provides standardized spatial features across subjects by:
        1. Using fsaverage template (same anatomy for all subjects)
        2. Using atlas-defined ROIs (same regions for all subjects)
        3. Aggregating within ROIs (reduces to manageable number of features)
    """
    if roi_names is None:
        roi_names = DEFAULT_MOTOR_ROIS

    # Load labels
    all_labels = load_fsaverage_labels(subjects_dir=subjects_dir, parc=parc)

    # Create label dictionary for easy lookup
    label_dict = {label.name: label for label in all_labels}

    # Select requested labels
    selected_labels = []
    for roi_name in roi_names:
        # Try to find label (handle both 'name' and 'name-lh'/'name-rh' formats)
        label = None
        if roi_name in label_dict:
            label = label_dict[roi_name]
        else:
            # Try adding hemisphere suffix if not present
            for hemi in ['-lh', '-rh']:
                full_name = f"{roi_name}{hemi}"
                if full_name in label_dict:
                    label = label_dict[full_name]
                    break

        if label is None:
            raise ValueError(
                f"ROI '{roi_name}' not found in {parc} atlas. "
                f"Available ROIs: {sorted(label_dict.keys())}"
            )

        selected_labels.append(label)

    # Get source space from first stc (need it for label extraction)
    # We need to create a proper source space object
    if subjects_dir is None:
        subjects_dir = mne.get_config("SUBJECTS_DIR")
        if subjects_dir is None:
            subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
    subjects_dir = Path(subjects_dir)

    # Create source space (needed for in_label)
    src = mne.setup_source_space(
        "fsaverage",
        spacing=spacing,  # Must match what was used in inverse operator
        subjects_dir=str(subjects_dir),
        add_dist=False,
        verbose=False,
    )

    # Extract time-series for each epoch
    roi_timeseries = []

    for stc in stcs:
        epoch_rois = []

        for label in selected_labels:
            # Extract label time course
            label_tc = stc.extract_label_time_course(
                label,
                src=src,
                mode=mode,
            )

            # label_tc is (n_vertices_in_label, n_timepoints) or (1, n_timepoints) if mode='mean'
            if mode in ['mean', 'max']:
                # Result is already averaged/maxed, shape (1, n_timepoints)
                epoch_rois.append(label_tc.ravel())  # Shape: (n_timepoints,)
            else:  # pca_flip
                # Take first component
                epoch_rois.append(label_tc[0])  # Shape: (n_timepoints,)

        roi_timeseries.append(np.array(epoch_rois))  # Shape: (n_rois, n_timepoints)

    # Stack all epochs
    roi_data = np.array(roi_timeseries)  # Shape: (n_epochs, n_rois, n_timepoints)

    return roi_data


def preprocess_pipeline_with_rois(
    epochs: mne.Epochs,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    roi_names: Union[List[str], None] = None,
    spacing: str = "oct5",
    method: str = "sLORETA",
    mode: str = "mean",
    subjects_dir: Union[str, Path, None] = None,
) -> np.ndarray:
    """Complete preprocessing pipeline: filtering → sLORETA → ROI extraction.

    This is the recommended pipeline for multi-subject transfer learning:
    1. Bandpass filter
    2. Source reconstruction using standardized fsaverage template
    3. Extract ROI time-series from motor-relevant regions

    Args:
        epochs: MNE Epochs object
        l_freq: Low frequency cutoff (Hz)
        h_freq: High frequency cutoff (Hz)
        roi_names: List of ROI names. If None, uses DEFAULT_MOTOR_ROIS
        spacing: Source space spacing ('oct5' or 'oct6')
        method: Inverse method ('sLORETA', 'dSPM', 'MNE')
        mode: ROI aggregation mode ('mean', 'max', 'pca_flip')
        subjects_dir: Path to MNE subjects directory

    Returns:
        ROI time-series array of shape (n_epochs, n_rois, n_timepoints)

    Example:
        >>> epochs, labels = load_subject_data('data/s01', return_epochs=True)
        >>> roi_data = preprocess_pipeline_with_rois(epochs, l_freq=8, h_freq=30)
        >>> print(roi_data.shape)  # (n_epochs, 10, n_timepoints)

    Note:
        This pipeline ensures spatial standardization across subjects by:
        - Using fsaverage template (identical anatomy)
        - Using atlas-defined ROIs (identical regions)
        - Using consistent spacing parameter (identical source count before ROI extraction)
    """
    # Step 1: Filter
    epochs_filt = filter_epochs(epochs.copy(), l_freq=l_freq, h_freq=h_freq)

    # Step 2: Create inverse operator
    inv_op = create_inverse_operator(
        epochs_filt.info,
        spacing=spacing,
        subjects_dir=subjects_dir,
    )

    # Step 3: Apply source reconstruction
    stcs = source_reconstruction(epochs_filt, inv_op, method=method)

    # Step 4: Extract ROI time-series
    roi_data = extract_roi_timeseries(
        stcs,
        roi_names=roi_names,
        subjects_dir=subjects_dir,
        mode=mode,
        spacing=spacing,
    )

    return roi_data
