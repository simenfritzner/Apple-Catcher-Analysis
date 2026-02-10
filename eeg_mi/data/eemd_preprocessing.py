"""EEMD (Ensemble Empirical Mode Decomposition) preprocessing for EEG.

EEMD decomposes each EEG channel independently into Intrinsic Mode Functions (IMFs).
Note: This processes channels independently, not as multivariate data.
For true multivariate decomposition, see memd_preprocessing.py.
Removing the first IMF can help eliminate high-frequency noise and artifacts.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def apply_eemd(
    data: np.ndarray,
    n_imfs: int = 5,
    remove_first_imf: bool = True,
    ensemble_trials: int = 50,
) -> np.ndarray:
    """Apply EEMD decomposition and optionally remove first IMF.

    Args:
        data: EEG data (n_trials, n_channels, n_timepoints)
        n_imfs: Number of IMFs to extract
        remove_first_imf: If True, remove first IMF (removes high-freq noise)
        ensemble_trials: Number of ensemble trials for EEMD

    Returns:
        Reconstructed data without first IMF (same shape as input)

    Note:
        This processes each channel independently using EEMD.
        For true multivariate EMD, use memd_preprocessing.py.

        You need to install a EEMD library:
        - PyEMD (pip install EMD-signal)
        - libeemd (pip install libeemd)
    """
    try:
        from PyEMD import EEMD
    except ImportError:
        raise ImportError(
            "PyEMD not installed. Install with: pip install EMD-signal\n"
            "Or use libeemd: pip install libeemd"
        )

    n_trials, n_channels, n_timepoints = data.shape
    reconstructed_data = np.zeros_like(data)

    logger.info(f"Applying EEMD: {n_trials} trials, {n_channels} channels")
    logger.info(f"Extracting {n_imfs} IMFs, remove_first={remove_first_imf}, ensemble_trials={ensemble_trials}")

    for trial_idx in range(n_trials):
        trial_data = data[trial_idx]  # (n_channels, n_timepoints)

        # Apply EEMD (ensemble version, more robust)
        eemd = EEMD(trials=ensemble_trials)

        # Decompose each channel
        trial_imfs = []
        for ch_idx in range(n_channels):
            channel_data = trial_data[ch_idx]

            # Get IMFs for this channel
            imfs = eemd.eemd(channel_data, max_imf=n_imfs)
            trial_imfs.append(imfs)

        # Reconstruct by summing IMFs (skip first if requested)
        start_imf = 1 if remove_first_imf else 0

        for ch_idx in range(n_channels):
            imfs = trial_imfs[ch_idx]
            # Sum all IMFs except the first (if remove_first_imf=True)
            if len(imfs) > start_imf:
                reconstructed_data[trial_idx, ch_idx] = np.sum(imfs[start_imf:], axis=0)
            else:
                # Not enough IMFs, use original
                reconstructed_data[trial_idx, ch_idx] = trial_data[ch_idx]

        if (trial_idx + 1) % 50 == 0:
            logger.info(f"  Processed {trial_idx + 1}/{n_trials} trials")

    logger.info("EEMD decomposition complete")
    return reconstructed_data


def apply_eemd_simple(
    data: np.ndarray,
    n_imfs: int = 5,
    remove_first_imf: bool = True,
    ensemble_trials: int = 50,
    noise_strength: float = 0.2,
) -> np.ndarray:
    """Simplified EEMD using libeemd (faster, but needs installation).

    Args:
        data: EEG data (n_trials, n_channels, n_timepoints)
        n_imfs: Number of IMFs to extract
        remove_first_imf: If True, remove first IMF
        ensemble_trials: Number of ensemble trials
        noise_strength: Noise strength for EEMD

    Returns:
        Reconstructed data without first IMF
    """
    try:
        from libeemd import eemd
    except ImportError:
        raise ImportError(
            "libeemd not installed. Install with: pip install libeemd\n"
            "Or use PyEMD version: apply_eemd()"
        )

    n_trials, n_channels, n_timepoints = data.shape
    reconstructed_data = np.zeros_like(data)

    logger.info(f"Applying EEMD (libeemd): {n_trials} trials, {n_channels} channels")
    logger.info(f"Extracting {n_imfs} IMFs, remove_first={remove_first_imf}, ensemble_trials={ensemble_trials}")

    for trial_idx in range(n_trials):
        trial_data = data[trial_idx]  # (n_channels, n_timepoints)

        # Apply ensemble EMD
        # Note: libeemd expects (n_timepoints, n_channels) format
        trial_data_T = trial_data.T  # (n_timepoints, n_channels)

        imfs = eemd(trial_data_T, num_imfs=n_imfs, ensemble_size=ensemble_trials, noise_strength=noise_strength)

        # imfs shape: (n_imfs, n_timepoints, n_channels)
        # Reconstruct by summing IMFs
        start_imf = 1 if remove_first_imf else 0
        reconstructed = np.sum(imfs[start_imf:], axis=0)  # (n_timepoints, n_channels)

        reconstructed_data[trial_idx] = reconstructed.T  # Back to (n_channels, n_timepoints)

        if (trial_idx + 1) % 50 == 0:
            logger.info(f"  Processed {trial_idx + 1}/{n_trials} trials")

    logger.info("EEMD decomposition complete")
    return reconstructed_data


def apply_eemd_parallel(
    data: np.ndarray,
    n_imfs: int = 5,
    remove_first_imf: bool = True,
    ensemble_trials: int = 50,
    n_jobs: int = -1,
) -> np.ndarray:
    """Apply EEMD with parallel processing (faster for many trials).

    Args:
        data: EEG data (n_trials, n_channels, n_timepoints)
        n_imfs: Number of IMFs to extract
        remove_first_imf: If True, remove first IMF
        ensemble_trials: Number of ensemble trials for EEMD
        n_jobs: Number of parallel jobs (-1 = all CPUs)

    Returns:
        Reconstructed data without first IMF
    """
    from joblib import Parallel, delayed

    n_trials, n_channels, n_timepoints = data.shape
    logger.info(f"Applying EEMD (parallel): {n_trials} trials, {n_jobs} jobs")
    logger.info(f"Extracting {n_imfs} IMFs, remove_first={remove_first_imf}, ensemble_trials={ensemble_trials}")

    def process_trial(trial_data, n_imfs, remove_first_imf, ensemble_trials):
        """Process single trial."""
        try:
            from PyEMD import EEMD
        except ImportError:
            raise ImportError("PyEMD not installed. Install with: pip install EMD-signal")

        eemd = EEMD(trials=ensemble_trials)
        reconstructed = np.zeros_like(trial_data)

        for ch_idx in range(trial_data.shape[0]):
            channel_data = trial_data[ch_idx]
            imfs = eemd.eemd(channel_data, max_imf=n_imfs)

            start_imf = 1 if remove_first_imf else 0
            if len(imfs) > start_imf:
                reconstructed[ch_idx] = np.sum(imfs[start_imf:], axis=0)
            else:
                reconstructed[ch_idx] = channel_data

        return reconstructed

    # Process all trials in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_trial)(data[i], n_imfs, remove_first_imf, ensemble_trials) for i in range(n_trials)
    )

    reconstructed_data = np.array(results)
    logger.info("EEMD decomposition complete")
    return reconstructed_data


def preprocess_with_eemd(
    data: np.ndarray,
    labels: np.ndarray,
    n_imfs: int = 5,
    remove_first_imf: bool = True,
    ensemble_trials: int = 50,
    use_parallel: bool = False,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Complete preprocessing pipeline with EEMD.

    Args:
        data: EEG data (n_trials, n_channels, n_timepoints)
        labels: Labels (n_trials,)
        n_imfs: Number of IMFs to extract
        remove_first_imf: If True, remove first IMF (removes noise)
        ensemble_trials: Number of ensemble trials for EEMD
        use_parallel: If True, use parallel processing
        n_jobs: Number of parallel jobs

    Returns:
        preprocessed_data: Data after EEMD
        labels: Original labels (unchanged)
    """
    logger.info("Starting EEMD preprocessing...")
    logger.info(f"Input shape: {data.shape}")

    if use_parallel:
        preprocessed_data = apply_eemd_parallel(
            data,
            n_imfs=n_imfs,
            remove_first_imf=remove_first_imf,
            ensemble_trials=ensemble_trials,
            n_jobs=n_jobs
        )
    else:
        preprocessed_data = apply_eemd(
            data,
            n_imfs=n_imfs,
            remove_first_imf=remove_first_imf,
            ensemble_trials=ensemble_trials
        )

    logger.info(f"Output shape: {preprocessed_data.shape}")
    logger.info("EEMD preprocessing complete")

    return preprocessed_data, labels


# Backward compatibility: old function names (deprecated)
def apply_memd(*args, **kwargs):
    """Deprecated: Use apply_eemd() instead."""
    logger.warning("apply_memd() is deprecated. Use apply_eemd() instead.")
    return apply_eemd(*args, **kwargs)


def apply_memd_simple(*args, **kwargs):
    """Deprecated: Use apply_eemd_simple() instead."""
    logger.warning("apply_memd_simple() is deprecated. Use apply_eemd_simple() instead.")
    return apply_eemd_simple(*args, **kwargs)


def apply_memd_parallel(*args, **kwargs):
    """Deprecated: Use apply_eemd_parallel() instead."""
    logger.warning("apply_memd_parallel() is deprecated. Use apply_eemd_parallel() instead.")
    return apply_eemd_parallel(*args, **kwargs)


def preprocess_with_memd(*args, **kwargs):
    """Deprecated: Use preprocess_with_eemd() instead."""
    logger.warning("preprocess_with_memd() is deprecated. Use preprocess_with_eemd() instead.")
    return preprocess_with_eemd(*args, **kwargs)


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    import time

    print("Testing EEMD preprocessing...")

    # Create dummy data
    n_trials = 10
    n_channels = 32
    n_timepoints = 501

    data = np.random.randn(n_trials, n_channels, n_timepoints)
    labels = np.random.randint(0, 2, n_trials)

    print(f"Input shape: {data.shape}")

    # Test EEMD
    start = time.time()
    try:
        preprocessed_data, _ = preprocess_with_eemd(
            data, labels,
            remove_first_imf=True,
            use_parallel=False
        )
        print(f"Output shape: {preprocessed_data.shape}")
        print(f"Time: {time.time() - start:.2f}s")
        print("✓ EEMD preprocessing works!")
    except ImportError as e:
        print(f"✗ Error: {e}")
        print("\nTo use EEMD, install one of:")
        print("  pip install EMD-signal  (PyEMD)")
        print("  pip install libeemd     (libeemd - faster)")
