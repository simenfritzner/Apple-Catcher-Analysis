"""MEMD (Multivariate Empirical Mode Decomposition) preprocessing for EEG.

This implements MEMD using the MEMD_all.py implementation, which processes
all channels simultaneously (multivariate) rather than channel-by-channel (EEMD).

Key difference from EEMD:
- MEMD: Processes all channels together, preserving inter-channel relationships
- EEMD: Processes each channel independently, loses inter-channel information

Reference:
    Rehman and D. P. Mandic, "Multivariate Empirical Mode Decomposition",
    Proceedings of the Royal Society A, 2010
"""

import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path
import sys

# Add MEMD_all to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "eeg_mi" / "data"))

from MEMD_all import memd

logger = logging.getLogger(__name__)


def apply_memd_to_trial(
    trial_data: np.ndarray,
    n_directions: int = 64,
    stop_criteria: str = 'stop',
    stop_vector: Tuple[float, float, float] = (0.075, 0.75, 0.075),
    remove_first_imf: bool = True,
) -> np.ndarray:
    """Apply MEMD decomposition to a single trial.

    Args:
        trial_data: EEG data for one trial (n_channels, n_timepoints)
        n_directions: Number of projection directions (default: 64)
                     Rule of thumb: >= 2 * n_channels
        stop_criteria: 'stop' or 'fix_h' (default: 'stop')
        stop_vector: Stopping criteria parameters (sd, sd2, tol)
        remove_first_imf: If True, remove first IMF (removes high-freq noise)

    Returns:
        Reconstructed data (n_channels, n_timepoints)

    Note:
        MEMD requires data in shape (n_timepoints, n_channels) internally.
        The function handles transposition automatically.
    """
    n_channels, n_timepoints = trial_data.shape

    # Ensure we have enough directions (rule of thumb: 2 * n_channels minimum)
    if n_directions < 2 * n_channels:
        logger.warning(
            f"n_directions={n_directions} is less than 2*n_channels={2*n_channels}. "
            f"Increasing to {2*n_channels} for better decomposition."
        )
        n_directions = 2 * n_channels

    # MEMD expects (n_timepoints, n_channels)
    trial_data_T = trial_data.T

    # Apply MEMD
    # Returns: imf shape (n_imfs, n_channels, n_timepoints)
    imfs = memd(trial_data_T, n_directions, stop_criteria, stop_vector)

    n_imfs = imfs.shape[0]

    # Reconstruct by summing IMFs
    if remove_first_imf and n_imfs > 1:
        # Skip first IMF (high-frequency noise)
        start_imf = 1
        logger.debug(f"Removing first IMF, using IMFs {start_imf} to {n_imfs-1}")
    else:
        start_imf = 0
        logger.debug(f"Keeping all {n_imfs} IMFs")

    # Sum IMFs: imfs shape is (n_imfs, n_channels, n_timepoints)
    # After summing along axis 0, result is already (n_channels, n_timepoints)
    if n_imfs > start_imf:
        reconstructed = np.sum(imfs[start_imf:-1], axis=0)  # Exclude residue (last "IMF")
        # reconstructed is already (n_channels, n_timepoints) - no transpose needed!
        return reconstructed
    else:
        # Not enough IMFs, return original (need to transpose trial_data_T back)
        return trial_data


def apply_memd_parallel(
    data: np.ndarray,
    n_directions: int = 64,
    remove_first_imf: bool = True,
    stop_criteria: str = 'stop',
    stop_vector: Tuple[float, float, float] = (0.075, 0.75, 0.075),
    n_jobs: int = -1,
) -> np.ndarray:
    """Apply MEMD with parallel processing (faster for many trials).

    Args:
        data: EEG data (n_trials, n_channels, n_timepoints)
        n_directions: Number of projection directions
        remove_first_imf: If True, remove first IMF
        stop_criteria: 'stop' or 'fix_h'
        stop_vector: Stopping criteria parameters
        n_jobs: Number of parallel jobs (-1 = all CPUs)

    Returns:
        Reconstructed data (n_trials, n_channels, n_timepoints)
    """
    from joblib import Parallel, delayed

    n_trials, n_channels, n_timepoints = data.shape
    logger.info(f"Applying MEMD (parallel): {n_trials} trials, {n_channels} channels")
    logger.info(f"  n_directions: {n_directions}, remove_first_imf: {remove_first_imf}")
    logger.info(f"  Parallel jobs: {n_jobs}")

    def process_trial(trial_idx, trial_data):
        """Process single trial."""
        try:
            reconstructed = apply_memd_to_trial(
                trial_data,
                n_directions=n_directions,
                stop_criteria=stop_criteria,
                stop_vector=stop_vector,
                remove_first_imf=remove_first_imf,
            )
            if (trial_idx + 1) % 10 == 0:
                logger.info(f"  Processed {trial_idx + 1}/{n_trials} trials")
            return reconstructed
        except Exception as e:
            logger.error(f"Error processing trial {trial_idx}: {e}")
            return trial_data  # Return original on error

    # Process all trials in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_trial)(i, data[i]) for i in range(n_trials)
    )

    reconstructed_data = np.array(results)
    logger.info("MEMD decomposition complete")
    return reconstructed_data


def preprocess_with_memd(
    data: np.ndarray,
    labels: np.ndarray,
    n_directions: int = 64,
    remove_first_imf: bool = True,
    stop_criteria: str = 'stop',
    stop_vector: Tuple[float, float, float] = (0.075, 0.75, 0.075),
    use_parallel: bool = True,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Complete preprocessing pipeline with TRUE MEMD (multivariate).

    Args:
        data: EEG data (n_trials, n_channels, n_timepoints)
        labels: Labels (n_trials,)
        n_directions: Number of projection directions for MEMD
                     Rule of thumb: >= 2 * n_channels
        remove_first_imf: If True, remove first IMF (removes noise)
        stop_criteria: 'stop' or 'fix_h'
        stop_vector: Stopping criteria parameters (sd, sd2, tol)
        use_parallel: If True, use parallel processing
        n_jobs: Number of parallel jobs

    Returns:
        preprocessed_data: Data after MEMD
        labels: Original labels (unchanged)
    """
    logger.info("Starting TRUE MEMD preprocessing...")
    logger.info(f"Input shape: {data.shape}")
    logger.info(f"This is MULTIVARIATE EMD - all channels processed together!")

    n_channels = data.shape[1]

    # Ensure adequate number of directions
    if n_directions < 2 * n_channels:
        n_directions = 2 * n_channels
        logger.info(f"Adjusted n_directions to {n_directions} (2 * n_channels)")

    if use_parallel:
        preprocessed_data = apply_memd_parallel(
            data,
            n_directions=n_directions,
            remove_first_imf=remove_first_imf,
            stop_criteria=stop_criteria,
            stop_vector=stop_vector,
            n_jobs=n_jobs
        )
    else:
        n_trials = data.shape[0]
        preprocessed_data = np.zeros_like(data)
        for trial_idx in range(n_trials):
            preprocessed_data[trial_idx] = apply_memd_to_trial(
                data[trial_idx],
                n_directions=n_directions,
                stop_criteria=stop_criteria,
                stop_vector=stop_vector,
                remove_first_imf=remove_first_imf,
            )
            if (trial_idx + 1) % 10 == 0:
                logger.info(f"  Processed {trial_idx + 1}/{n_trials} trials")

    logger.info(f"Output shape: {preprocessed_data.shape}")
    logger.info("TRUE MEMD preprocessing complete")

    return preprocessed_data, labels


# Example usage
if __name__ == "__main__":
    import time

    print("Testing TRUE MEMD preprocessing...")
    print("="*60)

    # Create dummy data
    n_trials = 5
    n_channels = 8
    n_timepoints = 501

    data = np.random.randn(n_trials, n_channels, n_timepoints)
    labels = np.random.randint(0, 2, n_trials)

    print(f"Input shape: {data.shape}")
    print(f"n_directions will be: {2 * n_channels} (2 * n_channels)")
    print("="*60)

    # Test MEMD
    start = time.time()
    try:
        preprocessed_data, _ = preprocess_with_memd(
            data, labels,
            n_directions=2 * n_channels,
            remove_first_imf=True,
            use_parallel=False
        )
        print(f"\nOutput shape: {preprocessed_data.shape}")
        print(f"Time: {time.time() - start:.2f}s")
        print("✓ TRUE MEMD preprocessing works!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
