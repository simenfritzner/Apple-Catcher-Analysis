"""Preprocessing pipelines for EEG data alignment and transformation.

This module provides complex preprocessing pipelines like Euclidean Alignment (EA)
and Common Spatial Patterns (CSP) that go beyond simple filtering.

Reference for EA:
    He, H., & Wu, D. (2019). Transfer learning for brain–computer interfaces:
    A Euclidean space data alignment approach. IEEE Transactions on Biomedical
    Engineering, 67(2), 399-410.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from scipy.linalg import sqrtm


class EuclideanAlignment:
    """Euclidean Alignment for domain adaptation in EEG classification.

    EA transforms covariance matrices to a common space where all domains
    have the identity matrix as their mean covariance. This enables better
    transfer learning across subjects by aligning their covariance structures.

    Attributes:
        reference_matrix: The mean covariance matrix computed from calibration data
    """

    def __init__(self):
        """Initialize Euclidean Alignment transformer."""
        self.reference_matrix: Union[np.ndarray, None] = None

    def fit(self, covariances: List[np.ndarray]) -> EuclideanAlignment:
        """Compute the reference matrix from calibration data.

        Args:
            covariances: List of covariance matrices (each n_channels x n_channels)

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If covariances list is empty
        """
        if len(covariances) == 0:
            raise ValueError("Cannot fit EA with empty covariance list")

        # Compute mean covariance (Euclidean mean)
        self.reference_matrix = np.mean(covariances, axis=0)

        # Ensure positive definiteness by adding small regularization if needed
        min_eig = np.min(np.linalg.eigvalsh(self.reference_matrix))
        if min_eig < 1e-10:
            n_channels = self.reference_matrix.shape[0]
            self.reference_matrix += (1e-10 - min_eig) * np.eye(n_channels)

        return self

    def transform(self, covariances: List[np.ndarray]) -> List[np.ndarray]:
        """Transform covariance matrices using Euclidean Alignment.

        For each covariance matrix C, applies: C_aligned = R^(-1/2) @ C @ R^(-1/2)
        where R is the reference matrix.

        Args:
            covariances: List of covariance matrices to transform

        Returns:
            List of aligned covariance matrices

        Raises:
            RuntimeError: If EA has not been fitted yet
        """
        if self.reference_matrix is None:
            raise RuntimeError("EA not fitted. Call fit() first.")

        # Compute R^(-1/2)
        R_inv_sqrt = self._matrix_inv_sqrt(self.reference_matrix)

        # Transform all covariance matrices
        aligned = []
        for C in covariances:
            C_aligned = R_inv_sqrt @ C @ R_inv_sqrt.T
            aligned.append(C_aligned)

        return aligned

    def fit_transform(self, covariances: List[np.ndarray]) -> List[np.ndarray]:
        """Fit to data and transform it in one step.

        Args:
            covariances: List of covariance matrices

        Returns:
            List of aligned covariance matrices
        """
        self.fit(covariances)
        return self.transform(covariances)

    @staticmethod
    def _matrix_inv_sqrt(C: np.ndarray) -> np.ndarray:
        """Compute inverse square root of a matrix via eigendecomposition.

        Args:
            C: Symmetric positive definite matrix

        Returns:
            C^(-1/2)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Ensure positivity
        return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    @staticmethod
    def _matrix_sqrt(C: np.ndarray) -> np.ndarray:
        """Compute matrix square root via eigendecomposition.

        Args:
            C: Symmetric positive definite matrix

        Returns:
            C^(1/2)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def compute_covariance_matrices(
    data: np.ndarray,
    method: str = "scm",
    normalize: bool = True,
) -> List[np.ndarray]:
    """Compute covariance matrices from EEG trials.

    Args:
        data: EEG data (n_trials, n_channels, n_timepoints)
        method: Covariance estimation method
               - 'scm': Sample Covariance Matrix (default)
               - 'ledoit_wolf': Ledoit-Wolf shrinkage estimator
        normalize: If True, normalize by trace (recommended for EA)

    Returns:
        List of covariance matrices (one per trial)

    Raises:
        ValueError: If unknown covariance method specified
    """
    n_trials = data.shape[0]
    covariances = []

    for trial_idx in range(n_trials):
        trial_data = data[trial_idx]  # (n_channels, n_timepoints)

        if method == "scm":
            # Sample Covariance Matrix: C = X @ X^T / n_timepoints
            cov = (trial_data @ trial_data.T) / trial_data.shape[1]
        elif method == "ledoit_wolf":
            from sklearn.covariance import ledoit_wolf

            cov, _ = ledoit_wolf(trial_data.T)
        else:
            raise ValueError(f"Unknown covariance method: {method}")

        # Normalize by trace (makes covariances scale-invariant)
        if normalize:
            trace = np.trace(cov)
            if trace > 0:
                cov = cov / trace

        covariances.append(cov)

    return covariances


def apply_ea_alignment(
    data: np.ndarray,
    calibration_indices: Union[np.ndarray, None] = None,
    ea_fitted: Union[EuclideanAlignment, None] = None,
) -> tuple[np.ndarray, EuclideanAlignment]:
    """Apply Euclidean Alignment to EEG data.

    Args:
        data: EEG data (n_trials, n_channels, n_timepoints)
        calibration_indices: Indices of trials to use for computing EA reference.
                           If None and ea_fitted is None, uses all trials.
        ea_fitted: Pre-fitted EuclideanAlignment object. If provided, uses this
                  instead of fitting new one.

    Returns:
        aligned_data: EA-aligned data (same shape as input)
        ea: Fitted EuclideanAlignment object

    Note:
        The alignment is performed in covariance space, then data is reconstructed
        to preserve the covariance structure while maintaining time-series form.
    """
    # Compute covariances for all trials
    all_covs = compute_covariance_matrices(data)

    # Fit EA if not provided
    if ea_fitted is None:
        if calibration_indices is None:
            calibration_covs = all_covs
        else:
            calibration_covs = [all_covs[i] for i in calibration_indices]

        ea = EuclideanAlignment()
        ea.fit(calibration_covs)
    else:
        ea = ea_fitted

    # Transform all covariances
    aligned_covs = ea.transform(all_covs)

    # Reconstruct time-series data from aligned covariances
    # This preserves the covariance structure in the data
    aligned_data = reconstruct_data_from_covariances(
        aligned_covs, data.shape[1], data.shape[2]
    )

    return aligned_data, ea


def reconstruct_data_from_covariances(
    covariances: List[np.ndarray],
    n_channels: int,
    n_timepoints: int,
    seed: Union[int, None] = None,
) -> np.ndarray:
    """Reconstruct time-series data from covariance matrices.

    Uses eigendecomposition to generate data with the desired covariance structure.

    Args:
        covariances: List of covariance matrices
        n_channels: Number of channels
        n_timepoints: Number of timepoints
        seed: Random seed for reproducibility

    Returns:
        Reconstructed data (n_trials, n_channels, n_timepoints)

    Note:
        This generates new time-series data that has the same covariance structure
        as specified. The temporal dynamics are randomized but the spatial structure
        (covariance) is preserved.
    """
    if seed is not None:
        np.random.seed(seed)

    n_trials = len(covariances)
    reconstructed = np.zeros((n_trials, n_channels, n_timepoints))

    for trial_idx, cov in enumerate(covariances):
        # Generate white noise
        noise = np.random.randn(n_channels, n_timepoints)

        # Compute covariance square root
        C_sqrt = EuclideanAlignment._matrix_sqrt(cov)

        # Apply covariance structure
        reconstructed[trial_idx] = C_sqrt @ noise

    return reconstructed
