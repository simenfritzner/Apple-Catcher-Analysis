"""
Riemannian geometry-based classifiers for motor imagery EEG.

This module implements Minimum Distance to Mean (MDM) classification on the
Riemannian manifold of symmetric positive definite (SPD) matrices. It includes
support for calibration-free BCI through Generic Recentering and evidence
accumulation for robust decision-making.

Classes:
    MDMClassifier: Basic MDM classifier using Riemannian distance to prototypes
    MDMWithRecentering: MDM with online adaptation via Generic Recentering
    EvidenceAccumulator: Temporal integration of predictions for robustness

References:
    Barachant, A., et al. (2010). "Classification of covariance matrices using a
    Riemannian-based kernel for BCI applications."

    Rodrigues, P. L. C., et al. (2019). "Riemannian Procrustes analysis:
    Transfer learning for brain-computer interfaces."
"""

import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from scipy.linalg import logm, sqrtm


# ============================================================================
# Riemannian Geometry Utilities
# ============================================================================

def matrix_sqrt(C: np.ndarray) -> np.ndarray:
    """
    Compute matrix square root via eigendecomposition.

    Args:
        C: Symmetric positive definite matrix of shape (n, n)

    Returns:
        Matrix square root C^(1/2) of shape (n, n)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def matrix_inv_sqrt(C: np.ndarray) -> np.ndarray:
    """
    Compute inverse matrix square root via eigendecomposition.

    Args:
        C: Symmetric positive definite matrix of shape (n, n)

    Returns:
        Inverse matrix square root C^(-1/2) of shape (n, n)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def matrix_log(C: np.ndarray) -> np.ndarray:
    """
    Compute matrix logarithm via eigendecomposition.

    Args:
        C: Symmetric positive definite matrix of shape (n, n)

    Returns:
        Matrix logarithm log(C) of shape (n, n)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability
    return eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T


def matrix_exp(C: np.ndarray) -> np.ndarray:
    """
    Compute matrix exponential via eigendecomposition.

    Args:
        C: Symmetric matrix of shape (n, n)

    Returns:
        Matrix exponential exp(C) of shape (n, n)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T


def matrix_power(C: np.ndarray, power: float) -> np.ndarray:
    """
    Compute matrix power via eigendecomposition: C^power.

    Args:
        C: Symmetric positive definite matrix of shape (n, n)
        power: Exponent value

    Returns:
        Matrix power C^power of shape (n, n)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability
    return eigenvectors @ np.diag(np.power(eigenvalues, power)) @ eigenvectors.T


def riemannian_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """
    Compute Riemannian distance between two covariance matrices.

    The Riemannian distance on the manifold of SPD matrices is defined as:
    δ(C₁, C₂) = ||log(C₁^(-1/2) × C₂ × C₁^(-1/2))||_F

    where ||·||_F denotes the Frobenius norm.

    Args:
        C1: First covariance matrix of shape (n, n)
        C2: Second covariance matrix of shape (n, n)

    Returns:
        Riemannian distance as a scalar float
    """
    C1_inv_sqrt = matrix_inv_sqrt(C1)
    temp = C1_inv_sqrt @ C2 @ C1_inv_sqrt
    log_temp = matrix_log(temp)
    return float(np.linalg.norm(log_temp, 'fro'))


def geodesic_interpolation(C1: np.ndarray, C2: np.ndarray, t: float) -> np.ndarray:
    """
    Geodesic interpolation between two covariance matrices on the SPD manifold.

    Computes the geodesic path between C1 and C2 at parameter t:
    γ(C₁, C₂, t) = C₁^(1/2) × (C₁^(-1/2) × C₂ × C₁^(-1/2))^t × C₁^(1/2)

    When t=0, returns C1; when t=1, returns C2.

    Args:
        C1: Starting covariance matrix of shape (n, n)
        C2: Ending covariance matrix of shape (n, n)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated covariance matrix of shape (n, n)
    """
    C1_sqrt = matrix_sqrt(C1)
    C1_inv_sqrt = matrix_inv_sqrt(C1)

    temp = C1_inv_sqrt @ C2 @ C1_inv_sqrt
    temp_power = matrix_power(temp, t)

    return C1_sqrt @ temp_power @ C1_sqrt


def riemannian_mean(
    covmats: List[np.ndarray],
    tol: float = 1e-6,
    max_iter: int = 50
) -> np.ndarray:
    """
    Compute Riemannian mean (Karcher/Fréchet mean) of covariance matrices.

    The Riemannian mean is the unique point that minimizes the sum of squared
    Riemannian distances to all input matrices:
    C̄ = argmin_C Σᵢ δ²(Cᵢ, C)

    Uses the iterative algorithm from Barachant et al. (2010).

    Args:
        covmats: List of covariance matrices, each of shape (n, n)
        tol: Convergence tolerance for the algorithm (default: 1e-6)
        max_iter: Maximum number of iterations (default: 50)

    Returns:
        Riemannian mean covariance matrix of shape (n, n)

    Raises:
        ValueError: If covmats is empty
    """
    if len(covmats) == 0:
        raise ValueError("Cannot compute mean of empty list")

    if len(covmats) == 1:
        return covmats[0].copy()

    # Initialize with Euclidean mean
    C_mean = np.mean(covmats, axis=0)

    for iteration in range(max_iter):
        # Compute inverse square root of current mean
        C_mean_inv_sqrt = matrix_inv_sqrt(C_mean)

        # Accumulate log differences
        log_sum = np.zeros_like(C_mean)
        for C in covmats:
            temp = C_mean_inv_sqrt @ C @ C_mean_inv_sqrt
            log_sum += matrix_log(temp)

        # Average the logs
        log_avg = log_sum / len(covmats)

        # Check convergence
        if np.linalg.norm(log_avg, 'fro') < tol:
            break

        # Update mean
        C_mean_sqrt = matrix_sqrt(C_mean)
        C_mean = C_mean_sqrt @ matrix_exp(log_avg) @ C_mean_sqrt

    return C_mean


def affine_transform(C: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply affine transformation for recentering on the SPD manifold.

    Transforms covariance matrix C using transform matrix T:
    C̃ = T^(-1/2) × C × T^(-1/2)ᵀ

    This transformation is used in Generic Recentering for online adaptation.

    Args:
        C: Covariance matrix to transform, shape (n, n)
        T: Transform matrix, shape (n, n)

    Returns:
        Transformed covariance matrix of shape (n, n)
    """
    T_inv_sqrt = matrix_inv_sqrt(T)
    return T_inv_sqrt @ C @ T_inv_sqrt.T


# ============================================================================
# MDM Classifier
# ============================================================================

class MDMClassifier:
    """
    Minimum Distance to Mean (MDM) classifier on the Riemannian manifold.

    Classifies covariance matrices based on their Riemannian distance to class
    prototypes (Riemannian means). This approach is particularly effective for
    motor imagery classification where spatial covariance patterns are key
    discriminative features.

    The classifier predicts the class with the minimum Riemannian distance:
    ŷ = argmin_k δ(C, P_k)

    where C is the test covariance matrix and P_k are the class prototypes.

    Attributes:
        n_classes: Number of classes in the classification problem
        prototypes: List of class prototype matrices (Riemannian means)

    Example:
        >>> # Create classifier for 2-class MI task
        >>> mdm = MDMClassifier(n_classes=2)
        >>>
        >>> # Fit using covariance matrices organized by class
        >>> covmats_class0 = [cov1, cov2, cov3]  # Left hand MI
        >>> covmats_class1 = [cov4, cov5, cov6]  # Right hand MI
        >>> mdm.fit([covmats_class0, covmats_class1])
        >>>
        >>> # Predict new sample
        >>> prediction = mdm.predict(test_cov)
        >>> probabilities = mdm.predict_proba(test_cov)
    """

    def __init__(self, n_classes: int = 2) -> None:
        """
        Initialize MDM classifier.

        Args:
            n_classes: Number of classes (default: 2 for binary MI classification)
        """
        self.n_classes = n_classes
        self.prototypes: Optional[List[np.ndarray]] = None

    def fit(self, covmats_by_class: List[List[np.ndarray]]) -> None:
        """
        Fit the classifier by computing Riemannian mean for each class.

        Computes the class prototypes as the Riemannian mean of all training
        covariance matrices for each class.

        Args:
            covmats_by_class: List of lists containing covariance matrices.
                Each inner list contains all covariance matrices for that class.
                Length must equal n_classes.

        Raises:
            ValueError: If number of classes doesn't match or if any class has
                no samples
        """
        if len(covmats_by_class) != self.n_classes:
            raise ValueError(
                f"Expected {self.n_classes} classes, got {len(covmats_by_class)}"
            )

        self.prototypes = []
        for class_idx, covmats in enumerate(covmats_by_class):
            if len(covmats) == 0:
                raise ValueError(f"No covariance matrices for class {class_idx}")

            prototype = riemannian_mean(covmats)
            self.prototypes.append(prototype)

    def predict(self, covmat: np.ndarray) -> int:
        """
        Predict class label for a single covariance matrix.

        Assigns the class with minimum Riemannian distance to the input.

        Args:
            covmat: Covariance matrix to classify, shape (n, n)

        Returns:
            Predicted class label (integer in range [0, n_classes-1])

        Raises:
            RuntimeError: If classifier has not been fitted
        """
        if self.prototypes is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        distances = self._compute_distances(covmat)
        return int(np.argmin(distances))

    def predict_proba(self, covmat: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for a single covariance matrix.

        Converts Riemannian distances to probabilities using softmax on
        negative squared distances:
        p_k = exp(-d_k²) / Σⱼ exp(-d_j²)

        Args:
            covmat: Covariance matrix to classify, shape (n, n)

        Returns:
            Array of class probabilities, shape (n_classes,)

        Raises:
            RuntimeError: If classifier has not been fitted
        """
        if self.prototypes is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        distances = self._compute_distances(covmat)

        # Convert distances to probabilities using softmax
        neg_sq_distances = -np.square(distances)
        # Subtract max for numerical stability
        exp_distances = np.exp(neg_sq_distances - np.max(neg_sq_distances))
        probas = exp_distances / np.sum(exp_distances)

        return probas

    def _compute_distances(self, covmat: np.ndarray) -> np.ndarray:
        """
        Compute Riemannian distances to all class prototypes.

        Args:
            covmat: Covariance matrix, shape (n, n)

        Returns:
            Array of distances to each class prototype, shape (n_classes,)
        """
        distances = np.zeros(self.n_classes)
        for class_idx, prototype in enumerate(self.prototypes):
            distances[class_idx] = riemannian_distance(covmat, prototype)
        return distances

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained classifier to disk.

        Serializes the classifier state including class prototypes using pickle.

        Args:
            filepath: Path to save file

        Raises:
            RuntimeError: If classifier has not been fitted
        """
        if self.prototypes is None:
            raise RuntimeError("Cannot save unfitted classifier.")

        filepath = Path(filepath)
        data = {
            'n_classes': self.n_classes,
            'prototypes': self.prototypes,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained classifier from disk.

        Args:
            filepath: Path to saved classifier file
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.n_classes = data['n_classes']
        self.prototypes = data['prototypes']


# ============================================================================
# MDM with Generic Recentering
# ============================================================================

class MDMWithRecentering:
    """
    MDM classifier with Generic Recentering for calibration-free online adaptation.

    Generic Recentering enables transfer learning by adapting a pre-trained MDM
    classifier to new users without requiring calibration data. It incrementally
    builds a transform matrix that aligns the new user's covariance distribution
    with the training distribution.

    The transform is updated using geodesic interpolation:
    T_test[i] = γ(T_test[i-1], C_subject[i], 1/(i+1))

    Each new covariance matrix is then transformed before classification:
    C̃ = T^(-1/2) × C × T^(-1/2)ᵀ

    Attributes:
        mdm: Underlying MDM classifier with pre-trained prototypes
        T_test: Current transform matrix for recentering
        sample_count: Number of samples processed for adaptation

    Example:
        >>> # Train MDM on source users
        >>> mdm = MDMClassifier(n_classes=2)
        >>> mdm.fit(source_data)
        >>>
        >>> # Wrap with recentering for new user
        >>> mdm_recenter = MDMWithRecentering(mdm)
        >>>
        >>> # Adapt and classify online (no calibration needed)
        >>> for test_cov in target_user_data:
        >>>     prediction = mdm_recenter.predict(test_cov)

    References:
        Rodrigues, P. L. C., et al. (2019). "Riemannian Procrustes analysis:
        Transfer learning for brain-computer interfaces."
    """

    def __init__(self, mdm_classifier: MDMClassifier) -> None:
        """
        Initialize MDM with Generic Recentering.

        Args:
            mdm_classifier: Pre-trained MDM classifier with class prototypes

        Raises:
            RuntimeError: If mdm_classifier has not been fitted
        """
        if mdm_classifier.prototypes is None:
            raise RuntimeError("MDM classifier must be fitted before use.")

        self.mdm = mdm_classifier
        self.T_test: Optional[np.ndarray] = None
        self.sample_count = 0

    def reset(self) -> None:
        """
        Reset recentering state.

        Clears the transform matrix and sample count. Useful when switching
        to a new user or restarting adaptation.
        """
        self.T_test = None
        self.sample_count = 0

    def predict(self, covmat: np.ndarray) -> int:
        """
        Predict class label with online recentering adaptation.

        Updates the transform matrix with the new sample, applies the transform,
        and classifies using the underlying MDM classifier.

        Args:
            covmat: Covariance matrix from new user, shape (n, n)

        Returns:
            Predicted class label (integer in range [0, n_classes-1])
        """
        self._update_transform(covmat)
        covmat_recentered = affine_transform(covmat, self.T_test)
        return self.mdm.predict(covmat_recentered)

    def predict_proba(self, covmat: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities with online recentering adaptation.

        Updates the transform matrix with the new sample, applies the transform,
        and returns class probabilities from the underlying MDM classifier.

        Args:
            covmat: Covariance matrix from new user, shape (n, n)

        Returns:
            Array of class probabilities, shape (n_classes,)
        """
        self._update_transform(covmat)
        covmat_recentered = affine_transform(covmat, self.T_test)
        return self.mdm.predict_proba(covmat_recentered)

    def _update_transform(self, covmat: np.ndarray) -> None:
        """
        Update the recentering transform using geodesic interpolation.

        Implements the Generic Recentering update rule:
        T_test[i] = γ(T_test[i-1], C_subject[i], 1/(i+1))

        The weight 1/(i+1) gives equal contribution to all samples seen so far,
        providing a running geometric mean.

        Args:
            covmat: New covariance matrix from subject, shape (n, n)
        """
        if self.T_test is None:
            # Initialize with first sample
            self.T_test = covmat.copy()
            self.sample_count = 1
        else:
            # Update via geodesic interpolation
            self.sample_count += 1
            weight = 1.0 / self.sample_count
            self.T_test = geodesic_interpolation(self.T_test, covmat, weight)


# ============================================================================
# Evidence Accumulator
# ============================================================================

class EvidenceAccumulator:
    """
    Accumulates evidence over time for robust command delivery in BCI.

    Integrates predictions from multiple time windows using exponential smoothing
    to reduce false positives and improve decision confidence. This is particularly
    important for real-time BCI applications where single-sample predictions may
    be noisy.

    Uses exponential smoothing for temporal integration:
    Prob[i] = (1 - α) × Prob[i-1] + α × p[i]

    A command is delivered only when accumulated probability exceeds a threshold.

    Attributes:
        n_classes: Number of classes
        alpha: Smoothing factor (weight for new samples)
        decision_threshold: Threshold for command delivery
        min_confidence: Minimum confidence to accumulate evidence
        accumulated_probs: Current accumulated probabilities

    Example:
        >>> accumulator = EvidenceAccumulator(
        >>>     n_classes=2,
        >>>     smoothing_factor=0.05,
        >>>     decision_threshold=0.75
        >>> )
        >>>
        >>> # Process streaming predictions
        >>> for test_cov in stream:
        >>>     probs = classifier.predict_proba(test_cov)
        >>>     decision = accumulator.update(probs)
        >>>
        >>>     if decision is not None:
        >>>         # Deliver command
        >>>         execute_action(decision)
        >>>         accumulator.reset()
    """

    def __init__(
        self,
        n_classes: int = 2,
        smoothing_factor: float = 0.05,
        decision_threshold: float = 0.75,
        min_confidence: float = 0.55,
    ) -> None:
        """
        Initialize evidence accumulator.

        Args:
            n_classes: Number of classes (default: 2)
            smoothing_factor: Weight for new samples in [0, 1]. Higher values
                give more weight to recent samples (default: 0.05)
            decision_threshold: Accumulated probability threshold for command
                delivery (default: 0.75)
            min_confidence: Minimum single-sample confidence to accumulate
                evidence. Samples below this are ignored (default: 0.55)
        """
        self.n_classes = n_classes
        self.alpha = smoothing_factor
        self.decision_threshold = decision_threshold
        self.min_confidence = min_confidence

        # Initialize with uniform distribution
        self.accumulated_probs = np.ones(n_classes) / n_classes

    def update(self, probas: np.ndarray) -> Optional[int]:
        """
        Update accumulated evidence with new probabilities.

        If the new sample has sufficient confidence (max probability ≥ min_confidence),
        updates the accumulated probabilities using exponential smoothing.
        Returns a decision if the accumulated evidence exceeds the threshold.

        Args:
            probas: Class probabilities from current sample, shape (n_classes,)

        Returns:
            Class label if threshold exceeded, None otherwise
        """
        max_prob = np.max(probas)

        # Only accumulate evidence if confidence is sufficient
        if max_prob >= self.min_confidence:
            self.accumulated_probs = (
                (1 - self.alpha) * self.accumulated_probs + self.alpha * probas
            )

        # Check if decision threshold is exceeded
        max_accumulated = np.max(self.accumulated_probs)

        if max_accumulated >= self.decision_threshold:
            decision = int(np.argmax(self.accumulated_probs))
            return decision

        return None

    def reset(self) -> None:
        """
        Reset accumulated evidence to uniform distribution.

        Should be called after delivering a command to start fresh accumulation.
        """
        self.accumulated_probs = np.ones(self.n_classes) / self.n_classes

    def get_accumulated_probs(self) -> np.ndarray:
        """
        Get current accumulated probabilities.

        Returns:
            Copy of accumulated probabilities, shape (n_classes,)
        """
        return self.accumulated_probs.copy()
