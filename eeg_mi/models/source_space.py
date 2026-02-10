"""Source space classifier using PCA+LDA pipeline.

This module provides a scikit-learn-like classifier that:
1. Creates an inverse operator for source reconstruction
2. Extracts features using sLORETA in source space
3. Trains a StandardScaler + PCA + LDA pipeline
"""

from __future__ import annotations

from typing import Union

import mne
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_mi.data.preprocessing import (
    create_inverse_operator,
    extract_features,
)


class SourceSpaceClassifier:
    """A classifier using source space features with PCA+LDA pipeline.

    This classifier follows the pattern from train_pca_lda.py:
    1. Creates inverse operator for source reconstruction
    2. Extracts bandpower features using sLORETA
    3. Trains StandardScaler + PCA + LDA pipeline

    Attributes:
        n_components: Number of PCA components or variance ratio (0.0-1.0)
        inverse_operator_: The fitted inverse operator (available after fit)
        pipeline_: The fitted sklearn pipeline (available after fit)

    Example:
        >>> clf = SourceSpaceClassifier(n_components=0.95)
        >>> clf.fit(epochs_train, y_train)
        >>> predictions = clf.predict(epochs_test)
        >>> probabilities = clf.predict_proba(epochs_test)
    """

    def __init__(
        self,
        n_components: Union[int, float] = 0.95,
        tmin: float = -0.5,
        tmax: float = 1.0,
        frequencies: Union[tuple[tuple[int, int], ...], None] = None,
        decimation_factor: int = 1,
        method: str = "sLORETA",
        snr: float = 3.0,
        spacing: str = "oct5",
    ):
        """Initialize the source space classifier.

        Args:
            n_components: Number of PCA components to keep.
                If float (0.0-1.0), keeps components explaining this variance ratio.
                If int, keeps exactly this many components.
            tmin: Start time for epoch crop (seconds)
            tmax: End time for epoch crop (seconds)
            frequencies: Frequency bands as ((low1, high1), (low2, high2), ...).
                If None, uses default alpha/mu bands: ((7, 11), (9, 13))
            decimation_factor: Decimation factor for downsampling
            method: Inverse solution method ('sLORETA', 'dSPM', 'MNE', etc.)
            snr: Signal-to-noise ratio for inverse operator
            spacing: Source space spacing ('oct5' or 'oct6')
        """
        self.n_components = n_components
        self.tmin = tmin
        self.tmax = tmax
        self.frequencies = frequencies
        self.decimation_factor = decimation_factor
        self.method = method
        self.snr = snr
        self.spacing = spacing

        # Fitted attributes (set during fit())
        self.inverse_operator_: Union[mne.minimum_norm.InverseOperator, None] = None
        self.pipeline_: Union[Pipeline, None] = None

    def fit(
        self,
        epochs: mne.Epochs,
        labels: np.ndarray,
    ) -> "SourceSpaceClassifier":
        """Fit the classifier to training data.

        This method:
        1. Creates inverse operator from epochs.info
        2. Extracts source space features
        3. Trains StandardScaler + PCA + LDA pipeline

        Args:
            epochs: Training epochs (MNE Epochs object)
            labels: Training labels (1D array of shape (n_epochs,))

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If training data has no variance
        """
        # Create inverse operator
        self.inverse_operator_ = create_inverse_operator(
            epochs.info,
            spacing=self.spacing,
        )

        # Extract features
        X = extract_features(
            epochs,
            self.inverse_operator_,
            frequencies=self.frequencies,
            tmin=self.tmin,
            tmax=self.tmax,
            decimation_factor=self.decimation_factor,
            method=self.method,
            snr=self.snr,
        )

        # Reshape to 2D if needed
        X = X.reshape(len(epochs), -1)

        # Guard against degenerate datasets
        if np.allclose(X, X[0]):
            raise ValueError(
                "Training data has no variance; feature extraction might have failed."
            )

        # Create and train pipeline
        self.pipeline_ = make_pipeline(
            StandardScaler(),
            PCA(n_components=self.n_components),
            LinearDiscriminantAnalysis(),
        )

        self.pipeline_.fit(X, labels)

        return self

    def predict(self, epochs: mne.Epochs) -> np.ndarray:
        """Predict class labels for epochs.

        Args:
            epochs: Test epochs (MNE Epochs object)

        Returns:
            Predicted class labels (1D array of shape (n_epochs,))

        Raises:
            ValueError: If classifier has not been fitted yet
        """
        if self.inverse_operator_ is None or self.pipeline_ is None:
            raise ValueError(
                "Classifier must be fitted before prediction. Call fit() first."
            )

        # Extract features
        X = extract_features(
            epochs,
            self.inverse_operator_,
            frequencies=self.frequencies,
            tmin=self.tmin,
            tmax=self.tmax,
            decimation_factor=self.decimation_factor,
            method=self.method,
            snr=self.snr,
        )

        # Reshape to 2D if needed
        X = X.reshape(len(epochs), -1)

        return self.pipeline_.predict(X)

    def predict_proba(self, epochs: mne.Epochs) -> np.ndarray:
        """Predict class probabilities for epochs.

        Args:
            epochs: Test epochs (MNE Epochs object)

        Returns:
            Class probabilities (2D array of shape (n_epochs, n_classes))

        Raises:
            ValueError: If classifier has not been fitted yet
        """
        if self.inverse_operator_ is None or self.pipeline_ is None:
            raise ValueError(
                "Classifier must be fitted before prediction. Call fit() first."
            )

        # Extract features
        X = extract_features(
            epochs,
            self.inverse_operator_,
            frequencies=self.frequencies,
            tmin=self.tmin,
            tmax=self.tmax,
            decimation_factor=self.decimation_factor,
            method=self.method,
            snr=self.snr,
        )

        # Reshape to 2D if needed
        X = X.reshape(len(epochs), -1)

        return self.pipeline_.predict_proba(X)

    def score(self, epochs: mne.Epochs, labels: np.ndarray) -> float:
        """Compute accuracy score on test data.

        Args:
            epochs: Test epochs (MNE Epochs object)
            labels: True labels (1D array of shape (n_epochs,))

        Returns:
            Accuracy score (float in range [0, 1])

        Raises:
            ValueError: If classifier has not been fitted yet
        """
        if self.inverse_operator_ is None or self.pipeline_ is None:
            raise ValueError(
                "Classifier must be fitted before scoring. Call fit() first."
            )

        # Extract features
        X = extract_features(
            epochs,
            self.inverse_operator_,
            frequencies=self.frequencies,
            tmin=self.tmin,
            tmax=self.tmax,
            decimation_factor=self.decimation_factor,
            method=self.method,
            snr=self.snr,
        )

        # Reshape to 2D if needed
        X = X.reshape(len(epochs), -1)

        return self.pipeline_.score(X, labels)

    def get_pca_components(self) -> np.ndarray:
        """Get the PCA components from the fitted pipeline.

        Returns:
            PCA components (2D array of shape (n_components, n_features))

        Raises:
            ValueError: If classifier has not been fitted yet
        """
        if self.pipeline_ is None:
            raise ValueError(
                "Classifier must be fitted before accessing PCA components. Call fit() first."
            )

        pca = self.pipeline_.named_steps["pca"]
        return pca.components_

    def get_lda_coef(self) -> np.ndarray:
        """Get the LDA coefficients from the fitted pipeline.

        Returns:
            LDA coefficients (2D array of shape (n_classes-1, n_pca_components))

        Raises:
            ValueError: If classifier has not been fitted yet
        """
        if self.pipeline_ is None:
            raise ValueError(
                "Classifier must be fitted before accessing LDA coefficients. Call fit() first."
            )

        lda = self.pipeline_.named_steps["lineardiscriminantanalysis"]
        return lda.coef_

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio of PCA components.

        Returns:
            Explained variance ratio for each PCA component (1D array)

        Raises:
            ValueError: If classifier has not been fitted yet
        """
        if self.pipeline_ is None:
            raise ValueError(
                "Classifier must be fitted before accessing explained variance. Call fit() first."
            )

        pca = self.pipeline_.named_steps["pca"]
        return pca.explained_variance_ratio_
