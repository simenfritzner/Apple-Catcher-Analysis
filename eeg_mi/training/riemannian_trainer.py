"""Trainer for Riemannian geometry-based classifiers with domain adaptation.

This trainer uses:
1. Covariance matrices as features (one per trial)
2. Riemannian mean for each class (geometric mean on SPD manifold)
3. Riemannian Procrustes Analysis (via recentering) for alignment [cite: 1036]
4. MDM classifier for prediction
"""

from typing import Any, Dict, Optional, List

import numpy as np
# We need pyriemann for correct Riemannian operations (mean and transform)
# pip install pyriemann
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm, powm

from eeg_mi.models.riemannian import MDMClassifier
# MDMWithRecentering is removed as its logic was flawed and is now
# handled directly within this trainer.
from eeg_mi.training.base_trainer import BaseTrainer


def compute_covariance_matrices(data: np.ndarray) -> np.ndarray:
    """Compute covariance matrix for each trial.

    Args:
        data: EEG data of shape (n_trials, n_channels, n_timepoints)

    Returns:
        Covariance matrices of shape (n_trials, n_channels, n_channels)
    """
    n_trials, n_channels, n_timepoints = data.shape
    covmats = np.zeros((n_trials, n_channels, n_channels))

    for i in range(n_trials):
        trial_data = data[i]
        # Use shrinkage estimator for robustness, as in paper [cite: 838]
        # This requires sklearn: pip install scikit-learn
        # For simplicity, np.cov is kept, but Ledoit-Wolf is better.
        covmats[i] = np.cov(trial_data)
        
        # Add regularization (small identity matrix) to ensure SPD
        covmats[i] += 1e-6 * np.eye(n_channels)

    return covmats


def transform_covs(covmats: np.ndarray, T_align: np.ndarray) -> np.ndarray:
    """Apply Riemannian alignment transform T^(-1/2) * C * T^(-1/2).

    This corresponds to Eq. 9 in the paper[cite: 886].

    Args:
        covmats: Covariance matrices to transform (n_trials, n_channels, n_channels)
        T_align: Alignment matrix (global mean) (n_channels, n_channels)

    Returns:
        Transformed covariance matrices (n_trials, n_channels, n_channels)
    """
    T_inv_sqrt = invsqrtm(T_align)
    n_trials = covmats.shape[0]
    aligned_covmats = np.zeros_like(covmats)
    
    for i in range(n_trials):
        aligned_covmats[i] = T_inv_sqrt @ covmats[i] @ T_inv_sqrt
        
    return aligned_covmats


class RiemannianTrainer(BaseTrainer):
    """Trainer for Riemannian classifiers with unsupervised domain adaptation.

    Implements Riemannian Procrustes Analysis (Generic Recentering)[cite: 17, 1036].

    The approach [cite: 872-890]:
    1. Compute covariance matrices from source trials
    2. Compute global source mean (T_source)
    3. Transform all source covmats using T_source
    4. Train MDM on *transformed* source covmats
    5. If adapting, compute global target mean (T_target) from calibration data
    6. Classify target test data by transforming it with T_target (or T_source
       if not adapting) and comparing to the MDM's (aligned) prototypes.
    """

    def __init__(
        self,
        n_classes: int = 2,
        use_recentering: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.use_recentering = use_recentering
        
        # Use Affine Invariant metric (riemann), as specified in paper [cite: 850]
        # Note: mean_riemann() uses affine invariant metric by default 

    def train(
        self,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        target_data: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train Riemannian classifier with unsupervised domain adaptation.

        Args:
            source_data: Source domain data (n_source, n_channels, n_timepoints)
            source_labels: Source domain labels (n_source,)
            target_data: Target domain calibration data (n_target, n_channels, n_timepoints)
            target_labels: Not used (unsupervised adaptation)
            val_data: Optional validation data
            val_labels: Optional validation labels

        Returns:
            Model dictionary containing the MDM classifier and alignment matrices
        """
        print("Computing covariance matrices...", flush=True)
        source_covmats = compute_covariance_matrices(source_data)

        print(f"Source covmats: {source_covmats.shape}", flush=True)

        # 1. Compute global source mean (T_source)
        print("Computing global source mean (T_source)...", flush=True)
        T_source = mean_riemann(source_covmats)

        # 2. Transform source covmats to aligned space 
        print("Aligning source data to T_source...", flush=True)
        source_covmats_aligned = transform_covs(source_covmats, T_source)

        # 3. Organize aligned source covariance matrices by class
        covmats_by_class_aligned = []
        for class_idx in range(self.n_classes):
            class_mask = source_labels == class_idx
            class_covmats = list(source_covmats_aligned[class_mask])
            covmats_by_class_aligned.append(class_covmats)
            print(f"Class {class_idx}: {len(class_covmats)} aligned samples", flush=True)

        # 4. Train MDM on *aligned* source domain
        print("Training MDM on aligned source domain...", flush=True)
        mdm = MDMClassifier(n_classes=self.n_classes)
        # The MDM's prototypes (class means) now exist in the T_source space
        mdm.fit(covmats_by_class_aligned) 

        # 5. Handle target domain alignment
        T_target = None
        if self.use_recentering and target_data is not None:
            print("Computing global target mean (T_target) from calibration data...", flush=True)
            target_covmats = compute_covariance_matrices(target_data)
            print(f"Target covmats: {target_covmats.shape}", flush=True)
            
            # Compute global target mean (T_target)
            T_target = mean_riemann(target_covmats)
            print(f"Recentering complete: {len(target_covmats)} calibration samples", flush=True)
            
        elif self.use_recentering:
            print("Warning: 'use_recentering' is True but no 'target_data' was provided.", flush=True)
            print("Falling back to source-only model.", flush=True)
        else:
            print("No recentering (source-only model)", flush=True)

        # Store model components
        model = {
            'mdm': mdm,
            'T_source': T_source,
            'T_target': T_target,  # Will be None if not adapting
            'use_recentering': self.use_recentering
        }

        # Evaluate on validation set if provided
        if val_data is not None and val_labels is not None:
            print("Evaluating on validation set...", flush=True)
            val_metrics = self.evaluate(model, val_data, val_labels)
            print(f"Validation accuracy: {val_metrics['accuracy']:.2%}", flush=True)

        return model

    def evaluate(
        self,
        model: Dict[str, Any],
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate trained Riemannian classifier.

        Args:
            model: Model dictionary from train()
            test_data: Test data (n_test, n_channels, n_timepoints)
            test_labels: Test labels (n_test,)

        Returns:
            Dictionary with accuracy and sample count
        """
        predictions = self.predict(model, test_data)
        accuracy = np.mean(predictions == test_labels)

        return {
            "accuracy": float(accuracy),
            "n_samples": len(test_labels),
        }

    def predict(
        self,
        model: Dict[str, Any],
        data: np.ndarray,
    ) -> np.ndarray:
        """Make predictions with trained Riemannian classifier.

        Args:
            model: Model dictionary from train()
            data: Data to predict (n_samples, n_channels, n_timepoints)

        Returns:
            Predictions (n_samples,)
        """
        mdm = model['mdm']
        T_source = model['T_source']
        T_target = model['T_target']
        
        # Compute covariance matrices
        covmats = compute_covariance_matrices(data)

        # 1. Select alignment matrix
        # If we adapted, use T_target. If not, use T_source.
        T_align = T_source
        if T_target is not None:
            T_align = T_target
            # print("Using T_target for alignment") # DEBUG
        # else:
            # print("Using T_source for alignment") # DEBUG

        # 2. Transform test data to the aligned space 
        covmats_aligned = transform_covs(covmats, T_align)

        # 3. Predict using the MDM (which operates in the aligned space)
        predictions = []
        for covmat in covmats_aligned:
            pred = mdm.predict(covmat) # mdm.predict expects a single covmat
            predictions.append(pred)

        return np.array(predictions)