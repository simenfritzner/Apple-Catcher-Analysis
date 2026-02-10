"""Trainer for scikit-learn based models (PCA+LDA, Riemannian, etc.)."""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from eeg_mi.training.base_trainer import BaseTrainer


class SklearnTrainer(BaseTrainer):
    """Trainer for scikit-learn compatible models.
    
    Works with models that implement fit/predict interface:
    - SourceSpaceClassifier (PCA+LDA)
    - MDMClassifier (Riemannian)
    - Any sklearn classifier
    """

    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: np.ndarray = None,
        val_labels: np.ndarray = None,
    ) -> Any:
        """Train sklearn model.

        Args:
            train_data: Training data (can be epochs or features)
            train_labels: Training labels
            val_data: Validation data (not used for sklearn, but kept for API consistency)
            val_labels: Validation labels (not used)

        Returns:
            Trained model instance
        """
        from eeg_mi.models import SourceSpaceClassifier

        # Create model based on config
        model_type = self.config.get("model_type", "pca_lda")
        
        if model_type == "pca_lda":
            model = SourceSpaceClassifier(
                n_components=self.config.get("pca_components", 0.95),
                tmin=self.config.get("tmin", -0.5),
                tmax=self.config.get("tmax", 1.0),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train
        model.fit(train_data, train_labels)

        # Store training accuracy
        train_pred = model.predict(train_data)
        train_acc = accuracy_score(train_labels, train_pred)
        self.history["train_accuracy"] = [train_acc]

        return model

    def evaluate(
        self,
        model: Any,
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate sklearn model.

        Args:
            model: Trained sklearn model
            test_data: Test data
            test_labels: Test labels

        Returns:
            Dictionary with accuracy and other metrics
        """
        predictions = model.predict(test_data)
        
        metrics = {
            "accuracy": accuracy_score(test_labels, predictions),
            "n_samples": len(test_labels),
        }

        # Add confusion matrix if binary classification
        if len(np.unique(test_labels)) == 2:
            cm = confusion_matrix(test_labels, predictions)
            metrics["confusion_matrix"] = cm.tolist()
            
            # Compute sensitivity and specificity
            tn, fp, fn, tp = cm.ravel()
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics

    def predict(
        self,
        model: Any,
        data: np.ndarray,
    ) -> np.ndarray:
        """Make predictions.

        Args:
            model: Trained sklearn model
            data: Input data

        Returns:
            Predicted labels
        """
        return model.predict(data)

    def predict_proba(
        self,
        model: Any,
        data: np.ndarray,
    ) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            model: Trained sklearn model
            data: Input data

        Returns:
            Class probabilities
        """
        if hasattr(model, "predict_proba"):
            return model.predict_proba(data)
        else:
            raise AttributeError("Model does not support predict_proba")
