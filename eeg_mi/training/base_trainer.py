"""Abstract base trainer for EEG classification models.

Defines the interface that all trainers must implement.
This avoids a monolithic trainer with if/else logic for different model types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseTrainer(ABC):
    """Abstract base class for model trainers.
    
    Each model type (deep learning, sklearn, etc.) should have its own
    trainer subclass that implements the specific training logic.
    """

    def __init__(self, **kwargs):
        """Initialize trainer with configuration parameters.
        
        Args:
            **kwargs: Trainer-specific configuration parameters
        """
        self.config = kwargs
        self.history: Dict[str, list] = {}

    @abstractmethod
    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: np.ndarray = None,
        val_labels: np.ndarray = None,
    ) -> Any:
        """Train the model.

        Args:
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data (optional)
            val_labels: Validation labels (optional)

        Returns:
            Trained model
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            model: Trained model
            test_data: Test data
            test_labels: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def predict(
        self,
        model: Any,
        data: np.ndarray,
    ) -> np.ndarray:
        """Make predictions with the model.

        Args:
            model: Trained model
            data: Input data

        Returns:
            Predicted labels
        """
        pass

    def get_history(self) -> Dict[str, list]:
        """Get training history.

        Returns:
            Dictionary of metrics over training
        """
        return self.history
