"""Domain Adversarial Neural Networks (DANN) for EEG classification.

Implements DANN architecture for unsupervised domain adaptation in EEG motor imagery.
Learns domain-invariant features by adversarial training between label predictor
and domain discriminator.

This adjusted version incorporates architectural improvements for better performance
and flexibility, including robust feature pooling and a more powerful label predictor.

Reference:
    Ganin, Y., et al. (2016). Domain-adversarial training of neural networks.
    The Journal of Machine Learning Research, 17(1), 2096-2030.
"""

from typing import Tuple

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer (GRL) from DANN paper.
    
    Forward pass: identity function
    Backward pass: negates gradients (multiplies by -lambda)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        lambda_ = ctx.lambda_
        return grad_output.neg() * lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer module wrapper."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_


class MLPClassifier(nn.Module):
    """General-purpose MLP classifier for DANN.
    
    Used for both the label predictor and the domain discriminator.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """Initialize the MLP classifier.
        
        Args:
            input_size: Size of input features.
            output_size: Number of output classes.
            hidden_size: Size of hidden layers.
            num_layers: Number of hidden layers (must be >= 1).
            dropout: Dropout rate.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
            
        layers = []
        
        # First layer
        layers.extend([
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        
        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DANN(nn.Module):
    """DANN with a flexible feature extractor and Flattening (matches EEGNet).

    Architecture:
        Input → Feature Extractor → Feature Map → Flatten → Flat Features
                                                            ↓
                                        ┌───────────────────┴───────────────────┐
                                        ↓                                       ↓
                              Label Predictor                  [GRL] → Domain Discriminator
                                        ↓                                       ↓
                                Task Classification                   Domain Classification
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_dim: int,
        n_classes: int,
        samples: int,
        n_domains: int = 2,
        predictor_hidden: int = 256,
        predictor_layers: int = 2,
        discriminator_hidden: int = 256,
        discriminator_layers: int = 2,
        dropout: float = 0.5,
    ):
        """Initialize the DANN model.

        Args:
            feature_extractor: A neural network module that extracts features.
                               (e.g., an initialized EEGNet instance).
            feature_dim: The number of output channels from the feature extractor
                         (e.g., F2 in EEGNet).
            n_classes: Number of task-specific classes (e.g., 2 for MI).
            samples: Total timepoints in input (e.g., 750).
            n_domains: Number of domains for domain discriminator (default: 2 for binary)
            predictor_hidden: Hidden layer size for the label predictor.
            predictor_layers: Number of hidden layers in the label predictor.
            discriminator_hidden: Hidden layer size for the domain discriminator.
            discriminator_layers: Number of hidden layers in the discriminator.
            dropout: Dropout rate for both classifiers.
        """
        super().__init__()

        self.feature_extractor = feature_extractor

        # Calculate flattened dimension (same logic as EEGNet)
        # EEGNet reduces time dimension by factor of 128 (8*16)
        self.output_timepoints = samples // 128
        self.flattened_dim = feature_dim * self.output_timepoints

        # Replace pooling with flatten
        self.pooling = nn.Flatten()

        # Strengthened label predictor (now uses flattened_dim)
        self.label_predictor = MLPClassifier(
            input_size=self.flattened_dim,
            output_size=n_classes,
            hidden_size=predictor_hidden,
            num_layers=predictor_layers,
            dropout=dropout,
        )

        # Gradient reversal layer
        self.grl = GradientReversalLayer()

        # Domain discriminator (now uses flattened_dim)
        self.domain_discriminator_mlp = MLPClassifier(
            input_size=self.flattened_dim,
            output_size=n_domains,  # Multi-domain support
            hidden_size=discriminator_hidden,
            num_layers=discriminator_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the DANN.

        Args:
            x: Input data (e.g., EEG: batch, 1, channels, timepoints).
            alpha: Lambda for gradient reversal, controlled by a scheduler.

        Returns:
            class_output: Task class predictions (batch, n_classes).
            domain_output: Domain predictions (batch, n_domains).
        """
        # 1. Extract feature maps
        features = self.feature_extractor(x)  # Shape: (Batch, F2, 1, Time/128)

        # 2. Flatten (preserves temporal structure)
        # Result: (Batch, F2 * Time/128)
        flat_features = self.pooling(features)

        # 3. Predict task label
        class_output = self.label_predictor(flat_features)

        # 4. Reverse gradients and predict domain
        self.grl.set_lambda(alpha)
        reversed_features = self.grl(flat_features)
        domain_output = self.domain_discriminator_mlp(reversed_features)

        return class_output, domain_output

    def domain_discriminator(self, flat_features: torch.Tensor, lambda_p: float) -> torch.Tensor:
        """Apply gradient reversal and domain discrimination.

        Args:
            flat_features: Flattened features from feature extractor
            lambda_p: Lambda value for gradient reversal

        Returns:
            Domain predictions
        """
        self.grl.set_lambda(lambda_p)
        reversed_features = self.grl(flat_features)
        return self.domain_discriminator_mlp(reversed_features)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict task classes for inference.

        Args:
            x: Input data.

        Returns:
            Predicted class indices.
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
            flat_features = self.pooling(features)
            class_output = self.label_predictor(flat_features)
            return torch.argmax(class_output, dim=1)