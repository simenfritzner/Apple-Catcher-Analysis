"""
EEGNet implementation for EEG classification and feature extraction.

This module provides a PyTorch implementation of EEGNet adapted for 250Hz sampling rate.
Original paper: http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
"""

from typing import Literal, Tuple

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet model for EEG signal classification and feature extraction.

    This implementation assumes input is in NCHW format: (batch, 1, channels, samples).
    The model has been adjusted for 250Hz sampling rate with modified kernel lengths
    and pooling factors.

    Supports two modes:
    - 'classifier': Full model for standard training/testing
    - 'feature_extractor': Returns features for transfer learning with Riemannian geometry

    Note:
        The original Keras implementation includes a max_norm constraint on convolutional
        layers. This must be applied manually in the training loop after each optimizer
        step in PyTorch.

    Args:
        nb_classes: Number of output classes.
        Chans: Number of EEG channels. Default: 64.
        Samples: Number of time samples per trial. Default: 256.
        dropoutRate: Dropout probability. Default: 0.5.
        kernLength: Length of temporal convolution kernel. Default: 125.
        F1: Number of temporal filters. Default: 8.
        D: Depth multiplier (number of spatial filters per temporal filter). Default: 2.
        F2: Number of pointwise filters. Default: 16.
        norm_rate: Max norm constraint rate (not enforced automatically). Default: 0.25.
        dropoutType: Type of dropout ('Dropout' or 'SpatialDropout2D'). Default: 'Dropout'.
        mode: Operating mode ('classifier' or 'feature_extractor'). Default: 'classifier'.

    Example:
        >>> model = EEGNet(nb_classes=4, Chans=22, Samples=1000)
        >>> x = torch.randn(32, 1, 22, 1000)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 4])
    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int = 64,
        Samples: int = 256,
        dropoutRate: float = 0.5,
        kernLength: int = 125,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        norm_rate: float = 0.25,
        dropoutType: Literal['Dropout', 'SpatialDropout2D'] = 'Dropout',
        mode: Literal['classifier', 'feature_extractor'] = 'classifier'
    ) -> None:
        super(EEGNet, self).__init__()

        self.mode = mode
        self.F2 = F2
        self.Samples = Samples
        self.norm_rate = norm_rate

        # Configure dropout type
        if dropoutType == 'SpatialDropout2D':
            self.dropout = nn.Dropout2d(p=dropoutRate)
        elif dropoutType == 'Dropout':
            self.dropout = nn.Dropout(p=dropoutRate)
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout')

        # Block 1: Temporal convolution and spatial filtering
        self.block1 = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial convolution
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 8))
        )

        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            # Depthwise separable convolution
            nn.Conv2d(F1 * D, F2, (1, 31), padding=(0, 31 // 2), groups=F1 * D, bias=False),
            # Pointwise convolution
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 16))
        )

        # Classifier head (only used in 'classifier' mode)
        if mode == 'classifier':
            # Total pooling factor: 8 * 16 = 128
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(F2 * (Samples // 128), nb_classes)
            )
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 1, channels, samples).

        Returns:
            If mode='classifier': Class logits of shape (batch, nb_classes).
            If mode='feature_extractor': Features of shape (batch, F2, 1, samples // 128).

        Raises:
            ValueError: If mode is not 'classifier' or 'feature_extractor'.
        """
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)

        if self.mode == 'classifier':
            x = self.classifier(x)
            return x
        elif self.mode == 'feature_extractor':
            # Return features for covariance matrix computation
            return x
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def set_mode(self, mode: Literal['classifier', 'feature_extractor']) -> None:
        """
        Switch between 'classifier' and 'feature_extractor' modes.

        Args:
            mode: Target mode ('classifier' or 'feature_extractor').

        Raises:
            ValueError: If mode is invalid.
        """
        if mode not in ['classifier', 'feature_extractor']:
            raise ValueError("Mode must be 'classifier' or 'feature_extractor'")
        self.mode = mode

    def get_feature_dim(self) -> int:
        """
        Get the number of output channels from the feature extractor block.
        This corresponds to F2, the number of pointwise filters.
        
        Returns:
            The number of feature channels (F2).
        """
        return self.F2

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features regardless of current mode.

        This is a convenience method that temporarily switches to feature_extractor
        mode, extracts features with no gradient computation, and restores the
        original mode.

        Args:
            x: Input tensor of shape (batch, 1, channels, samples).

        Returns:
            Features of shape (batch, F2, 1, samples // 128).
        """
        original_mode = self.mode
        self.set_mode('feature_extractor')
        with torch.no_grad():
            features = self.forward(x)
        self.set_mode(original_mode)
        return features
