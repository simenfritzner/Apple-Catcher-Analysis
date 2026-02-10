"""EEG classification models.

This package provides various classifiers for EEG motor imagery tasks.
"""

from eeg_mi.models.eegnet import EEGNet
from eeg_mi.models.source_space import SourceSpaceClassifier
from eeg_mi.models.riemannian import (
    MDMClassifier,
    MDMWithRecentering,
    EvidenceAccumulator,
)
from eeg_mi.models.dann import (
    DANN,
    GradientReversalLayer,
    MLPClassifier,
)

__all__ = [
    "EEGNet",
    "SourceSpaceClassifier",
    "MDMClassifier",
    "MDMWithRecentering",
    "EvidenceAccumulator",
    "DANN",
    "GradientReversalLayer",
    "MLPClassifier",
]
