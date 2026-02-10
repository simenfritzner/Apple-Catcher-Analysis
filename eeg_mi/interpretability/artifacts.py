"""
Artifact investigation analysis for EEG classification models.

Provides tools to assess whether a trained model relies on genuine
neural signals (e.g., motor imagery from central channels) or on
artifacts (e.g., blinks/eye movements from frontal channels).

Implements SPEC.md Phase 6: items 6.1, 6.3, 6.5.
"""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass

from .saliency import SaliencyMapGenerator
from .ablation import TemporalAblationStudy


# Standard 10-20 channel region prefixes
_REGION_PREFIXES: Dict[str, List[str]] = {
    'frontal': ['Fp', 'AF', 'F'],
    'central': ['C', 'FC'],
    'parietal': ['P', 'CP'],
    'occipital': ['O', 'PO'],
    'temporal': ['T', 'TP'],
}

# Prefixes that should NOT be matched by shorter prefixes
# e.g., 'FC3' should match 'central', not 'frontal'
_EXCLUSION_MAP: Dict[str, List[str]] = {
    'frontal': ['FC'],
    'central': ['CP'],
    'parietal': ['PO'],
    'occipital': [],
    'temporal': [],
}


@dataclass
class ChannelGroupResult:
    """Results from channel group saliency analysis."""
    group_importance: Dict[str, float]
    frontal_central_ratio: float
    per_channel_importance: np.ndarray
    channel_group_map: Dict[str, List[int]]


@dataclass
class ArtifactTrialResult:
    """Results from artifact trial detection."""
    artifact_mask: np.ndarray
    max_frontal_amplitude: np.ndarray
    n_artifact_trials: int
    n_clean_trials: int
    threshold_uv: float


def _classify_channel(name: str) -> Optional[str]:
    """
    Classify a channel name into a scalp region.

    Uses standard 10-20 naming conventions. Channels that do not
    match any region return None.
    """
    for region, prefixes in _REGION_PREFIXES.items():
        exclusions = _EXCLUSION_MAP[region]
        for prefix in prefixes:
            if name.startswith(prefix):
                # Check if a longer exclusion prefix matches instead
                excluded = False
                for exc in exclusions:
                    if name.startswith(exc):
                        excluded = True
                        break
                if not excluded:
                    return region
    return None


def _build_channel_groups(
    channel_names: List[str],
) -> Dict[str, List[int]]:
    """
    Map each channel to its scalp region and return group-to-index mapping.

    Returns:
        Dictionary mapping region name to list of channel indices.
    """
    groups: Dict[str, List[int]] = {
        'frontal': [],
        'central': [],
        'parietal': [],
        'occipital': [],
        'temporal': [],
    }

    for idx, name in enumerate(channel_names):
        region = _classify_channel(name)
        if region is not None:
            groups[region].append(idx)

    return groups


class ArtifactAnalyzer:
    """
    Analyze whether a trained EEG classifier relies on artifacts or genuine neural signals.

    Combines gradient-based saliency analysis with channel-group ablation
    and trial-level artifact detection to provide a comprehensive picture
    of potential artifact contamination in model decisions.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        batch_size: int = 32,
    ) -> None:
        """
        Initialize artifact analyzer.

        Args:
            model: Trained PyTorch model expecting input shape (batch, 1, channels, samples).
            device: Device to run computations on.
            batch_size: Batch size for model evaluation.
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.model.eval()

        self.saliency = SaliencyMapGenerator(model, device)
        self.ablation = TemporalAblationStudy(model, device, batch_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate model accuracy on the given data.

        Returns:
            accuracy: Classification accuracy as a float.
            predictions: Array of predicted labels.
        """
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())

        predictions = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        accuracy = float((predictions == labels).mean())

        return accuracy, predictions

    def _mask_channels(
        self,
        x: torch.Tensor,
        channel_indices: List[int],
        mask_type: Literal['zero', 'mean', 'noise'] = 'zero',
    ) -> torch.Tensor:
        """
        Mask specified channels in the input tensor.

        Args:
            x: Input tensor of shape (batch, 1, channels, samples).
            channel_indices: List of channel indices to mask.
            mask_type: Type of masking to apply.

        Returns:
            Masked copy of the input tensor.
        """
        x_masked = x.clone()

        if mask_type == 'zero':
            x_masked[:, :, channel_indices, :] = 0
        elif mask_type == 'mean':
            channel_means = x.mean(dim=3, keepdim=True)
            x_masked[:, :, channel_indices, :] = channel_means[:, :, channel_indices, :]
        elif mask_type == 'noise':
            std = x[:, :, channel_indices, :].std()
            noise = torch.randn_like(x[:, :, channel_indices, :]) * std
            x_masked[:, :, channel_indices, :] = noise
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        return x_masked

    # ------------------------------------------------------------------
    # 6.1  Frontal vs central channel saliency analysis
    # ------------------------------------------------------------------

    def analyze_channel_groups(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        channel_names: List[str],
        sfreq: float = 250.0,
        tmin: float = -1.0,
        method: str = 'integrated_gradients',
    ) -> ChannelGroupResult:
        """
        Compute per-region saliency importance and compare frontal vs central.

        Groups channels by scalp region and averages gradient-based
        importance within each group. A frontal/central ratio greater
        than 1 suggests the model may rely on eye-movement artifacts
        rather than genuine motor-imagery signals.

        Args:
            x: Input data of shape (N, 1, channels, samples).
            y: True labels of shape (N,).
            channel_names: List of channel names matching the channel dimension.
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.
            method: Saliency method ('integrated_gradients', 'vanilla', 'gradient_x_input').

        Returns:
            ChannelGroupResult with per-group importance and frontal/central ratio.
        """
        # Compute saliency map: (N, channels, samples)
        if method == 'vanilla':
            saliency_map = self.saliency.vanilla_gradient(x)
        elif method == 'integrated_gradients':
            saliency_map = self.saliency.integrated_gradients(x)
        elif method == 'gradient_x_input':
            saliency_map = self.saliency.gradient_x_input(x)
        else:
            raise ValueError(f"Unknown saliency method: {method}")

        # Per-channel importance averaged across trials and time
        per_channel = self.saliency.compute_channel_importance(saliency_map)

        # Build channel groups
        groups = _build_channel_groups(channel_names)

        # Compute per-group mean importance
        group_importance: Dict[str, float] = {}
        for region, indices in groups.items():
            if len(indices) > 0:
                group_importance[region] = float(per_channel[indices].mean())
            else:
                group_importance[region] = 0.0

        frontal_imp = group_importance.get('frontal', 0.0)
        central_imp = group_importance.get('central', 0.0)
        ratio = frontal_imp / (central_imp + 1e-10)

        return ChannelGroupResult(
            group_importance=group_importance,
            frontal_central_ratio=ratio,
            per_channel_importance=per_channel,
            channel_group_map=groups,
        )

    # ------------------------------------------------------------------
    # 6.5  Channel group ablation
    # ------------------------------------------------------------------

    def channel_group_ablation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        channel_names: List[str],
        mask_type: Literal['zero', 'mean', 'noise'] = 'zero',
    ) -> Dict[str, Dict[str, float]]:
        """
        Ablate entire channel groups and measure accuracy impact.

        For each scalp region, all channels in that group are masked
        and the model is re-evaluated. A large accuracy drop when
        ablating frontal channels suggests artifact reliance; a large
        drop for central channels suggests genuine motor-imagery usage.

        Args:
            x: Input data of shape (N, 1, channels, samples).
            y: True labels of shape (N,).
            channel_names: List of channel names.
            mask_type: How to mask channels ('zero', 'mean', 'noise').

        Returns:
            Dictionary mapping group name to a dict with keys:
                - 'accuracy': accuracy with the group ablated
                - 'accuracy_drop': baseline accuracy minus ablated accuracy
                - 'n_channels': number of channels in the group
            Also includes 'baseline' key with unablated accuracy.
        """
        baseline_acc, _ = self._evaluate(x, y)
        groups = _build_channel_groups(channel_names)

        results: Dict[str, Dict[str, float]] = {
            'baseline': {
                'accuracy': baseline_acc,
                'accuracy_drop': 0.0,
                'n_channels': 0.0,
            }
        }

        for region, indices in groups.items():
            if len(indices) == 0:
                results[region] = {
                    'accuracy': baseline_acc,
                    'accuracy_drop': 0.0,
                    'n_channels': 0.0,
                }
                continue

            x_ablated = self._mask_channels(x, indices, mask_type)
            acc, _ = self._evaluate(x_ablated, y)

            results[region] = {
                'accuracy': acc,
                'accuracy_drop': baseline_acc - acc,
                'n_channels': float(len(indices)),
            }

        return results

    # ------------------------------------------------------------------
    # 6.3  Blink / artifact epoch detection
    # ------------------------------------------------------------------

    def detect_artifact_trials(
        self,
        x: torch.Tensor,
        channel_names: List[str],
        threshold_uv: float = 100.0,
    ) -> ArtifactTrialResult:
        """
        Identify trials likely contaminated by blinks or eye-movement artifacts.

        Examines frontal channels (Fp*, AF*) for large amplitude excursions
        that typically indicate blink artifacts.

        Args:
            x: Input data of shape (N, 1, channels, samples).
            channel_names: List of channel names.
            threshold_uv: Amplitude threshold in microvolts. Trials where
                the max absolute amplitude on any frontal channel exceeds
                this value are flagged.

        Returns:
            ArtifactTrialResult with boolean mask and per-trial amplitudes.
        """
        # Identify frontal artifact channels (Fp and AF only)
        frontal_indices: List[int] = []
        for idx, name in enumerate(channel_names):
            if name.startswith('Fp') or name.startswith('AF'):
                frontal_indices.append(idx)

        # If no frontal channels found, return all-clean mask
        if len(frontal_indices) == 0:
            n_trials = x.shape[0]
            return ArtifactTrialResult(
                artifact_mask=np.zeros(n_trials, dtype=bool),
                max_frontal_amplitude=np.zeros(n_trials),
                n_artifact_trials=0,
                n_clean_trials=n_trials,
                threshold_uv=threshold_uv,
            )

        # Extract frontal channels: (N, 1, n_frontal, samples) -> (N, n_frontal, samples)
        x_np = x[:, 0, frontal_indices, :].cpu().numpy()

        # Max absolute amplitude per trial across frontal channels and time
        max_amp = np.abs(x_np).max(axis=(1, 2))

        artifact_mask = max_amp > threshold_uv
        n_artifact = int(artifact_mask.sum())

        return ArtifactTrialResult(
            artifact_mask=artifact_mask,
            max_frontal_amplitude=max_amp,
            n_artifact_trials=n_artifact,
            n_clean_trials=int(x.shape[0] - n_artifact),
            threshold_uv=threshold_uv,
        )

    # ------------------------------------------------------------------
    # 6.3 (cont.)  Clean vs artifact accuracy comparison
    # ------------------------------------------------------------------

    def compare_clean_vs_artifact_trials(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        channel_names: List[str],
        threshold_uv: float = 100.0,
    ) -> Dict[str, float]:
        """
        Compare model accuracy on clean vs artifact-contaminated trials.

        If accuracy is substantially higher on artifact trials than on
        clean trials, the model likely exploits artifact-correlated
        features rather than genuine neural signals.

        Args:
            x: Input data of shape (N, 1, channels, samples).
            y: True labels of shape (N,).
            channel_names: List of channel names.
            threshold_uv: Frontal amplitude threshold in microvolts.

        Returns:
            Dictionary with keys:
                - 'all_accuracy': accuracy on all trials
                - 'clean_accuracy': accuracy on clean trials only
                - 'artifact_accuracy': accuracy on artifact trials only
                - 'n_total': total number of trials
                - 'n_clean': number of clean trials
                - 'n_artifact': number of artifact trials
        """
        detection = self.detect_artifact_trials(x, channel_names, threshold_uv)

        # Baseline: all trials
        all_acc, _ = self._evaluate(x, y)

        clean_mask = ~detection.artifact_mask
        artifact_mask = detection.artifact_mask

        # Clean trials
        if clean_mask.sum() > 0:
            x_clean = x[clean_mask]
            y_clean = y[clean_mask]
            clean_acc, _ = self._evaluate(x_clean, y_clean)
        else:
            clean_acc = float('nan')

        # Artifact trials
        if artifact_mask.sum() > 0:
            x_artifact = x[artifact_mask]
            y_artifact = y[artifact_mask]
            artifact_acc, _ = self._evaluate(x_artifact, y_artifact)
        else:
            artifact_acc = float('nan')

        return {
            'all_accuracy': all_acc,
            'clean_accuracy': clean_acc,
            'artifact_accuracy': artifact_acc,
            'n_total': int(x.shape[0]),
            'n_clean': int(clean_mask.sum()),
            'n_artifact': int(artifact_mask.sum()),
        }

    # ------------------------------------------------------------------
    # 6.5  Single channel ablation
    # ------------------------------------------------------------------

    def single_channel_ablation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        channel_names: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Ablate one channel at a time and measure accuracy impact.

        Provides a fine-grained view of per-channel model dependency.
        Channels whose ablation causes large accuracy drops are critical
        to the model's decision.

        Args:
            x: Input data of shape (N, 1, channels, samples).
            y: True labels of shape (N,).
            channel_names: Optional list of channel names for labelling.

        Returns:
            Dictionary with keys:
                - 'baseline_accuracy': accuracy with no ablation
                - 'accuracy_drops': array of accuracy drops per channel (n_channels,)
                - 'accuracies': array of per-channel ablated accuracies
                - 'ranking': channel indices sorted by accuracy drop (descending)
                - 'channel_names': the channel names (if provided)
        """
        n_channels = x.shape[2]
        baseline_acc, _ = self._evaluate(x, y)

        accuracies = np.zeros(n_channels)
        drops = np.zeros(n_channels)

        for ch_idx in range(n_channels):
            x_ablated = self._mask_channels(x, [ch_idx], mask_type='zero')
            acc, _ = self._evaluate(x_ablated, y)
            accuracies[ch_idx] = acc
            drops[ch_idx] = baseline_acc - acc

        ranking = np.argsort(drops)[::-1]

        result: Dict[str, object] = {
            'baseline_accuracy': baseline_acc,
            'accuracy_drops': drops,
            'accuracies': accuracies,
            'ranking': ranking,
        }
        if channel_names is not None:
            result['channel_names'] = channel_names

        return result
