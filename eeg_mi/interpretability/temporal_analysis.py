"""
High-level temporal importance analysis combining multiple methods.

Provides unified interface for analyzing temporal patterns in EEG classification.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn

from .saliency import SaliencyMapGenerator
from .ablation import TemporalAblationStudy


class TemporalImportanceAnalyzer:
    """
    Unified analyzer for temporal importance in EEG classification.

    Combines gradient-based methods (saliency) and perturbation-based
    methods (ablation) to provide comprehensive view of temporal importance.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        batch_size: int = 32
    ):
        """
        Initialize temporal importance analyzer.

        Args:
            model: Trained PyTorch model
            device: Device to run on
            batch_size: Batch size for evaluation
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size

        # Initialize sub-analyzers
        self.saliency = SaliencyMapGenerator(model, device)
        self.ablation = TemporalAblationStudy(model, device, batch_size)

    def analyze_pre_vs_post_cue(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cue_onset: float = 0.0,
        sfreq: float = 250.0,
        tmin: float = -1.0,
        method: str = 'integrated_gradients'
    ) -> Dict[str, float]:
        """
        Compare model reliance on pre-cue vs post-cue information.

        This is the KEY analysis for your problem - understanding if
        the model uses pre-cue visual information vs post-cue motor imagery.

        Args:
            x: Input data (N, 1, channels, samples)
            y: True labels (N,)
            cue_onset: Time of cue onset in seconds
            sfreq: Sampling frequency
            tmin: Start time of epoch
            method: Saliency method to use

        Returns:
            Dictionary with:
                - pre_cue_importance: Average importance before cue
                - post_cue_importance: Average importance after cue
                - ratio: Pre-cue / post-cue importance
                - pre_cue_accuracy_drop: Performance drop when pre-cue masked
                - post_cue_accuracy_drop: Performance drop when post-cue masked
        """
        # Convert cue onset to sample index
        cue_idx = int((cue_onset - tmin) * sfreq)

        # 1. Gradient-based importance
        if method == 'vanilla':
            saliency_map = self.saliency.vanilla_gradient(x)
        elif method == 'integrated_gradients':
            saliency_map = self.saliency.integrated_gradients(x)
        elif method == 'gradient_x_input':
            saliency_map = self.saliency.gradient_x_input(x)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Average importance pre vs post cue
        pre_cue_importance = saliency_map[:, :, :cue_idx].mean()
        post_cue_importance = saliency_map[:, :, cue_idx:].mean()

        # 2. Ablation-based importance
        ablation_windows = [
            (tmin, cue_onset),  # Pre-cue
            (cue_onset, tmin + x.shape[-1] / sfreq)  # Post-cue
        ]

        ablation_results = self.ablation.fixed_windows_ablation(
            x, y, ablation_windows, sfreq, tmin, mask_type='zero'
        )

        pre_cue_acc_drop = ablation_results[0].accuracy_drop
        post_cue_acc_drop = ablation_results[1].accuracy_drop

        return {
            'pre_cue_importance': float(pre_cue_importance),
            'post_cue_importance': float(post_cue_importance),
            'importance_ratio': float(pre_cue_importance / (post_cue_importance + 1e-10)),
            'pre_cue_accuracy_drop': float(pre_cue_acc_drop),
            'post_cue_accuracy_drop': float(post_cue_acc_drop),
            'accuracy_drop_ratio': float(pre_cue_acc_drop / (post_cue_acc_drop + 1e-10)),
            'baseline_accuracy': float(ablation_results[0].accuracy + ablation_results[0].accuracy_drop)
        }

    def analyze_temporal_profile(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sfreq: float = 250.0,
        tmin: float = -1.0,
        window_size: float = 0.5,
        step_size: float = 0.1,
        method: str = 'integrated_gradients'
    ) -> Dict[str, np.ndarray]:
        """
        Get detailed temporal importance profile.

        Args:
            x: Input data
            y: True labels
            sfreq: Sampling frequency
            tmin: Start time
            window_size: Sliding window size in seconds
            step_size: Step size in seconds
            method: Saliency method

        Returns:
            Dictionary with temporal profiles from both methods
        """
        # Gradient-based temporal profile
        if method == 'vanilla':
            saliency_map = self.saliency.vanilla_gradient(x)
        elif method == 'integrated_gradients':
            saliency_map = self.saliency.integrated_gradients(x)
        elif method == 'gradient_x_input':
            saliency_map = self.saliency.gradient_x_input(x)
        else:
            raise ValueError(f"Unknown method: {method}")

        times, temporal_importance = self.saliency.compute_temporal_importance(
            saliency_map, sfreq, tmin
        )

        # Ablation-based temporal profile
        ablation_results = self.ablation.sliding_window_ablation(
            x, y, window_size, step_size, sfreq, tmin, mask_type='zero'
        )

        return {
            'times': times,
            'gradient_importance': temporal_importance,
            'ablation_window_centers': ablation_results['window_centers'],
            'ablation_accuracy_drops': ablation_results['accuracy_drops'],
            'baseline_accuracy': ablation_results['baseline_accuracy']
        }

    def find_critical_time_window(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sfreq: float = 250.0,
        tmin: float = -1.0,
        threshold: float = 0.05
    ) -> Tuple[float, float]:
        """
        Identify the critical time window using progressive ablation.

        Finds the minimal time window needed for good performance.

        Args:
            x: Input data
            y: True labels
            sfreq: Sampling frequency
            tmin: Start time
            threshold: Accuracy drop threshold

        Returns:
            (start_time, end_time) of critical window
        """
        # Progressive ablation from both directions
        forward_results = self.ablation.progressive_ablation(
            x, y, sfreq, tmin, direction='forward', step_size=0.1
        )

        backward_results = self.ablation.progressive_ablation(
            x, y, sfreq, tmin, direction='backward', step_size=0.1
        )

        # Find baseline accuracy
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_y.numpy())

        baseline_acc = (np.concatenate(all_preds) == np.concatenate(all_labels)).mean()

        # Find critical points
        forward_critical_idx = np.where(
            forward_results['accuracies'] < baseline_acc - threshold
        )[0]

        backward_critical_idx = np.where(
            backward_results['accuracies'] < baseline_acc - threshold
        )[0]

        if len(forward_critical_idx) > 0:
            critical_start = forward_results['cutoff_times'][forward_critical_idx[0]]
        else:
            critical_start = tmin

        if len(backward_critical_idx) > 0:
            critical_end = backward_results['cutoff_times'][backward_critical_idx[-1]]
        else:
            critical_end = tmin + x.shape[-1] / sfreq

        return critical_start, critical_end

    def analyze_channel_importance(
        self,
        x: torch.Tensor,
        channel_names: Optional[List[str]] = None,
        method: str = 'integrated_gradients',
        top_k: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Identify most important EEG channels.

        Args:
            x: Input data
            channel_names: List of channel names
            method: Saliency method
            top_k: Number of top channels to identify

        Returns:
            Dictionary with channel importance rankings
        """
        # Compute saliency map
        if method == 'vanilla':
            saliency_map = self.saliency.vanilla_gradient(x)
        elif method == 'integrated_gradients':
            saliency_map = self.saliency.integrated_gradients(x)
        elif method == 'gradient_x_input':
            saliency_map = self.saliency.gradient_x_input(x)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Get channel importance
        channel_importance = self.saliency.compute_channel_importance(saliency_map)

        # Get top channels
        top_indices = np.argsort(channel_importance)[::-1][:top_k]

        results = {
            'channel_importance': channel_importance,
            'top_indices': top_indices,
            'top_importance': channel_importance[top_indices]
        }

        if channel_names is not None:
            results['top_channels'] = [channel_names[i] for i in top_indices]

        return results
