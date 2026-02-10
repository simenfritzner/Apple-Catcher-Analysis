"""
Temporal ablation study for understanding time window importance.

Systematically masks different time windows and measures performance
degradation to identify which temporal regions are critical for classification.
"""

from typing import Tuple, Dict, List, Literal
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass


@dataclass
class AblationResult:
    """Results from temporal ablation study."""
    window_start: float
    window_end: float
    accuracy: float
    accuracy_drop: float
    predictions: np.ndarray
    true_labels: np.ndarray


class TemporalAblationStudy:
    """
    Perform temporal ablation to identify critical time windows.

    Tests model performance when different temporal windows are masked
    (set to zero, mean, or noise) to understand which time periods
    are most important for classification.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        batch_size: int = 32
    ):
        """
        Initialize temporal ablation study.

        Args:
            model: Trained PyTorch model
            device: Device to run on
            batch_size: Batch size for evaluation
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.model.eval()

    def _mask_window(
        self,
        x: torch.Tensor,
        start_idx: int,
        end_idx: int,
        mask_type: Literal['zero', 'mean', 'noise'] = 'zero'
    ) -> torch.Tensor:
        """
        Mask a temporal window in the input.

        Args:
            x: Input tensor (batch, 1, channels, samples)
            start_idx: Start sample index
            end_idx: End sample index
            mask_type: Type of masking to apply

        Returns:
            Masked input tensor
        """
        x_masked = x.clone()

        if mask_type == 'zero':
            x_masked[:, :, :, start_idx:end_idx] = 0
        elif mask_type == 'mean':
            # Use mean across all samples for each channel
            channel_means = x.mean(dim=3, keepdim=True)
            x_masked[:, :, :, start_idx:end_idx] = channel_means
        elif mask_type == 'noise':
            # Replace with Gaussian noise matching original std
            std = x[:, :, :, start_idx:end_idx].std()
            noise = torch.randn_like(x[:, :, :, start_idx:end_idx]) * std
            x_masked[:, :, :, start_idx:end_idx] = noise
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        return x_masked

    def evaluate_window_ablation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        window_start: int,
        window_end: int,
        mask_type: Literal['zero', 'mean', 'noise'] = 'zero'
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate model with a specific window masked.

        Args:
            x: Input data (N, 1, channels, samples)
            y: True labels (N,)
            window_start: Start sample index
            window_end: End sample index
            mask_type: Type of masking

        Returns:
            accuracy: Classification accuracy
            predictions: Model predictions
        """
        # Mask the window
        x_masked = self._mask_window(x, window_start, window_end, mask_type)

        # Create dataloader
        dataset = TensorDataset(x_masked, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Evaluate
        all_preds = []
        all_labels = []

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

        accuracy = (predictions == labels).mean()

        return accuracy, predictions

    def sliding_window_ablation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        window_size: float = 0.5,
        step_size: float = 0.1,
        sfreq: float = 250.0,
        tmin: float = -1.0,
        mask_type: Literal['zero', 'mean', 'noise'] = 'zero'
    ) -> Dict[str, np.ndarray]:
        """
        Perform sliding window ablation study.

        Args:
            x: Input data (N, 1, channels, samples)
            y: True labels (N,)
            window_size: Size of ablation window in seconds
            step_size: Step size for sliding in seconds
            sfreq: Sampling frequency
            tmin: Start time of epoch
            mask_type: Type of masking

        Returns:
            Dictionary with results:
                - window_centers: Time points of window centers
                - accuracies: Accuracy for each window
                - accuracy_drops: Drop from baseline
                - baseline_accuracy: Original model accuracy
        """
        # Get baseline accuracy (no masking)
        baseline_acc, _ = self.evaluate_window_ablation(
            x, y, 0, 0, mask_type='zero'
        )

        # Convert to sample indices
        n_samples = x.shape[-1]
        window_samples = int(window_size * sfreq)
        step_samples = int(step_size * sfreq)

        # Calculate time points
        times = np.arange(n_samples) / sfreq + tmin

        results = {
            'window_centers': [],
            'window_starts': [],
            'window_ends': [],
            'accuracies': [],
            'accuracy_drops': [],
            'baseline_accuracy': baseline_acc
        }

        # Slide window across time
        for start_idx in range(0, n_samples - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples

            # Evaluate with this window masked
            acc, _ = self.evaluate_window_ablation(
                x, y, start_idx, end_idx, mask_type
            )

            # Calculate center time
            center_idx = (start_idx + end_idx) // 2
            center_time = times[center_idx]

            results['window_centers'].append(center_time)
            results['window_starts'].append(times[start_idx])
            results['window_ends'].append(times[end_idx])
            results['accuracies'].append(acc)
            results['accuracy_drops'].append(baseline_acc - acc)

        # Convert to arrays
        for key in results:
            if key != 'baseline_accuracy':
                results[key] = np.array(results[key])

        return results

    def fixed_windows_ablation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        time_windows: List[Tuple[float, float]],
        sfreq: float = 250.0,
        tmin: float = -1.0,
        mask_type: Literal['zero', 'mean', 'noise'] = 'zero'
    ) -> List[AblationResult]:
        """
        Ablate specific pre-defined time windows.

        Useful for testing hypothesis about specific time periods
        (e.g., pre-cue vs post-cue).

        Args:
            x: Input data (N, 1, channels, samples)
            y: True labels (N,)
            time_windows: List of (start_time, end_time) tuples in seconds
            sfreq: Sampling frequency
            tmin: Start time of epoch
            mask_type: Type of masking

        Returns:
            List of AblationResult objects
        """
        # Get baseline accuracy
        baseline_acc, _ = self.evaluate_window_ablation(
            x, y, 0, 0, mask_type='zero'
        )

        results = []

        for window_start, window_end in time_windows:
            # Convert times to sample indices
            start_idx = int((window_start - tmin) * sfreq)
            end_idx = int((window_end - tmin) * sfreq)

            # Clip to valid range
            start_idx = max(0, start_idx)
            end_idx = min(x.shape[-1], end_idx)

            # Evaluate
            acc, preds = self.evaluate_window_ablation(
                x, y, start_idx, end_idx, mask_type
            )

            result = AblationResult(
                window_start=window_start,
                window_end=window_end,
                accuracy=acc,
                accuracy_drop=baseline_acc - acc,
                predictions=preds,
                true_labels=y.cpu().numpy()
            )

            results.append(result)

        return results

    def progressive_ablation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sfreq: float = 250.0,
        tmin: float = -1.0,
        direction: Literal['forward', 'backward'] = 'forward',
        step_size: float = 0.1,
        mask_type: Literal['zero', 'mean', 'noise'] = 'zero'
    ) -> Dict[str, np.ndarray]:
        """
        Progressive ablation from one end of the epoch.

        Progressively masks more of the signal from one end to identify
        the critical time point where performance drops.

        Args:
            x: Input data
            y: True labels
            sfreq: Sampling frequency
            tmin: Start time
            direction: 'forward' masks from start, 'backward' from end
            step_size: Step size in seconds
            mask_type: Type of masking

        Returns:
            Dictionary with results
        """
        n_samples = x.shape[-1]
        step_samples = int(step_size * sfreq)
        times = np.arange(n_samples) / sfreq + tmin

        results = {
            'cutoff_times': [],
            'accuracies': [],
            'n_samples_masked': []
        }

        if direction == 'forward':
            # Progressively mask from start
            for end_idx in range(step_samples, n_samples + 1, step_samples):
                acc, _ = self.evaluate_window_ablation(
                    x, y, 0, end_idx, mask_type
                )
                results['cutoff_times'].append(times[end_idx - 1])
                results['accuracies'].append(acc)
                results['n_samples_masked'].append(end_idx)
        else:
            # Progressively mask from end
            for start_idx in range(n_samples - step_samples, -1, -step_samples):
                acc, _ = self.evaluate_window_ablation(
                    x, y, start_idx, n_samples, mask_type
                )
                results['cutoff_times'].append(times[start_idx])
                results['accuracies'].append(acc)
                results['n_samples_masked'].append(n_samples - start_idx)

        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])

        return results
