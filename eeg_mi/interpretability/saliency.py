"""
Saliency map generation for EEG classification models.

Provides gradient-based visualization methods to understand which parts
of the input (time x channel) are most important for model predictions.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class SaliencyMapGenerator:
    """
    Generate saliency maps showing input importance for EEG classification.

    Supports multiple gradient-based methods:
    - Vanilla gradients: Simple gradient of output w.r.t. input
    - Integrated gradients: Path integral from baseline to input
    - Gradient x input: Element-wise multiplication
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize saliency map generator.

        Args:
            model: Trained PyTorch model
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def vanilla_gradient(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute vanilla gradient saliency map.

        Args:
            x: Input tensor of shape (batch, 1, channels, samples)
            target_class: Class to compute gradient for. If None, uses predicted class.

        Returns:
            Saliency map of shape (batch, channels, samples)
        """
        x = x.to(self.device)
        x.requires_grad = True

        # Forward pass
        output = self.model(x)

        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * x.shape[0], device=self.device)

        # Compute gradient for target class
        self.model.zero_grad()

        # One-hot encode target class
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1)

        # Backward pass
        output.backward(gradient=one_hot)

        # Get gradients and remove channel dimension
        saliency = x.grad.data.abs().squeeze(1).cpu().numpy()

        return saliency

    def integrated_gradients(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50
    ) -> np.ndarray:
        """
        Compute integrated gradients saliency map.

        Integrated gradients provide better attribution by integrating
        gradients along a path from baseline to input.

        Args:
            x: Input tensor of shape (batch, 1, channels, samples)
            target_class: Class to compute gradient for
            baseline: Baseline input (default: zeros)
            n_steps: Number of integration steps

        Returns:
            Saliency map of shape (batch, channels, samples)
        """
        x = x.to(self.device)

        # Use zero baseline if not provided
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)

        # Generate alphas for interpolation
        alphas = torch.linspace(0, 1, n_steps, device=self.device)

        # Accumulate gradients
        integrated_grads = torch.zeros_like(x)

        for alpha in alphas:
            # Interpolate between baseline and input
            x_step = baseline + alpha * (x - baseline)
            x_step.requires_grad = True

            # Forward pass
            output = self.model(x_step)

            # Use predicted class if not specified
            if target_class is None:
                target_class_step = output.argmax(dim=1)
            elif isinstance(target_class, int):
                target_class_step = torch.tensor([target_class] * x.shape[0], device=self.device)
            else:
                target_class_step = target_class

            # Compute gradient
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, target_class_step.unsqueeze(1), 1)
            output.backward(gradient=one_hot)

            # Accumulate gradients
            integrated_grads += x_step.grad.data

        # Average gradients and multiply by input difference
        integrated_grads = integrated_grads / n_steps
        integrated_grads = (x - baseline) * integrated_grads

        # Remove channel dimension and take absolute value
        saliency = integrated_grads.abs().squeeze(1).cpu().numpy()

        return saliency

    def gradient_x_input(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute gradient x input saliency map.

        Args:
            x: Input tensor of shape (batch, 1, channels, samples)
            target_class: Class to compute gradient for

        Returns:
            Saliency map of shape (batch, channels, samples)
        """
        # Get vanilla gradients
        x_copy = x.clone().to(self.device)
        x_copy.requires_grad = True

        output = self.model(x_copy)

        if target_class is None:
            target_class = output.argmax(dim=1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * x.shape[0], device=self.device)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1)
        output.backward(gradient=one_hot)

        # Multiply gradient by input
        saliency = (x_copy.grad.data * x_copy).abs().squeeze(1).cpu().numpy()

        return saliency

    def compute_temporal_importance(
        self,
        saliency_map: np.ndarray,
        sfreq: float = 250.0,
        tmin: float = -1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate saliency across channels to show temporal importance.

        Args:
            saliency_map: Saliency map of shape (batch, channels, samples)
            sfreq: Sampling frequency in Hz
            tmin: Start time in seconds

        Returns:
            times: Time points in seconds
            importance: Temporal importance averaged across channels and trials
        """
        # Average across channels and trials
        temporal_importance = saliency_map.mean(axis=(0, 1))

        # Create time vector
        n_samples = saliency_map.shape[-1]
        times = np.arange(n_samples) / sfreq + tmin

        return times, temporal_importance

    def compute_channel_importance(
        self,
        saliency_map: np.ndarray
    ) -> np.ndarray:
        """
        Aggregate saliency across time to show channel importance.

        Args:
            saliency_map: Saliency map of shape (batch, channels, samples)

        Returns:
            importance: Channel importance averaged across time and trials
        """
        # Average across time and trials
        channel_importance = saliency_map.mean(axis=(0, 2))

        return channel_importance

    def compute_time_channel_map(
        self,
        saliency_map: np.ndarray,
        sfreq: float = 250.0,
        tmin: float = -1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create average time x channel importance map.

        Args:
            saliency_map: Saliency map of shape (batch, channels, samples)
            sfreq: Sampling frequency
            tmin: Start time in seconds

        Returns:
            times: Time points
            importance_map: Average importance map (channels x samples)
            std_map: Standard deviation across trials
        """
        # Average across trials
        importance_map = saliency_map.mean(axis=0)
        std_map = saliency_map.std(axis=0)

        # Create time vector
        n_samples = saliency_map.shape[-1]
        times = np.arange(n_samples) / sfreq + tmin

        return times, importance_map, std_map

    def deeplift(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute DeepLIFT attribution saliency map.

        Uses captum's DeepLift implementation with a zero baseline.
        Complements gradient-based methods by using a reference-based
        contribution approach that handles non-linearities differently.

        Args:
            x: Input tensor of shape (batch, 1, channels, samples)
            target_class: Class to compute attribution for. If None, uses predicted class.

        Returns:
            Saliency map of shape (batch, channels, samples)
        """
        from captum.attr import DeepLift

        x = x.to(self.device)

        if target_class is None:
            with torch.no_grad():
                output = self.model(x)
                target_class = output.argmax(dim=1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * x.shape[0], device=self.device)

        dl = DeepLift(self.model)
        baseline = torch.zeros_like(x)

        attributions = dl.attribute(x, baselines=baseline, target=target_class)
        saliency = attributions.abs().squeeze(1).cpu().detach().numpy()

        return saliency

    def compare_methods(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        methods: Optional[List[str]] = None,
        sfreq: float = 250.0,
        tmin: float = -1.0
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run multiple attribution methods on the same data for comparison.

        Args:
            x: Input tensor of shape (batch, 1, channels, samples)
            target_class: Class to compute attribution for
            methods: List of method names. Default: all four methods.
            sfreq: Sampling frequency in Hz
            tmin: Start time in seconds

        Returns:
            Dictionary keyed by method name, each containing:
                - saliency_map: Raw saliency map (batch, channels, samples)
                - temporal_importance: Aggregated over channels (samples,)
                - channel_importance: Aggregated over time (channels,)
                - times: Time vector in seconds
        """
        if methods is None:
            methods = ['vanilla', 'integrated_gradients', 'gradient_x_input', 'deeplift']

        method_funcs = {
            'vanilla': self.vanilla_gradient,
            'integrated_gradients': self.integrated_gradients,
            'gradient_x_input': self.gradient_x_input,
            'deeplift': self.deeplift,
        }

        results: Dict[str, Dict[str, np.ndarray]] = {}

        for method in methods:
            if method not in method_funcs:
                raise ValueError(f"Unknown method: {method}")

            saliency_map = method_funcs[method](x, target_class)
            times, temporal_importance = self.compute_temporal_importance(saliency_map, sfreq, tmin)
            channel_importance = self.compute_channel_importance(saliency_map)

            results[method] = {
                'saliency_map': saliency_map,
                'temporal_importance': temporal_importance,
                'channel_importance': channel_importance,
                'times': times,
            }

        return results
