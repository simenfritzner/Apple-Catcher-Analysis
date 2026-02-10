"""
Filter and weight visualization for EEGNet models.

Provides extraction and visualization of learned convolutional filters,
including temporal filters, spatial filters, separable convolution filters,
and classifier weights.
"""

from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Sensorimotor channels commonly relevant for motor imagery BCI
SENSORIMOTOR_CHANNELS = {'C3', 'C4', 'Cz', 'FC3', 'FC4'}

# EEG frequency band definitions (Hz)
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'mu': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 125.0),
}

# Colors for frequency bands
BAND_COLORS: Dict[str, str] = {
    'delta': '#d4e6f1',
    'theta': '#a9dfbf',
    'mu': '#f9e79f',
    'beta': '#f5cba7',
    'gamma': '#fadbd8',
}


class FilterVisualizer:
    """
    Extract and visualize learned filters from a trained EEGNet model.

    Supports visualization of temporal convolution filters (time-domain and
    frequency response), spatial filters, separable convolution filters,
    and classifier weights.
    """

    def extract_temporal_filters(self, model: nn.Module) -> np.ndarray:
        """
        Extract temporal convolution kernels from EEGNet block1[0].

        Args:
            model: Trained EEGNet model.

        Returns:
            Temporal filter weights as numpy array of shape (F1, kernel_length).
        """
        temporal_conv = model.block1[0]
        # Shape: (F1, 1, 1, kernel_length) -> (F1, kernel_length)
        weights = temporal_conv.weight.detach().cpu().numpy()
        return weights.squeeze(axis=(1, 2))

    def extract_spatial_filters(self, model: nn.Module) -> np.ndarray:
        """
        Extract depthwise spatial convolution kernels from EEGNet block1[2].

        Args:
            model: Trained EEGNet model.

        Returns:
            Spatial filter weights as numpy array of shape (F1*D, Chans).
        """
        spatial_conv = model.block1[2]
        # Shape: (F1*D, 1, Chans, 1) -> (F1*D, Chans)
        weights = spatial_conv.weight.detach().cpu().numpy()
        return weights.squeeze(axis=(1, 3))

    def extract_separable_filters(
        self, model: nn.Module
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract separable convolution filters from EEGNet block2.

        Args:
            model: Trained EEGNet model.

        Returns:
            Tuple of (depthwise_filters, pointwise_filters).
            - depthwise_filters: shape (F2, kernel_length) from block2[0]
            - pointwise_filters: shape (F2, F2) from block2[1]
        """
        depthwise_conv = model.block2[0]
        pointwise_conv = model.block2[1]

        # Depthwise: (F2, 1, 1, kernel_length) -> (F2, kernel_length)
        dw_weights = depthwise_conv.weight.detach().cpu().numpy()
        dw_weights = dw_weights.squeeze(axis=(1, 2))

        # Pointwise: (F2, F2, 1, 1) -> (F2, F2)
        pw_weights = pointwise_conv.weight.detach().cpu().numpy()
        pw_weights = pw_weights.squeeze(axis=(2, 3))

        return dw_weights, pw_weights

    def extract_classifier_weights(self, model: nn.Module) -> np.ndarray:
        """
        Extract linear classifier weights from EEGNet.

        Args:
            model: Trained EEGNet model.

        Returns:
            Classifier weight matrix of shape (nb_classes, n_features).
        """
        # classifier is nn.Sequential(Flatten, Linear)
        linear_layer = model.classifier[1]
        weights = linear_layer.weight.detach().cpu().numpy()
        return weights

    def plot_temporal_filters(
        self,
        filters: np.ndarray,
        sfreq: float = 250.0,
        save_path: Optional[Path] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Plot temporal filters as time-domain waveforms and frequency responses.

        Creates a 2-column grid: left column shows the filter in time domain,
        right column shows FFT magnitude with mu (8-13 Hz) and beta (13-30 Hz)
        band annotations.

        Args:
            filters: Temporal filter array of shape (n_filters, kernel_length).
            sfreq: Sampling frequency in Hz.
            save_path: Path to save figure. If None, figure is not saved.
            figsize: Figure size as (width, height). Auto-computed if None.

        Returns:
            Matplotlib Figure object.
        """
        n_filters, kernel_length = filters.shape
        if figsize is None:
            figsize = (14, 3 * n_filters)

        fig, axes = plt.subplots(n_filters, 2, figsize=figsize)
        if n_filters == 1:
            axes = axes[np.newaxis, :]

        # Time vector for the filter kernel
        times = np.arange(kernel_length) / sfreq * 1000  # ms

        # Frequency axis for FFT
        freqs = np.fft.rfftfreq(kernel_length, d=1.0 / sfreq)

        for i in range(n_filters):
            kernel = filters[i]

            # --- Time domain ---
            ax_time = axes[i, 0]
            ax_time.plot(times, kernel, linewidth=1.5, color='steelblue')
            ax_time.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax_time.set_ylabel('Amplitude', fontsize=10)
            ax_time.set_title(
                f'Filter {i + 1} - Time Domain', fontsize=11, fontweight='bold'
            )
            ax_time.grid(True, alpha=0.3)

            if i == n_filters - 1:
                ax_time.set_xlabel('Time (ms)', fontsize=10)

            # --- Frequency response ---
            ax_freq = axes[i, 1]
            fft_magnitude = np.abs(np.fft.rfft(kernel))
            ax_freq.plot(freqs, fft_magnitude, linewidth=1.5, color='coral')

            # Shade mu and beta bands
            _shade_frequency_band(ax_freq, 'mu', freqs, alpha=0.25)
            _shade_frequency_band(ax_freq, 'beta', freqs, alpha=0.25)

            ax_freq.set_ylabel('Magnitude', fontsize=10)
            ax_freq.set_title(
                f'Filter {i + 1} - Frequency Response',
                fontsize=11,
                fontweight='bold',
            )
            ax_freq.set_xlim(0, sfreq / 2)
            ax_freq.grid(True, alpha=0.3)

            if i == n_filters - 1:
                ax_freq.set_xlabel('Frequency (Hz)', fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def plot_spatial_filters(
        self,
        filters: np.ndarray,
        channel_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Plot spatial filters as bar charts of per-channel weights.

        Highlights sensorimotor channels (C3, C4, Cz, FC3, FC4) when
        channel_names are provided.

        Args:
            filters: Spatial filter array of shape (n_filters, n_channels).
            channel_names: List of EEG channel names.
            save_path: Path to save figure. If None, figure is not saved.
            figsize: Figure size as (width, height). Auto-computed if None.

        Returns:
            Matplotlib Figure object.
        """
        n_filters, n_channels = filters.shape
        n_cols = 4
        n_rows = int(np.ceil(n_filters / n_cols))
        if figsize is None:
            figsize = (4 * n_cols, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes)

        labels = (
            channel_names
            if channel_names is not None
            else [f"Ch{j}" for j in range(n_channels)]
        )

        # Determine which channel indices are sensorimotor
        sm_indices = set()
        if channel_names is not None:
            for idx, name in enumerate(channel_names):
                if name in SENSORIMOTOR_CHANNELS:
                    sm_indices.add(idx)

        for i in range(n_filters):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]

            weights = filters[i]
            colors = [
                '#e74c3c' if j in sm_indices else 'steelblue'
                for j in range(n_channels)
            ]

            ax.bar(range(n_channels), weights, color=colors, alpha=0.8, width=0.8)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f'Spatial Filter {i + 1}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Only label x-axis on bottom row or if few channels
            if row == n_rows - 1 and n_channels <= 32:
                tick_step = max(1, n_channels // 8)
                tick_positions = list(range(0, n_channels, tick_step))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(
                    [labels[p] for p in tick_positions],
                    rotation=45,
                    ha='right',
                    fontsize=7,
                )
            else:
                ax.set_xticks([])

        # Hide unused subplots
        for i in range(n_filters, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].set_visible(False)

        # Add legend for sensorimotor highlighting
        if sm_indices:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#e74c3c', alpha=0.8, label='Sensorimotor'),
                Patch(facecolor='steelblue', alpha=0.8, label='Other'),
            ]
            fig.legend(
                handles=legend_elements,
                loc='upper right',
                fontsize=9,
                framealpha=0.9,
            )

        plt.suptitle(
            'Spatial Filters (Weight per Channel)',
            fontsize=13,
            fontweight='bold',
            y=1.01,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def plot_filter_frequency_response(
        self,
        filters: np.ndarray,
        sfreq: float = 250.0,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Overlay all temporal filter frequency responses on a single plot.

        Shades standard EEG frequency bands: delta (1-4 Hz), theta (4-8 Hz),
        mu (8-13 Hz), beta (13-30 Hz), and gamma (30+ Hz).

        Args:
            filters: Temporal filter array of shape (n_filters, kernel_length).
            sfreq: Sampling frequency in Hz.
            save_path: Path to save figure. If None, figure is not saved.
            figsize: Figure size as (width, height).

        Returns:
            Matplotlib Figure object.
        """
        n_filters, kernel_length = filters.shape
        freqs = np.fft.rfftfreq(kernel_length, d=1.0 / sfreq)

        fig, ax = plt.subplots(figsize=figsize)

        # Shade all frequency bands
        for band_name in ['delta', 'theta', 'mu', 'beta', 'gamma']:
            _shade_frequency_band(ax, band_name, freqs, alpha=0.15, label=True)

        # Plot each filter's frequency response
        cmap = plt.cm.tab10
        for i in range(n_filters):
            fft_magnitude = np.abs(np.fft.rfft(filters[i]))
            ax.plot(
                freqs,
                fft_magnitude,
                linewidth=1.5,
                color=cmap(i / max(n_filters - 1, 1)),
                label=f'Filter {i + 1}',
            )

        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Magnitude', fontsize=11)
        ax.set_title(
            'Temporal Filters - Frequency Response Comparison',
            fontsize=13,
            fontweight='bold',
        )
        ax.set_xlim(0, sfreq / 2)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def plot_classifier_weights(
        self,
        weights: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """
        Visualize classifier weights as a heatmap.

        Shows which features drive each class prediction in the final
        linear layer of EEGNet.

        Args:
            weights: Weight matrix of shape (nb_classes, n_features).
            class_names: Names for each class. Defaults to Class 0, Class 1, etc.
            save_path: Path to save figure. If None, figure is not saved.
            figsize: Figure size as (width, height).

        Returns:
            Matplotlib Figure object.
        """
        nb_classes, n_features = weights.shape

        if class_names is None:
            class_names = [f'Class {i}' for i in range(nb_classes)]

        fig, ax = plt.subplots(figsize=figsize)

        # Symmetric colorbar centered on zero
        vmax = np.abs(weights).max()

        im = sns.heatmap(
            weights,
            ax=ax,
            cmap='RdBu_r',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            yticklabels=class_names,
            xticklabels=False,
            cbar_kws={'label': 'Weight'},
        )

        ax.set_xlabel('Feature Index', fontsize=11)
        ax.set_ylabel('Class', fontsize=11)
        ax.set_title(
            'Classifier Weights Heatmap', fontsize=13, fontweight='bold'
        )

        # Add feature index ticks at regular intervals
        n_ticks = min(20, n_features)
        tick_step = max(1, n_features // n_ticks)
        tick_positions = list(range(0, n_features, tick_step))
        ax.set_xticks([p + 0.5 for p in tick_positions])
        ax.set_xticklabels(tick_positions, rotation=0, fontsize=8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig


def _shade_frequency_band(
    ax: plt.Axes,
    band_name: str,
    freqs: np.ndarray,
    alpha: float = 0.2,
    label: bool = False,
) -> None:
    """Shade a frequency band region on an axes object."""
    if band_name not in FREQ_BANDS:
        return

    low, high = FREQ_BANDS[band_name]
    # Clamp to frequency range
    low = max(low, freqs[0])
    high = min(high, freqs[-1])

    if low >= high:
        return

    ax.axvspan(
        low,
        high,
        alpha=alpha,
        color=BAND_COLORS[band_name],
        label=f'{band_name} ({FREQ_BANDS[band_name][0]}-{FREQ_BANDS[band_name][1]} Hz)'
        if label
        else None,
    )
