"""
MRCP and preparatory potential analysis for EEG classification models.

Provides tools to assess whether an EEGNet model exploits slow cortical
potentials (movement-related cortical potentials, MRCPs) and preparatory
activity in the <5 Hz range.  Implements SPEC.md Phase 5 items 5.1-5.4:

- 5.1  Low-frequency saliency (bandpass 0.1-5 Hz, then saliency)
- 5.2  Temporal filter frequency response check for MRCP sensitivity
- 5.3  Grand-average ERP overlay with model temporal importance
- 5.4  Lateralized readiness potential (LRP) vs model lateralized saliency
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

from .saliency import SaliencyMapGenerator
from .filters import FilterVisualizer


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass
class FilterMRCPResult:
    """Per-filter MRCP sensitivity result."""
    filter_index: int
    low_freq_energy_ratio: float
    peak_frequency: float
    is_mrcp_sensitive: bool


@dataclass
class LowFrequencySaliencyResult:
    """Results from low-frequency saliency analysis."""
    times: np.ndarray
    temporal_importance: np.ndarray
    filtered_saliency_map: np.ndarray


@dataclass
class ERPOverlayResult:
    """Results for ERP + temporal importance overlay."""
    times: np.ndarray
    erp_per_class: Dict[int, np.ndarray]
    channel_name: str
    temporal_importance: np.ndarray


@dataclass
class LRPResult:
    """Results from lateralized readiness potential analysis."""
    times: np.ndarray
    classical_lrp: np.ndarray
    model_lateralized_saliency: Optional[np.ndarray] = None


# ── Main analyser class ──────────────────────────────────────────────────

class MRCPAnalyzer:
    """
    Analyze MRCP and preparatory potential contributions to EEGNet decisions.

    Combines low-frequency filtering with gradient-based saliency, learned
    filter analysis, grand-average ERPs, and the lateralized readiness
    potential to determine whether the model leverages slow cortical
    potentials for classification.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        batch_size: int = 32,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.model.eval()

        self.saliency = SaliencyMapGenerator(model, device)
        self.filter_viz = FilterVisualizer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_saliency(
        self,
        x: torch.Tensor,
        method: str = 'integrated_gradients',
    ) -> np.ndarray:
        """Dispatch to the requested saliency method and return (batch, ch, samples)."""
        if method == 'vanilla':
            return self.saliency.vanilla_gradient(x)
        elif method == 'integrated_gradients':
            return self.saliency.integrated_gradients(x)
        elif method == 'gradient_x_input':
            return self.saliency.gradient_x_input(x)
        elif method == 'deeplift':
            return self.saliency.deeplift(x)
        else:
            raise ValueError(f"Unknown saliency method: {method}")

    @staticmethod
    def _bandpass_filter(
        data: np.ndarray,
        low_freq: float,
        high_freq: float,
        sfreq: float,
        order: int = 4,
    ) -> np.ndarray:
        """Apply zero-phase Butterworth bandpass filter along the last axis."""
        sos = butter(order, [low_freq, high_freq], btype='bandpass',
                     fs=sfreq, output='sos')
        return sosfiltfilt(sos, data, axis=-1)

    @staticmethod
    def _find_channel_index(
        channel_names: List[str],
        target: str,
    ) -> Optional[int]:
        """Return the index of *target* in *channel_names*, or None."""
        for idx, name in enumerate(channel_names):
            if name == target:
                return idx
        return None

    # ------------------------------------------------------------------
    # 5.1  Low-frequency saliency
    # ------------------------------------------------------------------

    def low_frequency_saliency(
        self,
        x: torch.Tensor,
        sfreq: float = 250.0,
        tmin: float = -1.0,
        low_freq: float = 0.1,
        high_freq: float = 5.0,
        method: str = 'integrated_gradients',
    ) -> LowFrequencySaliencyResult:
        """
        Bandpass-filter input to the MRCP range and compute saliency.

        Args:
            x: Input tensor (N, 1, channels, samples).
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.
            low_freq: Lower edge of the bandpass in Hz.
            high_freq: Upper edge of the bandpass in Hz.
            method: Saliency method name.

        Returns:
            LowFrequencySaliencyResult with times, temporal importance,
            and the full filtered saliency map.
        """
        # Filter the raw EEG in numpy then re-pack as tensor
        x_np = x.cpu().numpy()  # (N, 1, ch, samples)
        x_filt = self._bandpass_filter(x_np, low_freq, high_freq, sfreq)
        x_filt_t = torch.tensor(x_filt, dtype=torch.float32)

        # Run saliency on the filtered input
        saliency_map = self._get_saliency(x_filt_t, method)

        times, temporal_importance = self.saliency.compute_temporal_importance(
            saliency_map, sfreq, tmin,
        )

        return LowFrequencySaliencyResult(
            times=times,
            temporal_importance=temporal_importance,
            filtered_saliency_map=saliency_map,
        )

    # ------------------------------------------------------------------
    # 5.2  Temporal filter MRCP sensitivity
    # ------------------------------------------------------------------

    def check_filter_mrcp_sensitivity(
        self,
        sfreq: float = 250.0,
        mrcp_cutoff: float = 5.0,
        energy_threshold: float = 0.3,
    ) -> List[FilterMRCPResult]:
        """
        Assess each learned temporal filter for low-frequency MRCP sensitivity.

        Computes the FFT of every temporal filter and measures the fraction
        of spectral energy below *mrcp_cutoff* Hz.

        Args:
            sfreq: Sampling frequency in Hz.
            mrcp_cutoff: Upper frequency for the MRCP band in Hz.
            energy_threshold: Minimum low-freq energy ratio to flag a
                filter as MRCP-sensitive.

        Returns:
            List of FilterMRCPResult, one per temporal filter.
        """
        filters = self.filter_viz.extract_temporal_filters(self.model)
        n_filters, kernel_length = filters.shape

        freqs = np.fft.rfftfreq(kernel_length, d=1.0 / sfreq)

        results: List[FilterMRCPResult] = []

        for i in range(n_filters):
            fft_mag = np.abs(np.fft.rfft(filters[i]))
            total_energy = np.sum(fft_mag ** 2)

            # Energy below the MRCP cutoff
            low_mask = freqs <= mrcp_cutoff
            low_energy = np.sum(fft_mag[low_mask] ** 2)
            low_ratio = float(low_energy / (total_energy + 1e-12))

            peak_freq = float(freqs[np.argmax(fft_mag)])

            results.append(FilterMRCPResult(
                filter_index=i,
                low_freq_energy_ratio=low_ratio,
                peak_frequency=peak_freq,
                is_mrcp_sensitive=low_ratio >= energy_threshold,
            ))

        return results

    # ------------------------------------------------------------------
    # 5.3  Grand-average ERP + importance overlay
    # ------------------------------------------------------------------

    def compute_grand_average_erp(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        channel_names: List[str],
        sfreq: float = 250.0,
        tmin: float = -1.0,
        channels_of_interest: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Compute grand-average ERPs at specified channels, split by class.

        Args:
            data: Trial array (n_trials, n_channels, n_samples).
            labels: Class labels (n_trials,).
            channel_names: List of channel names.
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.
            channels_of_interest: Channel names to average. Defaults to
                ['Cz', 'FCz'], falling back to whatever is available.

        Returns:
            Dictionary with keys:
                - 'times': time vector (n_samples,)
                - 'erp_per_class': {class_label: {channel_name: erp_array}}
                - 'channels_used': list of channel names actually found
        """
        if channels_of_interest is None:
            channels_of_interest = ['Cz', 'FCz']

        # Resolve channel indices
        channels_used: List[str] = []
        channel_indices: List[int] = []
        for ch in channels_of_interest:
            idx = self._find_channel_index(channel_names, ch)
            if idx is not None:
                channels_used.append(ch)
                channel_indices.append(idx)

        if len(channels_used) == 0:
            # Fall back to first channel
            channels_used = [channel_names[0]]
            channel_indices = [0]

        n_samples = data.shape[-1]
        times = np.arange(n_samples) / sfreq + tmin

        unique_labels = np.unique(labels)
        erp_per_class: Dict[int, Dict[str, np.ndarray]] = {}

        for lbl in unique_labels:
            mask = labels == lbl
            class_data = data[mask]
            erp_per_class[int(lbl)] = {}
            for ch_name, ch_idx in zip(channels_used, channel_indices):
                erp_per_class[int(lbl)][ch_name] = class_data[:, ch_idx, :].mean(axis=0)

        return {
            'times': times,
            'erp_per_class': erp_per_class,
            'channels_used': channels_used,
        }

    def overlay_erp_with_importance(
        self,
        x: torch.Tensor,
        data: np.ndarray,
        labels: np.ndarray,
        channel_names: List[str],
        sfreq: float = 250.0,
        tmin: float = -1.0,
        method: str = 'integrated_gradients',
        channels_of_interest: Optional[List[str]] = None,
    ) -> List[ERPOverlayResult]:
        """
        Compute grand-average ERP and model temporal importance for overlay.

        Args:
            x: Input tensor (N, 1, channels, samples).
            data: Raw EEG array (n_trials, n_channels, n_samples).
            labels: Class labels (n_trials,).
            channel_names: List of channel names.
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.
            method: Saliency method name.
            channels_of_interest: Channels for ERP. Defaults to ['Cz', 'FCz'].

        Returns:
            List of ERPOverlayResult, one per channel found.
        """
        # ERP computation
        erp_info = self.compute_grand_average_erp(
            data, labels, channel_names, sfreq, tmin, channels_of_interest,
        )
        times = erp_info['times']
        erp_per_class = erp_info['erp_per_class']
        channels_used = erp_info['channels_used']

        # Model temporal importance
        saliency_map = self._get_saliency(x, method)
        _, temporal_importance = self.saliency.compute_temporal_importance(
            saliency_map, sfreq, tmin,
        )

        results: List[ERPOverlayResult] = []
        for ch_name in channels_used:
            ch_erp: Dict[int, np.ndarray] = {}
            for cls_label, ch_dict in erp_per_class.items():
                if ch_name in ch_dict:
                    ch_erp[cls_label] = ch_dict[ch_name]

            results.append(ERPOverlayResult(
                times=times,
                erp_per_class=ch_erp,
                channel_name=ch_name,
                temporal_importance=temporal_importance,
            ))

        return results

    # ------------------------------------------------------------------
    # 5.4  Lateralized readiness potential
    # ------------------------------------------------------------------

    def compute_lrp(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        channel_names: List[str],
        sfreq: float = 250.0,
        tmin: float = -1.0,
    ) -> LRPResult:
        """
        Compute the classical lateralized readiness potential (LRP).

        LRP = mean(C3_left - C3_right) - mean(C4_left - C4_right)

        where C3_left is the average C3 signal across left-MI trials, etc.
        This follows the double-subtraction method (Coles, 1989).

        Args:
            data: Trial array (n_trials, n_channels, n_samples).
            labels: Class labels (n_trials,). Label 0 = left MI, 1 = right MI.
            channel_names: List of channel names.
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.

        Returns:
            LRPResult with times and the classical LRP curve.
        """
        c3_idx = self._find_channel_index(channel_names, 'C3')
        c4_idx = self._find_channel_index(channel_names, 'C4')

        if c3_idx is None or c4_idx is None:
            raise ValueError(
                f"C3 and/or C4 not found in channel_names. "
                f"Available: {channel_names}"
            )

        n_samples = data.shape[-1]
        times = np.arange(n_samples) / sfreq + tmin

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError("Need at least two classes for LRP computation")

        # Assume label 0 = left MI, label 1 = right MI
        left_label = int(unique_labels[0])
        right_label = int(unique_labels[1])

        left_mask = labels == left_label
        right_mask = labels == right_label

        c3_left = data[left_mask, c3_idx, :].mean(axis=0)
        c3_right = data[right_mask, c3_idx, :].mean(axis=0)
        c4_left = data[left_mask, c4_idx, :].mean(axis=0)
        c4_right = data[right_mask, c4_idx, :].mean(axis=0)

        # Double subtraction: LRP = (C3_left - C3_right) - (C4_left - C4_right)
        classical_lrp = (c3_left - c3_right) - (c4_left - c4_right)

        return LRPResult(
            times=times,
            classical_lrp=classical_lrp,
            model_lateralized_saliency=None,
        )

    def compute_lateralized_saliency(
        self,
        x: torch.Tensor,
        labels: np.ndarray,
        channel_names: List[str],
        sfreq: float = 250.0,
        tmin: float = -1.0,
        method: str = 'integrated_gradients',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute lateralized model saliency at C3/C4 across MI classes.

        Mirrors the classical LRP double subtraction but uses gradient-based
        saliency instead of raw EEG amplitude:

        lateralized = (C3_sal_left - C3_sal_right) - (C4_sal_left - C4_sal_right)

        Args:
            x: Input tensor (N, 1, channels, samples).
            labels: Class labels (N,). Label 0 = left MI, 1 = right MI.
            channel_names: List of channel names.
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.
            method: Saliency method name.

        Returns:
            Tuple of (times, lateralized_saliency).
        """
        c3_idx = self._find_channel_index(channel_names, 'C3')
        c4_idx = self._find_channel_index(channel_names, 'C4')

        if c3_idx is None or c4_idx is None:
            raise ValueError(
                f"C3 and/or C4 not found in channel_names. "
                f"Available: {channel_names}"
            )

        unique_labels = np.unique(labels)
        left_label = int(unique_labels[0])
        right_label = int(unique_labels[1])

        left_mask = labels == left_label
        right_mask = labels == right_label

        # Saliency for left-MI trials
        saliency_left = self._get_saliency(x[left_mask], method)
        # Saliency for right-MI trials
        saliency_right = self._get_saliency(x[right_mask], method)

        # Average saliency at C3 and C4 per class
        c3_sal_left = saliency_left[:, c3_idx, :].mean(axis=0)
        c3_sal_right = saliency_right[:, c3_idx, :].mean(axis=0)
        c4_sal_left = saliency_left[:, c4_idx, :].mean(axis=0)
        c4_sal_right = saliency_right[:, c4_idx, :].mean(axis=0)

        lateralized = (c3_sal_left - c3_sal_right) - (c4_sal_left - c4_sal_right)

        n_samples = x.shape[-1]
        times = np.arange(n_samples) / sfreq + tmin

        return times, lateralized


# ── Module-level plotting functions ──────────────────────────────────────

def plot_low_frequency_saliency(
    times: np.ndarray,
    importance: np.ndarray,
    cue_onset: float = 0.0,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot low-frequency (<5 Hz) saliency temporal importance.

    Args:
        times: Time vector in seconds.
        importance: Temporal importance curve.
        cue_onset: Time of cue onset in seconds.
        save_path: Path to save figure at 300 dpi.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(times, importance, linewidth=2, color='steelblue')
    ax.fill_between(times, 0, importance, alpha=0.3, color='steelblue')
    ax.axvline(cue_onset, color='red', linestyle='--', linewidth=1.5,
               label='Cue Onset', alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Importance (0.1-5 Hz)', fontsize=11)
    ax.set_title('Low-Frequency Saliency (MRCP Range)', fontsize=13,
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_filter_mrcp_sensitivity(
    filter_results: List[FilterMRCPResult],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Bar chart of low-frequency energy ratio per temporal filter.

    Args:
        filter_results: List of FilterMRCPResult from check_filter_mrcp_sensitivity.
        save_path: Path to save figure at 300 dpi.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    indices = [r.filter_index for r in filter_results]
    ratios = [r.low_freq_energy_ratio for r in filter_results]
    sensitive = [r.is_mrcp_sensitive for r in filter_results]
    peak_freqs = [r.peak_frequency for r in filter_results]

    colors = ['#E74C3C' if s else '#3498DB' for s in sensitive]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        [f"F{i + 1}" for i in indices],
        ratios,
        color=colors,
        alpha=0.8,
        edgecolor='black',
    )

    # Annotate peak frequency on each bar
    for bar, ratio, pf in zip(bars, ratios, peak_freqs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{ratio:.2f}\n({pf:.1f} Hz)',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    ax.set_xlabel('Temporal Filter', fontsize=11)
    ax.set_ylabel('Low-Freq Energy Ratio (<5 Hz)', fontsize=11)
    ax.set_title('Temporal Filter MRCP Sensitivity', fontsize=13,
                 fontweight='bold')
    ax.axhline(0.3, color='gray', linestyle='--', alpha=0.6,
               label='Sensitivity threshold (0.3)')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='black', alpha=0.8,
              label='MRCP-Sensitive'),
        Patch(facecolor='#3498DB', edgecolor='black', alpha=0.8,
              label='Not Sensitive'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_erp_with_importance(
    times: np.ndarray,
    erp_data: Dict[int, np.ndarray],
    importance: np.ndarray,
    channel_name: str,
    cue_onset: float = 0.0,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Dual-axis plot: ERP on left y-axis, model importance on right y-axis.

    Args:
        times: Time vector in seconds.
        erp_data: Dict mapping class label to ERP array.
        importance: Model temporal importance curve.
        channel_name: Name of the ERP channel.
        cue_onset: Time of cue onset in seconds.
        save_path: Path to save figure at 300 dpi.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    class_colors = {0: '#3498DB', 1: '#E74C3C'}
    class_names = {0: 'Left MI', 1: 'Right MI'}

    fig, ax1 = plt.subplots(figsize=figsize)

    # ERP on left axis
    for cls_label, erp in erp_data.items():
        color = class_colors.get(cls_label, '#95A5A6')
        name = class_names.get(cls_label, f'Class {cls_label}')
        ax1.plot(times, erp, linewidth=1.5, color=color, label=f'ERP {name}')

    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('ERP Amplitude', fontsize=11, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Importance on right axis
    ax2 = ax1.twinx()
    ax2.plot(times, importance, linewidth=2, color='orange', alpha=0.7,
             label='Model Importance')
    ax2.fill_between(times, 0, importance, alpha=0.15, color='orange')
    ax2.set_ylabel('Model Importance', fontsize=11, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Cue onset
    ax1.axvline(cue_onset, color='gray', linestyle='--', linewidth=1.5,
                alpha=0.7, label='Cue Onset')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               fontsize=9)

    ax1.set_title(
        f'Grand-Average ERP at {channel_name} with Model Importance',
        fontsize=13,
        fontweight='bold',
    )
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_lrp_comparison(
    times: np.ndarray,
    classical_lrp: np.ndarray,
    model_lrp: Optional[np.ndarray] = None,
    cue_onset: float = 0.0,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Overlay classical LRP with model lateralized saliency.

    Args:
        times: Time vector in seconds.
        classical_lrp: Classical LRP curve.
        model_lrp: Model lateralized saliency curve. May be None.
        cue_onset: Time of cue onset in seconds.
        save_path: Path to save figure at 300 dpi.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(times, classical_lrp, linewidth=2, color='steelblue',
             label='Classical LRP')
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(cue_onset, color='red', linestyle='--', linewidth=1.5,
                alpha=0.7, label='Cue Onset')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Classical LRP Amplitude', fontsize=11, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    if model_lrp is not None:
        ax2 = ax1.twinx()
        ax2.plot(times, model_lrp, linewidth=2, color='coral',
                 label='Model Lateralized Saliency')
        ax2.set_ylabel('Model Lateralized Saliency', fontsize=11,
                        color='coral')
        ax2.tick_params(axis='y', labelcolor='coral')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                   fontsize=10)
    else:
        ax1.legend(loc='upper right', fontsize=10)

    ax1.set_title(
        'Lateralized Readiness Potential: Classical vs Model',
        fontsize=13,
        fontweight='bold',
    )
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig
