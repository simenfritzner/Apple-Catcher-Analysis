"""
ERD/ERS analysis for EEG classification models.

Provides tools to investigate event-related desynchronization (ERD) and
synchronization (ERS) patterns, both as derived from the trained model's
internal representations and from classical bandpass-filter-then-square
analysis of the raw EEG.

Implements SPEC.md Phase 4: items 4.1-4.4.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, morlet2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .saliency import SaliencyMapGenerator


# Default frequency bands for ERD/ERS analysis
DEFAULT_ERD_BANDS: Dict[str, Tuple[float, float]] = {
    'mu': (8, 13),
    'beta': (13, 30),
}

# Sub-bands for frequency-band saliency (Phase 4.1)
SALIENCY_BANDS: Dict[str, Tuple[float, float]] = {
    'delta': (1, 4),
    'theta': (4, 8),
    'mu': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}

# Standard sensorimotor channel names used for contralateral ERD analysis
SENSORIMOTOR_CHANNELS = ['C3', 'C4', 'Cz', 'FC3', 'FC4']


def _bandpass_filter(
    data: np.ndarray,
    low: float,
    high: float,
    sfreq: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter along last axis."""
    nyq = sfreq / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1)


class ERDERSAnalyzer:
    """
    Analyze ERD/ERS patterns using both model-derived and classical methods.

    Combines gradient-based saliency in frequency sub-bands, time-frequency
    saliency via Morlet wavelets, model-internal activation power analysis,
    and traditional ERD/ERS computation to determine whether the EEGNet model
    leverages genuine sensorimotor rhythms.
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

    # ------------------------------------------------------------------
    # 4.1  Frequency-band saliency
    # ------------------------------------------------------------------

    def frequency_band_saliency(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sfreq: float = 250.0,
        tmin: float = -1.0,
        method: str = 'integrated_gradients',
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Bandpass filter input into sub-bands and compute saliency on each.

        For every frequency band the input data is filtered, then the chosen
        saliency method is applied. The returned temporal importance curves
        reveal which frequency bands drive classification at each time point.

        Args:
            x: Input tensor of shape (N, 1, channels, samples).
            y: True labels of shape (N,).
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.
            method: Saliency method ('integrated_gradients', 'vanilla',
                    'gradient_x_input', 'deeplift').
            bands: Frequency band dict mapping name to (low, high) Hz.
                   Defaults to SALIENCY_BANDS (delta, theta, mu, beta, gamma).

        Returns:
            Dictionary mapping band name to a dict with keys:
                - 'times': time vector in seconds
                - 'temporal_importance': importance averaged across channels/trials
                - 'saliency_map': raw saliency map (N, channels, samples)
                - 'mean_importance': scalar mean importance for ranking
        """
        if bands is None:
            bands = SALIENCY_BANDS

        method_funcs = {
            'vanilla': self.saliency.vanilla_gradient,
            'integrated_gradients': self.saliency.integrated_gradients,
            'gradient_x_input': self.saliency.gradient_x_input,
            'deeplift': self.saliency.deeplift,
        }
        if method not in method_funcs:
            raise ValueError(f"Unknown saliency method: {method}")

        saliency_func = method_funcs[method]

        # Extract raw numpy data: (N, channels, samples)
        x_np = x[:, 0, :, :].cpu().numpy()

        results: Dict[str, Dict[str, np.ndarray]] = {}

        for band_name, (low, high) in bands.items():
            # Bandpass filter
            x_filtered = _bandpass_filter(x_np, low, high, sfreq)

            # Reconstruct tensor with channel dim: (N, 1, channels, samples)
            x_band = torch.tensor(x_filtered, dtype=torch.float32).unsqueeze(1)

            # Compute saliency on the filtered input
            saliency_map = saliency_func(x_band)

            # Temporal importance curve
            times, temporal_importance = self.saliency.compute_temporal_importance(
                saliency_map, sfreq, tmin
            )

            results[band_name] = {
                'times': times,
                'temporal_importance': temporal_importance,
                'saliency_map': saliency_map,
                'mean_importance': float(temporal_importance.mean()),
            }

        return results

    # ------------------------------------------------------------------
    # 4.2  Time-frequency saliency
    # ------------------------------------------------------------------

    def time_frequency_saliency(
        self,
        saliency_map: np.ndarray,
        sfreq: float = 250.0,
        tmin: float = -1.0,
        n_freqs: int = 30,
        freq_range: Tuple[float, float] = (1, 45),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute time-frequency representation of a saliency map via Morlet wavelets.

        The saliency map is first averaged across channels and trials to obtain
        a 1-D temporal importance signal. This signal is then convolved with a
        bank of Morlet wavelets spanning *freq_range* to produce a TF matrix.

        Args:
            saliency_map: Saliency array of shape (batch, channels, samples).
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.
            n_freqs: Number of frequency bins.
            freq_range: (low_hz, high_hz) frequency range.

        Returns:
            times: 1-D array of time points in seconds.
            freqs: 1-D array of frequency bins in Hz.
            tf_matrix: 2-D array of shape (n_freqs, n_samples) with power values.
        """
        # Average across batch and channels -> (n_samples,)
        signal = saliency_map.mean(axis=(0, 1))
        n_samples = signal.shape[0]

        times = np.arange(n_samples) / sfreq + tmin
        freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)

        tf_matrix = np.zeros((n_freqs, n_samples))

        for fi, freq in enumerate(freqs):
            # Number of cycles scales with frequency for balanced resolution
            n_cycles = max(3, freq / 2.0)
            # scipy morlet2 width parameter: w = n_cycles * pi (approx)
            w = n_cycles
            wavelet_length = int(2 * w * sfreq / freq)
            # Ensure odd length
            wavelet_length = max(wavelet_length, 7)

            wavelet = morlet2(wavelet_length, w, freq / sfreq)

            # Convolve signal with wavelet (using mode='same' to preserve length)
            convolved = np.convolve(signal, wavelet, mode='same')
            tf_matrix[fi, :] = np.abs(convolved) ** 2

        return times, freqs, tf_matrix

    # ------------------------------------------------------------------
    # 4.3  Model-derived ERD/ERS
    # ------------------------------------------------------------------

    def model_derived_erd_ers(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        channel_names: List[str],
        sfreq: float = 250.0,
        tmin: float = -1.0,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Extract Block 1 activations and compute power in mu/beta bands over time.

        Passes data through the EEGNet temporal and spatial convolutions
        (block1), then bandpass-filters the resulting activations into
        specified frequency bands and computes instantaneous power. Results
        are averaged per class so that contralateral ERD at C3/C4 can be
        compared between left-MI and right-MI.

        Args:
            x: Input tensor of shape (N, 1, channels, samples).
            labels: Class labels of shape (N,).
            channel_names: List of EEG channel names.
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.
            bands: Frequency band dict. Defaults to DEFAULT_ERD_BANDS.

        Returns:
            Nested dict: band_name -> class_label -> channel_name ->
            dict with keys 'times' (1-D) and 'power' (1-D).
        """
        if bands is None:
            bands = DEFAULT_ERD_BANDS

        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        unique_classes = np.unique(labels_np)

        # Extract Block 1 activations
        # block1 output: (N, F1*D, 1, samples//8) after AvgPool(1,8)
        # The effective sfreq is sfreq / 8 after pooling
        block1_activations = self._extract_block1_activations(x)
        act_np = block1_activations.cpu().numpy()  # (N, F1*D, 1, T_pooled)
        act_np = act_np.squeeze(2)  # (N, F1*D, T_pooled)

        n_filters = act_np.shape[1]
        n_timepoints = act_np.shape[2]
        pooled_sfreq = sfreq / 8.0

        # Time vector for pooled activations
        times = np.arange(n_timepoints) / pooled_sfreq + tmin

        results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        for band_name, (low, high) in bands.items():
            results[band_name] = {}

            # Clamp band edges to Nyquist of the pooled signal
            nyq = pooled_sfreq / 2.0
            band_low = max(low, 0.5)
            band_high = min(high, nyq - 0.5)

            if band_low >= band_high:
                # Band is outside the representable range after pooling
                for cls in unique_classes:
                    results[band_name][str(int(cls))] = {}
                    for ch_name in channel_names:
                        results[band_name][str(int(cls))][ch_name] = {
                            'times': times,
                            'power': np.zeros(n_timepoints),
                        }
                continue

            for cls in unique_classes:
                cls_key = str(int(cls))
                results[band_name][cls_key] = {}
                cls_mask = labels_np == cls
                cls_act = act_np[cls_mask]  # (N_cls, F1*D, T_pooled)

                # Average across trials for this class
                mean_act = cls_act.mean(axis=0)  # (F1*D, T_pooled)

                # Bandpass filter activations and compute power
                filtered = _bandpass_filter(mean_act, band_low, band_high, pooled_sfreq)
                power = filtered ** 2  # (F1*D, T_pooled)

                # Average power across all activation filters
                mean_power = power.mean(axis=0)  # (T_pooled,)

                # Store per channel name — map spatial filter responses back
                # Since spatial convolution mixes all channels, we report the
                # aggregate power curve for each requested channel. For channels
                # of interest (C3, C4 etc.) we try to use the spatial filter
                # weight to create a weighted power.
                spatial_weights = self._get_spatial_weights()  # (F1*D, Chans)

                for ch_idx, ch_name in enumerate(channel_names):
                    # Weight each filter's power by how much it loads on this channel
                    ch_weights = np.abs(spatial_weights[:, ch_idx])
                    ch_weights_norm = ch_weights / (ch_weights.sum() + 1e-10)

                    # Weighted power for this channel
                    ch_power_filtered = _bandpass_filter(
                        cls_act.mean(axis=0), band_low, band_high, pooled_sfreq
                    )
                    ch_power = (ch_power_filtered ** 2)  # (F1*D, T_pooled)
                    ch_weighted_power = (ch_power * ch_weights_norm[:, np.newaxis]).sum(axis=0)

                    results[band_name][cls_key][ch_name] = {
                        'times': times,
                        'power': ch_weighted_power,
                    }

        return results

    # ------------------------------------------------------------------
    # 4.4  Classical ERD/ERS
    # ------------------------------------------------------------------

    def classical_erd_ers(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        channel_names: List[str],
        sfreq: float = 250.0,
        tmin: float = -1.0,
        baseline_window: Tuple[float, float] = (-1.0, -0.5),
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Compute traditional ERD/ERS from raw EEG (no model involved).

        For each frequency band: bandpass filter -> square (instantaneous power)
        -> average across trials per class -> normalize to a pre-cue baseline.
        The resulting ERD/ERS percentages follow the Pfurtscheller convention:
        ERD% = (power - baseline) / baseline * 100, where negative values
        indicate desynchronization.

        Args:
            data: Raw EEG array of shape (N, channels, samples).
            labels: Class labels of shape (N,).
            channel_names: List of EEG channel names.
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.
            baseline_window: (start_s, end_s) defining the baseline period.
            bands: Frequency band dict. Defaults to DEFAULT_ERD_BANDS.

        Returns:
            Nested dict: band_name -> class_label -> channel_name ->
            dict with keys 'times' (1-D) and 'power' (1-D, ERD/ERS %).
        """
        if bands is None:
            bands = DEFAULT_ERD_BANDS

        unique_classes = np.unique(labels)
        n_samples = data.shape[-1]
        times = np.arange(n_samples) / sfreq + tmin

        # Baseline sample indices
        bl_start = int((baseline_window[0] - tmin) * sfreq)
        bl_end = int((baseline_window[1] - tmin) * sfreq)
        bl_start = max(0, bl_start)
        bl_end = min(n_samples, bl_end)

        results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        for band_name, (low, high) in bands.items():
            results[band_name] = {}

            # Bandpass filter all trials: (N, channels, samples)
            filtered = _bandpass_filter(data, low, high, sfreq)

            # Instantaneous power
            power = filtered ** 2  # (N, channels, samples)

            for cls in unique_classes:
                cls_key = str(int(cls))
                results[band_name][cls_key] = {}
                cls_mask = labels == cls
                cls_power = power[cls_mask].mean(axis=0)  # (channels, samples)

                for ch_idx, ch_name in enumerate(channel_names):
                    ch_power = cls_power[ch_idx]  # (samples,)

                    # Baseline normalization (Pfurtscheller ERD%)
                    baseline_power = ch_power[bl_start:bl_end].mean()
                    if baseline_power > 0:
                        erd_pct = (ch_power - baseline_power) / baseline_power * 100.0
                    else:
                        erd_pct = np.zeros_like(ch_power)

                    results[band_name][cls_key][ch_name] = {
                        'times': times,
                        'power': erd_pct,
                    }

        return results

    # ------------------------------------------------------------------
    # 4.4 (cont.)  Compare model-derived vs classical ERD/ERS
    # ------------------------------------------------------------------

    def compare_model_vs_classical(
        self,
        model_erd: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        classical_erd: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute Pearson correlation between model-derived and classical ERD/ERS curves.

        For every (band, class, channel) combination present in both dicts,
        the power timecourses are interpolated to a common length and
        correlated. High positive correlation suggests the model captures
        genuine ERD/ERS patterns.

        Args:
            model_erd: Output of model_derived_erd_ers().
            classical_erd: Output of classical_erd_ers().

        Returns:
            Nested dict: band_name -> class_label -> channel_name -> float
            (Pearson r).
        """
        correlations: Dict[str, Dict[str, Dict[str, float]]] = {}

        for band_name in model_erd:
            if band_name not in classical_erd:
                continue
            correlations[band_name] = {}

            for cls_key in model_erd[band_name]:
                if cls_key not in classical_erd[band_name]:
                    continue
                correlations[band_name][cls_key] = {}

                for ch_name in model_erd[band_name][cls_key]:
                    if ch_name not in classical_erd[band_name][cls_key]:
                        continue

                    model_power = model_erd[band_name][cls_key][ch_name]['power']
                    classical_power = classical_erd[band_name][cls_key][ch_name]['power']

                    # Interpolate to common length (shorter)
                    common_len = min(len(model_power), len(classical_power))
                    m_interp = np.interp(
                        np.linspace(0, 1, common_len),
                        np.linspace(0, 1, len(model_power)),
                        model_power,
                    )
                    c_interp = np.interp(
                        np.linspace(0, 1, common_len),
                        np.linspace(0, 1, len(classical_power)),
                        classical_power,
                    )

                    # Pearson correlation
                    if m_interp.std() > 0 and c_interp.std() > 0:
                        r = float(np.corrcoef(m_interp, c_interp)[0, 1])
                    else:
                        r = 0.0

                    correlations[band_name][cls_key][ch_name] = r

        return correlations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_block1_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Run input through EEGNet block1 and return activations."""
        x = x.to(self.device)
        with torch.no_grad():
            activations = self.model.block1(x)
        return activations.cpu()

    def _get_spatial_weights(self) -> np.ndarray:
        """
        Extract spatial convolution weights from block1[2].

        Returns:
            Weight array of shape (F1*D, Chans).
        """
        spatial_conv = self.model.block1[2]
        # Shape: (F1*D, 1, Chans, 1) -> (F1*D, Chans)
        weights = spatial_conv.weight.detach().cpu().numpy()
        return weights.squeeze(axis=(1, 3))


# ======================================================================
# Module-level plotting functions
# ======================================================================


def plot_frequency_band_saliency(
    band_results: Dict[str, Dict[str, np.ndarray]],
    cue_onset: float = 0.0,
    save_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Multi-panel plot showing temporal saliency importance per frequency band.

    Args:
        band_results: Output of ERDERSAnalyzer.frequency_band_saliency().
        cue_onset: Cue onset time in seconds.
        save_path: Optional path to save figure.
        figsize: Figure size. Auto-computed if None.

    Returns:
        Matplotlib Figure.
    """
    band_names = list(band_results.keys())
    n_bands = len(band_names)

    if figsize is None:
        figsize = (14, 3 * n_bands)

    fig, axes = plt.subplots(n_bands, 1, figsize=figsize, sharex=True)
    if n_bands == 1:
        axes = [axes]

    band_colors = {
        'delta': '#2980B9',
        'theta': '#27AE60',
        'mu': '#F39C12',
        'beta': '#E74C3C',
        'gamma': '#8E44AD',
    }

    for i, band_name in enumerate(band_names):
        ax = axes[i]
        times = band_results[band_name]['times']
        importance = band_results[band_name]['temporal_importance']
        mean_imp = band_results[band_name]['mean_importance']
        color = band_colors.get(band_name, 'steelblue')

        ax.plot(times, importance, linewidth=2, color=color)
        ax.fill_between(times, 0, importance, alpha=0.3, color=color)
        ax.axvline(cue_onset, color='red', linestyle='--', alpha=0.7, label='Cue Onset')

        band_hz = SALIENCY_BANDS.get(band_name, ('?', '?'))
        ax.set_ylabel('Importance', fontsize=10)
        ax.set_title(
            f'{band_name} ({band_hz[0]}-{band_hz[1]} Hz)  '
            f'[mean={mean_imp:.5f}]',
            fontsize=11, fontweight='bold',
        )
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=11)

    plt.suptitle(
        'Frequency-Band Saliency Analysis',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_time_frequency_saliency(
    times: np.ndarray,
    freqs: np.ndarray,
    tf_matrix: np.ndarray,
    cue_onset: float = 0.0,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Heatmap of time-frequency saliency with time on x and frequency on y.

    Args:
        times: 1-D time vector in seconds.
        freqs: 1-D frequency vector in Hz.
        tf_matrix: 2-D array (n_freqs, n_samples).
        cue_onset: Cue onset time in seconds.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        times, freqs, tf_matrix,
        cmap='hot', shading='gouraud',
    )
    ax.axvline(cue_onset, color='cyan', linestyle='--', linewidth=2, label='Cue Onset')

    # Mark mu and beta band boundaries
    ax.axhline(8, color='white', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(13, color='white', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(30, color='white', linestyle=':', alpha=0.5, linewidth=1)

    # Band labels
    ax.text(times[0] + 0.05, 10.5, 'mu', color='white', fontsize=9, fontweight='bold')
    ax.text(times[0] + 0.05, 21.0, 'beta', color='white', fontsize=9, fontweight='bold')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title('Time-Frequency Saliency', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Saliency Power', fontsize=11)
    ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_erd_ers_comparison(
    model_erd: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    classical_erd: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    channel_names: List[str],
    correlations: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    save_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of model-derived vs classical ERD/ERS at C3 and C4.

    Creates a grid: rows = frequency bands, columns = [C3 model, C3 classical,
    C4 model, C4 classical].

    Args:
        model_erd: Output of model_derived_erd_ers().
        classical_erd: Output of classical_erd_ers().
        channel_names: Full list of channel names.
        correlations: Optional correlations dict from compare_model_vs_classical().
        save_path: Optional path to save figure.
        figsize: Figure size. Auto-computed if None.

    Returns:
        Matplotlib Figure.
    """
    # Determine which channels to plot
    target_channels = ['C3', 'C4']
    available = [ch for ch in target_channels if ch in channel_names]
    if not available:
        # Fall back to first two channels
        available = channel_names[:2]

    bands = list(model_erd.keys())
    n_bands = len(bands)
    n_channels = len(available)

    if figsize is None:
        figsize = (7 * n_channels, 4 * n_bands)

    fig, axes = plt.subplots(n_bands, n_channels * 2, figsize=figsize, squeeze=False)

    class_colors = {'0': '#3498DB', '1': '#E74C3C'}
    class_labels = {'0': 'Left MI (class 0)', '1': 'Right MI (class 1)'}

    for bi, band_name in enumerate(bands):
        for ci, ch_name in enumerate(available):
            # Model-derived column
            ax_model = axes[bi, ci * 2]
            for cls_key in sorted(model_erd[band_name].keys()):
                if ch_name in model_erd[band_name][cls_key]:
                    entry = model_erd[band_name][cls_key][ch_name]
                    ax_model.plot(
                        entry['times'], entry['power'],
                        linewidth=1.5,
                        color=class_colors.get(cls_key, 'gray'),
                        label=class_labels.get(cls_key, f'Class {cls_key}'),
                    )
            ax_model.axvline(0, color='black', linestyle='--', alpha=0.5, label='Cue')
            ax_model.set_ylabel('Power (a.u.)', fontsize=9)
            ax_model.set_title(
                f'{ch_name} — Model ({band_name})',
                fontsize=10, fontweight='bold',
            )
            ax_model.legend(fontsize=7, loc='upper right')
            ax_model.grid(True, alpha=0.3)

            # Classical column
            ax_classical = axes[bi, ci * 2 + 1]
            for cls_key in sorted(classical_erd[band_name].keys()):
                if ch_name in classical_erd[band_name][cls_key]:
                    entry = classical_erd[band_name][cls_key][ch_name]
                    ax_classical.plot(
                        entry['times'], entry['power'],
                        linewidth=1.5,
                        color=class_colors.get(cls_key, 'gray'),
                        label=class_labels.get(cls_key, f'Class {cls_key}'),
                    )
            ax_classical.axvline(0, color='black', linestyle='--', alpha=0.5, label='Cue')
            ax_classical.set_ylabel('ERD/ERS (%)', fontsize=9)
            title_str = f'{ch_name} — Classical ({band_name})'
            if correlations and band_name in correlations:
                for cls_key in correlations[band_name]:
                    if ch_name in correlations[band_name][cls_key]:
                        r = correlations[band_name][cls_key][ch_name]
                        title_str += f'\nr={r:.3f}'
                        break
            ax_classical.set_title(title_str, fontsize=10, fontweight='bold')
            ax_classical.legend(fontsize=7, loc='upper right')
            ax_classical.grid(True, alpha=0.3)

        # Label x-axis only on bottom row
        if bi == n_bands - 1:
            for col_idx in range(n_channels * 2):
                axes[bi, col_idx].set_xlabel('Time (s)', fontsize=10)

    plt.suptitle(
        'Model-Derived vs Classical ERD/ERS',
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_erd_ers_curves(
    erd_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    channel_name: str,
    bands: Optional[List[str]] = None,
    cue_onset: float = 0.0,
    save_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    ERD/ERS timecourse for one channel, overlaying classes per band.

    Creates one subplot per frequency band, each showing the power
    timecourse for all classes at the specified channel.

    Args:
        erd_data: Output of classical_erd_ers() or model_derived_erd_ers().
        channel_name: Channel to plot (e.g. 'C3').
        bands: List of band names to plot. Defaults to all bands in erd_data.
        cue_onset: Cue onset time in seconds.
        save_path: Optional path to save figure.
        figsize: Figure size. Auto-computed if None.

    Returns:
        Matplotlib Figure.
    """
    if bands is None:
        bands = list(erd_data.keys())

    n_bands = len(bands)
    if figsize is None:
        figsize = (12, 4 * n_bands)

    fig, axes = plt.subplots(n_bands, 1, figsize=figsize, sharex=True)
    if n_bands == 1:
        axes = [axes]

    class_colors = {'0': '#3498DB', '1': '#E74C3C'}
    class_labels = {'0': 'Left MI (class 0)', '1': 'Right MI (class 1)'}

    for bi, band_name in enumerate(bands):
        ax = axes[bi]

        if band_name not in erd_data:
            ax.set_title(f'{band_name} — not available', fontsize=11)
            continue

        for cls_key in sorted(erd_data[band_name].keys()):
            if channel_name not in erd_data[band_name][cls_key]:
                continue
            entry = erd_data[band_name][cls_key][channel_name]
            ax.plot(
                entry['times'], entry['power'],
                linewidth=2,
                color=class_colors.get(cls_key, 'gray'),
                label=class_labels.get(cls_key, f'Class {cls_key}'),
            )

        ax.axvline(cue_onset, color='red', linestyle='--', alpha=0.7, label='Cue Onset')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_ylabel('Power', fontsize=10)
        ax.set_title(
            f'{channel_name} — {band_name} band',
            fontsize=11, fontweight='bold',
        )
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=11)

    plt.suptitle(
        f'ERD/ERS Curves — {channel_name}',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig
