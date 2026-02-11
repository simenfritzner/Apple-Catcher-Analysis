"""
Publication-quality figure generation and statistical testing for EEG interpretability.

Provides topographic saliency maps, temporal importance figures, filter galleries,
summary tables, and statistical tests (permutation tests, bootstrap CIs) for
the Phase 8 analysis of the Apple-Catcher EEGNet model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import mne
import torch
import torch.nn as nn

from .saliency import SaliencyMapGenerator
from .filters import FilterVisualizer, FREQ_BANDS, BAND_COLORS, _shade_frequency_band


# ── Publication rcParams ──────────────────────────────────────────────────
_PUB_RCPARAMS: Dict[str, object] = {
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
}


# ── Dataclass for statistical results ─────────────────────────────────────

@dataclass
class PermutationTestResult:
    """Result from a permutation test."""
    observed_difference: float
    p_value: float
    null_distribution: np.ndarray


@dataclass
class BootstrapCIResult:
    """Result from bootstrap confidence interval estimation."""
    means: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray


# ── Helper: create MNE info from channel names ───────────────────────────

def _make_mne_info(
    channel_names: List[str],
    sfreq: float = 250.0,
) -> mne.Info:
    """Create an MNE Info object with a standard 10-20 montage."""
    info = mne.create_info(ch_names=list(channel_names), sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='warn')
    return info


# ═══════════════════════════════════════════════════════════════════════════
# PublicationFigureGenerator
# ═══════════════════════════════════════════════════════════════════════════

class PublicationFigureGenerator:
    """
    Generate publication-quality figures for the EEGNet interpretability analysis.

    Combines saliency analysis, filter visualization, and MNE topographic
    mapping into cohesive multi-panel figures suitable for thesis/paper
    inclusion.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size

        self.saliency_gen = SaliencyMapGenerator(model, device=device)
        self.filter_viz = FilterVisualizer()

    # ── 8.1  Topographic saliency maps ────────────────────────────────────

    def generate_topographic_maps(
        self,
        x: torch.Tensor,
        channel_names: List[str],
        sfreq: float = 250.0,
        tmin: float = -1.0,
        time_points: Optional[List[float]] = None,
        method: str = 'integrated_gradients',
    ) -> plt.Figure:
        """
        MNE-style topographic maps of channel saliency at key time points.

        Args:
            x: Input tensor (N, 1, channels, samples).
            channel_names: EEG channel names matching the montage.
            sfreq: Sampling frequency in Hz.
            tmin: Epoch start time in seconds.
            time_points: Times (s) at which to plot topomaps.
                Defaults to [-0.5, 0.0, 0.5, 1.0].
            method: Attribution method forwarded to SaliencyMapGenerator.

        Returns:
            Matplotlib Figure with one topomap per requested time point.
        """
        if time_points is None:
            time_points = [-0.5, 0.0, 0.5, 1.0]

        with plt.rc_context(_PUB_RCPARAMS):
            # Compute saliency map — shape (N, channels, samples)
            saliency_func = {
                'vanilla': self.saliency_gen.vanilla_gradient,
                'integrated_gradients': self.saliency_gen.integrated_gradients,
                'gradient_x_input': self.saliency_gen.gradient_x_input,
            }
            if method not in saliency_func:
                raise ValueError(f"Unknown saliency method: {method}")
            saliency_map = saliency_func[method](x)

            # Average across trials: (channels, samples)
            avg_saliency = saliency_map.mean(axis=0)

            # Build time vector
            n_samples = avg_saliency.shape[1]
            times = np.arange(n_samples) / sfreq + tmin

            info = _make_mne_info(channel_names, sfreq)

            n_tp = len(time_points)
            fig, axes = plt.subplots(1, n_tp, figsize=(4 * n_tp, 4))
            if n_tp == 1:
                axes = [axes]

            # Compute global vmax for consistent colour scale
            channel_at_tps = []
            for tp in time_points:
                idx = int(np.argmin(np.abs(times - tp)))
                # Use a small window around the time point for stability
                half_win = max(1, int(0.05 * sfreq))
                start = max(0, idx - half_win)
                end = min(n_samples, idx + half_win + 1)
                channel_at_tps.append(avg_saliency[:, start:end].mean(axis=1))
            vmax = max(c.max() for c in channel_at_tps)

            for ax, tp, ch_vals in zip(axes, time_points, channel_at_tps):
                mne.viz.plot_topomap(
                    ch_vals,
                    info,
                    axes=ax,
                    show=False,
                    cmap='hot',
                    vlim=(0, vmax),
                    contours=6,
                )
                label = 'Cue onset' if tp == 0.0 else f'{tp:+.1f} s'
                ax.set_title(label, fontsize=14, fontweight='bold')

            fig.suptitle(
                'Topographic Saliency Maps',
                fontsize=16,
                fontweight='bold',
                y=1.04,
            )
            fig.tight_layout()

        return fig

    # ── 8.2  Temporal importance figure ───────────────────────────────────

    def generate_temporal_importance_figure(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        channel_names: List[str],
        sfreq: float = 250.0,
        tmin: float = -1.0,
        cue_onset: float = 0.0,
    ) -> plt.Figure:
        """
        Multi-panel publication figure for temporal importance.

        Top panel: temporal importance curve with pre-cue (blue) and
        post-cue (green) shading and cue onset marker.
        Bottom panel: time x channel saliency heatmap.

        Args:
            x: Input tensor (N, 1, channels, samples).
            y: Labels tensor (N,).
            channel_names: Channel name list.
            sfreq: Sampling frequency.
            tmin: Epoch start time.
            cue_onset: Cue onset time in seconds.

        Returns:
            Matplotlib Figure.
        """
        with plt.rc_context(_PUB_RCPARAMS):
            saliency_map = self.saliency_gen.integrated_gradients(x)
            times, temporal_importance = self.saliency_gen.compute_temporal_importance(
                saliency_map, sfreq, tmin
            )
            _, importance_map, _ = self.saliency_gen.compute_time_channel_map(
                saliency_map, sfreq, tmin
            )

            cue_idx_time = cue_onset

            fig = plt.figure(figsize=(14, 9))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.30)

            # ── Top panel: temporal importance curve ──
            ax_top = fig.add_subplot(gs[0])

            ax_top.plot(times, temporal_importance, linewidth=2, color='black', zorder=3)

            # Pre-cue shading
            pre_mask = times < cue_idx_time
            ax_top.fill_between(
                times[pre_mask], 0, temporal_importance[pre_mask],
                alpha=0.30, color='#3498DB', label='Pre-cue',
            )
            # Post-cue shading
            post_mask = times >= cue_idx_time
            ax_top.fill_between(
                times[post_mask], 0, temporal_importance[post_mask],
                alpha=0.30, color='#2ECC71', label='Post-cue',
            )
            ax_top.axvline(
                cue_idx_time, color='red', linestyle='--', linewidth=1.5,
                label='Cue onset', zorder=4,
            )

            ax_top.set_ylabel('Mean Saliency (a.u.)')
            ax_top.set_title('Temporal Importance Profile', fontsize=14, fontweight='bold')
            ax_top.legend(loc='upper right', framealpha=0.9)
            ax_top.grid(True, alpha=0.25)
            ax_top.set_xlim(times[0], times[-1])

            # ── Bottom panel: time x channel heatmap ──
            ax_bot = fig.add_subplot(gs[1])

            im = ax_bot.imshow(
                importance_map,
                aspect='auto',
                cmap='hot',
                interpolation='bilinear',
                extent=[times[0], times[-1], len(channel_names), 0],
                vmin=0,
            )
            ax_bot.axvline(
                cue_idx_time, color='cyan', linestyle='--', linewidth=1.5,
                label='Cue onset',
            )

            # Y-axis channel labels
            tick_step = max(1, len(channel_names) // 16)
            tick_positions = np.arange(0, len(channel_names), tick_step)
            ax_bot.set_yticks(tick_positions + 0.5)
            ax_bot.set_yticklabels([channel_names[i] for i in tick_positions], fontsize=9)

            ax_bot.set_xlabel('Time (s)')
            ax_bot.set_ylabel('Channel')
            ax_bot.set_title('Time x Channel Saliency', fontsize=14, fontweight='bold')
            ax_bot.legend(loc='upper right', framealpha=0.9)

            cbar = fig.colorbar(im, ax=ax_bot, fraction=0.025, pad=0.02)
            cbar.set_label('Importance', fontsize=11)

            fig.suptitle(
                'EEGNet Temporal Importance Analysis',
                fontsize=16, fontweight='bold', y=1.01,
            )

        return fig

    # ── 8.3  Filter gallery ───────────────────────────────────────────────

    def generate_filter_gallery(
        self,
        channel_names: List[str],
        sfreq: float = 250.0,
    ) -> plt.Figure:
        """
        Comprehensive filter gallery: temporal waveforms, frequency responses, spatial topomaps.

        Row 1: Temporal filter kernels in the time domain.
        Row 2: FFT magnitude with EEG band shading.
        Row 3: Spatial filter topographic maps (one per spatial filter, first 8 shown).

        Args:
            channel_names: Channel names for topomap montage.
            sfreq: Sampling frequency.

        Returns:
            Matplotlib Figure.
        """
        with plt.rc_context(_PUB_RCPARAMS):
            temporal_filters = self.filter_viz.extract_temporal_filters(self.model)
            spatial_filters = self.filter_viz.extract_spatial_filters(self.model)

            n_temp = temporal_filters.shape[0]  # F1 = 8
            n_spat = min(spatial_filters.shape[0], 8)  # show at most 8 spatial
            n_cols = max(n_temp, n_spat)

            fig = plt.figure(figsize=(2.5 * n_cols, 10))
            gs = gridspec.GridSpec(3, n_cols, hspace=0.45, wspace=0.35)

            kernel_length = temporal_filters.shape[1]
            t_ms = np.arange(kernel_length) / sfreq * 1000
            freqs = np.fft.rfftfreq(kernel_length, d=1.0 / sfreq)

            info = _make_mne_info(channel_names, sfreq)

            # ── Row 1: temporal waveforms ──
            for i in range(n_temp):
                ax = fig.add_subplot(gs[0, i])
                ax.plot(t_ms, temporal_filters[i], linewidth=1.2, color='steelblue')
                ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
                ax.set_title(f'T{i+1}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.2)
                if i == 0:
                    ax.set_ylabel('Amplitude', fontsize=10)
                if i == n_temp // 2:
                    ax.set_xlabel('Time (ms)', fontsize=10)
                ax.tick_params(labelsize=8)

            # ── Row 2: frequency responses ──
            for i in range(n_temp):
                ax = fig.add_subplot(gs[1, i])
                fft_mag = np.abs(np.fft.rfft(temporal_filters[i]))
                ax.plot(freqs, fft_mag, linewidth=1.2, color='coral')
                _shade_frequency_band(ax, 'mu', freqs, alpha=0.20)
                _shade_frequency_band(ax, 'beta', freqs, alpha=0.20)
                ax.set_xlim(0, sfreq / 2)
                ax.set_title(f'T{i+1} FFT', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.2)
                if i == 0:
                    ax.set_ylabel('Magnitude', fontsize=10)
                if i == n_temp // 2:
                    ax.set_xlabel('Frequency (Hz)', fontsize=10)
                ax.tick_params(labelsize=8)

            # ── Row 3: spatial topomaps ──
            # Compute global vmax across spatial filters for consistent scale
            spat_vmax = np.abs(spatial_filters[:n_spat]).max()
            for i in range(n_spat):
                ax = fig.add_subplot(gs[2, i])
                mne.viz.plot_topomap(
                    spatial_filters[i],
                    info,
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(-spat_vmax, spat_vmax),
                    contours=4,
                )
                ax.set_title(f'S{i+1}', fontsize=11, fontweight='bold')

            # Hide unused axes in row 3 if n_spat < n_cols
            for i in range(n_spat, n_cols):
                ax = fig.add_subplot(gs[2, i])
                ax.set_visible(False)

            fig.suptitle(
                'Learned EEGNet Filters',
                fontsize=16, fontweight='bold', y=1.01,
            )

        return fig

    # ── 8.4  Summary table ────────────────────────────────────────────────

    def generate_summary_table(
        self,
        analysis_results: dict,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Render a summary metrics table as a matplotlib figure.

        Args:
            analysis_results: Dictionary with keys such as
                'pre_cue_importance', 'post_cue_importance',
                'importance_ratio', 'critical_window',
                'top_channels', 'frontal_central_ratio',
                'baseline_accuracy', 'pre_cue_accuracy_drop',
                'post_cue_accuracy_drop', 'permutation_p_value',
                'bootstrap_ci'.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib Figure containing the rendered table.
        """
        with plt.rc_context(_PUB_RCPARAMS):
            # Build rows from available results
            rows: List[Tuple[str, str]] = []

            def _fmt(key: str, label: str, fmt_str: str = '.4f') -> None:
                val = analysis_results.get(key)
                if val is not None:
                    if isinstance(val, (list, tuple)):
                        rows.append((label, ', '.join(str(v) for v in val)))
                    elif isinstance(val, str):
                        rows.append((label, val))
                    else:
                        rows.append((label, f'{val:{fmt_str}}'))

            _fmt('baseline_accuracy', 'Baseline Accuracy', '.3f')
            _fmt('pre_cue_importance', 'Pre-cue Importance', '.6f')
            _fmt('post_cue_importance', 'Post-cue Importance', '.6f')
            _fmt('importance_ratio', 'Pre/Post Importance Ratio', '.3f')
            _fmt('pre_cue_accuracy_drop', 'Pre-cue Accuracy Drop', '.4f')
            _fmt('post_cue_accuracy_drop', 'Post-cue Accuracy Drop', '.4f')
            _fmt('critical_window', 'Critical Time Window (s)')
            _fmt('top_channels', 'Top Channels')
            _fmt('frontal_central_ratio', 'Frontal/Central Ratio', '.3f')
            _fmt('permutation_p_value', 'Permutation p-value (pre vs post)', '.4f')
            _fmt('bootstrap_ci', 'Bootstrap 95% CI (pre-post diff)')

            if not rows:
                rows.append(('(no results)', ''))

            col_labels = ['Metric', 'Value']
            cell_text = [[r[0], r[1]] for r in rows]

            fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.5))
            ax.axis('off')

            table = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                loc='center',
                cellLoc='left',
                colWidths=[0.50, 0.50],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.0, 1.6)

            # Style header
            for j in range(2):
                cell = table[0, j]
                cell.set_facecolor('#2C3E50')
                cell.set_text_props(color='white', fontweight='bold')

            # Alternate row colours
            for i in range(1, len(rows) + 1):
                for j in range(2):
                    cell = table[i, j]
                    cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')

            ax.set_title(
                'Analysis Summary',
                fontsize=16, fontweight='bold', pad=20,
            )

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved summary table to {save_path}")

        return fig


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Tests  (module-level functions)
# ═══════════════════════════════════════════════════════════════════════════

def permutation_test_pre_vs_post(
    saliency_map: np.ndarray,
    cue_idx: int,
    n_permutations: int = 10_000,
) -> PermutationTestResult:
    """
    Permutation test for the difference in mean saliency pre- vs post-cue.

    H0: The mean importance before and after the cue sample index is equal.
    The time labels are shuffled within each trial to build the null
    distribution.

    Args:
        saliency_map: Array of shape (n_trials, channels, samples).
        cue_idx: Sample index of cue onset.
        n_permutations: Number of random permutations.

    Returns:
        PermutationTestResult with observed difference, p-value, and null
        distribution array.
    """
    # Collapse channels: (n_trials, samples)
    trial_temporal = saliency_map.mean(axis=1)
    n_trials, n_samples = trial_temporal.shape

    pre_mean = trial_temporal[:, :cue_idx].mean()
    post_mean = trial_temporal[:, cue_idx:].mean()
    observed = float(pre_mean - post_mean)

    rng = np.random.default_rng(seed=42)
    null_dist = np.empty(n_permutations, dtype=np.float64)

    for i in range(n_permutations):
        # Shuffle time indices independently per trial
        shuffled = trial_temporal.copy()
        for t in range(n_trials):
            rng.shuffle(shuffled[t])
        null_pre = shuffled[:, :cue_idx].mean()
        null_post = shuffled[:, cue_idx:].mean()
        null_dist[i] = null_pre - null_post

    # Two-sided p-value
    p_value = float((np.abs(null_dist) >= np.abs(observed)).sum() + 1) / (n_permutations + 1)

    return PermutationTestResult(
        observed_difference=observed,
        p_value=p_value,
        null_distribution=null_dist,
    )


def bootstrap_channel_importance(
    saliency_map: np.ndarray,
    n_bootstrap: int = 5_000,
    ci: float = 0.95,
) -> BootstrapCIResult:
    """
    Bootstrap confidence intervals for per-channel importance rankings.

    Resamples trials with replacement and recomputes channel importance
    to estimate variability.

    Args:
        saliency_map: Array of shape (n_trials, channels, samples).
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level (e.g. 0.95 for 95% CI).

    Returns:
        BootstrapCIResult with per-channel mean, lower CI, and upper CI.
    """
    n_trials, n_channels, _ = saliency_map.shape
    rng = np.random.default_rng(seed=42)

    # Channel importance per bootstrap sample
    boot_importance = np.empty((n_bootstrap, n_channels), dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_trials, size=n_trials)
        boot_importance[b] = saliency_map[idx].mean(axis=(0, 2))

    alpha = (1.0 - ci) / 2.0
    means = boot_importance.mean(axis=0)
    lower = np.quantile(boot_importance, alpha, axis=0)
    upper = np.quantile(boot_importance, 1.0 - alpha, axis=0)

    return BootstrapCIResult(means=means, lower_ci=lower, upper_ci=upper)


def bootstrap_temporal_difference(
    saliency_map: np.ndarray,
    cue_idx: int,
    n_bootstrap: int = 5_000,
    ci: float = 0.95,
) -> BootstrapCIResult:
    """
    Bootstrap CI for the pre-cue minus post-cue mean importance difference.

    Args:
        saliency_map: Array of shape (n_trials, channels, samples).
        cue_idx: Sample index of cue onset.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level.

    Returns:
        BootstrapCIResult where means/lower_ci/upper_ci are scalar arrays
        (length 1) representing the pre-post difference statistic.
    """
    n_trials = saliency_map.shape[0]
    rng = np.random.default_rng(seed=42)

    # Temporal profile per trial: collapse channels
    trial_temporal = saliency_map.mean(axis=1)  # (n_trials, samples)

    boot_diffs = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_trials, size=n_trials)
        sample = trial_temporal[idx]
        boot_diffs[b] = sample[:, :cue_idx].mean() - sample[:, cue_idx:].mean()

    alpha = (1.0 - ci) / 2.0
    mean_diff = np.array([boot_diffs.mean()])
    lower = np.array([np.quantile(boot_diffs, alpha)])
    upper = np.array([np.quantile(boot_diffs, 1.0 - alpha)])

    return BootstrapCIResult(means=mean_diff, lower_ci=lower, upper_ci=upper)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers for statistical results
# ═══════════════════════════════════════════════════════════════════════════

def plot_permutation_test(
    observed: float,
    null_distribution: np.ndarray,
    p_value: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Histogram of the null distribution with the observed statistic marked.

    Args:
        observed: Observed test statistic.
        null_distribution: Array of null-distribution values.
        p_value: p-value from the permutation test.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    with plt.rc_context(_PUB_RCPARAMS):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            null_distribution, bins=80, density=True, alpha=0.7,
            color='#95A5A6', edgecolor='white', linewidth=0.5,
            label='Null distribution',
        )
        ax.axvline(
            observed, color='#E74C3C', linewidth=2.5, linestyle='--',
            label=f'Observed = {observed:.4f}',
        )
        # Mirror line for two-sided
        ax.axvline(
            -observed, color='#E74C3C', linewidth=1.5, linestyle=':',
            alpha=0.5,
        )

        ax.set_xlabel('Pre-cue minus Post-cue Importance')
        ax.set_ylabel('Density')
        ax.set_title(
            f'Permutation Test  (p = {p_value:.4f})',
            fontsize=14, fontweight='bold',
        )
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved permutation test figure to {save_path}")

    return fig


def plot_bootstrap_ci(
    means: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    channel_names: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of channel importance with bootstrap CI error bars.

    Args:
        means: Per-channel mean importance.
        lower_ci: Lower CI bound per channel.
        upper_ci: Upper CI bound per channel.
        channel_names: Channel name labels.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    with plt.rc_context(_PUB_RCPARAMS):
        # Sort by mean importance (descending)
        order = np.argsort(means)[::-1]
        sorted_means = means[order]
        sorted_lower = lower_ci[order]
        sorted_upper = upper_ci[order]
        sorted_names = [channel_names[i] for i in order]

        xerr_lower = sorted_means - sorted_lower
        xerr_upper = sorted_upper - sorted_means
        xerr = np.array([xerr_lower, xerr_upper])

        n_ch = len(sorted_names)
        fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * n_ch)))

        y_pos = np.arange(n_ch)
        colors = plt.cm.viridis(np.linspace(0.3, 0.85, n_ch))

        ax.barh(
            y_pos, sorted_means, xerr=xerr, color=colors,
            edgecolor='black', alpha=0.85, capsize=3,
            error_kw={'linewidth': 1.0, 'color': '#2C3E50'},
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mean Importance')
        ax.set_title(
            'Channel Importance with 95% Bootstrap CI',
            fontsize=14, fontweight='bold',
        )
        ax.grid(True, alpha=0.25, axis='x')

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved bootstrap CI figure to {save_path}")

    return fig
