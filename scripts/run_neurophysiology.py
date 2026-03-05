"""
Standalone neurophysiological characterization of raw EEG (thesis §5.1, RQ1).

Model-independent ground truth: ERDS time-frequency maps, MRCP waveforms,
frontal lateralization index, and per-subject ERD magnitudes for correlation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import ttest_ind

DATA_DIR = project_root / "data" / "raw" / "apple_catcher"
OUT_DIR = project_root / "results" / "interpretability" / "neurophysiology"
TMIN, TMAX = -1.0, 2.0
SFREQ = 250.0
N_SAMPLES = int((TMAX - TMIN) * SFREQ)
ALL_SUBJECTS = [f"s{i:02d}" for i in range(1, 41)]
CLASS_NAMES = {0: 'MI_left', 1: 'MI_right'}


def load_all_epochs() -> tuple[dict[str, mne.Epochs], list[str]]:
    """Load epochs per subject, return dict and channel names."""
    mne.set_log_level("ERROR")
    subject_epochs: dict[str, mne.Epochs] = {}
    ch_names = None

    for sid in ALL_SUBJECTS:
        subj_dir = DATA_DIR / sid
        fif_files = sorted(subj_dir.glob("*_epo.fif"))
        epoch_list = []
        for fif_path in fif_files:
            ep = mne.read_epochs(fif_path, preload=True)
            ep.crop(tmin=TMIN, tmax=TMAX)
            epoch_list.append(ep)
        if epoch_list:
            combined = mne.concatenate_epochs(epoch_list)
            subject_epochs[sid] = combined
            if ch_names is None:
                ch_names = combined.ch_names
            print(f"  {sid}: {len(combined)} epochs")

    return subject_epochs, ch_names


def get_class_data(
    subject_epochs: dict[str, mne.Epochs],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split all data into left MI and right MI arrays."""
    left_data, right_data = [], []
    for ep in subject_epochs.values():
        data = ep.get_data()[:, :, :N_SAMPLES]
        labels = ep.events[:, -1]
        left_data.append(data[labels == 0])
        right_data.append(data[labels == 1])
    return np.concatenate(left_data), np.concatenate(right_data)


def run_erds_analysis(
    subject_epochs: dict[str, mne.Epochs],
    ch_names: list[str],
) -> None:
    """§5.1.1: Grand average ERDS time-frequency maps at C3/C4."""
    print("\n── §5.1.1: ERDS Time-Frequency Maps ──")

    left_data, right_data = get_class_data(subject_epochs)
    times = np.arange(N_SAMPLES) / SFREQ + TMIN

    c3_idx = ch_names.index('C3')
    c4_idx = ch_names.index('C4')

    freqs = np.arange(2, 36, 1.0)
    baseline_start, baseline_end = -1.0, -0.5
    bl_s = int((baseline_start - TMIN) * SFREQ)
    bl_e = int((baseline_end - TMIN) * SFREQ)

    for cls_name, cls_data in [('MI_left', left_data), ('MI_right', right_data)]:
        for ch_name, ch_idx in [('C3', c3_idx), ('C4', c4_idx)]:
            signal = cls_data[:, ch_idx, :]  # (n_trials, n_samples)

            # Morlet wavelet TF decomposition
            n_trials, n_samp = signal.shape
            tf_power = np.zeros((len(freqs), n_samp))

            for fi, freq in enumerate(freqs):
                n_cycles = freq / 2.0
                sigma_t = n_cycles / (2 * np.pi * freq)
                t_wavelet = np.arange(-3 * sigma_t, 3 * sigma_t, 1.0 / SFREQ)
                wavelet = np.exp(2j * np.pi * freq * t_wavelet) * np.exp(-t_wavelet**2 / (2 * sigma_t**2))

                for trial in range(n_trials):
                    analytic = np.convolve(signal[trial], wavelet, mode='same')
                    tf_power[fi] += np.abs(analytic)**2

                tf_power[fi] /= n_trials

            # ERDS: (power - baseline) / baseline * 100
            baseline_power = tf_power[:, bl_s:bl_e].mean(axis=1, keepdims=True)
            erds = (tf_power - baseline_power) / (baseline_power + 1e-12) * 100

            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            vmax = 100
            im = ax.pcolormesh(times, freqs, erds, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax, shading='auto')
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Cue onset')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Frequency (Hz)', fontsize=12)
            ax.set_title(f'ERDS — {ch_name}, {cls_name} (N={cls_data.shape[0]})',
                         fontsize=13, fontweight='bold')

            # Band annotations
            ax.axhline(8, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(13, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(30, color='gray', linestyle=':', alpha=0.5)
            ax.text(TMAX - 0.05, 10.5, 'mu', ha='right', fontsize=9, color='gray')
            ax.text(TMAX - 0.05, 21, 'beta', ha='right', fontsize=9, color='gray')

            plt.colorbar(im, ax=ax, label='ERDS (%)')
            ax.legend(loc='upper right')
            plt.tight_layout()
            fig.savefig(OUT_DIR / f'erds_tf_{ch_name}_{cls_name}.png', dpi=200, bbox_inches='tight')
            plt.close(fig)

    print("  Saved ERDS TF maps for C3/C4 × left/right")


def run_erds_topography(
    subject_epochs: dict[str, mne.Epochs],
    ch_names: list[str],
) -> None:
    """§5.1.2: ERDS topographic maps in mu and beta bands."""
    print("\n── §5.1.2: ERDS Topographic Maps ──")

    left_data, right_data = get_class_data(subject_epochs)

    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=list(ch_names), sfreq=SFREQ, ch_types='eeg')
    info.set_montage(montage, on_missing='warn')

    baseline_start, baseline_end = -1.0, -0.5
    post_start, post_end = 0.5, 1.5
    bl_s = int((baseline_start - TMIN) * SFREQ)
    bl_e = int((baseline_end - TMIN) * SFREQ)
    post_s = int((post_start - TMIN) * SFREQ)
    post_e = int((post_end - TMIN) * SFREQ)

    bands = {'mu (8-13 Hz)': (8, 13), 'beta (13-30 Hz)': (13, 30)}

    fig, axes = plt.subplots(2, len(bands) * 2, figsize=(5 * len(bands) * 2, 10))

    col = 0
    for band_name, (flo, fhi) in bands.items():
        for cls_name, cls_data in [('MI_left', left_data), ('MI_right', right_data)]:
            n_ch = cls_data.shape[1]
            erds_per_ch = np.zeros(n_ch)

            nyq = SFREQ / 2.0
            sos = butter(4, [flo / nyq, fhi / nyq], btype='band', output='sos')

            for ch in range(n_ch):
                filtered = sosfiltfilt(sos, cls_data[:, ch, :], axis=-1)
                power = filtered**2

                bl_power = power[:, bl_s:bl_e].mean()
                post_power = power[:, post_s:post_e].mean()
                erds_per_ch[ch] = (post_power - bl_power) / (bl_power + 1e-12) * 100

            ax = axes[0, col] if len(axes.shape) > 1 else axes[col]
            vmax_topo = max(abs(erds_per_ch.min()), abs(erds_per_ch.max())) * 0.8
            mne.viz.plot_topomap(
                erds_per_ch, info, axes=ax, show=False,
                cmap='RdBu_r', vlim=(-vmax_topo, vmax_topo), contours=4,
            )
            ax.set_title(f'{band_name}\n{cls_name}', fontsize=10, fontweight='bold')
            col += 1

    plt.suptitle('ERDS Topography (0.5–1.5s post-cue)', fontsize=14, fontweight='bold', y=1.02)
    # Remove unused bottom row
    if len(axes.shape) > 1:
        for ax in axes[1, :]:
            ax.set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'erds_topography.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved ERDS topography")


def run_mrcp_analysis(
    subject_epochs: dict[str, mne.Epochs],
    ch_names: list[str],
) -> None:
    """§5.1.3: MRCP waveforms at Cz, C3, C4."""
    print("\n── §5.1.3: MRCP Waveforms ──")

    left_data, right_data = get_class_data(subject_epochs)
    times = np.arange(N_SAMPLES) / SFREQ + TMIN

    # Low-pass at 5 Hz for MRCP
    nyq = SFREQ / 2.0
    sos = butter(4, 5.0 / nyq, btype='low', output='sos')

    bl_s = int((-1.0 - TMIN) * SFREQ)
    bl_e = int((-0.5 - TMIN) * SFREQ)

    channels = ['Cz', 'C3', 'C4']

    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 4 * len(channels)), sharex=True)

    for i, ch_name in enumerate(channels):
        ch_idx = ch_names.index(ch_name)
        ax = axes[i]

        for cls_name, cls_data, color in [
            ('MI_left', left_data, '#E74C3C'),
            ('MI_right', right_data, '#3498DB'),
        ]:
            signal = cls_data[:, ch_idx, :]
            filtered = sosfiltfilt(sos, signal, axis=-1)

            # Baseline correction
            bl_mean = filtered[:, bl_s:bl_e].mean(axis=1, keepdims=True)
            corrected = filtered - bl_mean

            mean_erp = corrected.mean(axis=0)
            sem = corrected.std(axis=0) / np.sqrt(len(corrected))

            # Scale to µV (assuming data is in V)
            scale = 1e6
            ax.plot(times, mean_erp * scale, linewidth=2, color=color, label=cls_name)
            ax.fill_between(times, (mean_erp - sem) * scale, (mean_erp + sem) * scale,
                            alpha=0.2, color=color)

        ax.axvline(0, color='black', linestyle='--', alpha=0.7, label='Cue onset')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_ylabel(f'{ch_name} (µV)', fontsize=11)
        ax.set_title(f'MRCP — {ch_name} (low-pass 5 Hz)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # EEG convention: negative up

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'mrcp_waveforms.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved MRCP waveforms")


def run_frontal_lateralization(
    subject_epochs: dict[str, mne.Epochs],
    ch_names: list[str],
) -> None:
    """§5.1.4: Frontal Lateralization Index (FLI)."""
    print("\n── §5.1.4: Frontal Lateralization Index ──")

    # No Fp1/Fp2 — use F3/F1 (left) vs F4/F2 (right)
    left_frontal = [ch_names.index(ch) for ch in ['F3', 'F1'] if ch in ch_names]
    right_frontal = [ch_names.index(ch) for ch in ['F4', 'F2'] if ch in ch_names]

    left_data, right_data = get_class_data(subject_epochs)

    fli_per_trial = {}
    for cls_name, cls_data in [('MI_left', left_data), ('MI_right', right_data)]:
        # Power in post-cue window
        post_s = int((0.0 - TMIN) * SFREQ)
        post_e = int((1.5 - TMIN) * SFREQ)
        post_data = cls_data[:, :, post_s:post_e]

        left_power = (post_data[:, left_frontal, :]**2).mean(axis=(1, 2))
        right_power = (post_data[:, right_frontal, :]**2).mean(axis=(1, 2))

        fli = (right_power - left_power) / (right_power + left_power + 1e-12)
        fli_per_trial[cls_name] = fli

    # Statistical test
    t_stat, p_val = ttest_ind(fli_per_trial['MI_left'], fli_per_trial['MI_right'])
    fli_left_mean = fli_per_trial['MI_left'].mean()
    fli_right_mean = fli_per_trial['MI_right'].mean()

    print(f"  FLI MI_left:  {fli_left_mean:.4f} ± {fli_per_trial['MI_left'].std():.4f}")
    print(f"  FLI MI_right: {fli_right_mean:.4f} ± {fli_per_trial['MI_right'].std():.4f}")
    print(f"  t={t_stat:.3f}, p={p_val:.4e}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot([fli_per_trial['MI_left'], fli_per_trial['MI_right']],
                          positions=[0, 1], showmeans=True, showextrema=True)
    for i, color in enumerate(['#E74C3C', '#3498DB']):
        parts['bodies'][i].set_facecolor(color)
        parts['bodies'][i].set_alpha(0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['MI_left', 'MI_right'])
    ax.set_ylabel('Frontal Lateralization Index', fontsize=11)
    ax.set_title(f'FLI Distribution (F3,F1 vs F4,F2)\nt={t_stat:.3f}, p={p_val:.4e}',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'frontal_lateralization.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    with open(OUT_DIR / 'frontal_lateralization.txt', 'w') as f:
        f.write(f"Frontal Lateralization Index (F3,F1 vs F4,F2)\n")
        f.write(f"MI_left:  {fli_left_mean:.6f} ± {fli_per_trial['MI_left'].std():.6f}\n")
        f.write(f"MI_right: {fli_right_mean:.6f} ± {fli_per_trial['MI_right'].std():.6f}\n")
        f.write(f"t-test: t={t_stat:.4f}, p={p_val:.6e}\n")
        f.write(f"Note: No Fp1/Fp2 in montage; using F3,F1 (left) vs F4,F2 (right)\n")

    print("  Saved FLI results")


def compute_per_subject_erd(
    subject_epochs: dict[str, mne.Epochs],
    ch_names: list[str],
) -> None:
    """Compute per-subject mu-band ERD at C3/C4 for correlation analysis."""
    print("\n── Per-Subject ERD Magnitudes ──")

    c3_idx = ch_names.index('C3')
    c4_idx = ch_names.index('C4')

    nyq = SFREQ / 2.0
    sos_mu = butter(4, [8.0 / nyq, 13.0 / nyq], btype='band', output='sos')

    bl_s = int((-1.0 - TMIN) * SFREQ)
    bl_e = int((-0.5 - TMIN) * SFREQ)
    post_s = int((0.5 - TMIN) * SFREQ)
    post_e = int((1.5 - TMIN) * SFREQ)

    lines = ["Subject  ERD_C3_mu  ERD_C4_mu  ERD_mean_mu"]

    for sid in ALL_SUBJECTS:
        if sid not in subject_epochs:
            continue
        ep = subject_epochs[sid]
        data = ep.get_data()[:, :, :N_SAMPLES]

        erd_vals = {}
        for ch_name, ch_idx in [('C3', c3_idx), ('C4', c4_idx)]:
            filtered = sosfiltfilt(sos_mu, data[:, ch_idx, :], axis=-1)
            power = filtered**2
            bl_power = power[:, bl_s:bl_e].mean()
            post_power = power[:, post_s:post_e].mean()
            erd = (post_power - bl_power) / (bl_power + 1e-12) * 100
            erd_vals[ch_name] = erd

        mean_erd = (erd_vals['C3'] + erd_vals['C4']) / 2
        lines.append(f"{sid:>7}  {erd_vals['C3']:>9.2f}  {erd_vals['C4']:>9.2f}  {mean_erd:>11.2f}")
        print(f"  {sid}: C3={erd_vals['C3']:.2f}%, C4={erd_vals['C4']:.2f}%")

    with open(OUT_DIR / 'per_subject_erd.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print("  Saved per-subject ERD magnitudes")


def main() -> None:
    print("=" * 60)
    print("RAW EEG NEUROPHYSIOLOGY (thesis §5.1, RQ1)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading all subjects...")
    subject_epochs, ch_names = load_all_epochs()
    print(f"Loaded {len(subject_epochs)} subjects, channels: {ch_names}")

    run_erds_analysis(subject_epochs, ch_names)
    run_erds_topography(subject_epochs, ch_names)
    run_mrcp_analysis(subject_epochs, ch_names)
    run_frontal_lateralization(subject_epochs, ch_names)
    compute_per_subject_erd(subject_epochs, ch_names)

    print("\n" + "=" * 60)
    print("NEUROPHYSIOLOGY ANALYSIS COMPLETE")
    print(f"Results: {OUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
