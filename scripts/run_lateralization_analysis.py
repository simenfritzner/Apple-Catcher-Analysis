"""
Class-conditional lateralization analysis for the all-subjects EEGNet model.

Tests whether the model uses contralateral sensorimotor patterns for
left vs right MI classification — the hallmark of genuine motor imagery.

If genuine MI:
  - Left MI → stronger saliency at right hemisphere (C4, FC4, FC6, ...)
  - Right MI → stronger saliency at left hemisphere (C3, FC3, FC5, ...)
  - Laterality index should be positive and develop post-cue (0.5–1.5s)

If cue-evoked ERP:
  - Both classes show similar spatial distribution
  - No contralateral pattern, or lateralization locked to cue onset
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_1samp

from eeg_mi.models.eegnet import EEGNet
from eeg_mi.interpretability.saliency import SaliencyMapGenerator


# ── Configuration ─────────────────────────────────────────────────────────
MODEL_PATH = project_root / "all_subjects_interpretation" / "models" / "model.pt"
DATA_DIR = project_root / "data" / "raw" / "apple_catcher"
OUTPUT_DIR = project_root / "results" / "interpretability" / "lateralization"
TMIN, TMAX = -1.0, 2.0
SFREQ = 250.0
N_SAMPLES = int((TMAX - TMIN) * SFREQ)  # 750
CUE_ONSET = 0.0
CUE_IDX = int((CUE_ONSET - TMIN) * SFREQ)  # 250
N_CHANNELS = 32
N_CLASSES = 2

# Use all 40 subjects
ALL_SUBJECTS = [f"s{i:02d}" for i in range(1, 41)]

# Homologous left-right channel pairs (sensorimotor focus)
HOMOLOGOUS_PAIRS: list[tuple[str, str]] = [
    ('C3', 'C4'),
    ('C1', 'C2'),
    ('C5', 'C6'),
    ('FC3', 'FC4'),
    ('FC1', 'FC2'),
    ('FC5', 'FC6'),
    ('CP3', 'CP4'),
    ('CP1', 'CP2'),
    ('CP5', 'CP6'),
    ('F3', 'F4'),
    ('F1', 'F2'),
    ('P5', 'P6'),
    ('P1', 'P2'),
]

# Sensorimotor pairs only (for focused analysis)
SENSORIMOTOR_PAIRS: list[tuple[str, str]] = [
    ('C3', 'C4'),
    ('C1', 'C2'),
    ('C5', 'C6'),
    ('FC3', 'FC4'),
    ('FC1', 'FC2'),
    ('FC5', 'FC6'),
    ('CP3', 'CP4'),
    ('CP1', 'CP2'),
    ('CP5', 'CP6'),
]

CLASS_NAMES = {0: 'MI_left', 1: 'MI_right'}


def load_model() -> EEGNet:
    model = EEGNet(
        nb_classes=N_CLASSES, Chans=N_CHANNELS, Samples=N_SAMPLES,
        kernLength=64, F1=8, D=2, F2=16, dropoutRate=0.3,
    )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")
    return model


def load_subject_data(subject_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    subj_dir = DATA_DIR / subject_id
    fif_files = sorted(subj_dir.glob("*_epo.fif"))
    mne.set_log_level("ERROR")

    all_data, all_labels = [], []
    ch_names = None

    for fif_path in fif_files:
        epochs = mne.read_epochs(fif_path, preload=True)
        epochs.crop(tmin=TMIN, tmax=TMAX)
        data = epochs.get_data()[:, :, :N_SAMPLES]
        labels = epochs.events[:, -1]
        all_data.append(data)
        all_labels.append(labels)
        if ch_names is None:
            ch_names = epochs.ch_names

    return np.concatenate(all_data), np.concatenate(all_labels), ch_names


def load_all_subjects() -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_data, all_labels = [], []
    ch_names = None

    for sid in ALL_SUBJECTS:
        try:
            data, labels, names = load_subject_data(sid)
            all_data.append(data)
            all_labels.append(labels)
            if ch_names is None:
                ch_names = names
            print(f"  {sid}: {len(labels)} trials")
        except Exception as e:
            print(f"  {sid}: FAILED ({e})")

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)
    print(f"Total: {len(labels)} trials, shape={data.shape}")
    return data, labels, ch_names


def get_hemisphere_indices(
    ch_names: list[str],
    pairs: list[tuple[str, str]],
) -> tuple[list[int], list[int]]:
    """Return (left_indices, right_indices) for the given homologous pairs."""
    left_idx, right_idx = [], []
    for left_ch, right_ch in pairs:
        if left_ch in ch_names and right_ch in ch_names:
            left_idx.append(ch_names.index(left_ch))
            right_idx.append(ch_names.index(right_ch))
    return left_idx, right_idx


def compute_class_saliency(
    saliency_gen: SaliencyMapGenerator,
    data: np.ndarray,
    labels: np.ndarray,
    method: str = 'integrated_gradients',
    batch_size: int = 64,
) -> dict[int, np.ndarray]:
    """Compute saliency maps per class, batched for memory efficiency.

    Returns dict mapping class_id -> saliency array (n_trials, channels, samples).
    """
    results = {}
    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        cls_data = data[mask]
        all_sal = []

        n = len(cls_data)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_batch = torch.tensor(cls_data[start:end], dtype=torch.float32).unsqueeze(1)

            if method == 'integrated_gradients':
                sal = saliency_gen.integrated_gradients(x_batch, target_class=int(cls))
            elif method == 'deeplift':
                sal = saliency_gen.deeplift(x_batch, target_class=int(cls))
            else:
                sal = saliency_gen.vanilla_gradient(x_batch, target_class=int(cls))

            all_sal.append(sal)

        results[int(cls)] = np.concatenate(all_sal, axis=0)
        print(f"    Class {cls} ({CLASS_NAMES.get(cls, '?')}): "
              f"{results[int(cls)].shape[0]} trials, saliency shape={results[int(cls)].shape}")

    return results


def compute_laterality_index_timecourse(
    saliency: np.ndarray,
    left_idx: list[int],
    right_idx: list[int],
    class_label: int,
    window_ms: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute laterality index over time for one class.

    For left MI (class 0): contralateral = right hemisphere
      LI = (right - left) / (right + left)
    For right MI (class 1): contralateral = left hemisphere
      LI = (left - right) / (left + right)

    Positive LI = contralateral dominance (expected for genuine MI).

    Returns (times, LI_mean, LI_trials) where LI_trials has shape (n_trials, n_timepoints).
    """
    n_trials, n_channels, n_samples = saliency.shape
    times = np.arange(n_samples) / SFREQ + TMIN

    # Average across homologous channels per hemisphere
    left_sal = saliency[:, left_idx, :].mean(axis=1)   # (n_trials, n_samples)
    right_sal = saliency[:, right_idx, :].mean(axis=1)  # (n_trials, n_samples)

    # Smooth with sliding window
    win = int(window_ms / 1000 * SFREQ)
    kernel = np.ones(win) / win
    left_smooth = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, left_sal)
    right_smooth = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, right_sal)

    denom = left_smooth + right_smooth + 1e-12

    if class_label == 0:  # Left MI → contralateral = right
        li_trials = (right_smooth - left_smooth) / denom
    else:  # Right MI → contralateral = left
        li_trials = (left_smooth - right_smooth) / denom

    li_mean = li_trials.mean(axis=0)

    return times, li_mean, li_trials


def run_analysis_1_hemisphere_saliency(
    class_saliency: dict[int, np.ndarray],
    ch_names: list[str],
    out: Path,
) -> dict:
    """Analysis 1: Hemispheric saliency comparison per class."""
    print("\n── Analysis 1: Hemispheric Saliency Per Class ──")

    left_idx, right_idx = get_hemisphere_indices(ch_names, SENSORIMOTOR_PAIRS)
    midline_names = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'AFz']
    midline_idx = [ch_names.index(ch) for ch in midline_names if ch in ch_names]

    results = {}

    for cls, sal in class_saliency.items():
        left_mean = sal[:, left_idx, :].mean()
        right_mean = sal[:, right_idx, :].mean()
        midline_mean = sal[:, midline_idx, :].mean()

        # Post-cue only (where MI should occur)
        left_post = sal[:, left_idx, CUE_IDX:].mean()
        right_post = sal[:, right_idx, CUE_IDX:].mean()

        results[cls] = {
            'left_hemisphere': float(left_mean),
            'right_hemisphere': float(right_mean),
            'midline': float(midline_mean),
            'left_post_cue': float(left_post),
            'right_post_cue': float(right_post),
        }

        name = CLASS_NAMES[cls]
        expected_contra = 'right' if cls == 0 else 'left'
        print(f"  {name}:")
        print(f"    Left hemisphere:  {left_mean:.6f}  (post-cue: {left_post:.6f})")
        print(f"    Right hemisphere: {right_mean:.6f}  (post-cue: {right_post:.6f})")
        print(f"    Midline:          {midline_mean:.6f}")
        print(f"    Expected contralateral dominant: {expected_contra}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, period_label, use_post in [(0, 'Full Epoch', False), (1, 'Post-Cue Only', True)]:
        ax = axes[ax_idx]
        x_pos = np.arange(2)
        width = 0.35

        for i, cls in enumerate(sorted(results.keys())):
            r = results[cls]
            if use_post:
                vals = [r['left_post_cue'], r['right_post_cue']]
            else:
                vals = [r['left_hemisphere'], r['right_hemisphere']]

            offset = -width / 2 + i * width
            color = '#E74C3C' if cls == 0 else '#3498DB'
            bars = ax.bar(x_pos + offset, vals, width, label=CLASS_NAMES[cls],
                          color=color, alpha=0.8, edgecolor='black')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.5f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Left Hemisphere\n(C3, FC3, FC5, ...)',
                            'Right Hemisphere\n(C4, FC4, FC6, ...)'])
        ax.set_ylabel('Mean Saliency', fontsize=11)
        ax.set_title(f'Hemispheric Saliency — {period_label}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate expected pattern
        ax.annotate(
            'Genuine MI → contralateral dominance\n'
            '(Left MI → Right hemi, Right MI → Left hemi)',
            xy=(0.5, 0.02), xycoords='axes fraction', ha='center',
            fontsize=8, style='italic', color='gray',
        )

    plt.tight_layout()
    fig.savefig(out / 'hemispheric_saliency.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: hemispheric_saliency.png")

    return results


def run_analysis_2_laterality_timecourse(
    class_saliency: dict[int, np.ndarray],
    ch_names: list[str],
    out: Path,
) -> dict:
    """Analysis 2: Laterality index time course."""
    print("\n── Analysis 2: Laterality Index Over Time ──")

    left_idx, right_idx = get_hemisphere_indices(ch_names, SENSORIMOTOR_PAIRS)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    li_results = {}

    for cls in sorted(class_saliency.keys()):
        times, li_mean, li_trials = compute_laterality_index_timecourse(
            class_saliency[cls], left_idx, right_idx, cls,
        )
        li_sem = li_trials.std(axis=0) / np.sqrt(li_trials.shape[0])

        li_results[cls] = {
            'times': times,
            'li_mean': li_mean,
            'li_sem': li_sem,
            'li_trials': li_trials,
        }

        name = CLASS_NAMES[cls]
        color = '#E74C3C' if cls == 0 else '#3498DB'
        ax = axes[cls]

        ax.plot(times, li_mean, linewidth=2, color=color, label=f'{name} LI')
        ax.fill_between(times, li_mean - li_sem, li_mean + li_sem,
                         alpha=0.3, color=color)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(CUE_ONSET, color='red', linestyle='--', alpha=0.7, label='Cue onset')
        ax.set_ylabel('Laterality Index', fontsize=11)
        ax.set_title(f'{name} — Contralateral Laterality Index', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.15, 0.15)

        # Annotate
        post_cue_li = li_mean[CUE_IDX:].mean()
        pre_cue_li = li_mean[:CUE_IDX].mean()
        ax.annotate(f'Pre-cue mean LI: {pre_cue_li:.4f}\nPost-cue mean LI: {post_cue_li:.4f}',
                     xy=(0.02, 0.95), xycoords='axes fraction', va='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        print(f"  {name}: pre-cue LI={pre_cue_li:.4f}, post-cue LI={post_cue_li:.4f}")

    # Panel 3: Both classes overlaid
    ax = axes[2]
    for cls in sorted(li_results.keys()):
        color = '#E74C3C' if cls == 0 else '#3498DB'
        ax.plot(li_results[cls]['times'], li_results[cls]['li_mean'],
                linewidth=2, color=color, label=f'{CLASS_NAMES[cls]}')
        ax.fill_between(li_results[cls]['times'],
                         li_results[cls]['li_mean'] - li_results[cls]['li_sem'],
                         li_results[cls]['li_mean'] + li_results[cls]['li_sem'],
                         alpha=0.2, color=color)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(CUE_ONSET, color='red', linestyle='--', alpha=0.7, label='Cue onset')
    ax.set_ylabel('Laterality Index', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Both Classes — Contralateral Laterality Index', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.15, 0.15)
    ax.annotate('Positive = contralateral dominance (expected for MI)',
                 xy=(0.5, 0.02), xycoords='axes fraction', ha='center',
                 fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    fig.savefig(out / 'laterality_timecourse.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: laterality_timecourse.png")

    return li_results


def run_analysis_3_channel_pair_lateralization(
    class_saliency: dict[int, np.ndarray],
    ch_names: list[str],
    out: Path,
) -> None:
    """Analysis 3: Per-channel-pair lateralization (post-cue)."""
    print("\n── Analysis 3: Channel Pair Lateralization (Post-Cue) ──")

    pair_names = []
    left_minus_right_cls0 = []
    left_minus_right_cls1 = []

    for left_ch, right_ch in HOMOLOGOUS_PAIRS:
        if left_ch not in ch_names or right_ch not in ch_names:
            continue
        li = ch_names.index(left_ch)
        ri = ch_names.index(right_ch)

        pair_names.append(f'{left_ch}/{right_ch}')

        # Post-cue saliency
        for cls, storage in [(0, left_minus_right_cls0), (1, left_minus_right_cls1)]:
            sal = class_saliency[cls]
            left_val = sal[:, li, CUE_IDX:].mean()
            right_val = sal[:, ri, CUE_IDX:].mean()
            storage.append(float(left_val - right_val))

    left_minus_right_cls0 = np.array(left_minus_right_cls0)
    left_minus_right_cls1 = np.array(left_minus_right_cls1)

    # For left MI: genuine MI → negative (right > left)
    # For right MI: genuine MI → positive (left > right)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(pair_names))
    width = 0.35

    bars0 = ax.bar(x - width / 2, left_minus_right_cls0, width,
                   label='MI_left', color='#E74C3C', alpha=0.8, edgecolor='black')
    bars1 = ax.bar(x + width / 2, left_minus_right_cls1, width,
                   label='MI_right', color='#3498DB', alpha=0.8, edgecolor='black')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, rotation=45, ha='right')
    ax.set_ylabel('Left Hemi − Right Hemi Saliency (post-cue)', fontsize=11)
    ax.set_title('Per-Pair Hemispheric Difference (Post-Cue)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    ax.annotate(
        'Genuine MI pattern:\n'
        '  MI_left → negative bars (right hemi dominant)\n'
        '  MI_right → positive bars (left hemi dominant)',
        xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(out / 'channel_pair_lateralization.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: channel_pair_lateralization.png")

    # Print key pairs
    for i, pair in enumerate(pair_names):
        print(f"  {pair:>8}: MI_left={left_minus_right_cls0[i]:+.6f}  "
              f"MI_right={left_minus_right_cls1[i]:+.6f}")


def run_analysis_4_topographic_maps(
    class_saliency: dict[int, np.ndarray],
    ch_names: list[str],
    out: Path,
) -> None:
    """Analysis 4: Topographic saliency maps per class at key timepoints."""
    print("\n── Analysis 4: Class-Conditional Topographic Maps ──")

    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types='eeg')
    info.set_montage(montage, on_missing='warn')

    timepoints = [('−0.5s', -0.5), ('Cue (0s)', 0.0), ('+0.5s', 0.5), ('+1.0s', 1.0)]
    window_half = int(0.1 * SFREQ)  # ±100ms window for averaging

    fig, axes = plt.subplots(2, len(timepoints), figsize=(4 * len(timepoints), 8))

    # Find global vmax for consistent colorscale
    vmax = 0
    for cls in class_saliency:
        vmax = max(vmax, class_saliency[cls].mean(axis=0).max())

    for row, cls in enumerate(sorted(class_saliency.keys())):
        sal_mean = class_saliency[cls].mean(axis=0)  # (channels, samples)

        for col, (label, t) in enumerate(timepoints):
            t_idx = int((t - TMIN) * SFREQ)
            start = max(0, t_idx - window_half)
            end = min(sal_mean.shape[1], t_idx + window_half)
            data_at_t = sal_mean[:, start:end].mean(axis=1)

            ax = axes[row, col]
            mne.viz.plot_topomap(
                data_at_t, info, axes=ax, show=False,
                cmap='hot', vlim=(0, vmax * 0.8),
                contours=4,
            )
            if row == 0:
                ax.set_title(label, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[cls], fontsize=12, fontweight='bold')

    plt.suptitle('Class-Conditional Saliency Topography', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out / 'class_topographic_maps.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: class_topographic_maps.png")

    # Also plot the DIFFERENCE (left MI - right MI) at post-cue timepoints
    fig_diff, axes_diff = plt.subplots(1, len(timepoints), figsize=(4 * len(timepoints), 4))

    sal_left = class_saliency[0].mean(axis=0)
    sal_right = class_saliency[1].mean(axis=0)
    diff = sal_left - sal_right

    diff_max = np.abs(diff).max() * 0.8

    for col, (label, t) in enumerate(timepoints):
        t_idx = int((t - TMIN) * SFREQ)
        start = max(0, t_idx - window_half)
        end = min(diff.shape[1], t_idx + window_half)
        data_at_t = diff[:, start:end].mean(axis=1)

        ax = axes_diff[col]
        mne.viz.plot_topomap(
            data_at_t, info, axes=ax, show=False,
            cmap='RdBu_r', vlim=(-diff_max, diff_max),
            contours=4,
        )
        ax.set_title(label, fontsize=12, fontweight='bold')

    plt.suptitle('Saliency Difference (MI_left − MI_right)', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig_diff.savefig(out / 'class_difference_topography.png', dpi=300, bbox_inches='tight')
    plt.close(fig_diff)
    print(f"  Saved: class_difference_topography.png")


def run_analysis_5_statistical_tests(
    li_results: dict,
    class_saliency: dict[int, np.ndarray],
    ch_names: list[str],
    out: Path,
) -> None:
    """Analysis 5: Statistical testing of lateralization."""
    print("\n── Analysis 5: Statistical Tests ──")

    left_idx, right_idx = get_hemisphere_indices(ch_names, SENSORIMOTOR_PAIRS)
    lines = ["Lateralization Statistical Tests", "=" * 60, ""]

    for cls in sorted(li_results.keys()):
        name = CLASS_NAMES[cls]
        li_trials = li_results[cls]['li_trials']

        # Post-cue mean LI per trial
        post_cue_li_per_trial = li_trials[:, CUE_IDX:].mean(axis=1)
        n = len(post_cue_li_per_trial)
        mean_li = post_cue_li_per_trial.mean()
        std_li = post_cue_li_per_trial.std()

        # One-sample t-test: is post-cue LI significantly > 0?
        t_stat, p_val = ttest_1samp(post_cue_li_per_trial, 0)

        # Sign-flip permutation test for robustness
        rng = np.random.default_rng(42)
        n_perm = 10000
        observed_mean = post_cue_li_per_trial.mean()
        null_means = np.empty(n_perm)
        for p_i in range(n_perm):
            signs = rng.choice([-1, 1], size=n)
            null_means[p_i] = (post_cue_li_per_trial * signs).mean()
        p_perm = (np.sum(null_means >= observed_mean) + 1) / (n_perm + 1)

        lines.append(f"{name} (class {cls}):")
        lines.append(f"  N trials: {n}")
        lines.append(f"  Post-cue LI: {mean_li:.6f} ± {std_li:.6f}")
        lines.append(f"  One-sample t-test (LI > 0): t={t_stat:.4f}, p={p_val:.6f}")
        lines.append(f"  Sign-flip permutation test (LI > 0): p={p_perm:.6f}")
        lines.append(f"  Interpretation: {'Significant' if p_perm < 0.05 else 'NOT significant'} "
                      f"contralateral lateralization")
        lines.append("")

        print(f"  {name}: LI={mean_li:.6f}±{std_li:.6f}, "
              f"t={t_stat:.4f}, p_perm={p_perm:.6f}")

    # Cross-class difference: do the two classes lateralize in opposite directions?
    li0_post = li_results[0]['li_trials'][:, CUE_IDX:].mean(axis=1)
    li1_post = li_results[1]['li_trials'][:, CUE_IDX:].mean(axis=1)

    # Use the shorter array
    n_min = min(len(li0_post), len(li1_post))
    combined_li = np.concatenate([li0_post[:n_min], li1_post[:n_min]])
    mean_combined = combined_li.mean()

    lines.append("Cross-class combined laterality:")
    lines.append(f"  Mean combined post-cue LI: {mean_combined:.6f}")
    lines.append(f"  (Positive = both classes show contralateral dominance)")
    lines.append("")

    # Hemisphere × Class interaction
    # For each trial, compute: right_hemi_sal - left_hemi_sal
    # If MI is genuine, this should be positive for left MI and negative for right MI
    hemi_diffs = {}
    for cls in sorted(class_saliency.keys()):
        sal = class_saliency[cls]
        left_sal = sal[:, left_idx, CUE_IDX:].mean(axis=(1, 2))
        right_sal = sal[:, right_idx, CUE_IDX:].mean(axis=(1, 2))
        hemi_diffs[cls] = right_sal - left_sal  # per-trial

    # Interaction: does the hemisphere difference flip between classes?
    diff0_mean = hemi_diffs[0].mean()
    diff1_mean = hemi_diffs[1].mean()
    interaction = diff0_mean - diff1_mean  # Should be positive for genuine MI

    lines.append("Hemisphere × Class interaction (post-cue):")
    lines.append(f"  MI_left  (R−L): {diff0_mean:.6f}")
    lines.append(f"  MI_right (R−L): {diff1_mean:.6f}")
    lines.append(f"  Interaction (MI_left minus MI_right): {interaction:.6f}")
    lines.append(f"  (Positive interaction = classes use opposite hemispheres)")

    print(f"  Interaction (R-L): MI_left={diff0_mean:.6f}, MI_right={diff1_mean:.6f}, "
          f"diff={interaction:.6f}")

    with open(out / 'statistical_tests.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: statistical_tests.txt")


def run_analysis_6_c3c4_focus(
    class_saliency: dict[int, np.ndarray],
    ch_names: list[str],
    out: Path,
) -> None:
    """Analysis 6: Focused C3/C4 temporal saliency comparison."""
    print("\n── Analysis 6: C3 vs C4 Saliency Per Class Over Time ──")

    c3_idx = ch_names.index('C3')
    c4_idx = ch_names.index('C4')
    times = np.arange(N_SAMPLES) / SFREQ + TMIN

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for cls in sorted(class_saliency.keys()):
        sal = class_saliency[cls]
        c3_sal = sal[:, c3_idx, :].mean(axis=0)
        c4_sal = sal[:, c4_idx, :].mean(axis=0)

        # Smooth
        win = int(0.1 * SFREQ)
        kernel = np.ones(win) / win
        c3_smooth = np.convolve(c3_sal, kernel, mode='same')
        c4_smooth = np.convolve(c4_sal, kernel, mode='same')

        ax = axes[cls]
        name = CLASS_NAMES[cls]
        ax.plot(times, c3_smooth, linewidth=2, color='#E74C3C', label='C3 (left motor)')
        ax.plot(times, c4_smooth, linewidth=2, color='#3498DB', label='C4 (right motor)')
        ax.fill_between(times, c3_smooth, c4_smooth, alpha=0.15, color='gray')
        ax.axvline(CUE_ONSET, color='black', linestyle='--', alpha=0.7, label='Cue onset')
        ax.set_ylabel('Mean Saliency', fontsize=11)
        ax.set_title(f'{name} — C3 vs C4 Saliency', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Stats
        c3_post = sal[:, c3_idx, CUE_IDX:].mean()
        c4_post = sal[:, c4_idx, CUE_IDX:].mean()
        expected = 'C4 > C3' if cls == 0 else 'C3 > C4'
        actual = 'C4 > C3' if c4_post > c3_post else 'C3 > C4'
        match = 'MATCHES' if expected == actual else 'DOES NOT MATCH'

        ax.annotate(
            f'Post-cue: C3={c3_post:.6f}, C4={c4_post:.6f}\n'
            f'Expected for MI: {expected}\n'
            f'Observed: {actual} ({match} MI prediction)',
            xy=(0.02, 0.95), xycoords='axes fraction', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        )
        print(f"  {name}: C3={c3_post:.6f}, C4={c4_post:.6f} → {actual} ({match})")

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    fig.savefig(out / 'c3_c4_saliency.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: c3_c4_saliency.png")


def write_summary(
    hemi_results: dict,
    li_results: dict,
    out: Path,
) -> None:
    """Write a concise summary of all lateralization findings."""
    lines = [
        "# Lateralization Analysis Summary",
        "",
        "## Research Question",
        "Does the EEGNet model use contralateral sensorimotor patterns",
        "(the hallmark of genuine motor imagery classification)?",
        "",
        "## Key Results",
        "",
    ]

    for cls in sorted(hemi_results.keys()):
        name = CLASS_NAMES[cls]
        r = hemi_results[cls]
        li_mean = li_results[cls]['li_mean']
        post_cue_li = li_mean[CUE_IDX:].mean()

        lines.append(f"### {name}")
        lines.append(f"- Left hemisphere saliency:  {r['left_hemisphere']:.6f}")
        lines.append(f"- Right hemisphere saliency: {r['right_hemisphere']:.6f}")
        lines.append(f"- Post-cue laterality index: {post_cue_li:.4f}")

        if cls == 0:
            expected = "right hemisphere > left hemisphere"
            observed = ("right > left" if r['right_post_cue'] > r['left_post_cue']
                        else "left > right")
        else:
            expected = "left hemisphere > right hemisphere"
            observed = ("left > right" if r['left_post_cue'] > r['right_post_cue']
                        else "right > left")

        lines.append(f"- Expected for genuine MI: {expected}")
        lines.append(f"- Observed: {observed}")
        lines.append("")

    # Overall verdict
    li0_post = li_results[0]['li_mean'][CUE_IDX:].mean()
    li1_post = li_results[1]['li_mean'][CUE_IDX:].mean()

    if li0_post > 0.02 and li1_post > 0.02:
        verdict = "CONSISTENT with genuine motor imagery lateralization"
    elif li0_post > 0 and li1_post > 0:
        verdict = "WEAK contralateral lateralization — insufficient for MI"
    else:
        verdict = "NO contralateral lateralization — NOT genuine MI patterns"

    lines.append(f"## Verdict")
    lines.append(f"MI_left post-cue LI: {li0_post:.4f}")
    lines.append(f"MI_right post-cue LI: {li1_post:.4f}")
    lines.append(f"**{verdict}**")

    with open(out / 'lateralization_summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved: lateralization_summary.md")


def main() -> None:
    print("=" * 60)
    print("CLASS-CONDITIONAL LATERALIZATION ANALYSIS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model()

    # Load all 40 subjects
    print(f"\nLoading all {len(ALL_SUBJECTS)} subjects...")
    data, labels, ch_names = load_all_subjects()
    print(f"Channel names: {ch_names}")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Compute class-conditional saliency maps (integrated gradients)
    print("\nComputing class-conditional saliency (integrated gradients)...")
    saliency_gen = SaliencyMapGenerator(model, device='cpu')
    class_saliency = compute_class_saliency(saliency_gen, data, labels,
                                             method='integrated_gradients')

    # Run analyses
    hemi_results = run_analysis_1_hemisphere_saliency(class_saliency, ch_names, OUTPUT_DIR)
    li_results = run_analysis_2_laterality_timecourse(class_saliency, ch_names, OUTPUT_DIR)
    run_analysis_3_channel_pair_lateralization(class_saliency, ch_names, OUTPUT_DIR)
    run_analysis_4_topographic_maps(class_saliency, ch_names, OUTPUT_DIR)
    run_analysis_5_statistical_tests(li_results, class_saliency, ch_names, OUTPUT_DIR)
    run_analysis_6_c3c4_focus(class_saliency, ch_names, OUTPUT_DIR)

    # Write summary
    write_summary(hemi_results, li_results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("LATERALIZATION ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
