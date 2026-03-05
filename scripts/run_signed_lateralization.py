"""
Signed lateralization analysis — addresses the binary classification argument.

In a binary classifier, the model can use a SINGLE hemisphere to discriminate:
  "Right hemisphere active → left MI, not active → right MI"

With absolute saliency, both classes would show high importance at the same
hemisphere. The .abs() destroys the crucial sign information.

This script computes SIGNED integrated gradients w.r.t. a SINGLE class
(class 0 = MI_left) for ALL trials. Then compares:
  - Left MI trials: signed IG at C4 should be POSITIVE
    (right hemisphere activity is evidence FOR left MI)
  - Right MI trials: signed IG at C4 should be NEGATIVE
    (right hemisphere activity is evidence AGAINST right MI → pushes toward left MI)

If the sign FLIPS between classes at sensorimotor channels, the model IS using
lateralized motor activity, even though absolute saliency looks the same.
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
from scipy.stats import ttest_ind

from eeg_mi.models.eegnet import EEGNet


# ── Configuration ─────────────────────────────────────────────────────────
MODEL_PATH = project_root / "all_subjects_interpretation" / "models" / "model.pt"
DATA_DIR = project_root / "data" / "raw" / "apple_catcher"
OUTPUT_DIR = project_root / "results" / "interpretability" / "lateralization"
TMIN, TMAX = -1.0, 2.0
SFREQ = 250.0
N_SAMPLES = int((TMAX - TMIN) * SFREQ)
CUE_IDX = int((0.0 - TMIN) * SFREQ)
N_CHANNELS = 32

ALL_SUBJECTS = [f"s{i:02d}" for i in range(1, 41)]
CLASS_NAMES = {0: 'MI_left', 1: 'MI_right'}

SENSORIMOTOR_PAIRS = [
    ('C3', 'C4'), ('C1', 'C2'), ('C5', 'C6'),
    ('FC3', 'FC4'), ('FC1', 'FC2'), ('FC5', 'FC6'),
    ('CP3', 'CP4'), ('CP1', 'CP2'), ('CP5', 'CP6'),
]


def load_model() -> EEGNet:
    model = EEGNet(
        nb_classes=2, Chans=N_CHANNELS, Samples=N_SAMPLES,
        kernLength=64, F1=8, D=2, F2=16, dropoutRate=0.3,
    )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_all_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_data, all_labels = [], []
    ch_names = None
    mne.set_log_level("ERROR")

    for sid in ALL_SUBJECTS:
        subj_dir = DATA_DIR / sid
        for fif_path in sorted(subj_dir.glob("*_epo.fif")):
            epochs = mne.read_epochs(fif_path, preload=True)
            epochs.crop(tmin=TMIN, tmax=TMAX)
            data = epochs.get_data()[:, :, :N_SAMPLES]
            all_data.append(data)
            all_labels.append(epochs.events[:, -1])
            if ch_names is None:
                ch_names = epochs.ch_names
        print(f"  {sid}: loaded")

    return np.concatenate(all_data), np.concatenate(all_labels), ch_names


def signed_integrated_gradients(
    model: EEGNet,
    x: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
) -> np.ndarray:
    """Compute integrated gradients WITHOUT taking abs().

    Returns signed saliency: (batch, channels, samples).
    Positive = input feature pushes output TOWARD target_class.
    Negative = input feature pushes output AWAY from target_class.
    """
    model.eval()
    x = x.detach()
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, n_steps)

    integrated_grads = torch.zeros_like(x)

    target_tensor = torch.tensor([target_class] * x.shape[0])

    for alpha in alphas:
        x_step = (baseline + alpha * (x - baseline)).detach()
        x_step.requires_grad_(True)

        output = model(x_step)

        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_tensor.unsqueeze(1), 1)
        output.backward(gradient=one_hot)

        integrated_grads += x_step.grad.data

    integrated_grads = integrated_grads / n_steps
    integrated_grads = (x - baseline) * integrated_grads

    # NO .abs() — keep the sign
    return integrated_grads.squeeze(1).cpu().numpy()


def compute_signed_saliency_batched(
    model: EEGNet,
    data: np.ndarray,
    target_class: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute signed IG for all data w.r.t. a single target class."""
    all_sal = []
    n = len(data)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.tensor(data[start:end], dtype=torch.float32).unsqueeze(1)
        sal = signed_integrated_gradients(model, x, target_class)
        all_sal.append(sal)
        if (start // batch_size) % 20 == 0:
            print(f"    Batch {start//batch_size + 1}/{(n + batch_size - 1)//batch_size}")
    return np.concatenate(all_sal, axis=0)


def main() -> None:
    print("=" * 60)
    print("SIGNED LATERALIZATION ANALYSIS")
    print("(Addressing binary classification argument)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    print(f"\nLoading all {len(ALL_SUBJECTS)} subjects...")
    data, labels, ch_names = load_all_data()
    print(f"Total: {len(labels)} trials, classes: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # ── Compute signed IG w.r.t. class 0 (MI_left) for ALL trials ────────
    print("\nComputing SIGNED integrated gradients w.r.t. class 0 (MI_left)...")
    print("  (Positive = pushes toward MI_left, Negative = pushes toward MI_right)")
    signed_sal = compute_signed_saliency_batched(model, data, target_class=0)
    print(f"  Shape: {signed_sal.shape}")

    # Split by true label
    mask_left = labels == 0
    mask_right = labels == 1
    sal_left_trials = signed_sal[mask_left]    # Left MI trials
    sal_right_trials = signed_sal[mask_right]  # Right MI trials

    # ── Key channel indices ───────────────────────────────────────────────
    c3_idx = ch_names.index('C3')
    c4_idx = ch_names.index('C4')
    fc5_idx = ch_names.index('FC5')
    fc6_idx = ch_names.index('FC6')

    left_hemi_idx = [ch_names.index(p[0]) for p in SENSORIMOTOR_PAIRS if p[0] in ch_names and p[1] in ch_names]
    right_hemi_idx = [ch_names.index(p[1]) for p in SENSORIMOTOR_PAIRS if p[0] in ch_names and p[1] in ch_names]

    times = np.arange(N_SAMPLES) / SFREQ + TMIN

    # ── Analysis A: Signed saliency at key channels ──────────────────────
    print("\n── Analysis A: Signed Saliency at Key Channels (post-cue mean) ──")
    print("  (Positive = evidence FOR MI_left, Negative = evidence FOR MI_right)")

    key_channels = [('C3', c3_idx), ('C4', c4_idx), ('FC5', fc5_idx), ('FC6', fc6_idx)]
    lines = [
        "Signed Integrated Gradients w.r.t. MI_left (class 0)",
        "=" * 65,
        "",
        "Positive = input pushes toward MI_left",
        "Negative = input pushes toward MI_right",
        "",
        "If model uses MI lateralization at C4:",
        "  Left MI trials → positive at C4 (right hemi activity = evidence for left MI)",
        "  Right MI trials → negative at C4 (right hemi activity absent = evidence for right MI)",
        "",
        f"{'Channel':>8}  {'Left MI trials':>16}  {'Right MI trials':>16}  {'Difference':>12}  {'Sign flips?':>12}",
        "-" * 75,
    ]

    for name, idx in key_channels:
        left_val = sal_left_trials[:, idx, CUE_IDX:].mean()
        right_val = sal_right_trials[:, idx, CUE_IDX:].mean()
        flips = "YES" if (left_val > 0) != (right_val > 0) else "NO"

        line = f"{name:>8}  {left_val:>+16.7f}  {right_val:>+16.7f}  {left_val - right_val:>+12.7f}  {flips:>12}"
        lines.append(line)
        print(f"  {line}")

    # Hemisphere averages
    print("\n  Hemisphere averages (sensorimotor channels, post-cue):")
    lines.append("")
    lines.append("Hemisphere averages (sensorimotor, post-cue):")

    for hemi_name, hemi_idx in [("Left hemi", left_hemi_idx), ("Right hemi", right_hemi_idx)]:
        left_val = sal_left_trials[:, hemi_idx, CUE_IDX:].mean()
        right_val = sal_right_trials[:, hemi_idx, CUE_IDX:].mean()
        flips = "YES" if (left_val > 0) != (right_val > 0) else "NO"
        line = f"{hemi_name:>12}  {left_val:>+16.7f}  {right_val:>+16.7f}  {left_val - right_val:>+12.7f}  {flips:>12}"
        lines.append(line)
        print(f"  {line}")

    # ── Analysis B: Signed saliency time course at C3/C4 ─────────────────
    print("\n── Analysis B: Signed Saliency Time Course ──")

    # Smooth for plotting
    win = int(0.1 * SFREQ)
    kernel = np.ones(win) / win

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, (ch_name, ch_idx) in enumerate([('C3', c3_idx), ('C4', c4_idx)]):
        # Top row: per-class mean signed saliency
        ax = axes[0, col]
        left_ts = np.convolve(sal_left_trials[:, ch_idx, :].mean(axis=0), kernel, mode='same')
        right_ts = np.convolve(sal_right_trials[:, ch_idx, :].mean(axis=0), kernel, mode='same')

        ax.plot(times, left_ts, linewidth=2, color='#E74C3C', label='MI_left trials')
        ax.plot(times, right_ts, linewidth=2, color='#3498DB', label='MI_right trials')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.7, label='Cue onset')
        ax.fill_between(times, left_ts, right_ts, alpha=0.15, color='gray')
        ax.set_ylabel('Signed IG (→ MI_left)', fontsize=11)
        ax.set_title(f'{ch_name} — Signed Saliency w.r.t. MI_left', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Bottom row: difference (left MI - right MI)
        ax = axes[1, col]
        diff = left_ts - right_ts
        ax.plot(times, diff, linewidth=2, color='purple')
        ax.fill_between(times, 0, diff, where=diff > 0, alpha=0.3, color='green', label='Left MI > Right MI')
        ax.fill_between(times, 0, diff, where=diff < 0, alpha=0.3, color='orange', label='Right MI > Left MI')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_ylabel('IG difference (left − right trials)', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_title(f'{ch_name} — Class Difference in Signed Saliency', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Signed Integrated Gradients w.r.t. MI_left\n'
        '(Positive = evidence FOR left MI, Negative = evidence FOR right MI)',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'signed_c3_c4_timecourse.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: signed_c3_c4_timecourse.png")

    # ── Analysis C: Signed saliency at ALL channels (post-cue) ───────────
    print("\n── Analysis C: Signed Channel Importance (Post-Cue) ──")

    left_post = sal_left_trials[:, :, CUE_IDX:].mean(axis=(0, 2))
    right_post = sal_right_trials[:, :, CUE_IDX:].mean(axis=(0, 2))
    diff_post = left_post - right_post

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # Sort by difference magnitude
    sort_idx = np.argsort(np.abs(diff_post))[::-1]

    for ax_idx, (data_arr, title, cmap_pos, cmap_neg) in enumerate([
        (left_post, 'MI_left trials — Signed IG w.r.t. MI_left (post-cue)', '#E74C3C', '#3498DB'),
        (right_post, 'MI_right trials — Signed IG w.r.t. MI_left (post-cue)', '#E74C3C', '#3498DB'),
        (diff_post, 'Difference (MI_left − MI_right trials)', '#2ECC71', '#E67E22'),
    ]):
        ax = axes[ax_idx]
        vals = data_arr[sort_idx]
        names = [ch_names[i] for i in sort_idx]
        colors = [cmap_pos if v >= 0 else cmap_neg for v in vals]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Signed IG', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'signed_channel_importance.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: signed_channel_importance.png")

    # ── Analysis D: Topographic difference map (signed) ───────────────────
    print("\n── Analysis D: Signed Topographic Difference ──")

    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types='eeg')
    info.set_montage(montage, on_missing='warn')

    timepoints = [('−0.5s', -0.5), ('Cue (0s)', 0.0), ('+0.5s', 0.5), ('+1.0s', 1.0)]
    window_half = int(0.1 * SFREQ)

    fig, axes = plt.subplots(3, len(timepoints), figsize=(4 * len(timepoints), 12))

    row_data = [
        ('MI_left trials (signed IG → MI_left)', sal_left_trials.mean(axis=0)),
        ('MI_right trials (signed IG → MI_left)', sal_right_trials.mean(axis=0)),
        ('Difference (left − right trials)', sal_left_trials.mean(axis=0) - sal_right_trials.mean(axis=0)),
    ]

    for row, (row_title, sal_mean) in enumerate(row_data):
        vmax = np.abs(sal_mean).max() * 0.8
        cmap = 'RdBu_r'

        for col, (label, t) in enumerate(timepoints):
            t_idx = int((t - TMIN) * SFREQ)
            start = max(0, t_idx - window_half)
            end = min(sal_mean.shape[1], t_idx + window_half)
            data_at_t = sal_mean[:, start:end].mean(axis=1)

            ax = axes[row, col]
            mne.viz.plot_topomap(
                data_at_t, info, axes=ax, show=False,
                cmap=cmap, vlim=(-vmax, vmax), contours=4,
            )
            if row == 0:
                ax.set_title(label, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(row_title, fontsize=9, fontweight='bold')

    plt.suptitle(
        'Signed Saliency Topography (w.r.t. MI_left)\n'
        'Red = pushes toward MI_left, Blue = pushes toward MI_right',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'signed_topographic_maps.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: signed_topographic_maps.png")

    # ── Analysis E: The critical test — does the sign flip? ───────────────
    print("\n── Analysis E: Sign Flip Test (The Key Question) ──")
    print("  Does the signed saliency FLIP direction between classes?")
    print("  If yes → model uses that channel differently for each class (MI-like)")
    print("  If no  → model uses the same direction (cue-ERP-like)")
    print()

    lines.append("")
    lines.append("=" * 65)
    lines.append("SIGN FLIP TEST — The Critical Question")
    lines.append("=" * 65)
    lines.append("")
    lines.append("For each channel: does the signed IG flip between MI_left and MI_right?")
    lines.append("Sign flip = model uses that channel DIFFERENTLY per class")
    lines.append("")

    n_flip = 0
    n_total = 0
    flip_channels = []

    header = f"{'Channel':>8}  {'MI_left':>12}  {'MI_right':>12}  {'Flips?':>8}  {'t-stat':>10}  {'p-value':>10}"
    lines.append(header)
    lines.append("-" * 75)
    print(f"  {header}")

    for i, ch in enumerate(ch_names):
        left_vals = sal_left_trials[:, i, CUE_IDX:].mean(axis=1)   # per-trial mean
        right_vals = sal_right_trials[:, i, CUE_IDX:].mean(axis=1)

        left_mean = left_vals.mean()
        right_mean = right_vals.mean()
        flips = (left_mean > 0) != (right_mean > 0)

        t_stat, p_val = ttest_ind(left_vals, right_vals)

        n_total += 1
        if flips:
            n_flip += 1
            flip_channels.append(ch)

        line = f"{ch:>8}  {left_mean:>+12.7f}  {right_mean:>+12.7f}  {'YES' if flips else 'no':>8}  {t_stat:>10.3f}  {p_val:>10.2e}"
        lines.append(line)
        if flips or ch in ['C3', 'C4', 'FC5', 'FC6', 'FC3', 'FC4']:
            print(f"  {line}")

    lines.append("")
    lines.append(f"Channels with sign flip: {n_flip}/{n_total} ({n_flip/n_total*100:.1f}%)")
    lines.append(f"Flip channels: {flip_channels}")
    print(f"\n  Sign flips: {n_flip}/{n_total} channels ({n_flip/n_total*100:.1f}%)")
    print(f"  Flip channels: {flip_channels}")

    # ── Classify the result ───────────────────────────────────────────────
    # Key sensorimotor channels
    sm_channels = ['C3', 'C4', 'C1', 'C2', 'FC3', 'FC4']
    sm_flips = [ch for ch in sm_channels if ch in flip_channels]

    lines.append("")
    lines.append("Key sensorimotor channels with sign flip:")
    lines.append(f"  {sm_flips} out of {sm_channels}")

    if len(sm_flips) >= 3:
        verdict = ("SIGN FLIPS at sensorimotor channels → model may use lateralized "
                   "motor activity for discrimination (consistent with MI)")
    elif len(sm_flips) >= 1:
        verdict = ("PARTIAL sign flips at sensorimotor channels → weak/mixed evidence "
                   "for lateralized motor activity")
    else:
        verdict = ("NO sign flips at key sensorimotor channels → model does NOT use "
                   "lateralized motor activity (not genuine MI)")

    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    print(f"\n  VERDICT: {verdict}")

    with open(OUTPUT_DIR / 'signed_lateralization_results.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n  Saved: signed_lateralization_results.txt")

    print("\n" + "=" * 60)
    print("SIGNED LATERALIZATION ANALYSIS COMPLETE")
    print(f"Results: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
