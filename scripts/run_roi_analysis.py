"""
ROI ratio analysis for spatial filters and attributions (thesis §4.4.2, §5.2, §5.3).

Computes quantitative Region of Interest ratios for:
1. Spatial filter weights (sensorimotor vs frontal vs parietal)
2. Attribution maps per class
3. Temporal filter characterization table
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

from eeg_mi.models.eegnet import EEGNet
from eeg_mi.interpretability.saliency import SaliencyMapGenerator

MODEL_PATH = project_root / "all_subjects_interpretation" / "models" / "model.pt"
DATA_DIR = project_root / "data" / "raw" / "apple_catcher"
OUT_DIR = project_root / "results" / "interpretability" / "roi_analysis"
TMIN, TMAX = -1.0, 2.0
SFREQ = 250.0
N_SAMPLES = int((TMAX - TMIN) * SFREQ)
CUE_IDX = int((0.0 - TMIN) * SFREQ)

CH_NAMES = ['F3', 'F1', 'Fz', 'FC1', 'FCz', 'Cz', 'P5', 'CP5', 'CP3', 'CP1',
            'FC3', 'FC5', 'C5', 'C3', 'C1', 'P1', 'C4', 'C6', 'CP6', 'CP4',
            'CP2', 'CPz', 'AFz', 'F4', 'F2', 'C2', 'FC6', 'FC4', 'FC2', 'Pz',
            'P2', 'P6']

# ROI definitions (thesis §4.4.2, adapted to actual montage — no occipital channels)
ROIS: dict[str, list[str]] = {
    'sensorimotor': ['C3', 'C4', 'C1', 'C2', 'C5', 'C6', 'FC3', 'FC4', 'FC1', 'FC2', 'CP3', 'CP4', 'CP1', 'CP2'],
    'frontal': ['F3', 'F4', 'F1', 'F2', 'Fz', 'AFz', 'FC5', 'FC6', 'FCz'],
    'parietal': ['P5', 'P6', 'P1', 'P2', 'Pz', 'CP5', 'CP6', 'CPz'],
}


def get_roi_indices(ch_names: list[str]) -> dict[str, list[int]]:
    return {roi: [ch_names.index(ch) for ch in chs if ch in ch_names]
            for roi, chs in ROIS.items()}


def load_model() -> EEGNet:
    model = EEGNet(nb_classes=2, Chans=32, Samples=N_SAMPLES,
                   kernLength=64, F1=8, D=2, F2=16, dropoutRate=0.3)
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_spatial_filter_roi(model: EEGNet) -> None:
    """§5.2.2: ROI ratios for each of the 16 spatial filters."""
    print("\n── Spatial Filter ROI Analysis ──")

    # Extract depthwise spatial filters: shape (F1*D, 1, Chans, 1)
    spatial_weights = model.block1[2].weight.detach().cpu().numpy()  # depthwise conv
    n_filters = spatial_weights.shape[0]

    roi_idx = get_roi_indices(CH_NAMES)

    lines = [
        "Spatial Filter ROI Ratios",
        "=" * 70,
        f"ROIs: sensorimotor ({len(roi_idx['sensorimotor'])} ch), "
        f"frontal ({len(roi_idx['frontal'])} ch), parietal ({len(roi_idx['parietal'])} ch)",
        f"Note: No occipital channels in montage; parietal used as nearest alternative",
        "",
        f"{'Filter':>8}  {'R_sensorimotor':>15}  {'R_frontal':>10}  {'R_parietal':>11}  {'Dominant ROI':>13}",
        "-" * 70,
    ]

    roi_ratios = {roi: [] for roi in ROIS}

    for fi in range(n_filters):
        w = np.abs(spatial_weights[fi, 0, :, 0])  # (Chans,)
        total = w.sum()

        ratios = {}
        for roi_name, indices in roi_idx.items():
            ratios[roi_name] = w[indices].sum() / total
            roi_ratios[roi_name].append(ratios[roi_name])

        dominant = max(ratios, key=ratios.get)
        lines.append(f"{fi:>8}  {ratios['sensorimotor']:>15.4f}  "
                      f"{ratios['frontal']:>10.4f}  {ratios['parietal']:>11.4f}  "
                      f"{dominant:>13}")

    lines.append("-" * 70)
    means = {roi: np.mean(vals) for roi, vals in roi_ratios.items()}
    lines.append(f"{'MEAN':>8}  {means['sensorimotor']:>15.4f}  "
                  f"{means['frontal']:>10.4f}  {means['parietal']:>11.4f}")

    with open(OUT_DIR / 'spatial_filter_roi_ratios.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_filters)
    width = 0.25
    colors = {'sensorimotor': '#3498DB', 'frontal': '#E74C3C', 'parietal': '#2ECC71'}

    for i, (roi_name, vals) in enumerate(roi_ratios.items()):
        ax.bar(x + i * width, vals, width, label=roi_name.capitalize(),
               color=colors[roi_name], alpha=0.8, edgecolor='black')

    ax.set_xticks(x + width)
    ax.set_xticklabels([f'F{i}' for i in range(n_filters)])
    ax.set_ylabel('ROI Ratio', fontsize=11)
    ax.set_title('Spatial Filter ROI Ratios', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'spatial_filter_roi_ratios.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    for roi_name, mean_val in means.items():
        print(f"  Mean R_{roi_name}: {mean_val:.4f}")
    print("  Saved spatial filter ROI ratios")


def run_temporal_filter_table(model: EEGNet) -> None:
    """§5.2.1: Temporal filter characterization table."""
    print("\n── Temporal Filter Characterization ──")

    # Extract temporal filters: shape (F1, 1, 1, kernLength)
    temp_weights = model.block1[0].weight.detach().cpu().numpy()
    n_filters = temp_weights.shape[0]
    kern_len = temp_weights.shape[3]

    lines = [
        "Temporal Filter Characterization",
        "=" * 70,
        f"{'Filter':>8}  {'Peak (Hz)':>10}  {'BW_-3dB (Hz)':>13}  {'Band':>10}",
        "-" * 50,
    ]

    band_defs = [
        ('delta', 1, 4), ('theta', 4, 8), ('mu', 8, 13),
        ('beta', 13, 30), ('gamma', 30, 45),
    ]

    for fi in range(n_filters):
        kernel = temp_weights[fi, 0, 0, :]

        # FFT
        n_fft = 512
        spectrum = np.abs(np.fft.rfft(kernel, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / SFREQ)

        # Peak frequency
        peak_idx = np.argmax(spectrum[1:]) + 1  # skip DC
        peak_freq = freqs[peak_idx]

        # -3dB bandwidth
        peak_val = spectrum[peak_idx]
        threshold = peak_val / np.sqrt(2)
        above = spectrum >= threshold
        nonzero = np.where(above)[0]
        if len(nonzero) > 1:
            bw = freqs[nonzero[-1]] - freqs[nonzero[0]]
        else:
            bw = 0.0

        # Band classification
        band = 'broadband'
        for bname, blo, bhi in band_defs:
            if blo <= peak_freq <= bhi:
                band = bname
                break

        lines.append(f"{fi:>8}  {peak_freq:>10.1f}  {bw:>13.1f}  {band:>10}")

    with open(OUT_DIR / 'filter_characterization.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print("  Saved filter characterization table")


def run_attribution_roi(model: EEGNet) -> None:
    """§5.3.2: Attribution ROI analysis per class."""
    print("\n── Attribution ROI Analysis ──")

    # Load all data
    mne.set_log_level("ERROR")
    all_data, all_labels = [], []
    for sid in [f"s{i:02d}" for i in range(1, 41)]:
        subj_dir = DATA_DIR / sid
        for fif_path in sorted(subj_dir.glob("*_epo.fif")):
            ep = mne.read_epochs(fif_path, preload=True)
            ep.crop(tmin=TMIN, tmax=TMAX)
            data = ep.get_data()[:, :, :N_SAMPLES]
            all_data.append(data)
            all_labels.append(ep.events[:, -1])
        print(f"    {sid}")

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)
    print(f"  Total: {len(labels)} trials")

    saliency_gen = SaliencyMapGenerator(model, device='cpu')
    roi_idx = get_roi_indices(CH_NAMES)

    results = {}
    for cls in [0, 1]:
        cls_name = {0: 'MI_left', 1: 'MI_right'}[cls]
        mask = labels == cls
        cls_data = data[mask]

        # Compute IG in batches
        all_sal = []
        batch_size = 64
        for start in range(0, len(cls_data), batch_size):
            end = min(start + batch_size, len(cls_data))
            x = torch.tensor(cls_data[start:end], dtype=torch.float32).unsqueeze(1)
            sal = saliency_gen.integrated_gradients(x, target_class=cls)
            all_sal.append(sal)
        sal_map = np.concatenate(all_sal)  # (n_trials, channels, samples)

        # ROI percentages
        total_sal = sal_map.mean()
        results[cls_name] = {}

        for period_name, s_idx, e_idx in [
            ('full_epoch', 0, N_SAMPLES),
            ('post_cue', CUE_IDX, N_SAMPLES),
        ]:
            period_sal = sal_map[:, :, s_idx:e_idx]
            period_total = period_sal.mean()

            for roi_name, indices in roi_idx.items():
                roi_sal = period_sal[:, indices, :].mean()
                # Percentage: roi contribution vs all channels in same period
                all_ch_sal = period_sal.mean(axis=(0, 2))  # per-channel means
                roi_pct = all_ch_sal[indices].sum() / all_ch_sal.sum() * 100
                results[cls_name][f'{roi_name}_{period_name}'] = roi_pct

        print(f"  {cls_name}: computed")

    # Write table
    lines = [
        "Attribution ROI Percentages",
        "=" * 70,
        f"Note: percentages show fraction of total channel importance in each ROI",
        "",
        f"{'Class':>10}  {'Period':>12}  {'Sensorimotor%':>14}  {'Frontal%':>10}  {'Parietal%':>11}",
        "-" * 65,
    ]

    for cls_name in ['MI_left', 'MI_right']:
        for period in ['full_epoch', 'post_cue']:
            sm = results[cls_name][f'sensorimotor_{period}']
            fr = results[cls_name][f'frontal_{period}']
            pa = results[cls_name][f'parietal_{period}']
            lines.append(f"{cls_name:>10}  {period:>12}  {sm:>14.1f}  {fr:>10.1f}  {pa:>11.1f}")

    with open(OUT_DIR / 'attribution_roi_percentages.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, period in enumerate(['full_epoch', 'post_cue']):
        ax = axes[ax_idx]
        x = np.arange(len(ROIS))
        width = 0.35
        colors_cls = ['#E74C3C', '#3498DB']

        for i, cls_name in enumerate(['MI_left', 'MI_right']):
            vals = [results[cls_name][f'{roi}_{period}'] for roi in ROIS]
            ax.bar(x + i * width, vals, width, label=cls_name,
                   color=colors_cls[i], alpha=0.8, edgecolor='black')
            for j, v in enumerate(vals):
                ax.text(x[j] + i * width, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([r.capitalize() for r in ROIS])
        ax.set_ylabel('% of Total Attribution', fontsize=11)
        ax.set_title(f'Attribution ROI — {period.replace("_", " ").title()}',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'attribution_roi_percentages.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved attribution ROI percentages")


def main() -> None:
    print("=" * 60)
    print("ROI RATIO ANALYSIS (thesis §5.2, §5.3)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    print("Model loaded")

    run_temporal_filter_table(model)
    run_spatial_filter_roi(model)
    run_attribution_roi(model)

    print("\n" + "=" * 60)
    print("ROI ANALYSIS COMPLETE")
    print(f"Results: {OUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
