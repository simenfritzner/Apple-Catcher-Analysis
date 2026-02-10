"""
Run Phase 5: MRCP & Preparatory Potential Analysis.

Investigates whether the all-subjects EEGNet model exploits slow cortical
potentials (MRCPs, Bereitschaftspotential, lateralized readiness potentials)
for cross-subject motor imagery classification.

Analyses implemented:
    5.1  Low-frequency (<5 Hz) saliency
    5.2  Temporal filter MRCP sensitivity
    5.3  Grand-average ERP overlay with model importance at Cz/FCz
    5.4  Classical LRP vs model lateralized saliency at C3/C4
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import mne
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from eeg_mi.models.eegnet import EEGNet
from eeg_mi.interpretability.mrcp import (
    MRCPAnalyzer,
    plot_low_frequency_saliency,
    plot_filter_mrcp_sensitivity,
    plot_erp_with_importance,
    plot_lrp_comparison,
)


# ── Configuration ──────────────────────────────────────────────────────────
MODEL_PATH = project_root / "all_subjects_interpretation" / "models" / "model.pt"
DATA_DIR = project_root / "data" / "raw" / "apple_catcher"
OUTPUT_DIR = project_root / "results" / "interpretability" / "phase5_mrcp"
TMIN, TMAX = -1.0, 2.0
SFREQ = 250.0
N_SAMPLES = int((TMAX - TMIN) * SFREQ)  # 750
CUE_ONSET = 0.0
N_CHANNELS = 32
N_CLASSES = 2
SUBJECTS_TO_LOAD = ["s01", "s02", "s03", "s04", "s05"]


def load_model() -> EEGNet:
    """Load the trained EEGNet model."""
    model = EEGNet(
        nb_classes=N_CLASSES,
        Chans=N_CHANNELS,
        Samples=N_SAMPLES,
        kernLength=64,
        F1=8, D=2, F2=16,
        dropoutRate=0.3,
    )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")
    return model


def load_subject_data(subject_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and crop FIF epochs for a single subject."""
    subj_dir = DATA_DIR / subject_id
    fif_files = sorted(subj_dir.glob("*_epo.fif"))

    all_data = []
    all_labels = []

    mne.set_log_level("ERROR")

    for fif_path in fif_files:
        epochs = mne.read_epochs(fif_path, preload=True)
        epochs.crop(tmin=TMIN, tmax=TMAX)

        # Ensure exactly N_SAMPLES timepoints
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        data = data[:, :, :N_SAMPLES]

        labels = epochs.events[:, -1]

        all_data.append(data)
        all_labels.append(labels)

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return data, labels


def load_multi_subject_data(
    subject_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load data from multiple subjects."""
    all_data = []
    all_labels = []
    channel_names = None

    for sid in subject_ids:
        print(f"  Loading {sid}...", end=" ")
        try:
            data, labels = load_subject_data(sid)
            all_data.append(data)
            all_labels.append(labels)
            print(f"{len(labels)} trials")

            if channel_names is None:
                subj_dir = DATA_DIR / sid
                fif_files = sorted(subj_dir.glob("*_epo.fif"))
                mne.set_log_level("ERROR")
                epochs = mne.read_epochs(fif_files[0], preload=False)
                channel_names = epochs.ch_names
        except Exception as e:
            print(f"FAILED: {e}")

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Total: {len(labels)} trials, shape={data.shape}")

    return data, labels, channel_names


# ── Phase 5 analyses ──────────────────────────────────────────────────────

def run_phase5_1_low_freq_saliency(
    analyzer: MRCPAnalyzer,
    x: torch.Tensor,
    out: Path,
) -> None:
    """5.1: Low-frequency saliency in the MRCP band (0.1-5 Hz)."""
    print("\n  5.1  Low-frequency saliency (0.1-5 Hz)...")

    result = analyzer.low_frequency_saliency(
        x, sfreq=SFREQ, tmin=TMIN,
        low_freq=0.1, high_freq=5.0,
        method='integrated_gradients',
    )

    # Identify pre-cue vs post-cue importance
    cue_idx = int((CUE_ONSET - TMIN) * SFREQ)
    pre_mean = float(result.temporal_importance[:cue_idx].mean())
    post_mean = float(result.temporal_importance[cue_idx:].mean())
    ratio = pre_mean / (post_mean + 1e-10)

    print(f"    Pre-cue importance  (0.1-5 Hz): {pre_mean:.6f}")
    print(f"    Post-cue importance (0.1-5 Hz): {post_mean:.6f}")
    print(f"    Pre/Post ratio:                 {ratio:.3f}")

    if ratio > 1.0:
        print("    --> Pre-cue low-freq saliency is HIGHER: "
              "slow cortical potentials may drive pre-cue classification")
    else:
        print("    --> Post-cue low-freq saliency is higher")

    plot_low_frequency_saliency(
        result.times, result.temporal_importance,
        cue_onset=CUE_ONSET,
        save_path=out / "low_freq_saliency.png",
    )
    plt.close("all")

    # Save numerical summary
    with open(out / "low_freq_saliency_results.txt", "w") as f:
        f.write("Phase 5.1 — Low-Frequency Saliency (0.1-5 Hz)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"pre_cue_importance:  {pre_mean:.6f}\n")
        f.write(f"post_cue_importance: {post_mean:.6f}\n")
        f.write(f"pre_post_ratio:      {ratio:.6f}\n")


def run_phase5_2_filter_sensitivity(
    analyzer: MRCPAnalyzer,
    out: Path,
) -> None:
    """5.2: Temporal filter frequency response check for MRCP sensitivity."""
    print("\n  5.2  Temporal filter MRCP sensitivity check...")

    filter_results = analyzer.check_filter_mrcp_sensitivity(
        sfreq=SFREQ, mrcp_cutoff=5.0, energy_threshold=0.3,
    )

    n_sensitive = sum(1 for r in filter_results if r.is_mrcp_sensitive)
    print(f"    {n_sensitive}/{len(filter_results)} filters are MRCP-sensitive (>30% energy <5 Hz)")

    for r in filter_results:
        flag = " *** MRCP" if r.is_mrcp_sensitive else ""
        print(f"    Filter {r.filter_index + 1}: "
              f"low-freq ratio={r.low_freq_energy_ratio:.3f}, "
              f"peak={r.peak_frequency:.1f} Hz{flag}")

    plot_filter_mrcp_sensitivity(
        filter_results,
        save_path=out / "filter_mrcp_sensitivity.png",
    )
    plt.close("all")

    with open(out / "filter_mrcp_sensitivity.txt", "w") as f:
        f.write("Phase 5.2 — Temporal Filter MRCP Sensitivity\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"MRCP-sensitive filters: {n_sensitive}/{len(filter_results)}\n\n")
        for r in filter_results:
            f.write(f"Filter {r.filter_index + 1}: "
                    f"low_freq_ratio={r.low_freq_energy_ratio:.4f}, "
                    f"peak_freq={r.peak_frequency:.2f} Hz, "
                    f"mrcp_sensitive={r.is_mrcp_sensitive}\n")


def run_phase5_3_erp_overlay(
    analyzer: MRCPAnalyzer,
    x: torch.Tensor,
    data: np.ndarray,
    labels: np.ndarray,
    channel_names: list[str],
    out: Path,
) -> None:
    """5.3: Grand-average ERP overlay with model temporal importance."""
    print("\n  5.3  Grand-average ERP overlay at Cz/FCz...")

    overlay_results = analyzer.overlay_erp_with_importance(
        x, data, labels, channel_names,
        sfreq=SFREQ, tmin=TMIN,
        method='integrated_gradients',
        channels_of_interest=['Cz', 'FCz'],
    )

    for result in overlay_results:
        ch = result.channel_name
        print(f"    Channel {ch}: ERP computed for classes {list(result.erp_per_class.keys())}")

        plot_erp_with_importance(
            result.times,
            result.erp_per_class,
            result.temporal_importance,
            result.channel_name,
            cue_onset=CUE_ONSET,
            save_path=out / f"erp_overlay_{ch}.png",
        )
        plt.close("all")

    print(f"    Saved {len(overlay_results)} overlay plot(s)")


def run_phase5_4_lrp(
    analyzer: MRCPAnalyzer,
    x: torch.Tensor,
    data: np.ndarray,
    labels: np.ndarray,
    channel_names: list[str],
    out: Path,
) -> None:
    """5.4: Lateralized readiness potential vs model lateralized saliency."""
    print("\n  5.4  Lateralized readiness potential (LRP) at C3/C4...")

    # Classical LRP
    try:
        lrp_result = analyzer.compute_lrp(
            data, labels, channel_names,
            sfreq=SFREQ, tmin=TMIN,
        )
    except ValueError as e:
        print(f"    Skipping classical LRP: {e}")
        return

    # Model lateralized saliency
    model_lat_sal = None
    try:
        times_sal, model_lat_sal = analyzer.compute_lateralized_saliency(
            x, labels, channel_names,
            sfreq=SFREQ, tmin=TMIN,
            method='integrated_gradients',
        )
    except ValueError as e:
        print(f"    Skipping model lateralized saliency: {e}")

    # Print summary statistics
    cue_idx = int((CUE_ONSET - TMIN) * SFREQ)
    pre_lrp = float(np.abs(lrp_result.classical_lrp[:cue_idx]).mean())
    post_lrp = float(np.abs(lrp_result.classical_lrp[cue_idx:]).mean())
    print(f"    Classical LRP |amplitude|: pre-cue={pre_lrp:.6f}, post-cue={post_lrp:.6f}")

    if model_lat_sal is not None:
        pre_sal = float(np.abs(model_lat_sal[:cue_idx]).mean())
        post_sal = float(np.abs(model_lat_sal[cue_idx:]).mean())
        print(f"    Model lat. saliency |amp|: pre-cue={pre_sal:.6f}, post-cue={post_sal:.6f}")

    plot_lrp_comparison(
        lrp_result.times,
        lrp_result.classical_lrp,
        model_lrp=model_lat_sal,
        cue_onset=CUE_ONSET,
        save_path=out / "lrp_comparison.png",
    )
    plt.close("all")

    with open(out / "lrp_results.txt", "w") as f:
        f.write("Phase 5.4 — Lateralized Readiness Potential\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Classical LRP |amp| pre-cue:  {pre_lrp:.6f}\n")
        f.write(f"Classical LRP |amp| post-cue: {post_lrp:.6f}\n")
        if model_lat_sal is not None:
            f.write(f"Model lat. sal |amp| pre-cue:  {pre_sal:.6f}\n")
            f.write(f"Model lat. sal |amp| post-cue: {post_sal:.6f}\n")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("Phase 5: MRCP & Preparatory Potential Analysis")
    print("=" * 60)

    # Load model
    model = load_model()

    # Load data
    print(f"\nLoading data from {len(SUBJECTS_TO_LOAD)} subjects...")
    data, labels, channel_names = load_multi_subject_data(SUBJECTS_TO_LOAD)
    print(f"Channel names: {channel_names}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare tensor input — use a subset for gradient computations
    n_subset = min(200, len(labels))
    x = torch.tensor(data[:n_subset], dtype=torch.float32).unsqueeze(1)
    print(f"\nUsing {n_subset} trials for saliency, shape={x.shape}")

    # Also keep corresponding labels for the subset
    labels_subset = labels[:n_subset]

    analyzer = MRCPAnalyzer(model, device="cpu", batch_size=32)

    # Run all four sub-analyses
    print("\n" + "=" * 60)
    print("PHASE 5: MRCP & Preparatory Potential Analysis")
    print("=" * 60)

    run_phase5_1_low_freq_saliency(analyzer, x, OUTPUT_DIR)
    run_phase5_2_filter_sensitivity(analyzer, OUTPUT_DIR)
    run_phase5_3_erp_overlay(
        analyzer, x, data[:n_subset], labels_subset, channel_names, OUTPUT_DIR,
    )
    run_phase5_4_lrp(
        analyzer, x, data[:n_subset], labels_subset, channel_names, OUTPUT_DIR,
    )

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
