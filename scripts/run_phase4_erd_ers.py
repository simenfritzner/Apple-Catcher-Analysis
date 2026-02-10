"""
Run Phase 4 ERD/ERS analysis on the all-subjects EEGNet model.

Executes all four sub-analyses from SPEC.md Phase 4:
  4.1 Frequency-band saliency
  4.2 Time-frequency saliency
  4.3 Model-derived ERD/ERS curves
  4.4 Classical ERD/ERS and comparison with model
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
from eeg_mi.interpretability.erd_ers import (
    ERDERSAnalyzer,
    plot_frequency_band_saliency,
    plot_time_frequency_saliency,
    plot_erd_ers_comparison,
    plot_erd_ers_curves,
)


# ── Configuration ──────────────────────────────────────────────────────────
MODEL_PATH = project_root / "all_subjects_interpretation" / "models" / "model.pt"
DATA_DIR = project_root / "data" / "raw" / "apple_catcher"
OUTPUT_DIR = project_root / "results" / "interpretability" / "phase4_erd_ers"
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


# ── Analysis functions ─────────────────────────────────────────────────────


def run_phase4_1_frequency_band_saliency(
    analyzer: ERDERSAnalyzer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    """Phase 4.1: Frequency-band saliency analysis."""
    print("\n" + "=" * 60)
    print("PHASE 4.1: Frequency-Band Saliency")
    print("=" * 60)

    out = OUTPUT_DIR / "4_1_band_saliency"
    out.mkdir(parents=True, exist_ok=True)

    print("  Computing saliency per frequency band (integrated gradients)...")
    band_results = analyzer.frequency_band_saliency(
        x, y, sfreq=SFREQ, tmin=TMIN, method='integrated_gradients',
    )

    # Print ranking
    print("\n  Band importance ranking:")
    ranked = sorted(
        band_results.items(),
        key=lambda kv: kv[1]['mean_importance'],
        reverse=True,
    )
    for band_name, res in ranked:
        print(f"    {band_name:>6}: mean_importance = {res['mean_importance']:.6f}")

    # Save ranking
    with open(out / "band_importance_ranking.txt", "w") as f:
        f.write("Frequency-Band Saliency Ranking\n")
        f.write("=" * 40 + "\n\n")
        for band_name, res in ranked:
            f.write(f"{band_name}: {res['mean_importance']:.6f}\n")

    # Plot
    plot_frequency_band_saliency(band_results, cue_onset=CUE_ONSET, save_path=out / "band_saliency.png")
    plt.close("all")

    print(f"  Saved to {out}")


def run_phase4_2_time_frequency_saliency(
    analyzer: ERDERSAnalyzer,
    x: torch.Tensor,
) -> None:
    """Phase 4.2: Time-frequency saliency analysis."""
    print("\n" + "=" * 60)
    print("PHASE 4.2: Time-Frequency Saliency")
    print("=" * 60)

    out = OUTPUT_DIR / "4_2_tf_saliency"
    out.mkdir(parents=True, exist_ok=True)

    # First compute saliency map on the original data
    print("  Computing integrated gradients saliency map...")
    saliency_map = analyzer.saliency.integrated_gradients(x)

    print("  Computing time-frequency decomposition...")
    times, freqs, tf_matrix = analyzer.time_frequency_saliency(
        saliency_map, sfreq=SFREQ, tmin=TMIN, n_freqs=40, freq_range=(1, 45),
    )
    print(f"    TF matrix shape: {tf_matrix.shape} (freqs x time)")

    # Identify peak frequency
    freq_power = tf_matrix.mean(axis=1)
    peak_freq_idx = np.argmax(freq_power)
    print(f"    Peak saliency frequency: {freqs[peak_freq_idx]:.1f} Hz")

    # Plot
    plot_time_frequency_saliency(
        times, freqs, tf_matrix, cue_onset=CUE_ONSET, save_path=out / "tf_saliency.png",
    )
    plt.close("all")

    # Save peak info
    with open(out / "tf_summary.txt", "w") as f:
        f.write("Time-Frequency Saliency Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Peak frequency: {freqs[peak_freq_idx]:.1f} Hz\n")
        f.write(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz\n")
        f.write(f"TF matrix shape: {tf_matrix.shape}\n\n")
        f.write("Frequency-averaged power (top 5 bins):\n")
        top_freq_idx = np.argsort(freq_power)[::-1][:5]
        for idx in top_freq_idx:
            f.write(f"  {freqs[idx]:.1f} Hz: {freq_power[idx]:.6f}\n")

    print(f"  Saved to {out}")


def run_phase4_3_model_derived_erd(
    analyzer: ERDERSAnalyzer,
    x: torch.Tensor,
    y: torch.Tensor,
    channel_names: list[str],
) -> dict:
    """Phase 4.3: Model-derived ERD/ERS curves."""
    print("\n" + "=" * 60)
    print("PHASE 4.3: Model-Derived ERD/ERS")
    print("=" * 60)

    out = OUTPUT_DIR / "4_3_model_erd"
    out.mkdir(parents=True, exist_ok=True)

    print("  Extracting Block 1 activations and computing ERD/ERS...")
    model_erd = analyzer.model_derived_erd_ers(
        x, y, channel_names, sfreq=SFREQ, tmin=TMIN,
    )

    # Report C3/C4 findings
    target_channels = [ch for ch in ['C3', 'C4'] if ch in channel_names]
    if not target_channels:
        target_channels = channel_names[:2]
        print(f"  C3/C4 not found, using {target_channels} instead")

    for ch_name in target_channels:
        print(f"\n  {ch_name}:")
        for band_name in model_erd:
            print(f"    {band_name} band:")
            for cls_key in sorted(model_erd[band_name].keys()):
                if ch_name in model_erd[band_name][cls_key]:
                    power = model_erd[band_name][cls_key][ch_name]['power']
                    print(f"      Class {cls_key}: mean_power={power.mean():.6f}, "
                          f"std={power.std():.6f}")

        # Plot per-channel curves
        plot_erd_ers_curves(
            model_erd, ch_name, cue_onset=CUE_ONSET,
            save_path=out / f"model_erd_{ch_name}.png",
        )
        plt.close("all")

    print(f"  Saved to {out}")
    return model_erd


def run_phase4_4_classical_erd(
    analyzer: ERDERSAnalyzer,
    data: np.ndarray,
    labels: np.ndarray,
    channel_names: list[str],
    model_erd: dict,
) -> None:
    """Phase 4.4: Classical ERD/ERS and comparison with model."""
    print("\n" + "=" * 60)
    print("PHASE 4.4: Classical ERD/ERS & Model Comparison")
    print("=" * 60)

    out = OUTPUT_DIR / "4_4_comparison"
    out.mkdir(parents=True, exist_ok=True)

    print("  Computing classical ERD/ERS from raw EEG...")
    classical_erd = analyzer.classical_erd_ers(
        data, labels, channel_names,
        sfreq=SFREQ, tmin=TMIN, baseline_window=(-1.0, -0.5),
    )

    # Report C3/C4 findings
    target_channels = [ch for ch in ['C3', 'C4'] if ch in channel_names]
    if not target_channels:
        target_channels = channel_names[:2]

    for ch_name in target_channels:
        print(f"\n  Classical ERD/ERS at {ch_name}:")
        for band_name in classical_erd:
            print(f"    {band_name} band:")
            for cls_key in sorted(classical_erd[band_name].keys()):
                if ch_name in classical_erd[band_name][cls_key]:
                    power = classical_erd[band_name][cls_key][ch_name]['power']
                    # Check post-cue ERD (0 to 2s)
                    times = classical_erd[band_name][cls_key][ch_name]['times']
                    post_mask = times >= 0
                    post_mean = power[post_mask].mean()
                    print(f"      Class {cls_key}: post-cue mean ERD%={post_mean:.1f}")

        # Plot per-channel classical curves
        plot_erd_ers_curves(
            classical_erd, ch_name, cue_onset=CUE_ONSET,
            save_path=out / f"classical_erd_{ch_name}.png",
        )
        plt.close("all")

    # Compare model vs classical
    print("\n  Computing correlation between model and classical ERD/ERS...")
    correlations = analyzer.compare_model_vs_classical(model_erd, classical_erd)

    print("\n  Correlation results:")
    with open(out / "correlation_results.txt", "w") as f:
        f.write("Model vs Classical ERD/ERS Correlation\n")
        f.write("=" * 50 + "\n\n")

        for band_name in correlations:
            for cls_key in correlations[band_name]:
                for ch_name in correlations[band_name][cls_key]:
                    r = correlations[band_name][cls_key][ch_name]
                    line = f"{band_name} | Class {cls_key} | {ch_name}: r = {r:.4f}"
                    print(f"    {line}")
                    f.write(f"{line}\n")

    # Side-by-side comparison plot
    plot_erd_ers_comparison(
        model_erd, classical_erd, channel_names,
        correlations=correlations,
        save_path=out / "model_vs_classical.png",
    )
    plt.close("all")

    print(f"  Saved to {out}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("Phase 4: ERD/ERS Analysis")
    print("=" * 60)

    # Load model
    model = load_model()

    # Load data
    print(f"\nLoading data from {len(SUBJECTS_TO_LOAD)} subjects...")
    data, labels, channel_names = load_multi_subject_data(SUBJECTS_TO_LOAD)
    print(f"Channel names: {channel_names}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare tensors — use subset for speed
    n_subset = min(200, len(labels))
    x = torch.tensor(data[:n_subset], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels[:n_subset], dtype=torch.long)
    print(f"\nUsing {n_subset} trials for analysis, shape={x.shape}")

    # Initialize analyzer
    analyzer = ERDERSAnalyzer(model, device="cpu", batch_size=32)

    # Run all four sub-analyses
    run_phase4_1_frequency_band_saliency(analyzer, x, y)
    run_phase4_2_time_frequency_saliency(analyzer, x)
    model_erd = run_phase4_3_model_derived_erd(analyzer, x, y, channel_names)
    run_phase4_4_classical_erd(
        analyzer, data[:n_subset], labels[:n_subset], channel_names, model_erd,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE — KEY FINDINGS")
    print("=" * 60)
    print("  Check the following output directories:")
    print(f"    4.1 Band saliency:    {OUTPUT_DIR / '4_1_band_saliency'}")
    print(f"    4.2 TF saliency:      {OUTPUT_DIR / '4_2_tf_saliency'}")
    print(f"    4.3 Model ERD/ERS:    {OUTPUT_DIR / '4_3_model_erd'}")
    print(f"    4.4 Classical + comp:  {OUTPUT_DIR / '4_4_comparison'}")
    print()
    print("  Look for:")
    print("    - Which frequency bands drive classification (4.1)")
    print("    - Whether mu/beta ERD is visible in model saliency (4.2)")
    print("    - Contralateral ERD at C3/C4 in model activations (4.3)")
    print("    - Correlation between model and classical ERD/ERS (4.4)")
    print("=" * 60)


if __name__ == "__main__":
    main()
