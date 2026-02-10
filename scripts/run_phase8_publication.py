"""
Phase 8: Publication Figures & Statistical Testing.

Generates publication-quality figures (topographic saliency maps, temporal
importance panels, filter galleries, summary tables) and runs statistical
tests (permutation test, bootstrap CIs) on the all-subjects EEGNet model.
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eeg_mi.models.eegnet import EEGNet
from eeg_mi.interpretability.publication import (
    PublicationFigureGenerator,
    permutation_test_pre_vs_post,
    bootstrap_channel_importance,
    bootstrap_temporal_difference,
    plot_permutation_test,
    plot_bootstrap_ci,
)
from eeg_mi.interpretability import SaliencyMapGenerator, TemporalImportanceAnalyzer


# ── Configuration ──────────────────────────────────────────────────────────
MODEL_PATH = project_root / "all_subjects_interpretation" / "models" / "model.pt"
DATA_DIR = project_root / "data" / "raw" / "apple_catcher"
OUTPUT_DIR = project_root / "results" / "interpretability" / "phase8_publication"
TMIN, TMAX = -1.0, 2.0
SFREQ = 250.0
N_SAMPLES = int((TMAX - TMIN) * SFREQ)  # 750
CUE_ONSET = 0.0
CUE_IDX = int((CUE_ONSET - TMIN) * SFREQ)  # 250
N_CHANNELS = 32
N_CLASSES = 2
SUBJECTS_TO_LOAD = ["s01", "s02", "s03", "s04", "s05"]


# ── Data & model loading (same pattern as run_analysis.py) ─────────────────

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

    all_data: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

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
    all_data: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    channel_names: list[str] | None = None

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


# ── Phase 8 runners ───────────────────────────────────────────────────────

def run_phase8_figures(
    model: EEGNet,
    data: np.ndarray,
    labels: np.ndarray,
    channel_names: list[str],
) -> dict:
    """Generate all Phase 8 publication figures (8.1 -- 8.4)."""
    print("\n" + "=" * 60)
    print("PHASE 8: Publication Figures & Statistical Testing")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare tensors (subset for speed)
    n_subset = min(200, len(labels))
    x = torch.tensor(data[:n_subset], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels[:n_subset], dtype=torch.long)
    print(f"  Using {n_subset} trials, shape={x.shape}")

    pub = PublicationFigureGenerator(model, device="cpu", batch_size=32)

    # ── 8.1 Topographic saliency maps ──
    print("\n  8.1 Topographic saliency maps...")
    fig_topo = pub.generate_topographic_maps(
        x, channel_names, sfreq=SFREQ, tmin=TMIN,
        time_points=[-0.5, 0.0, 0.5, 1.0],
        method='integrated_gradients',
    )
    fig_topo.savefig(OUTPUT_DIR / "topographic_saliency.png", dpi=300, bbox_inches='tight')
    fig_topo.savefig(OUTPUT_DIR / "topographic_saliency.pdf", bbox_inches='tight')
    plt.close(fig_topo)
    print("    Saved topographic_saliency.png/.pdf")

    # ── 8.2 Temporal importance figure ──
    print("  8.2 Temporal importance figure...")
    fig_temp = pub.generate_temporal_importance_figure(
        x, y, channel_names, sfreq=SFREQ, tmin=TMIN, cue_onset=CUE_ONSET,
    )
    fig_temp.savefig(OUTPUT_DIR / "temporal_importance.png", dpi=300, bbox_inches='tight')
    fig_temp.savefig(OUTPUT_DIR / "temporal_importance.pdf", bbox_inches='tight')
    plt.close(fig_temp)
    print("    Saved temporal_importance.png/.pdf")

    # ── 8.3 Filter gallery ──
    print("  8.3 Filter gallery...")
    fig_filt = pub.generate_filter_gallery(channel_names, sfreq=SFREQ)
    fig_filt.savefig(OUTPUT_DIR / "filter_gallery.png", dpi=300, bbox_inches='tight')
    fig_filt.savefig(OUTPUT_DIR / "filter_gallery.pdf", bbox_inches='tight')
    plt.close(fig_filt)
    print("    Saved filter_gallery.png/.pdf")

    # ── Collect analysis results for summary table and stats ──
    print("  Computing analysis results for summary table...")
    analyzer = TemporalImportanceAnalyzer(model, device="cpu", batch_size=32)
    pre_post = analyzer.analyze_pre_vs_post_cue(
        x, y, cue_onset=CUE_ONSET, sfreq=SFREQ, tmin=TMIN,
        method="integrated_gradients",
    )
    chan_results = analyzer.analyze_channel_importance(
        x, channel_names=channel_names, method="integrated_gradients", top_k=5,
    )

    analysis_results: dict = {
        'baseline_accuracy': pre_post['baseline_accuracy'],
        'pre_cue_importance': pre_post['pre_cue_importance'],
        'post_cue_importance': pre_post['post_cue_importance'],
        'importance_ratio': pre_post['importance_ratio'],
        'pre_cue_accuracy_drop': pre_post['pre_cue_accuracy_drop'],
        'post_cue_accuracy_drop': pre_post['post_cue_accuracy_drop'],
        'top_channels': chan_results.get('top_channels', []),
    }

    # ── 8.5 Statistical tests ──
    print("\n  8.5 Statistical tests...")
    saliency_gen = SaliencyMapGenerator(model, device="cpu")
    saliency_map = saliency_gen.integrated_gradients(x)

    # Permutation test
    print("    Permutation test (pre vs post cue)...")
    perm_result = permutation_test_pre_vs_post(
        saliency_map, cue_idx=CUE_IDX, n_permutations=10_000,
    )
    print(f"      Observed diff: {perm_result.observed_difference:.6f}")
    print(f"      p-value:       {perm_result.p_value:.4f}")
    analysis_results['permutation_p_value'] = perm_result.p_value

    fig_perm = plot_permutation_test(
        perm_result.observed_difference,
        perm_result.null_distribution,
        perm_result.p_value,
        save_path=OUTPUT_DIR / "permutation_test.png",
    )
    plt.close(fig_perm)

    # Bootstrap CI for pre-post difference
    print("    Bootstrap CI for pre-post importance difference...")
    boot_diff = bootstrap_temporal_difference(
        saliency_map, cue_idx=CUE_IDX, n_bootstrap=5_000, ci=0.95,
    )
    print(f"      Mean diff:  {boot_diff.means[0]:.6f}")
    print(f"      95% CI:     [{boot_diff.lower_ci[0]:.6f}, {boot_diff.upper_ci[0]:.6f}]")
    analysis_results['bootstrap_ci'] = (
        f"{boot_diff.means[0]:.4f} "
        f"[{boot_diff.lower_ci[0]:.4f}, {boot_diff.upper_ci[0]:.4f}]"
    )

    # Bootstrap CI for channel importance
    print("    Bootstrap CI for channel importance...")
    boot_chan = bootstrap_channel_importance(
        saliency_map, n_bootstrap=5_000, ci=0.95,
    )
    fig_boot = plot_bootstrap_ci(
        boot_chan.means, boot_chan.lower_ci, boot_chan.upper_ci,
        channel_names,
        save_path=OUTPUT_DIR / "channel_importance_bootstrap.png",
    )
    plt.close(fig_boot)

    # ── 8.4 Summary table ──
    print("\n  8.4 Summary table...")
    fig_table = pub.generate_summary_table(
        analysis_results,
        save_path=OUTPUT_DIR / "summary_table.png",
    )
    plt.close(fig_table)
    print("    Saved summary_table.png")

    # ── Save statistical results to text ──
    stats_path = OUTPUT_DIR / "statistical_results.txt"
    with open(stats_path, "w") as f:
        f.write("Phase 8 Statistical Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("PERMUTATION TEST: Pre-cue vs Post-cue Importance\n")
        f.write("-" * 40 + "\n")
        f.write(f"Observed difference (pre - post): {perm_result.observed_difference:.6f}\n")
        f.write(f"p-value (two-sided):              {perm_result.p_value:.4f}\n")
        f.write(f"Number of permutations:           10000\n\n")

        f.write("BOOTSTRAP CI: Pre-Post Importance Difference\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean difference:  {boot_diff.means[0]:.6f}\n")
        f.write(f"95% CI lower:     {boot_diff.lower_ci[0]:.6f}\n")
        f.write(f"95% CI upper:     {boot_diff.upper_ci[0]:.6f}\n\n")

        f.write("BOOTSTRAP CI: Channel Importance Rankings\n")
        f.write("-" * 40 + "\n")
        order = np.argsort(boot_chan.means)[::-1]
        f.write(f"{'Rank':<6}{'Channel':<10}{'Mean':>10}{'Lower CI':>12}{'Upper CI':>12}\n")
        for rank, idx in enumerate(order, 1):
            f.write(
                f"{rank:<6}{channel_names[idx]:<10}"
                f"{boot_chan.means[idx]:>10.6f}"
                f"{boot_chan.lower_ci[idx]:>12.6f}"
                f"{boot_chan.upper_ci[idx]:>12.6f}\n"
            )
        f.write("\n")

        f.write("ANALYSIS SUMMARY\n")
        f.write("-" * 40 + "\n")
        for key, val in analysis_results.items():
            f.write(f"{key}: {val}\n")

    print(f"    Saved statistical results to {stats_path}")

    return analysis_results


def print_key_findings(analysis_results: dict) -> None:
    """Print a concise summary of the key findings."""
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    pre = analysis_results.get('pre_cue_importance', 0)
    post = analysis_results.get('post_cue_importance', 0)
    ratio = analysis_results.get('importance_ratio', 0)
    p_val = analysis_results.get('permutation_p_value', 1.0)
    top_ch = analysis_results.get('top_channels', [])

    print(f"  Pre-cue importance:   {pre:.6f}")
    print(f"  Post-cue importance:  {post:.6f}")
    print(f"  Ratio (pre/post):     {ratio:.3f}")
    print(f"  Permutation p-value:  {p_val:.4f}")

    if ratio > 1.0:
        print("  --> Model relies MORE on pre-cue period")
    else:
        print("  --> Model relies MORE on post-cue period")

    if p_val < 0.05:
        print("  --> The pre-post difference is statistically significant (p < 0.05)")
    else:
        print("  --> The pre-post difference is NOT statistically significant")

    if top_ch:
        print(f"  Top channels: {', '.join(str(c) for c in top_ch[:5])}")

    print(f"\n  Bootstrap CI: {analysis_results.get('bootstrap_ci', 'N/A')}")


def main() -> None:
    print("Phase 8: Publication Figures & Statistical Testing")
    print("=" * 60)

    # Set publication-quality defaults
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })

    # Load model
    model = load_model()

    # Load data
    print(f"\nLoading data from {len(SUBJECTS_TO_LOAD)} subjects...")
    data, labels, channel_names = load_multi_subject_data(SUBJECTS_TO_LOAD)
    print(f"Channel names: {channel_names}")

    # Run Phase 8
    analysis_results = run_phase8_figures(model, data, labels, channel_names)

    # Print key findings
    print_key_findings(analysis_results)

    print("\n" + "=" * 60)
    print("PHASE 8 COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
