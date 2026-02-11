"""
Run comprehensive interpretability analysis on the all-subjects EEGNet model.

Executes Phase 1 (filter visualization), Phase 2 (saliency/ablation),
Phase 3 (DeepLIFT cross-validation), and Phase 6 (artifact investigation)
from SPEC.md.
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
from eeg_mi.interpretability.filters import FilterVisualizer
from eeg_mi.interpretability import (
    TemporalImportanceAnalyzer,
    SaliencyMapGenerator,
)
from eeg_mi.interpretability.artifacts import ArtifactAnalyzer


# ── Configuration ──────────────────────────────────────────────────────────
MODEL_PATH = project_root / "all_subjects_interpretation" / "models" / "model.pt"
DATA_DIR = project_root / "data" / "raw" / "apple_catcher"
OUTPUT_DIR = project_root / "results" / "interpretability"
TMIN, TMAX = -1.0, 2.0
SFREQ = 250.0
N_SAMPLES = int((TMAX - TMIN) * SFREQ)  # 750
CUE_ONSET = 0.0
N_CHANNELS = 32
N_CLASSES = 2
SUBJECTS_TO_LOAD = ["s01", "s02", "s03", "s04", "s05"]  # Subset for speed


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


def run_phase1_filters(model: EEGNet, channel_names: list[str]) -> None:
    """Phase 1: Weight & filter visualization."""
    print("\n" + "=" * 60)
    print("PHASE 1: Filter & Weight Visualization")
    print("=" * 60)

    out = OUTPUT_DIR / "phase1_filters"
    out.mkdir(parents=True, exist_ok=True)

    fv = FilterVisualizer()

    # 1.1 Temporal filters
    print("  Temporal filters...")
    temp_filters = fv.extract_temporal_filters(model)
    print(f"    Shape: {temp_filters.shape}")
    fv.plot_temporal_filters(temp_filters, sfreq=SFREQ, save_path=out / "temporal_filters.png")
    plt.close("all")

    # 1.1b Frequency response overlay
    print("  Frequency response overlay...")
    fv.plot_filter_frequency_response(temp_filters, sfreq=SFREQ, save_path=out / "freq_response_overlay.png")
    plt.close("all")

    # 1.2 Spatial filters
    print("  Spatial filters...")
    spat_filters = fv.extract_spatial_filters(model)
    print(f"    Shape: {spat_filters.shape}")
    fv.plot_spatial_filters(spat_filters, channel_names=channel_names, save_path=out / "spatial_filters.png")
    plt.close("all")

    # 1.3 Separable conv filters
    print("  Separable convolution filters...")
    dw, pw = fv.extract_separable_filters(model)
    print(f"    Depthwise: {dw.shape}, Pointwise: {pw.shape}")

    # 1.4 Classifier weights
    print("  Classifier weights...")
    cls_weights = fv.extract_classifier_weights(model)
    print(f"    Shape: {cls_weights.shape}")
    fv.plot_classifier_weights(
        cls_weights, class_names=["MI_left", "MI_right"],
        save_path=out / "classifier_weights.png",
    )
    plt.close("all")

    print(f"  Saved to {out}")


def run_phase2_saliency(
    model: EEGNet,
    data: np.ndarray,
    labels: np.ndarray,
    channel_names: list[str],
) -> None:
    """Phase 2: Gradient-based attribution with existing tools."""
    print("\n" + "=" * 60)
    print("PHASE 2: Gradient-Based Attribution")
    print("=" * 60)

    out = OUTPUT_DIR / "phase2_saliency"
    out.mkdir(parents=True, exist_ok=True)

    # Prepare tensors — use subset for speed
    n_subset = min(200, len(labels))
    x = torch.tensor(data[:n_subset], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels[:n_subset], dtype=torch.long)
    print(f"  Using {n_subset} trials, shape={x.shape}")

    analyzer = TemporalImportanceAnalyzer(model, device="cpu", batch_size=32)

    # 2.1 Pre-cue vs post-cue
    print("  Pre-cue vs post-cue analysis (integrated gradients)...")
    pre_post = analyzer.analyze_pre_vs_post_cue(
        x, y, cue_onset=CUE_ONSET, sfreq=SFREQ, tmin=TMIN,
        method="integrated_gradients",
    )
    print(f"    Pre-cue importance:  {pre_post['pre_cue_importance']:.6f}")
    print(f"    Post-cue importance: {pre_post['post_cue_importance']:.6f}")
    print(f"    Ratio (pre/post):    {pre_post['importance_ratio']:.3f}")
    print(f"    Pre-cue acc drop:    {pre_post['pre_cue_accuracy_drop']:.4f}")
    print(f"    Post-cue acc drop:   {pre_post['post_cue_accuracy_drop']:.4f}")
    print(f"    Baseline accuracy:   {pre_post['baseline_accuracy']:.4f}")

    # Save numerical results
    with open(out / "pre_post_results.txt", "w") as f:
        for k, v in pre_post.items():
            f.write(f"{k}: {v:.6f}\n")

    # 2.2 Temporal profile
    print("  Temporal importance profile...")
    temporal = analyzer.analyze_temporal_profile(
        x, y, sfreq=SFREQ, tmin=TMIN, window_size=0.5, step_size=0.1,
        method="integrated_gradients",
    )

    from eeg_mi.interpretability.visualization import (
        plot_temporal_importance,
        plot_pre_vs_post_cue_comparison,
        plot_saliency_map,
        plot_channel_importance,
    )

    plot_pre_vs_post_cue_comparison(pre_post, save_path=out / "pre_post_comparison.png")
    plt.close("all")

    plot_temporal_importance(
        temporal["times"], temporal["gradient_importance"],
        temporal["ablation_window_centers"], temporal["ablation_accuracy_drops"],
        cue_onset=CUE_ONSET, save_path=out / "temporal_profile.png",
    )
    plt.close("all")

    # 2.3 Channel importance
    print("  Channel importance...")
    chan_results = analyzer.analyze_channel_importance(
        x, channel_names=channel_names, method="integrated_gradients", top_k=15,
    )
    print(f"    Top 5 channels: {chan_results.get('top_channels', chan_results['top_indices'][:5])}")

    plot_channel_importance(
        chan_results["channel_importance"], channel_names, top_k=15,
        save_path=out / "channel_importance.png",
    )
    plt.close("all")

    # 2.4 Saliency heatmap
    print("  Saliency heatmap...")
    saliency_gen = SaliencyMapGenerator(model, device="cpu")
    saliency_map = saliency_gen.integrated_gradients(x)
    times, importance_map, _ = saliency_gen.compute_time_channel_map(saliency_map, SFREQ, TMIN)

    plot_saliency_map(
        importance_map, times, channel_names, cue_onset=CUE_ONSET,
        save_path=out / "saliency_heatmap.png",
    )
    plt.close("all")

    # 2.4b Class-conditional saliency
    print("  Class-conditional saliency (left vs right)...")
    for cls_idx, cls_name in [(0, "MI_left"), (1, "MI_right")]:
        mask = labels[:n_subset] == cls_idx
        if mask.sum() > 0:
            x_cls = x[mask]
            sal_cls = saliency_gen.integrated_gradients(x_cls)
            _, imp_cls, _ = saliency_gen.compute_time_channel_map(sal_cls, SFREQ, TMIN)
            plot_saliency_map(
                imp_cls, times, channel_names, cue_onset=CUE_ONSET,
                title=f"Saliency Map — {cls_name}",
                save_path=out / f"saliency_{cls_name}.png",
            )
            plt.close("all")

    # 2.5 Critical time window
    print("  Finding critical time window...")
    crit_start, crit_end = analyzer.find_critical_time_window(
        x, y, sfreq=SFREQ, tmin=TMIN, threshold=0.05,
    )
    print(f"    Critical window: {crit_start:.2f}s to {crit_end:.2f}s")

    print(f"  Saved to {out}")


def run_phase6_artifacts(
    model: EEGNet,
    data: np.ndarray,
    labels: np.ndarray,
    channel_names: list[str],
) -> None:
    """Phase 6: Artifact investigation."""
    print("\n" + "=" * 60)
    print("PHASE 6: Artifact Investigation")
    print("=" * 60)

    out = OUTPUT_DIR / "phase6_artifacts"
    out.mkdir(parents=True, exist_ok=True)

    n_subset = min(200, len(labels))
    x = torch.tensor(data[:n_subset], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels[:n_subset], dtype=torch.long)

    artifact_analyzer = ArtifactAnalyzer(model, device="cpu", batch_size=32)

    # 6.1 Channel group analysis
    print("  Channel group saliency analysis...")
    group_result = artifact_analyzer.analyze_channel_groups(
        x, y, channel_names, sfreq=SFREQ, tmin=TMIN,
        method="integrated_gradients",
    )
    print(f"    Group importance: {group_result.group_importance}")
    print(f"    Frontal/Central ratio: {group_result.frontal_central_ratio:.3f}")

    from eeg_mi.interpretability.visualization import (
        plot_channel_group_comparison,
        plot_artifact_trial_analysis,
        plot_single_channel_ablation,
    )

    plot_channel_group_comparison(group_result, save_path=out / "channel_group_importance.png")
    plt.close("all")

    # 6.5 Channel group ablation
    print("  Channel group ablation...")
    ablation_result = artifact_analyzer.channel_group_ablation(x, y, channel_names)
    baseline_key = "baseline_accuracy" if "baseline_accuracy" in ablation_result else "baseline"
    baseline_val = ablation_result[baseline_key]
    if isinstance(baseline_val, dict):
        baseline_acc = baseline_val.get("accuracy", baseline_val)
    else:
        baseline_acc = baseline_val
    print(f"    Baseline accuracy: {baseline_acc:.4f}")
    for group in ["frontal", "central", "parietal", "temporal"]:
        if group in ablation_result:
            v = ablation_result[group]
            if isinstance(v, dict):
                print(f"    {group:>10} ablated: acc={v.get('accuracy', 'N/A')}, "
                      f"drop={v.get('accuracy_drop', v.get('drop', 'N/A'))}")
            else:
                print(f"    {group:>10} ablated: {v}")

    with open(out / "channel_group_ablation.txt", "w") as f:
        f.write(f"baseline_accuracy: {baseline_acc}\n\n")
        for k, v in ablation_result.items():
            if isinstance(v, dict):
                f.write(f"{k}: {v}\n")

    # 6.3 Artifact trial detection
    print("  Artifact trial detection...")
    artifact_result = artifact_analyzer.detect_artifact_trials(x, channel_names, threshold_uv=100.0)
    n_art = artifact_result.n_artifact_trials
    n_total = artifact_result.n_artifact_trials + artifact_result.n_clean_trials
    print(f"    Artifact trials: {n_art}/{n_total} ({n_art/n_total*100:.1f}%)")

    # 6.3b Clean vs artifact comparison
    print("  Clean vs artifact trial comparison...")
    clean_vs = artifact_analyzer.compare_clean_vs_artifact_trials(x, y, channel_names, threshold_uv=100.0)
    print(f"    All trials accuracy:      {clean_vs['all_accuracy']:.4f} (n={clean_vs['n_total']})")
    print(f"    Clean trials accuracy:    {clean_vs['clean_accuracy']:.4f} (n={clean_vs['n_clean']})")
    artifact_acc = clean_vs['artifact_accuracy']
    if not np.isnan(artifact_acc):
        print(f"    Artifact trials accuracy: {artifact_acc:.4f} (n={clean_vs['n_artifact']})")
    else:
        print(f"    Artifact trials accuracy: N/A (n={clean_vs['n_artifact']})")

    plot_artifact_trial_analysis(clean_vs, save_path=out / "clean_vs_artifact.png")
    plt.close("all")

    # 6.5b Single channel ablation
    print("  Single channel ablation (this takes a minute)...")
    sca_result = artifact_analyzer.single_channel_ablation(x, y, channel_names)
    drops = sca_result["accuracy_drops"]
    ranking = sca_result["ranking"]
    print(f"    Most important channels:")
    for i in range(min(5, len(ranking))):
        idx = ranking[i]
        print(f"      {channel_names[idx]:>5}: drop={drops[idx]:.4f}")

    plot_single_channel_ablation(drops, channel_names, save_path=out / "single_channel_ablation.png")
    plt.close("all")

    print(f"  Saved to {out}")


def run_phase3_deeplift(
    model: EEGNet,
    data: np.ndarray,
    labels: np.ndarray,
    channel_names: list[str],
) -> None:
    """Phase 3: DeepLIFT attribution — cross-validate with gradient methods."""
    print("\n" + "=" * 60)
    print("PHASE 3: DeepLIFT Attribution")
    print("=" * 60)

    out = OUTPUT_DIR / "phase3_deeplift"
    out.mkdir(parents=True, exist_ok=True)

    n_subset = min(200, len(labels))
    x = torch.tensor(data[:n_subset], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels[:n_subset], dtype=torch.long)
    print(f"  Using {n_subset} trials, shape={x.shape}")

    saliency_gen = SaliencyMapGenerator(model, device="cpu")
    analyzer = TemporalImportanceAnalyzer(model, device="cpu", batch_size=32)

    from eeg_mi.interpretability.visualization import (
        plot_method_comparison,
        plot_temporal_importance,
        plot_pre_vs_post_cue_comparison,
        plot_channel_importance,
        plot_saliency_map,
    )

    # ── 3.2 Compare all attribution methods ──────────────────────────────
    print("  Running compare_methods() (all 4 methods)...")
    comparison = saliency_gen.compare_methods(
        x, target_class=None,
        methods=["vanilla", "integrated_gradients", "gradient_x_input", "deeplift"],
        sfreq=SFREQ, tmin=TMIN,
    )

    plot_method_comparison(comparison, cue_onset=CUE_ONSET, save_path=out / "method_comparison.png")
    plt.close("all")

    # Compute inter-method correlation (temporal importance profiles)
    from scipy.stats import pearsonr, spearmanr

    method_names = list(comparison.keys())
    n_methods = len(method_names)

    print("\n  Inter-method temporal importance correlations (Pearson r):")
    corr_lines = ["Inter-method Temporal Importance Correlations", "=" * 55, ""]

    pearson_matrix = np.zeros((n_methods, n_methods))
    spearman_matrix = np.zeros((n_methods, n_methods))

    for i in range(n_methods):
        for j in range(n_methods):
            ti = comparison[method_names[i]]["temporal_importance"]
            tj = comparison[method_names[j]]["temporal_importance"]
            r_p, _ = pearsonr(ti, tj)
            r_s, _ = spearmanr(ti, tj)
            pearson_matrix[i, j] = r_p
            spearman_matrix[i, j] = r_s

    # Print Pearson matrix
    header = f"{'':>22}" + "".join(f"{m:>22}" for m in method_names)
    corr_lines.append("Pearson r:")
    corr_lines.append(header)
    for i, m in enumerate(method_names):
        row = f"{m:>22}" + "".join(f"{pearson_matrix[i, j]:>22.4f}" for j in range(n_methods))
        corr_lines.append(row)
        print(f"    {row}")

    corr_lines.append("")
    corr_lines.append("Spearman rho:")
    corr_lines.append(header)
    for i, m in enumerate(method_names):
        row = f"{m:>22}" + "".join(f"{spearman_matrix[i, j]:>22.4f}" for j in range(n_methods))
        corr_lines.append(row)

    with open(out / "method_correlations.txt", "w") as f:
        f.write("\n".join(corr_lines) + "\n")

    # Per-method pre/post-cue importance summary
    print("\n  Per-method pre/post-cue importance:")
    summary_lines = [
        "Per-Method Pre/Post-Cue Importance",
        "=" * 55,
        "",
        f"{'Method':>22}  {'Pre-Cue':>10}  {'Post-Cue':>10}  {'Ratio':>8}",
    ]

    cue_idx = int((CUE_ONSET - TMIN) * SFREQ)
    for method in method_names:
        sal_map = comparison[method]["saliency_map"]
        pre = sal_map[:, :, :cue_idx].mean()
        post = sal_map[:, :, cue_idx:].mean()
        ratio = pre / (post + 1e-10)
        line = f"{method:>22}  {pre:>10.6f}  {post:>10.6f}  {ratio:>8.3f}"
        summary_lines.append(line)
        print(f"    {line}")

    with open(out / "pre_post_by_method.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # ── 3.2b DeepLIFT-specific pre/post-cue analysis (with ablation) ─────
    print("\n  DeepLIFT pre-cue vs post-cue (with ablation)...")
    dl_pre_post = analyzer.analyze_pre_vs_post_cue(
        x, y, cue_onset=CUE_ONSET, sfreq=SFREQ, tmin=TMIN,
        method="deeplift",
    )

    print(f"    Pre-cue importance:  {dl_pre_post['pre_cue_importance']:.6f}")
    print(f"    Post-cue importance: {dl_pre_post['post_cue_importance']:.6f}")
    print(f"    Ratio (pre/post):    {dl_pre_post['importance_ratio']:.3f}")
    print(f"    Baseline accuracy:   {dl_pre_post['baseline_accuracy']:.4f}")

    plot_pre_vs_post_cue_comparison(
        dl_pre_post, cue_onset=CUE_ONSET,
        save_path=out / "deeplift_pre_post_comparison.png",
    )
    plt.close("all")

    with open(out / "deeplift_pre_post_results.txt", "w") as f:
        for k, v in dl_pre_post.items():
            f.write(f"{k}: {v:.6f}\n")

    # ── 3.3 DeepLIFT temporal profile ────────────────────────────────────
    print("  DeepLIFT temporal profile (sliding window)...")
    dl_temporal = analyzer.analyze_temporal_profile(
        x, y, sfreq=SFREQ, tmin=TMIN, window_size=0.5, step_size=0.1,
        method="deeplift",
    )

    plot_temporal_importance(
        dl_temporal["times"], dl_temporal["gradient_importance"],
        dl_temporal["ablation_window_centers"], dl_temporal["ablation_accuracy_drops"],
        cue_onset=CUE_ONSET, title="DeepLIFT Temporal Importance",
        save_path=out / "deeplift_temporal_profile.png",
    )
    plt.close("all")

    # ── 3.3b DeepLIFT channel importance ─────────────────────────────────
    print("  DeepLIFT channel importance...")
    dl_channels = analyzer.analyze_channel_importance(
        x, channel_names=channel_names, method="deeplift", top_k=15,
    )
    top_ch = dl_channels.get("top_channels", dl_channels["top_indices"][:5])
    print(f"    Top 5 channels (DeepLIFT): {top_ch}")

    plot_channel_importance(
        dl_channels["channel_importance"], channel_names, top_k=15,
        title="Channel Importance (DeepLIFT)",
        save_path=out / "deeplift_channel_importance.png",
    )
    plt.close("all")

    # ── 3.3c DeepLIFT saliency heatmap ───────────────────────────────────
    print("  DeepLIFT saliency heatmap...")
    dl_saliency = saliency_gen.deeplift(x)
    times, dl_imp_map, _ = saliency_gen.compute_time_channel_map(dl_saliency, SFREQ, TMIN)

    plot_saliency_map(
        dl_imp_map, times, channel_names, cue_onset=CUE_ONSET,
        title="DeepLIFT Saliency Map",
        save_path=out / "deeplift_saliency_heatmap.png",
    )
    plt.close("all")

    # ── Channel importance comparison across methods ─────────────────────
    print("  Channel importance comparison across methods...")
    chan_corr_lines = ["Channel Importance Comparison Across Methods", "=" * 55, ""]

    ig_channels = comparison["integrated_gradients"]["channel_importance"]
    dl_channels_arr = comparison["deeplift"]["channel_importance"]
    r_chan, p_chan = pearsonr(ig_channels, dl_channels_arr)
    rho_chan, p_rho = spearmanr(ig_channels, dl_channels_arr)

    chan_corr_lines.append(f"IG vs DeepLIFT channel importance:")
    chan_corr_lines.append(f"  Pearson r  = {r_chan:.4f} (p = {p_chan:.2e})")
    chan_corr_lines.append(f"  Spearman ρ = {rho_chan:.4f} (p = {p_rho:.2e})")

    print(f"    IG vs DeepLIFT channel importance: r={r_chan:.4f}, ρ={rho_chan:.4f}")

    # Top-5 overlap
    ig_top5 = set(np.argsort(ig_channels)[::-1][:5])
    dl_top5 = set(np.argsort(dl_channels_arr)[::-1][:5])
    overlap = ig_top5 & dl_top5
    chan_corr_lines.append(f"  Top-5 overlap: {len(overlap)}/5 channels")
    if channel_names:
        chan_corr_lines.append(f"  IG top-5:      {[channel_names[i] for i in sorted(ig_top5)]}")
        chan_corr_lines.append(f"  DeepLIFT top-5: {[channel_names[i] for i in sorted(dl_top5)]}")
        chan_corr_lines.append(f"  Shared:         {[channel_names[i] for i in sorted(overlap)]}")

    print(f"    Top-5 overlap: {len(overlap)}/5")

    with open(out / "channel_importance_comparison.txt", "w") as f:
        f.write("\n".join(chan_corr_lines) + "\n")

    print(f"  Saved to {out}")


def main() -> None:
    print("EEGNet Interpretability Analysis")
    print("=" * 60)

    # Load model
    model = load_model()

    # Load data
    print(f"\nLoading data from {len(SUBJECTS_TO_LOAD)} subjects...")
    data, labels, channel_names = load_multi_subject_data(SUBJECTS_TO_LOAD)
    print(f"Channel names: {channel_names}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run analyses
    run_phase1_filters(model, channel_names)
    run_phase2_saliency(model, data, labels, channel_names)
    run_phase3_deeplift(model, data, labels, channel_names)
    run_phase6_artifacts(model, data, labels, channel_names)

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
