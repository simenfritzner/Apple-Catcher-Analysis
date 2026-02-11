"""
Run Phase 7: Cross-subject consistency analysis.

Computes per-subject saliency patterns, inter-subject correlations,
subject clustering, and consistency metrics to assess whether the
EEGNet model relies on shared neural signals or subject-specific artifacts.
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
from eeg_mi.interpretability.cross_subject import (
    CrossSubjectAnalyzer,
    plot_inter_subject_correlation,
    plot_subject_clustering,
    plot_per_subject_temporal_profiles,
    plot_consistency_summary,
)


# ── Configuration ──────────────────────────────────────────────────────────
MODEL_PATH = project_root / "all_subjects_interpretation" / "models" / "model.pt"
DATA_DIR = project_root / "data" / "raw" / "apple_catcher"
OUTPUT_DIR = project_root / "results" / "interpretability" / "phase7_cross_subject"
TMIN, TMAX = -1.0, 2.0
SFREQ = 250.0
N_SAMPLES = int((TMAX - TMIN) * SFREQ)  # 750
CUE_ONSET = 0.0
N_CHANNELS = 32
N_CLASSES = 2
MAX_TRIALS_PER_SUBJECT = 50
SUBJECTS = [f"s{i:02d}" for i in range(1, 41)]


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


def load_all_subjects() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load data for all 40 subjects."""
    subject_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for sid in SUBJECTS:
        print(f"  Loading {sid}...", end=" ")
        try:
            data, labels = load_subject_data(sid)
            subject_data[sid] = (data, labels)
            print(f"{len(labels)} trials, shape={data.shape}")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"Loaded {len(subject_data)}/{len(SUBJECTS)} subjects")
    return subject_data


def save_text_results(
    per_subject_results: dict,
    corr_matrix: np.ndarray,
    subject_ids: list[str],
    cluster_labels: np.ndarray,
    sil_score: float,
    consistency: object,
) -> None:
    """Save all numerical results to text files."""
    # Per-subject summary
    with open(OUTPUT_DIR / "per_subject_summary.txt", "w") as f:
        f.write("Subject  Accuracy  N_Trials  MeanTemporalImportance  MeanChannelImportance\n")
        f.write("-" * 80 + "\n")
        for sid in sorted(per_subject_results.keys()):
            r = per_subject_results[sid]
            f.write(
                f"{sid:>7s}  {r.accuracy:8.4f}  {r.n_trials:8d}  "
                f"{r.temporal_importance.mean():22.8f}  "
                f"{r.channel_importance.mean():21.8f}\n"
            )

    # Correlation matrix
    with open(OUTPUT_DIR / "inter_subject_correlation.txt", "w") as f:
        f.write("Inter-subject temporal saliency correlation matrix\n")
        f.write(f"Subjects: {', '.join(subject_ids)}\n\n")
        header = "        " + "  ".join(f"{sid:>6s}" for sid in subject_ids)
        f.write(header + "\n")
        for i, sid in enumerate(subject_ids):
            row = f"{sid:>6s}  " + "  ".join(f"{corr_matrix[i, j]:6.3f}" for j in range(len(subject_ids)))
            f.write(row + "\n")

        n = len(subject_ids)
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]
        f.write(f"\nMean pairwise correlation: {np.mean(upper_tri):.4f}\n")
        f.write(f"Median pairwise correlation: {np.median(upper_tri):.4f}\n")
        f.write(f"Min pairwise correlation: {np.min(upper_tri):.4f}\n")
        f.write(f"Max pairwise correlation: {np.max(upper_tri):.4f}\n")

    # Cluster assignments
    with open(OUTPUT_DIR / "cluster_assignments.txt", "w") as f:
        f.write(f"Agglomerative clustering (n_clusters=3)\n")
        f.write(f"Silhouette score: {sil_score:.4f}\n\n")
        f.write("Subject  Cluster  Accuracy\n")
        f.write("-" * 35 + "\n")
        for i, sid in enumerate(subject_ids):
            acc = per_subject_results[sid].accuracy
            f.write(f"{sid:>7s}  {cluster_labels[i]:7d}  {acc:8.4f}\n")

        # Per-cluster accuracy stats
        f.write("\nCluster  N_Subjects  MeanAcc   StdAcc\n")
        f.write("-" * 45 + "\n")
        for k in sorted(set(cluster_labels)):
            mask = cluster_labels == k
            accs = [per_subject_results[subject_ids[i]].accuracy for i in range(len(subject_ids)) if mask[i]]
            f.write(f"{k:7d}  {sum(mask):10d}  {np.mean(accs):.4f}  {np.std(accs):.4f}\n")

    # Consistency summary
    with open(OUTPUT_DIR / "consistency_summary.txt", "w") as f:
        f.write("Cross-Subject Consistency Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mean inter-subject correlation:   {consistency.mean_inter_subject_correlation:.4f}\n")
        f.write(f"Median inter-subject correlation: {consistency.median_inter_subject_correlation:.4f}\n")
        f.write(f"Temporal CV:                      {consistency.temporal_cv:.4f}\n")
        f.write(f"Channel CV:                       {consistency.channel_cv:.4f}\n")
        f.write(f"Outlier threshold:                {consistency.outlier_threshold:.4f}\n")
        f.write(f"Outlier subjects:                 {consistency.outlier_subjects}\n\n")

        f.write("Per-subject correlation with group mean:\n")
        f.write("-" * 35 + "\n")
        for sid in sorted(consistency.per_subject_group_correlation.keys()):
            r = consistency.per_subject_group_correlation[sid]
            marker = " ** OUTLIER" if sid in consistency.outlier_subjects else ""
            f.write(f"  {sid}: {r:.4f}{marker}\n")


def main() -> None:
    print("=" * 60)
    print("PHASE 7: Cross-Subject Consistency Analysis")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model()

    # Load all subjects
    print(f"\nLoading data from {len(SUBJECTS)} subjects...")
    subject_data = load_all_subjects()

    if len(subject_data) < 3:
        print("ERROR: Need at least 3 subjects for meaningful analysis.")
        return

    # Initialize analyzer
    analyzer = CrossSubjectAnalyzer(model, device="cpu", batch_size=32)

    # ── 7.1 Per-subject saliency patterns ──────────────────────────────
    print("\n" + "-" * 60)
    print("7.1 Computing per-subject saliency patterns...")
    print("-" * 60)
    per_subject_results = analyzer.compute_per_subject_saliency(
        subject_data,
        sfreq=SFREQ,
        tmin=TMIN,
        method='integrated_gradients',
        max_trials=MAX_TRIALS_PER_SUBJECT,
    )

    # Inter-subject correlation
    print("\nComputing inter-subject correlation...")
    corr_matrix, subject_ids = analyzer.compute_inter_subject_correlation(per_subject_results)
    n = len(subject_ids)
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    print(f"  Mean pairwise correlation:   {np.mean(upper_tri):.4f}")
    print(f"  Median pairwise correlation: {np.median(upper_tri):.4f}")
    print(f"  Range: [{np.min(upper_tri):.4f}, {np.max(upper_tri):.4f}]")

    plot_inter_subject_correlation(
        corr_matrix, subject_ids,
        save_path=OUTPUT_DIR / "inter_subject_correlation.png",
    )

    plot_per_subject_temporal_profiles(
        per_subject_results,
        cue_onset=CUE_ONSET, sfreq=SFREQ, tmin=TMIN,
        save_path=OUTPUT_DIR / "temporal_profiles_overlay.png",
    )

    # ── 7.2 Subject clustering ─────────────────────────────────────────
    print("\n" + "-" * 60)
    print("7.2 Clustering subjects by saliency patterns...")
    print("-" * 60)

    cluster_labels, sil_score, cluster_centers, cluster_sids = analyzer.cluster_subjects(
        per_subject_results, n_clusters=3, feature_type='temporal',
    )
    print(f"  Silhouette score: {sil_score:.4f}")

    for k in sorted(set(cluster_labels)):
        members = [cluster_sids[i] for i in range(len(cluster_sids)) if cluster_labels[i] == k]
        accs = [per_subject_results[sid].accuracy for sid in members]
        print(f"  Cluster {k}: {len(members)} subjects, mean acc={np.mean(accs):.3f} +/- {np.std(accs):.3f}")
        print(f"    Members: {', '.join(members)}")

    plot_subject_clustering(
        per_subject_results, cluster_labels, cluster_sids,
        save_path=OUTPUT_DIR / "subject_clustering.png",
    )

    # ── 7.3 Consistency analysis ───────────────────────────────────────
    print("\n" + "-" * 60)
    print("7.3 Analyzing cross-subject consistency...")
    print("-" * 60)

    consistency = analyzer.analyze_consistency(per_subject_results)

    print(f"  Mean inter-subject correlation:   {consistency.mean_inter_subject_correlation:.4f}")
    print(f"  Median inter-subject correlation: {consistency.median_inter_subject_correlation:.4f}")
    print(f"  Temporal importance CV:           {consistency.temporal_cv:.4f}")
    print(f"  Channel importance CV:            {consistency.channel_cv:.4f}")
    print(f"  Outlier threshold (mean - 2*SD):  {consistency.outlier_threshold:.4f}")
    print(f"  Outlier subjects:                 {consistency.outlier_subjects}")

    plot_consistency_summary(
        consistency,
        save_path=OUTPUT_DIR / "consistency_summary.png",
    )

    # ── Save results ───────────────────────────────────────────────────
    print("\nSaving numerical results...")
    save_text_results(
        per_subject_results, corr_matrix, subject_ids,
        cluster_labels, sil_score, consistency,
    )

    # ── Key findings ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    high_corr = np.mean(upper_tri) > 0.7
    print(f"\n1. Inter-subject consistency: {'HIGH' if high_corr else 'LOW/MODERATE'}")
    print(f"   Mean pairwise r = {np.mean(upper_tri):.4f}")
    if high_corr:
        print("   -> The model appears to rely on a shared signal across subjects.")
    else:
        print("   -> The model may exploit subject-specific patterns or artifacts.")

    sorted_accs = sorted(
        [(sid, per_subject_results[sid].accuracy) for sid in subject_ids],
        key=lambda x: x[1],
    )
    print(f"\n2. Accuracy range: {sorted_accs[0][1]:.3f} ({sorted_accs[0][0]}) to {sorted_accs[-1][1]:.3f} ({sorted_accs[-1][0]})")

    print(f"\n3. Subject clustering (silhouette={sil_score:.3f}):")
    if sil_score > 0.4:
        print("   -> Clear clustering structure detected.")
    elif sil_score > 0.2:
        print("   -> Moderate clustering structure.")
    else:
        print("   -> Weak clustering structure; subjects are relatively homogeneous.")

    if consistency.outlier_subjects:
        print(f"\n4. Outlier subjects: {', '.join(consistency.outlier_subjects)}")
        for sid in consistency.outlier_subjects:
            r = consistency.per_subject_group_correlation[sid]
            acc = per_subject_results[sid].accuracy
            print(f"   {sid}: group_corr={r:.3f}, accuracy={acc:.3f}")
    else:
        print("\n4. No outlier subjects detected (all within 2 SD of mean).")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
