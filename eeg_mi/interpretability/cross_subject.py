"""
Cross-subject consistency analysis for EEG classification models.

Analyzes inter-subject variability in saliency patterns to determine
whether the model relies on shared neural signals or subject-specific
artifacts. Phase 7 of the interpretability analysis pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .saliency import SaliencyMapGenerator


@dataclass
class SubjectResult:
    """Saliency analysis results for a single subject."""
    temporal_importance: np.ndarray
    channel_importance: np.ndarray
    saliency_map_mean: np.ndarray
    accuracy: float
    n_trials: int


@dataclass
class ConsistencyResult:
    """Summary of cross-subject consistency analysis."""
    mean_inter_subject_correlation: float
    median_inter_subject_correlation: float
    temporal_cv: float
    channel_cv: float
    per_subject_group_correlation: dict[str, float]
    outlier_subjects: list[str]
    outlier_threshold: float


class CrossSubjectAnalyzer:
    """Analyze consistency of saliency patterns across subjects."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.saliency_gen = SaliencyMapGenerator(model, device)
        self.model.eval()

    def compute_subject_accuracy(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Evaluate model accuracy on a single subject's data."""
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(batch_y.numpy())

        preds_all = np.concatenate(all_preds)
        labels_all = np.concatenate(all_labels)
        return float((preds_all == labels_all).mean())

    def compute_per_subject_saliency(
        self,
        subject_data: dict[str, tuple[np.ndarray, np.ndarray]],
        sfreq: float = 250.0,
        tmin: float = -1.0,
        method: str = 'integrated_gradients',
        max_trials: int = 100,
    ) -> dict[str, SubjectResult]:
        """Compute saliency maps and aggregate statistics per subject."""
        method_func = {
            'vanilla': self.saliency_gen.vanilla_gradient,
            'integrated_gradients': self.saliency_gen.integrated_gradients,
            'gradient_x_input': self.saliency_gen.gradient_x_input,
        }
        if method not in method_func:
            raise ValueError(f"Unknown method: {method}. Use one of {list(method_func.keys())}")

        results: dict[str, SubjectResult] = {}

        for subject_id, (data, labels) in subject_data.items():
            n_trials = min(max_trials, len(labels))
            data_subset = data[:n_trials]
            labels_subset = labels[:n_trials]

            # Compute accuracy on the subset
            accuracy = self.compute_subject_accuracy(data_subset, labels_subset)

            # Compute saliency in batches to manage memory
            x_tensor = torch.tensor(data_subset, dtype=torch.float32).unsqueeze(1)
            saliency_maps: list[np.ndarray] = []

            for start in range(0, n_trials, self.batch_size):
                end = min(start + self.batch_size, n_trials)
                batch = x_tensor[start:end]
                smap = method_func[method](batch)
                saliency_maps.append(smap)

            saliency_all = np.concatenate(saliency_maps, axis=0)

            _, temporal_importance = self.saliency_gen.compute_temporal_importance(
                saliency_all, sfreq, tmin
            )
            channel_importance = self.saliency_gen.compute_channel_importance(saliency_all)
            saliency_mean = saliency_all.mean(axis=0)

            results[subject_id] = SubjectResult(
                temporal_importance=temporal_importance,
                channel_importance=channel_importance,
                saliency_map_mean=saliency_mean,
                accuracy=accuracy,
                n_trials=n_trials,
            )
            print(f"  {subject_id}: {n_trials} trials, acc={accuracy:.3f}")

        return results

    def compute_inter_subject_correlation(
        self,
        per_subject_results: dict[str, SubjectResult],
    ) -> tuple[np.ndarray, list[str]]:
        """Compute pairwise Pearson correlation of temporal importance profiles."""
        subject_ids = sorted(per_subject_results.keys())
        n = len(subject_ids)
        corr_matrix = np.ones((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                ti_i = per_subject_results[subject_ids[i]].temporal_importance
                ti_j = per_subject_results[subject_ids[j]].temporal_importance
                r, _ = pearsonr(ti_i, ti_j)
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r

        return corr_matrix, subject_ids

    def cluster_subjects(
        self,
        per_subject_results: dict[str, SubjectResult],
        n_clusters: int = 3,
        feature_type: str = 'temporal',
    ) -> tuple[np.ndarray, float, np.ndarray, list[str]]:
        """Cluster subjects by their saliency patterns.

        Returns:
            Tuple of (cluster_labels, silhouette_score, cluster_centers, subject_ids).
        """
        subject_ids = sorted(per_subject_results.keys())

        if feature_type == 'temporal':
            features = np.array([
                per_subject_results[sid].temporal_importance for sid in subject_ids
            ])
        elif feature_type == 'channel':
            features = np.array([
                per_subject_results[sid].channel_importance for sid in subject_ids
            ])
        elif feature_type == 'combined':
            features = np.array([
                np.concatenate([
                    per_subject_results[sid].temporal_importance,
                    per_subject_results[sid].channel_importance,
                ])
                for sid in subject_ids
            ])
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cap n_clusters at number of subjects minus 1
        effective_n_clusters = min(n_clusters, len(subject_ids) - 1)

        clustering = AgglomerativeClustering(n_clusters=effective_n_clusters)
        cluster_labels = clustering.fit_predict(features_scaled)

        sil_score = silhouette_score(features_scaled, cluster_labels)

        # Compute cluster centers (mean of features per cluster)
        cluster_centers = np.array([
            features_scaled[cluster_labels == k].mean(axis=0)
            for k in range(effective_n_clusters)
        ])

        return cluster_labels, sil_score, cluster_centers, subject_ids

    def analyze_consistency(
        self,
        per_subject_results: dict[str, SubjectResult],
    ) -> ConsistencyResult:
        """Compute summary statistics of cross-subject consistency."""
        corr_matrix, subject_ids = self.compute_inter_subject_correlation(per_subject_results)
        n = len(subject_ids)

        # Extract upper triangle (excluding diagonal)
        upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
        mean_corr = float(np.mean(upper_triangle))
        median_corr = float(np.median(upper_triangle))

        # Coefficient of variation of temporal profiles
        temporal_profiles = np.array([
            per_subject_results[sid].temporal_importance for sid in subject_ids
        ])
        # CV across subjects at each timepoint, then average
        temporal_std = temporal_profiles.std(axis=0)
        temporal_mean = temporal_profiles.mean(axis=0)
        temporal_cv = float(np.mean(temporal_std / (temporal_mean + 1e-10)))

        # CV for channel importance
        channel_profiles = np.array([
            per_subject_results[sid].channel_importance for sid in subject_ids
        ])
        channel_std = channel_profiles.std(axis=0)
        channel_mean = channel_profiles.mean(axis=0)
        channel_cv = float(np.mean(channel_std / (channel_mean + 1e-10)))

        # Per-subject correlation with group mean temporal profile
        group_mean_temporal = temporal_profiles.mean(axis=0)
        per_subject_corr: dict[str, float] = {}
        for i, sid in enumerate(subject_ids):
            r, _ = pearsonr(temporal_profiles[i], group_mean_temporal)
            per_subject_corr[sid] = float(r)

        # Identify outlier subjects (correlation with group mean below threshold)
        corr_values = np.array(list(per_subject_corr.values()))
        outlier_threshold = float(np.mean(corr_values) - 2.0 * np.std(corr_values))
        outlier_subjects = [
            sid for sid, r in per_subject_corr.items() if r < outlier_threshold
        ]

        return ConsistencyResult(
            mean_inter_subject_correlation=mean_corr,
            median_inter_subject_correlation=median_corr,
            temporal_cv=temporal_cv,
            channel_cv=channel_cv,
            per_subject_group_correlation=per_subject_corr,
            outlier_subjects=outlier_subjects,
            outlier_threshold=outlier_threshold,
        )


# ── Plotting functions ────────────────────────────────────────────────────


def plot_inter_subject_correlation(
    corr_matrix: np.ndarray,
    subject_ids: list[str],
    save_path: Optional[Path] = None,
) -> None:
    """Heatmap of pairwise inter-subject temporal correlation."""
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson r', fontsize=12)

    ax.set_xticks(range(len(subject_ids)))
    ax.set_xticklabels(subject_ids, rotation=90, fontsize=7)
    ax.set_yticks(range(len(subject_ids)))
    ax.set_yticklabels(subject_ids, fontsize=7)
    ax.set_title('Inter-Subject Temporal Saliency Correlation', fontsize=14)

    # Annotate mean off-diagonal correlation
    n = len(subject_ids)
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    ax.text(
        0.02, 0.98,
        f"Mean r = {np.mean(upper_tri):.3f}",
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_subject_clustering(
    per_subject_results: dict[str, SubjectResult],
    cluster_labels: np.ndarray,
    subject_ids: list[str],
    save_path: Optional[Path] = None,
) -> None:
    """2D PCA scatter of subjects colored by cluster assignment."""
    # Build combined feature matrix
    features = np.array([
        np.concatenate([
            per_subject_results[sid].temporal_importance,
            per_subject_results[sid].channel_importance,
        ])
        for sid in subject_ids
    ])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(features_scaled)

    accuracies = np.array([per_subject_results[sid].accuracy for sid in subject_ids])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: colored by cluster
    unique_labels = np.unique(cluster_labels)
    cmap = plt.cm.Set2
    for k in unique_labels:
        mask = cluster_labels == k
        axes[0].scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(k / max(unique_labels.max(), 1))],
            label=f'Cluster {k}', s=60, edgecolors='k', linewidths=0.5,
        )
    for i, sid in enumerate(subject_ids):
        axes[0].annotate(
            sid, (coords[i, 0], coords[i, 1]),
            fontsize=6, ha='center', va='bottom', textcoords='offset points',
            xytext=(0, 4),
        )
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    axes[0].set_title('Subject Clusters (PCA)', fontsize=13)
    axes[0].legend(fontsize=9)

    # Panel 2: colored by accuracy
    sc = axes[1].scatter(
        coords[:, 0], coords[:, 1],
        c=accuracies, cmap='viridis', s=60, edgecolors='k', linewidths=0.5,
    )
    cbar = fig.colorbar(sc, ax=axes[1], shrink=0.8)
    cbar.set_label('Model Accuracy', fontsize=11)
    for i, sid in enumerate(subject_ids):
        axes[1].annotate(
            sid, (coords[i, 0], coords[i, 1]),
            fontsize=6, ha='center', va='bottom', textcoords='offset points',
            xytext=(0, 4),
        )
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    axes[1].set_title('Subjects Colored by Accuracy', fontsize=13)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_per_subject_temporal_profiles(
    per_subject_results: dict[str, SubjectResult],
    cue_onset: float = 0.0,
    sfreq: float = 250.0,
    tmin: float = -1.0,
    save_path: Optional[Path] = None,
) -> None:
    """Overlay all subjects' temporal importance profiles with mean +/- std."""
    subject_ids = sorted(per_subject_results.keys())
    profiles = np.array([
        per_subject_results[sid].temporal_importance for sid in subject_ids
    ])

    n_samples = profiles.shape[1]
    times = np.arange(n_samples) / sfreq + tmin
    mean_profile = profiles.mean(axis=0)
    std_profile = profiles.std(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Individual subjects in light gray
    for i in range(len(subject_ids)):
        ax.plot(times, profiles[i], color='gray', alpha=0.2, linewidth=0.7)

    # Mean +/- std
    ax.fill_between(
        times, mean_profile - std_profile, mean_profile + std_profile,
        alpha=0.3, color='steelblue', label='Mean +/- 1 SD',
    )
    ax.plot(times, mean_profile, color='steelblue', linewidth=2, label='Grand mean')

    # Cue onset line
    ax.axvline(cue_onset, color='red', linestyle='--', linewidth=1.5, label='Cue onset')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Saliency Importance', fontsize=12)
    ax.set_title(f'Temporal Saliency Profiles Across {len(subject_ids)} Subjects', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(times[0], times[-1])

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_consistency_summary(
    consistency_results: ConsistencyResult,
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart of per-subject correlation with group mean temporal profile."""
    subject_ids = sorted(consistency_results.per_subject_group_correlation.keys())
    correlations = [consistency_results.per_subject_group_correlation[sid] for sid in subject_ids]

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = [
        'salmon' if sid in consistency_results.outlier_subjects else 'steelblue'
        for sid in subject_ids
    ]

    bars = ax.bar(range(len(subject_ids)), correlations, color=colors, edgecolor='k', linewidth=0.5)
    ax.set_xticks(range(len(subject_ids)))
    ax.set_xticklabels(subject_ids, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Correlation with Group Mean', fontsize=12)
    ax.set_title('Per-Subject Consistency with Group Temporal Profile', fontsize=14)

    # Threshold line
    ax.axhline(
        consistency_results.outlier_threshold, color='red', linestyle='--',
        linewidth=1, label=f'Outlier threshold ({consistency_results.outlier_threshold:.3f})',
    )
    # Mean line
    ax.axhline(
        consistency_results.mean_inter_subject_correlation, color='green', linestyle='-',
        linewidth=1, label=f'Mean pairwise r ({consistency_results.mean_inter_subject_correlation:.3f})',
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
