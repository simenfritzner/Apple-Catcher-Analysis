"""
Visualization tools for interpretability results.

Provides plotting functions for saliency maps, ablation results,
and temporal importance profiles.
"""

from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_temporal_importance(
    times: np.ndarray,
    gradient_importance: np.ndarray,
    ablation_centers: Optional[np.ndarray] = None,
    ablation_drops: Optional[np.ndarray] = None,
    cue_onset: float = 0.0,
    title: str = "Temporal Importance",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot temporal importance from gradient and ablation methods.

    Args:
        times: Time points for gradient importance
        gradient_importance: Gradient-based importance scores
        ablation_centers: Window centers for ablation
        ablation_drops: Accuracy drops for ablation
        cue_onset: Time of cue onset
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot gradient-based importance
    axes[0].plot(times, gradient_importance, linewidth=2, color='steelblue')
    axes[0].axvline(cue_onset, color='red', linestyle='--', label='Cue Onset', alpha=0.7)
    axes[0].fill_between(times, 0, gradient_importance, alpha=0.3, color='steelblue')
    axes[0].set_ylabel('Gradient Importance', fontsize=11)
    axes[0].set_title(f'{title} - Gradient-Based', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot ablation-based importance (if provided)
    if ablation_centers is not None and ablation_drops is not None:
        axes[1].plot(ablation_centers, ablation_drops, linewidth=2,
                     marker='o', markersize=4, color='coral')
        axes[1].axvline(cue_onset, color='red', linestyle='--', label='Cue Onset', alpha=0.7)
        axes[1].fill_between(ablation_centers, 0, ablation_drops, alpha=0.3, color='coral')
        axes[1].set_ylabel('Accuracy Drop', fontsize=11)
        axes[1].set_title(f'{title} - Ablation-Based', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_pre_vs_post_cue_comparison(
    results: dict,
    cue_onset: float = 0.0,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Visualize pre-cue vs post-cue importance comparison.

    Args:
        results: Dictionary from analyze_pre_vs_post_cue
        cue_onset: Time of cue onset
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Gradient importance
    categories = ['Pre-Cue', 'Post-Cue']
    importance = [
        results['pre_cue_importance'],
        results['post_cue_importance']
    ]

    colors = ['#FF6B6B', '#4ECDC4']
    bars = axes[0].bar(categories, importance, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Average Gradient Importance', fontsize=11)
    axes[0].set_title('Gradient-Based Importance', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, importance):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Ablation accuracy drop
    accuracy_drops = [
        results['pre_cue_accuracy_drop'],
        results['post_cue_accuracy_drop']
    ]

    bars = axes[1].bar(categories, accuracy_drops, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Accuracy Drop When Masked', fontsize=11)
    axes[1].set_title('Ablation-Based Importance', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, accuracy_drops):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Add baseline accuracy annotation
    fig.text(0.5, 0.02, f"Baseline Accuracy: {results['baseline_accuracy']:.3f}",
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_saliency_map(
    saliency_map: np.ndarray,
    times: np.ndarray,
    channel_names: Optional[List[str]] = None,
    cue_onset: float = 0.0,
    title: str = "Saliency Map",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
    vmax: Optional[float] = None
) -> plt.Figure:
    """
    Plot time x channel saliency map.

    Args:
        saliency_map: Saliency values (channels, samples)
        times: Time points
        channel_names: Channel names
        cue_onset: Time of cue onset
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        vmax: Maximum value for colorbar

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(
        saliency_map,
        aspect='auto',
        cmap='hot',
        interpolation='bilinear',
        extent=[times[0], times[-1], saliency_map.shape[0], 0],
        vmin=0,
        vmax=vmax
    )

    # Add cue onset line
    ax.axvline(cue_onset, color='cyan', linestyle='--', linewidth=2, label='Cue Onset')

    # Set labels
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Importance', fontsize=11)

    # Set y-axis ticks to channel names if provided
    if channel_names is not None:
        n_channels = len(channel_names)
        tick_step = max(1, n_channels // 20)  # Show max 20 labels
        tick_positions = np.arange(0, n_channels, tick_step)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([channel_names[i] for i in tick_positions])

    ax.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_channel_importance(
    channel_importance: np.ndarray,
    channel_names: Optional[List[str]] = None,
    top_k: int = 15,
    title: str = "Channel Importance",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot channel importance ranking.

    Args:
        channel_importance: Importance scores for each channel
        channel_names: Channel names
        top_k: Number of top channels to show
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Get top channels
    top_indices = np.argsort(channel_importance)[::-1][:top_k]
    top_importance = channel_importance[top_indices]

    if channel_names is not None:
        labels = [channel_names[i] for i in top_indices]
    else:
        labels = [f"Ch {i}" for i in top_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))

    bars = ax.barh(y_pos, top_importance, color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, top_importance)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_dann_dynamics(
    history: dict,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Plot DANN training dynamics including losses, accuracies, and lambda progression.

    Args:
        history: Training history dictionary from DANN trainer
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    epochs = np.arange(len(history['train_loss']))

    # 1. Total loss
    axes[0, 0].plot(epochs, history['train_loss'], linewidth=2, color='darkblue', label='Total Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=10)
    axes[0, 0].set_ylabel('Loss', fontsize=10)
    axes[0, 0].set_title('Total Training Loss', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Classification vs Domain loss
    axes[0, 1].plot(epochs, history['class_loss'], linewidth=2, color='green', label='Class Loss')
    axes[0, 1].plot(epochs, history['domain_loss'], linewidth=2, color='orange', label='Domain Loss')
    axes[0, 1].set_xlabel('Epoch', fontsize=10)
    axes[0, 1].set_ylabel('Loss', fontsize=10)
    axes[0, 1].set_title('Classification vs Domain Loss', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Lambda progression
    axes[0, 2].plot(epochs, history['lambda'], linewidth=2, color='purple')
    axes[0, 2].set_xlabel('Epoch', fontsize=10)
    axes[0, 2].set_ylabel('Lambda (λ)', fontsize=10)
    axes[0, 2].set_title('Gradient Reversal Lambda', fontsize=11, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Source accuracy
    axes[1, 0].plot(epochs, history['source_acc'], linewidth=2, color='steelblue', label='Source')
    axes[1, 0].set_xlabel('Epoch', fontsize=10)
    axes[1, 0].set_ylabel('Accuracy', fontsize=10)
    axes[1, 0].set_title('Source Domain Accuracy', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])

    # 5. Target accuracy (if available)
    if 'target_acc_test' in history and len(history['target_acc_test']) > 0:
        # Adjust epochs for test accuracy if logged less frequently
        test_epochs = np.linspace(0, len(history['train_loss'])-1, len(history['target_acc_test']))
        axes[1, 1].plot(test_epochs, history['target_acc_test'], linewidth=2, color='coral', label='Target (Val)')
        axes[1, 1].set_xlabel('Epoch', fontsize=10)
        axes[1, 1].set_ylabel('Accuracy', fontsize=10)
        axes[1, 1].set_title('Target Domain Accuracy', fontsize=11, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1])

    # 6. Domain discriminator accuracy
    axes[1, 2].plot(epochs, history['domain_acc'], linewidth=2, color='darkorange')
    axes[1, 2].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    axes[1, 2].set_xlabel('Epoch', fontsize=10)
    axes[1, 2].set_ylabel('Accuracy', fontsize=10)
    axes[1, 2].set_title('Domain Discriminator Accuracy', fontsize=11, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    axes[1, 2].set_ylim([0, 1])

    plt.suptitle('DANN Training Dynamics', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved DANN dynamics plot to {save_path}")

    return fig


def plot_dann_feature_space(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    method: str = "tsne",
    perplexity: int = 30,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Visualize feature space using t-SNE or UMAP, colored by domain and class.

    Args:
        source_features: Source domain features (n_source, feature_dim)
        target_features: Target domain features (n_target, feature_dim)
        source_labels: Source domain class labels
        target_labels: Target domain class labels
        method: Dimensionality reduction method ("tsne" or "umap")
        perplexity: Perplexity for t-SNE
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Combine features
    all_features = np.concatenate([source_features, target_features], axis=0)
    all_labels = np.concatenate([source_labels, target_labels], axis=0)
    domain_labels = np.concatenate([
        np.zeros(len(source_features)),
        np.ones(len(target_features))
    ])

    # Dimensionality reduction
    if method.lower() == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedding = reducer.fit_transform(all_features)
    elif method.lower() == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(all_features)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            embedding = reducer.fit_transform(all_features)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'")

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Colored by domain
    for domain_idx, (domain_name, color) in enumerate([('Source', 'steelblue'), ('Target', 'coral')]):
        mask = domain_labels == domain_idx
        axes[0].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            label=domain_name,
            alpha=0.6,
            s=30,
            edgecolors='black',
            linewidth=0.5
        )
    axes[0].set_xlabel(f'{method.upper()} 1', fontsize=11)
    axes[0].set_ylabel(f'{method.upper()} 2', fontsize=11)
    axes[0].set_title('Feature Space - Colored by Domain', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Colored by class (with different markers for domains)
    unique_classes = np.unique(all_labels)
    colors_class = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

    for class_idx in unique_classes:
        # Source samples
        mask_source = (all_labels == class_idx) & (domain_labels == 0)
        if mask_source.sum() > 0:
            axes[1].scatter(
                embedding[mask_source, 0],
                embedding[mask_source, 1],
                c=[colors_class[class_idx]],
                label=f'Class {class_idx} (Source)',
                alpha=0.6,
                s=30,
                marker='o',
                edgecolors='black',
                linewidth=0.5
            )

        # Target samples
        mask_target = (all_labels == class_idx) & (domain_labels == 1)
        if mask_target.sum() > 0:
            axes[1].scatter(
                embedding[mask_target, 0],
                embedding[mask_target, 1],
                c=[colors_class[class_idx]],
                label=f'Class {class_idx} (Target)',
                alpha=0.6,
                s=30,
                marker='^',
                edgecolors='black',
                linewidth=0.5
            )

    axes[1].set_xlabel(f'{method.upper()} 1', fontsize=11)
    axes[1].set_ylabel(f'{method.upper()} 2', fontsize=11)
    axes[1].set_title('Feature Space - Colored by Class', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=8, loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature space plot to {save_path}")

    return fig


def create_comprehensive_report(
    analyzer,
    x: torch.Tensor,
    y: torch.Tensor,
    channel_names: Optional[List[str]] = None,
    sfreq: float = 250.0,
    tmin: float = -1.0,
    cue_onset: float = 0.0,
    output_dir: Path = Path("interpretability_results"),
    subject_id: Optional[str] = None
) -> None:
    """
    Create comprehensive interpretability report with all visualizations.

    Args:
        analyzer: TemporalImportanceAnalyzer instance
        x: Input data
        y: True labels
        channel_names: Channel names
        sfreq: Sampling frequency
        tmin: Start time
        cue_onset: Cue onset time
        output_dir: Directory to save results
        subject_id: Subject identifier for filenames
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{subject_id}_" if subject_id else ""

    print("Generating interpretability report...")

    # 1. Pre vs post cue analysis
    print("1. Analyzing pre-cue vs post-cue importance...")
    pre_post_results = analyzer.analyze_pre_vs_post_cue(
        x, y, cue_onset, sfreq, tmin
    )

    # Save numerical results
    with open(output_dir / f"{prefix}pre_post_analysis.txt", "w") as f:
        f.write("Pre-Cue vs Post-Cue Analysis\n")
        f.write("=" * 50 + "\n\n")
        for key, value in pre_post_results.items():
            f.write(f"{key}: {value:.6f}\n")

    # Plot comparison
    plot_pre_vs_post_cue_comparison(
        pre_post_results,
        cue_onset=cue_onset,
        save_path=output_dir / f"{prefix}pre_post_comparison.png"
    )
    plt.close()

    # 2. Temporal profile
    print("2. Computing temporal importance profile...")
    temporal_results = analyzer.analyze_temporal_profile(
        x, y, sfreq, tmin, window_size=0.5, step_size=0.1
    )

    plot_temporal_importance(
        temporal_results['times'],
        temporal_results['gradient_importance'],
        temporal_results['ablation_window_centers'],
        temporal_results['ablation_accuracy_drops'],
        cue_onset=cue_onset,
        save_path=output_dir / f"{prefix}temporal_profile.png"
    )
    plt.close()

    # 3. Channel importance
    print("3. Analyzing channel importance...")
    channel_results = analyzer.analyze_channel_importance(
        x, channel_names, top_k=15
    )

    plot_channel_importance(
        channel_results['channel_importance'],
        channel_names,
        top_k=15,
        save_path=output_dir / f"{prefix}channel_importance.png"
    )
    plt.close()

    # 4. Full saliency map
    print("4. Generating saliency maps...")
    saliency_map = analyzer.saliency.integrated_gradients(x)
    times, importance_map, _ = analyzer.saliency.compute_time_channel_map(
        saliency_map, sfreq, tmin
    )

    plot_saliency_map(
        importance_map,
        times,
        channel_names,
        cue_onset=cue_onset,
        save_path=output_dir / f"{prefix}saliency_map.png"
    )
    plt.close()

    print(f"\nReport saved to {output_dir}")
    print(f"\nKey Findings:")
    print(f"  - Pre-cue importance: {pre_post_results['pre_cue_importance']:.4f}")
    print(f"  - Post-cue importance: {pre_post_results['post_cue_importance']:.4f}")
    print(f"  - Ratio (pre/post): {pre_post_results['importance_ratio']:.2f}")
    print(f"  - Pre-cue accuracy drop: {pre_post_results['pre_cue_accuracy_drop']:.3f}")
    print(f"  - Post-cue accuracy drop: {pre_post_results['post_cue_accuracy_drop']:.3f}")


# --------------------------------------------------------------------------
# Artifact investigation visualizations (Phase 6)
# --------------------------------------------------------------------------

def plot_channel_group_comparison(
    group_results: "Union[Dict, object]",
    title: str = "Channel Group Saliency Importance",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Bar chart comparing saliency importance across channel groups.

    Annotates the frontal/central ratio so readers can immediately see
    whether the model over-relies on frontal (artifact-prone) channels.

    Args:
        group_results: A ChannelGroupResult dataclass (from
            ArtifactAnalyzer.analyze_channel_groups) or a dict with
            'group_importance' and 'frontal_central_ratio' keys.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    # Accept both dataclass and plain dict
    if hasattr(group_results, 'group_importance'):
        importance = group_results.group_importance
        ratio = group_results.frontal_central_ratio
    else:
        importance = group_results['group_importance']
        ratio = group_results['frontal_central_ratio']

    regions = list(importance.keys())
    values = [importance[r] for r in regions]

    region_colors = {
        'frontal': '#E74C3C',
        'central': '#3498DB',
        'parietal': '#2ECC71',
        'occipital': '#9B59B6',
        'temporal': '#F39C12',
    }
    colors = [region_colors.get(r, '#95A5A6') for r in regions]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(regions, values, color=colors, alpha=0.8, edgecolor='black')

    # Annotate bar values
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{val:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    ax.set_ylabel('Mean Saliency Importance', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate frontal/central ratio
    ax.annotate(
        f'Frontal / Central ratio: {ratio:.2f}',
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_artifact_trial_analysis(
    clean_vs_artifact_results: Dict[str, float],
    title: str = "Clean vs Artifact Trial Accuracy",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Side-by-side bar chart comparing accuracy on clean, artifact, and all trials.

    Args:
        clean_vs_artifact_results: Dictionary returned by
            ArtifactAnalyzer.compare_clean_vs_artifact_trials.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    r = clean_vs_artifact_results

    categories = ['All Trials', 'Clean Trials', 'Artifact Trials']
    accuracies = [r['all_accuracy'], r['clean_accuracy'], r['artifact_accuracy']]
    counts = [r['n_total'], r['n_clean'], r['n_artifact']]
    colors = ['#3498DB', '#2ECC71', '#E74C3C']

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black')

    # Annotate values and trial counts
    for bar, acc, n in zip(bars, accuracies, counts):
        height = bar.get_height()
        if np.isnan(acc):
            label = f'N/A\n(n={n})'
        else:
            label = f'{acc:.3f}\n(n={n})'
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            label,
            ha='center',
            va='bottom',
            fontsize=10,
        )

    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_single_channel_ablation(
    drops: np.ndarray,
    channel_names: List[str],
    title: str = "Single Channel Ablation Impact",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Horizontal bar chart of per-channel accuracy drops, sorted by impact.

    Channels are color-coded by scalp region:
        frontal=red, central=blue, parietal=green,
        occipital=purple, temporal=orange, other=grey.

    Args:
        drops: Array of accuracy drops per channel (n_channels,).
        channel_names: List of channel names.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    from .artifacts import _classify_channel

    region_colors = {
        'frontal': '#E74C3C',
        'central': '#3498DB',
        'parietal': '#2ECC71',
        'occipital': '#9B59B6',
        'temporal': '#F39C12',
        None: '#95A5A6',
    }

    # Sort by accuracy drop (descending)
    sorted_indices = np.argsort(drops)[::-1]
    sorted_drops = drops[sorted_indices]
    sorted_names = [channel_names[i] for i in sorted_indices]
    sorted_colors = [
        region_colors.get(_classify_channel(name), '#95A5A6')
        for name in sorted_names
    ]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(sorted_names))

    ax.barh(y_pos, sorted_drops, color=sorted_colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy Drop', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Legend for region colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=region_colors['frontal'], edgecolor='black', label='Frontal'),
        Patch(facecolor=region_colors['central'], edgecolor='black', label='Central'),
        Patch(facecolor=region_colors['parietal'], edgecolor='black', label='Parietal'),
        Patch(facecolor=region_colors['occipital'], edgecolor='black', label='Occipital'),
        Patch(facecolor=region_colors['temporal'], edgecolor='black', label='Temporal'),
        Patch(facecolor=region_colors[None], edgecolor='black', label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig
