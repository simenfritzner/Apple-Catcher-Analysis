"""
Correlation analysis: ERD magnitude vs classification accuracy (thesis §5.5).

Requires per_subject_erd.txt from run_neurophysiology.py and
per_subject_summary.txt from Phase 7.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind

NEURO_DIR = project_root / "results" / "interpretability" / "neurophysiology"
PHASE7_DIR = project_root / "results" / "interpretability" / "phase7_cross_subject"
OUT_DIR = project_root / "results" / "interpretability" / "correlation"


def load_per_subject_accuracy() -> dict[str, float]:
    """Load per-subject accuracy from Phase 7 results."""
    acc = {}
    with open(PHASE7_DIR / "per_subject_summary.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith('s'):
                acc[parts[0]] = float(parts[1])
    return acc


def load_per_subject_erd() -> dict[str, tuple[float, float, float]]:
    """Load per-subject ERD from neurophysiology results.
    Returns {subject: (erd_c3, erd_c4, erd_mean)}.
    """
    erd = {}
    with open(NEURO_DIR / "per_subject_erd.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0].startswith('s'):
                erd[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return erd


def main() -> None:
    print("=" * 60)
    print("CORRELATION ANALYSIS (thesis §5.5)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    acc_data = load_per_subject_accuracy()
    erd_data = load_per_subject_erd()

    # Match subjects
    subjects = sorted(set(acc_data.keys()) & set(erd_data.keys()))
    print(f"  Matched {len(subjects)} subjects")

    accuracies = np.array([acc_data[s] for s in subjects])
    erd_c3 = np.array([erd_data[s][0] for s in subjects])
    erd_c4 = np.array([erd_data[s][1] for s in subjects])
    erd_mean = np.array([erd_data[s][2] for s in subjects])

    lines = ["Correlation Analysis Results", "=" * 60, ""]

    # ── §5.5.1: ERD vs Accuracy scatter plots ────────────────────────────
    print("\n── §5.5.1: ERD Magnitude vs Accuracy ──")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, erd_vals, label in [
        (axes[0], erd_c3, 'C3 mu-ERD (%)'),
        (axes[1], erd_c4, 'C4 mu-ERD (%)'),
        (axes[2], erd_mean, 'Mean C3/C4 mu-ERD (%)'),
    ]:
        r, p = pearsonr(erd_vals, accuracies)

        ax.scatter(erd_vals, accuracies, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

        # Regression line
        z = np.polyfit(erd_vals, accuracies, 1)
        x_line = np.linspace(erd_vals.min(), erd_vals.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=2)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Classification Accuracy', fontsize=11)
        ax.set_title(f'{label}\nr={r:.3f}, p={p:.4f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        lines.append(f"{label} vs Accuracy:")
        lines.append(f"  Pearson r = {r:.4f}, p = {p:.6f}")
        lines.append(f"  {'Significant' if p < 0.05 else 'NOT significant'} (p {'<' if p < 0.05 else '>'} 0.05)")
        lines.append("")
        print(f"  {label}: r={r:.3f}, p={p:.4f}")

    plt.suptitle('ERD Magnitude vs Classification Accuracy (N=40)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'erd_vs_accuracy.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ── §5.5.3: BCI Inefficiency Analysis ─────────────────────────────────
    print("\n── §5.5.3: BCI Inefficiency Analysis ──")

    low_mask = accuracies < 0.70
    high_mask = accuracies > 0.80

    low_erd = erd_mean[low_mask]
    high_erd = erd_mean[high_mask]

    n_low = low_mask.sum()
    n_high = high_mask.sum()

    lines.append("BCI Inefficiency Analysis")
    lines.append("-" * 40)
    lines.append(f"Low performers (acc < 0.70):  N={n_low}")
    lines.append(f"High performers (acc > 0.80): N={n_high}")

    if n_low >= 2 and n_high >= 2:
        t_stat, p_val = ttest_ind(high_erd, low_erd)
        # Cohen's d
        pooled_std = np.sqrt(((n_high - 1) * high_erd.std()**2 + (n_low - 1) * low_erd.std()**2) /
                              (n_high + n_low - 2))
        d = (high_erd.mean() - low_erd.mean()) / (pooled_std + 1e-12)

        lines.append(f"Low performers mean ERD:  {low_erd.mean():.2f}% ± {low_erd.std():.2f}")
        lines.append(f"High performers mean ERD: {high_erd.mean():.2f}% ± {high_erd.std():.2f}")
        lines.append(f"t-test: t={t_stat:.3f}, p={p_val:.6f}")
        lines.append(f"Cohen's d: {d:.3f}")
        lines.append(f"Interpretation: {'ERD predicts performance' if p_val < 0.05 else 'ERD does NOT predict performance'}")

        print(f"  Low:  ERD={low_erd.mean():.2f}% (N={n_low})")
        print(f"  High: ERD={high_erd.mean():.2f}% (N={n_high})")
        print(f"  t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        positions = [0, 1]
        bp = ax.boxplot([low_erd, high_erd], positions=positions, widths=0.4, patch_artist=True)
        bp['boxes'][0].set_facecolor('#E74C3C')
        bp['boxes'][1].set_facecolor('#2ECC71')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels([f'Low (<0.70)\nN={n_low}', f'High (>0.80)\nN={n_high}'])
        ax.set_ylabel('Mean mu-ERD at C3/C4 (%)', fontsize=11)
        ax.set_title(f'ERD Magnitude by Performance Group\nt={t_stat:.3f}, p={p_val:.4f}, d={d:.2f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(OUT_DIR / 'bci_inefficiency.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        lines.append("Insufficient subjects in one or both groups for comparison")

    with open(OUT_DIR / 'correlation_results.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\nResults saved to {OUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
