"""Plotting utilities for EEG-MI experiments."""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_dann_training_curves(
    history: Dict[str, list],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot DANN training curves showing all key metrics.

    Args:
        history: Dictionary with training history from DANNTrainer
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    epochs = np.arange(len(history["train_loss"]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DANN Training Curves", fontsize=16, fontweight='bold')

    # 1. Losses (top left)
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Total Loss", linewidth=2)
    ax.plot(epochs, history["class_loss"], label="Class Loss", linewidth=2, alpha=0.7)
    ax.plot(epochs, history["domain_loss"], label="Domain Loss", linewidth=2, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Lambda schedule (top right)
    ax = axes[0, 1]
    ax.plot(epochs, history["lambda"], color='purple', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("λ (Lambda)")
    ax.set_title("Gradient Reversal Lambda Schedule")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 3. Classification accuracy (bottom left)
    ax = axes[1, 0]
    ax.plot(epochs, history["source_acc"], label="Source Accuracy", linewidth=2, color='green')
    if "target_acc" in history and len(history["target_acc"]) > 0:
        # Target acc is only logged every 10 epochs
        target_epochs = np.arange(0, len(history["train_loss"]), 10)[:len(history["target_acc"])]
        ax.plot(target_epochs, history["target_acc"], label="Target Accuracy",
                linewidth=2, color='orange', marker='o', markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

    # 4. Domain discriminator accuracy (bottom right)
    ax = axes[1, 1]
    ax.plot(epochs, history["domain_acc"], label="Domain Discriminator",
            linewidth=2, color='red')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(len(epochs)*0.5, 0.52, 'Random Guess (50%)',
            ha='center', va='bottom', color='gray', fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Domain Discriminator Accuracy\n(Lower = CNN is fooling discriminator)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Add explanation text
    fig.text(0.5, 0.02,
             "Ideally: Domain accuracy → 50% (CNN creates domain-invariant features) "
             "while Classification accuracy → 100%",
             ha='center', fontsize=10, style='italic', color='dimgray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_dann_summary(
    history: Dict[str, list],
    final_results: Dict[str, float],
    save_path: Optional[str] = None,
) -> None:
    """Create a summary plot with key DANN metrics and final results.

    Args:
        history: Training history
        final_results: Dictionary with test_accuracy and other metrics
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))

    # Main training curves (top 2/3 of figure)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    epochs = np.arange(len(history["train_loss"]))

    # Loss plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, history["train_loss"], label="Total", linewidth=2)
    ax1.plot(epochs, history["class_loss"], label="Class", linewidth=2, alpha=0.7)
    ax1.plot(epochs, history["domain_loss"], label="Domain", linewidth=2, alpha=0.7)
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Losses", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(epochs, history["source_acc"], label="Source (train)", linewidth=2, color='green')
    ax2.plot(epochs, history["domain_acc"], label="Domain disc.", linewidth=2, color='red')
    if "target_acc" in history and len(history["target_acc"]) > 0:
        target_epochs = np.arange(0, len(history["train_loss"]), 10)[:len(history["target_acc"])]
        ax2.plot(target_epochs, history["target_acc"], label="Target (unsup.)",
                linewidth=2, color='orange', marker='o', markersize=4)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Accuracy Metrics", fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Final results (bottom 1/3)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')

    results_text = "Final Test Results\n" + "="*30 + "\n"
    results_text += f"Test Accuracy: {final_results.get('accuracy', 0):.2%}\n"
    results_text += f"Number of test samples: {final_results.get('n_samples', 0)}\n"
    results_text += f"\nFinal Training Metrics:\n"
    results_text += f"Source Accuracy: {history['source_acc'][-1]:.2%}\n"
    results_text += f"Domain Accuracy: {history['domain_acc'][-1]:.2%}\n"
    if "target_acc" in history and len(history["target_acc"]) > 0:
        results_text += f"Target Accuracy: {history['target_acc'][-1]:.2%}\n"

    ax3.text(0.1, 0.9, results_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Lambda schedule
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(epochs, history["lambda"], color='purple', linewidth=2)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("λ")
    ax4.set_title("GRL Lambda Schedule", fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary plot to: {save_path}")

    plt.close()
