"""Figures shared by the model scripts (Version 0.3 onwards). Styling comes from src/exploration.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src import exploration as ex


def plot_confusion(threshold: float, metrics: dict[str, Any], title: str, path: Path) -> Path:
    """2 x 2 confusion matrix with counts and row shares; `metrics` comes from classification_metrics."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ex.apply_style()
    cells = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    row_share = cells / cells.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(row_share, cmap=LinearSegmentedColormap.from_list("blue", ex.BLUE_RAMP), vmin=0, vmax=1)
    names = ["Susceptible", "Resistant"]
    for i in range(2):
        for j in range(2):
            dark = row_share[i, j] > 0.55
            ax.text(j, i, f"{cells[i, j]:,}\n{row_share[i, j]:.1%} of true {names[i].lower()}", ha="center",
                    va="center", fontsize=10, color=ex.SURFACE if dark else ex.INK)
    ax.set_xticks([0, 1], [f"Predicted {n.lower()}" for n in names])
    ax.set_yticks([0, 1], [f"True {n.lower()}" for n in names])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, pad=22)
    ex._subtitle(ax, f"Threshold {threshold:.3g} (from validation); sensitivity {metrics['sensitivity']:.3f}, "
                     f"specificity {metrics['specificity']:.3f}")
    return ex._save(fig, path)


def plot_loss_curves(histories: list[tuple[str, dict[str, Any]]], title: str, path: Path) -> Path:
    """Training curves of the networks (Version 0.5): one column per model, loss above, AUROC below.

    `histories` are TorchClassifier.history dicts. The early-stopping curves come from an inner split of
    the training part, never from the validation or test parts, so the subtitle says so.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not histories:
        raise ValueError("No training histories to plot.")
    ex.apply_style()
    fig, axes = plt.subplots(2, len(histories), figsize=(5.4 * len(histories), 7.2), squeeze=False)
    for column, (label, history) in enumerate(histories):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        best = int(history["best_epoch"]) + 1
        loss_ax, auc_ax = axes[0][column], axes[1][column]
        loss_ax.plot(epochs, history["train_loss"], "-", color=ex.SERIES[0], lw=1.8, label="Training loss")
        loss_ax.plot(epochs, history["validation_loss"], "-", color=ex.SERIES[1], lw=1.8,
                     label="Inner-validation loss")
        auc_ax.plot(epochs, history["validation_roc_auc"], "-", color=ex.SERIES[2], lw=1.8,
                    label="Inner-validation AUROC")
        for ax, name in ((loss_ax, "Binary cross-entropy"), (auc_ax, "AUROC")):
            ax.axvline(best, ls="--", lw=1, color=ex.AXIS)
            ax.set(xlabel="Epoch", ylabel=name, xlim=(1, max(int(epochs[-1]), 2)))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))      # epochs are whole numbers
            ax.legend(loc="best", fontsize=8.5)
        stopped = "stopped early" if history["stopped_early"] else "ran to the epoch limit"
        loss_ax.set_title(label, pad=22)
        ex._subtitle(loss_ax, f"{len(epochs)} epochs, {stopped}; kept epoch {best} (dashed line)")
        auc_ax.set_title(f"{label}: inner-validation AUROC", pad=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return ex._save(fig, path)
