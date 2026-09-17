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
