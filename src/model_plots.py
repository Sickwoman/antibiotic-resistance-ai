"""Figures shared by the model scripts (Version 0.3 onwards). Styling comes from src/exploration.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def plot_regions(regions: pd.DataFrame, mean_resistant: np.ndarray, mean_susceptible: np.ndarray,
                 mz: np.ndarray, title: str, path: Path) -> Path:
    """Reported m/z regions (Version 0.6) over the mean spectrum of each class.

    Two panels on one shared m/z axis, never a twin y-axis: intensity and mean |contribution| have no
    common scale, so one curve drawn against a second y-axis could be read as the other. The top panel says
    where the regions sit in the signal, the bottom one how strong each region is and which class it pushes
    towards. Every region is named by its m/z interval only.
    """
    import matplotlib.pyplot as plt

    ex.apply_style()
    centres = np.asarray(mz, dtype=np.float64).ravel()
    if centres.size == 0:
        raise ValueError("No m/z axis to plot the regions against.")
    means = {"R": np.asarray(mean_resistant, dtype=np.float64).ravel(),
             "S": np.asarray(mean_susceptible, dtype=np.float64).ravel()}
    for key, values in means.items():
        if values.size not in (0, centres.size):
            raise ValueError(f"Mean {ex.LABEL_NAMES[key]} spectrum has {values.size} points for "
                             f"{centres.size} m/z bins.")
    drawable = [k for k, v in means.items() if v.size == centres.size and np.isfinite(v).any()]
    span = float(centres.max() - centres.min()) or 1.0
    min_visible = span * 0.0035               # a one-column region is 3 Da wide: too thin to see unaided

    frame = regions if isinstance(regions, pd.DataFrame) else pd.DataFrame()
    have = not frame.empty
    starts = frame["mz_start"].to_numpy(dtype=np.float64) if have else np.zeros(0)
    ends = frame["mz_end"].to_numpy(dtype=np.float64) if have else np.zeros(0)
    strength = np.nan_to_num(frame["total_abs"].to_numpy(dtype=np.float64)) if have else np.zeros(0)
    towards = frame["towards"].astype(str).to_numpy() if have else np.zeros(0, dtype=object)
    middles = (starts + ends) / 2

    # Rotated interval labels collide when two regions are close in m/z, so each label takes the lowest
    # row in which nothing sits within `apart`; past four rows the roomiest row is reused.
    apart, taken, level_of = span * 0.014, [], np.zeros(len(middles), dtype=np.int64)
    for i in np.argsort(middles):
        free = next((lv for lv, last in enumerate(taken) if middles[i] - last >= apart), None)
        if free is None and len(taken) < 4:
            taken.append(-np.inf)
            free = len(taken) - 1
        free = int(np.argmin(taken)) if free is None else free
        taken[free], level_of[i] = float(middles[i]), free
    levels = max(len(taken), 1)

    # A rotated label is as tall as its text is long, so the label rows, not the bars, set the panel height.
    label_chars = max((len(f"{s:,.0f}-{e:,.0f}") for s, e in zip(starts, ends, strict=True)), default=11)
    bars_in, row_in = 1.7, label_chars * 0.55 * 7.5 / 72 + 0.06
    lower_in = (bars_in + levels * row_in) / 0.72
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.5, 3.8 + lower_in), sharex=True,
                                        gridspec_kw={"height_ratios": [3.8, lower_in]})
    for key, dashes in (("S", (None, None)), ("R", (4.5, 1.8))):
        if key in drawable:
            ax_top.plot(centres, means[key], lw=1.1, color=ex.LABEL_COLORS[key], dashes=dashes,
                        label=ex.LABEL_NAMES[key], zorder=3)
    for start, end in zip(starts, ends, strict=True):
        pad = max(min_visible - (end - start), 0.0) / 2
        ax_top.axvspan(start - pad, end + pad, color=ex.GRID, zorder=1)
    ax_top.set_ylabel("Mean preprocessed intensity")
    ax_top.set_xlim(float(centres.min()), float(centres.max()))
    ax_top.set_title("Mean validation spectrum per class, reported regions shaded", pad=22)
    missing = [ex.LABEL_NAMES[k] for k in ("S", "R") if k not in drawable]
    note = f"{len(starts)} region(s) shaded, drawn at a minimum visible width" if have else "No region reported"
    if missing:
        note += f"; no rows for {', '.join(missing)}"
    ex._subtitle(ax_top, f"{note}. Regions are named by their m/z interval only.")
    if drawable:
        ax_top.legend(loc="upper right", fontsize=8.5)

    top = float(strength.max()) if have and strength.max() > 0 else 1.0
    base, step = top * 1.06, top * row_in / bars_in
    ax_bot.set_ylim(0, base + levels * step if have else 1.0)
    for direction, key, hatch, marker in (("susceptible", "S", "\\\\\\", "v"), ("resistant", "R", "///", "^")):
        pick = towards == direction
        if not pick.any():
            continue
        widths = np.maximum(ends[pick] - starts[pick], min_visible)
        ax_bot.bar(middles[pick], strength[pick], width=widths, color=ex.LABEL_COLORS[key], hatch=hatch,
                   edgecolor=ex.INK_2, linewidth=0.45, zorder=3)
        # the marker repeats the direction as a shape: a thin bar carries neither colour nor hatch well
        ax_bot.plot(middles[pick], strength[pick], ls="none", marker=marker, ms=5, color=ex.LABEL_COLORS[key],
                    markeredgecolor=ex.INK_2, markeredgewidth=0.5, zorder=4,
                    label=f"Pushes towards {direction}")
    for i in range(len(middles)):
        y = base + level_of[i] * step
        ax_bot.plot([middles[i]] * 2, [strength[i], y], lw=0.6, color=ex.AXIS, zorder=2)
        ax_bot.text(middles[i], y, f"{starts[i]:,.0f}-{ends[i]:,.0f}", rotation=90, ha="center", va="bottom",
                    fontsize=7.5, color=ex.INK_2)
    if not have:
        ax_bot.text(0.5, 0.5, "No region reported", transform=ax_bot.transAxes, ha="center", va="center",
                    fontsize=10, color=ex.INK_2)
    else:
        ax_bot.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8.5)
    ax_bot.set(xlabel="m/z (Da), bin centre", ylabel="Region total mean |contribution|")
    ax_bot.set_title("Strength and direction of each region", pad=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return ex._save(fig, path)


def plot_importance_agreement(shap_importance: np.ndarray, permutation_drop: np.ndarray,
                              mz: np.ndarray, title: str, path: Path) -> Path:
    """Do the two importance methods rank the same blocks? One scatter panel, one point per block.

    The two numbers answer different questions (how much a block moves one prediction, versus how much the
    model's AUROC depends on it) and are in different units, so only their ordering can agree: plotted
    against each other, agreement is a cloud rising to the top right. Both spans usually cover orders of
    magnitude and a permutation can raise AUROC, so the axes switch to symlog when the values demand it
    instead of heaping most blocks at the origin. Annotated blocks carry their m/z and nothing else.
    """
    import matplotlib.pyplot as plt

    shap = np.asarray(shap_importance, dtype=np.float64).ravel()
    drop = np.asarray(permutation_drop, dtype=np.float64).ravel()
    block_mz = np.asarray(mz, dtype=np.float64).ravel()
    if shap.size == 0 or shap.size != drop.size or block_mz.size != shap.size:
        raise ValueError(f"Need one SHAP value, one AUROC drop and one m/z per block, got {shap.size}, "
                         f"{drop.size} and {block_mz.size}.")
    ex.apply_style()
    finite = np.isfinite(shap) & np.isfinite(drop)

    def linear_threshold(values: np.ndarray) -> float | None:
        """Symlog linthresh when the magnitudes span orders of magnitude, else None (keep it linear)."""
        size = np.abs(values[np.isfinite(values)])
        size = size[size > 0]
        middle = float(np.median(size)) if size.size else 0.0
        if size.size < 3 or middle <= 0 or float(size.max()) / middle < 30:
            return None
        return max(float(np.quantile(size, 0.2)), float(size.max()) * 1e-3)

    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    ax.axhline(0, ls="--", lw=1, color=ex.AXIS, zorder=1)
    ax.scatter(shap[finite], drop[finite], s=15, color=ex.SERIES[0], alpha=0.45, linewidths=0, zorder=2,
               label=f"{int(finite.sum()):,} blocks")
    for set_scale, values in ((ax.set_xscale, shap), (ax.set_yscale, drop)):
        linthresh = linear_threshold(values)
        if linthresh is not None:
            set_scale("symlog", linthresh=linthresh, linscale=0.45)

    k = min(3, shap.size)
    spread = bool(np.ptp(shap[finite]) > 0 or np.ptp(drop[finite]) > 0) if finite.any() else False
    strongest = sorted(set(np.argsort(-shap)[:k].tolist()) | set(np.argsort(-drop)[:k].tolist()),
                       key=lambda i: (shap[i], drop[i])) if spread else []
    strongest = [i for i in strongest if finite[i]]
    if strongest:
        middle = float(np.median(shap[finite]))
        ax.scatter(shap[strongest], drop[strongest], s=56, facecolor="none", edgecolor=ex.INK, linewidths=1.1,
                   zorder=4, label=f"strongest {len(strongest)} blocks (ringed)")
        for n, i in enumerate(strongest):
            left = shap[i] > middle                      # keep labels of the strongest blocks off the edge
            ax.annotate(f"{block_mz[i]:,.0f} Da", (shap[i], drop[i]), textcoords="offset points",
                        xytext=(-11 if left else 11, 11 if n % 2 == 0 else -17), fontsize=8, color=ex.INK_2,
                        ha="right" if left else "left", zorder=5,
                        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": ex.AXIS, "shrinkA": 0, "shrinkB": 4})

    rho = float("nan")
    if int(finite.sum()) > 2 and spread:
        ranks = np.vstack([pd.Series(shap[finite]).rank().to_numpy(), pd.Series(drop[finite]).rank().to_numpy()])
        if ranks.std(axis=1).min() > 0:
            rho = float(np.corrcoef(ranks)[0, 1])
    ax.set(xlabel="SHAP mean |contribution| per block (margin units)",
           ylabel="Permutation AUROC drop of the same block (negative: permuting raised AUROC)")
    ax.set_title(title, pad=22)
    agreement = f"Spearman {rho:.2f}" if np.isfinite(rho) else "Spearman undefined: the values do not vary"
    ex._subtitle(ax, f"Agreement is a cloud rising to the top right; the units differ, so only the ordering "
                     f"can agree ({agreement})")
    ax.legend(loc="upper left", fontsize=8.5)
    return ex._save(fig, path)


def plot_uncertainty_zones(curve: pd.DataFrame, zones, probabilities: np.ndarray, labels: np.ndarray,
                           title: str, path: Path) -> Path:
    """The fitted confidence band (Version 0.6): what each edge costs, and where the rows actually are.

    Top panel is the trade-off the rule walks along, one line per side, so a side that met its target and a
    side that never could look different at a glance. Bottom panel is the probabilities the edges were
    fitted on, split by true class, because a zone is only as good as the rows inside it. Two panels rather
    than one panel with two y-scales: quality-against-coverage and counts-per-probability share no axis. An
    edge that does not exist is drawn as no edge and said in words, never as a phantom line.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    ex.apply_style()
    prob = np.asarray(probabilities, dtype=np.float64).ravel()
    truth = np.asarray(labels).astype(np.float64).ravel()
    if prob.size != truth.size:
        raise ValueError(f"Need one label per probability, got {truth.size} for {prob.size}.")
    rule = getattr(zones, "rule", None)
    targets = {"susceptible": float(getattr(rule, "target_npv", float("nan"))),
               "resistant": float(getattr(rule, "target_precision", float("nan")))}
    edges = {"susceptible": getattr(zones, "lower", None), "resistant": getattr(zones, "upper", None)}
    floor = float(getattr(rule, "min_coverage", float("nan")))
    threshold = float(getattr(zones, "threshold", float("nan")))
    rows = curve if isinstance(curve, pd.DataFrame) and not curve.empty else pd.DataFrame()

    fig, (ax_curve, ax_hist) = plt.subplots(2, 1, figsize=(10.6, 8.8), gridspec_kw={"height_ratios": [1, 1.2]})
    notes, floors, achieved = [], [], []
    for side, key, dashes, marker, measure in (("susceptible", "S", (None, None), "o", "NPV"),
                                              ("resistant", "R", (4.5, 1.8), "^", "precision")):
        side_rows = rows[rows["side"].astype(str) == side] if "side" in rows else pd.DataFrame()
        if len(side_rows):
            side_rows = side_rows[np.isfinite(side_rows["achieved"].to_numpy(dtype=np.float64))]
            side_rows = side_rows.sort_values("coverage")
        colour, edge = ex.LABEL_COLORS[key], edges[side]
        name = f"{side.capitalize()} side ({measure})"
        if not len(side_rows):
            notes.append(f"no {side}-side cut covered a row")
            continue
        cover = side_rows["coverage"].to_numpy(dtype=np.float64)
        value = side_rows["achieved"].to_numpy(dtype=np.float64)
        achieved.extend(value.tolist())
        # the reason a zone is missing belongs to that side, so it goes in that side's legend entry
        missed = "" if edge is not None else f", none reached {targets[side]:.0%}: no zone"
        ax_curve.plot(cover, value, lw=1.6, color=colour, dashes=dashes, marker=marker, ms=3.4,
                      label=f"{name}, {len(side_rows)} cuts{missed}", zorder=3)
        if edge is None:
            continue
        i = int(np.argmin(np.abs(side_rows["cut"].to_numpy(dtype=np.float64) - float(edge))))
        ax_curve.scatter([cover[i]], [value[i]], s=95, marker=marker, facecolor=ex.SURFACE, edgecolor=colour,
                         linewidths=1.8, zorder=5)
        ax_curve.annotate(f"chosen {side} edge {float(edge):.3g}\n{cover[i]:.0%} of rows at {value[i]:.1%}",
                          (cover[i], value[i]), textcoords="offset points",   # above: nothing is plotted there
                          xytext=(14, 12), fontsize=8, color=ex.INK_2,
                          ha="left", zorder=6, bbox={"facecolor": ex.SURFACE, "edgecolor": "none", "alpha": 0.8},
                          arrowprops={"arrowstyle": "-", "lw": 0.6, "color": ex.AXIS, "shrinkA": 0, "shrinkB": 7})
    if not achieved:
        ax_curve.text(0.5, 0.5, "No candidate cut covered any row", transform=ax_curve.transAxes, ha="center",
                      va="center", fontsize=10, color=ex.INK_2)
    for level in sorted({t for t in targets.values() if np.isfinite(t)}):
        sides = " and ".join(s for s, t in targets.items() if t == level)
        ax_curve.axhline(level, dashes=(5, 2.5), lw=1, color=ex.AXIS, zorder=2)
        floors.append(level)
        ax_curve.text(0.995, level + 0.004, f"target {level:.0%} ({sides})", transform=ax_curve.get_yaxis_transform(),
                      ha="right", va="bottom", fontsize=8, color=ex.INK_2)
    bottom = max(0.0, min(achieved + floors, default=0.0) - 0.06)
    ax_curve.set_ylim(bottom, 1.12)                    # headroom above 100%: the edge notes live there
    ax_curve.set_yticks([t for t in ax_curve.get_yticks() if bottom <= t <= 1.0])
    handles, _ = ax_curve.get_legend_handles_labels()
    if np.isfinite(floor) and 0 < floor < 1:
        # the floor is a legend entry, not a label in the band: the band is where the curves start
        ax_curve.axvspan(0, floor, color=ex.GRID, zorder=0,
                         label=f"Under the {floor:.0%} coverage floor, no edge allowed")
        handles, _ = ax_curve.get_legend_handles_labels()
    ax_curve.set_xlim(0, 1)
    ax_curve.set(xlabel="Coverage: share of the fitted rows the zone would claim",
                 ylabel="Achieved NPV / precision inside the zone")
    for axis in (ax_curve.xaxis, ax_curve.yaxis):
        axis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_curve.set_title("What each candidate edge buys, per side", pad=22)
    fitted = f"{int(getattr(zones, 'n_fitted', 0)):,} rows of the {getattr(zones, 'fitted_on', 'validation')} part"
    spelled = "; ".join(notes)
    ex._subtitle(ax_curve, f"Edges fitted on {fitted}; the model cut-off is unchanged"
                           + (f". {spelled[:1].upper()}{spelled[1:]}" if notes else ""))
    if handles:
        ax_curve.legend(loc="lower left", fontsize=8.5)

    good = np.isfinite(prob)
    lower, upper = edges["susceptible"], edges["resistant"]
    ends = [v for v in (prob[good].max() if good.any() else None, lower, upper, threshold)
            if v is not None and np.isfinite(v)]
    high = float(min(1.0, max(max(ends, default=1.0) * 1.05, 1e-3)))
    bins = np.linspace(0.0, high, 41)
    blend = ax_hist.get_xaxis_transform()
    series = [(prob[good & (truth == v)], ex.LABEL_COLORS[k], f"True {n} (n={int((truth == v).sum()):,})", h)
              for v, k, n, h in ((0.0, "S", "susceptible", ""), (1.0, "R", "resistant", "///"))]
    series = [s for s in series if s[0].size]
    tallest = 1
    if series:
        width = (bins[1] - bins[0]) / (len(series) + 0.6)
        middles = (bins[:-1] + bins[1:]) / 2
        for n, (values, colour, name, hatch) in enumerate(series):
            counts, _ = np.histogram(values, bins=bins)
            tallest = max(tallest, int(counts.max()))
            ax_hist.bar(middles + (n - (len(series) - 1) / 2) * width, counts, width=width * 0.92, color=colour,
                        hatch=hatch or None, edgecolor=ex.INK_2 if hatch else ex.SURFACE, linewidth=0.45,
                        label=name, zorder=3)
    else:
        ax_hist.text(0.5, 0.5, "No probability to plot", transform=ax_hist.transAxes, ha="center", va="center",
                     fontsize=10, color=ex.INK_2)
    ax_hist.set_ylim(0, tallest * 1.42)                             # room for the zone names and the legend
    zone_bands = []
    if lower is not None:
        zone_bands.append((0.0, float(lower), "High-confidence susceptible", ex.LABEL_COLORS["S"]))
    zone_bands.append((0.0 if lower is None else float(lower), high if upper is None else float(upper),
                       "Uncertain", ex.MUTED))
    if upper is not None:
        zone_bands.append((float(upper), high, "High-confidence resistant", ex.LABEL_COLORS["R"]))
    plate = {"facecolor": ex.SURFACE, "edgecolor": "none", "alpha": 0.8, "pad": 1.2}
    for start, stop, name, colour in zone_bands:
        ax_hist.axvspan(start, stop, color=colour, alpha=0.13, zorder=0)
        wide = (stop - start) / high > 0.22
        ax_hist.text((start + stop) / 2, 0.985, name, transform=blend, ha="center", fontsize=8.5, color=ex.INK_2,
                     va="top", rotation=0 if wide else 90, bbox=plate, zorder=6)
    for value, name, dashes in ((threshold, f"cut-off {threshold:.3g}", (1, 2)),
                                (lower, f"lower edge {float(lower):.3g}" if lower is not None else "", (None, None)),
                                (upper, f"upper edge {float(upper):.3g}" if upper is not None else "", (None, None))):
        if value is None or not np.isfinite(value):
            continue
        ax_hist.axvline(float(value), dashes=dashes, lw=1.3, color=ex.INK_2, zorder=4)
        ax_hist.text(float(value) + high * 0.006, 0.3, name, transform=blend, rotation=90, ha="left", va="bottom",
                     fontsize=8, color=ex.INK_2, zorder=5, bbox=plate)
    ax_hist.set_xlim(0, high)
    ax_hist.set(xlabel="Calibrated probability of resistance", ylabel="Rows")
    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))      # counts of rows are whole numbers
    ax_hist.set_title("Where the rows sit, by true class, with the three zones", pad=8)
    if series:
        ax_hist.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return ex._save(fig, path)


def plot_generalisation(frame: pd.DataFrame, reference: float, reference_label: str,
                        title: str, path: Path) -> Path:
    """AUROC where the model was not trained, one row per experiment and site, with its interval.

    A forest plot rather than bars: the interval is the point of the figure, and bars invite reading the
    distance from zero, which is meaningless for AUROC. The reference line is the split the model was
    developed on, and 0.5 is drawn because a site whose interval reaches it has no demonstrated skill there.
    """
    import matplotlib.pyplot as plt

    if frame.empty:
        raise ValueError("Nothing to plot: no experiment was scored.")
    ex.apply_style()
    frame = frame.iloc[::-1].reset_index(drop=True)          # first experiment at the top
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.6, 1.0 + 0.62 * len(frame)))

    ax.axvline(0.5, ls=":", lw=1.2, color=ex.MUTED, zorder=1)
    ax.annotate("no skill", (0.5, len(frame) - 0.42), fontsize=8, color=ex.MUTED, ha="center", va="bottom")
    ax.axvline(reference, ls="--", lw=1.3, color=ex.SERIES[1], zorder=1,
               label=f"{reference_label} ({reference:.3f})")

    low = frame["roc_auc"].to_numpy() - frame["low"].to_numpy()
    high = frame["high"].to_numpy() - frame["roc_auc"].to_numpy()
    ax.errorbar(frame["roc_auc"], y, xerr=np.vstack([low, high]), fmt="o", ms=7, lw=1.6, capsize=4,
                color=ex.SERIES[0], markeredgecolor=ex.SURFACE, markeredgewidth=1.2, zorder=3,
                label="AUROC with a 95 % interval")
    for i, r in frame.iterrows():
        ax.annotate(f"{r['roc_auc']:.3f}", (r["high"], i), textcoords="offset points", xytext=(9, 0),
                    fontsize=8.5, color=ex.INK_2, va="center")
    ax.set_yticks(y, [f"{r['label']}\n{int(r['n']):,} spectra, {int(r['n_resistant']):,} resistant"
                      for _, r in frame.iterrows()], fontsize=9)
    ax.set_xlim(min(0.42, float(frame["low"].min()) - 0.04), max(0.9, float(frame["high"].max()) + 0.1))
    ax.set_ylim(-0.7, len(frame) - 0.2)
    ax.set_xlabel("AUROC on that test part")
    ax.set_title(title, pad=22)
    ex._subtitle(ax, "An interval that reaches the dotted line means no skill was demonstrated at that site")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    return ex._save(fig, path)


def plot_zone_transfer(frame: pd.DataFrame, target: float, title: str, path: Path) -> Path:
    """Does the confidence zone fitted at one site still hold at another?

    The zone was fitted once, on the development site, and is applied here unchanged. Rows that isolate the
    change of site are drawn filled; rows that also carry a recalibrated probability scale are drawn hollow,
    because those two things do not mean the same thing and the figure should not hide the difference.
    """
    import matplotlib.pyplot as plt

    if frame.empty:
        raise ValueError("Nothing to plot: the zone was not carried anywhere.")
    ex.apply_style()
    frame = frame.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.8, 1.0 + 0.62 * len(frame)))

    ax.axvline(target, ls="--", lw=1.3, color=ex.SERIES[1], zorder=1, label=f"target ({target:.0%})")
    pure = frame["pure_site_test"].to_numpy(dtype=bool)
    low = frame["npv_susceptible"].to_numpy() - frame["npv_low"].to_numpy()
    high = frame["npv_high"].to_numpy() - frame["npv_susceptible"].to_numpy()
    for mask, face, label in ((pure, ex.SERIES[0], "same model, new site"),
                              (~pure, "none", "new site and a refitted, recalibrated model")):
        if not mask.any():
            continue
        ax.errorbar(frame["npv_susceptible"][mask], y[mask],
                    xerr=np.vstack([low[mask], high[mask]]), fmt="o", ms=7.5, lw=1.6, capsize=4,
                    color=ex.SERIES[0], markerfacecolor=face, markeredgecolor=ex.SERIES[0],
                    markeredgewidth=1.4, zorder=3, label=label)
    for i, r in frame.iterrows():
        ax.annotate(f"{r['npv_susceptible']:.3f}  ({r['share_susceptible']:.0%} of spectra)",
                    (r["npv_high"], i), textcoords="offset points", xytext=(9, 0), fontsize=8.5,
                    color=ex.INK_2, va="center")
    ax.set_yticks(y, [f"{r['part']}\n{r['model']}" for _, r in frame.iterrows()], fontsize=9)
    ax.set_xlim(min(0.5, float(frame["npv_low"].min()) - 0.05), 1.06)
    ax.set_ylim(-0.7, len(frame) - 0.2)
    ax.set_xlabel("Share of the confident-susceptible calls that were truly susceptible")
    ax.set_title(title, pad=22)
    ex._subtitle(ax, "The edge is the one fitted on the development site, applied unchanged and never refitted")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    return ex._save(fig, path)


def plot_region_shift(frame: pd.DataFrame, title: str, path: Path, keep: int = 10) -> Path:
    """How different do the regions the model relies on look at each site? No label is involved.

    Grouped bars, one group per m/z region and one bar per test part, so a drop in performance can be read
    next to how much the input actually moved. A standardised mean difference of 0 means the region looks
    the same as at the training site.
    """
    import matplotlib.pyplot as plt

    if frame.empty:
        raise ValueError("Nothing to plot: no region shift was computed.")
    ex.apply_style()
    regions = sorted(frame["rank"].unique())[:keep]
    frame = frame[frame["rank"].isin(regions)]
    parts = list(dict.fromkeys(frame["part"]))
    labels = [f"{r.mz_start:,.0f}–{r.mz_end:,.0f}"
              for r in frame.drop_duplicates("rank").sort_values("rank").itertuples(index=False)]
    x = np.arange(len(regions))
    width = min(0.8 / max(len(parts), 1), 0.26)

    fig, ax = plt.subplots(figsize=(max(8.4, 1.0 + 1.15 * len(regions)), 5.4))
    ax.axhline(0, lw=1.1, color=ex.AXIS, zorder=1)
    for i, part in enumerate(parts):
        values = [float(frame[(frame["rank"] == r) & (frame["part"] == part)]["smd"].iloc[0])
                  if len(frame[(frame["rank"] == r) & (frame["part"] == part)]) else np.nan
                  for r in regions]
        ax.bar(x + (i - (len(parts) - 1) / 2) * width, values, width * 0.92, label=part,
               color=ex.SERIES[i % len(ex.SERIES)], edgecolor=ex.SURFACE, linewidth=1.2, zorder=2)
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_xlabel("m/z region the model relies on (Version 0.6 ranking, strongest first)")
    ax.set_ylabel("Standardised mean difference\nagainst the training site")
    ax.set_title(title, pad=22)
    ex._subtitle(ax, "0 means the region looks the same as where the model was trained; no AST label is used here")
    ax.legend(title="Tested on", fontsize=8.5, title_fontsize=8.5)
    ax.grid(axis="x", visible=False)
    return ex._save(fig, path)
