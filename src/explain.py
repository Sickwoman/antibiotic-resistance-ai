"""Version 0.6: what a saved model used, expressed in m/z instead of column numbers.

Three views of the same fitted model, because they answer different questions:

- **Exact TreeSHAP** (`contributions`): how much each bin moved *one* prediction. LightGBM computes this
  itself, so the numbers are exact rather than sampled, and every spectrum's contributions sum to its
  margin. The calibration step that follows is a monotone sigmoid, so a positive contribution raises the
  calibrated probability too; the size of the change in probability is not additive, which is why
  contributions are reported on the margin and never as "percentage points of risk".
- **Permutation importance** (`permutation_importance`): how much the model's *AUROC* depends on a region.
  It is model-agnostic, so the same function runs on a network, and it is computed on blocks of
  neighbouring bins by default: bins of one peak are strongly correlated, so permuting a single bin lets
  the model read the same peak from its neighbour and understates it.
- **Class contrast** (`class_contrast`): the plain resistant-versus-susceptible intensity difference of a
  region, with no model involved. A region the model relies on that shows no univariate difference is
  being used in combination with others.

Regions, not bins, are the reporting unit: one 3 Da bin is below the resolution at which anything can be
said. No region is given a protein or peptide identity anywhere in this project (see
docs/v0.6_explainability_plan.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.preprocessing import PreprocessingConfig
from src.tuning import CoarsenBins, uncalibrated


class ExplainError(ValueError):
    """The model or the input cannot be explained the way it was asked for."""


def inner_pipeline(model: Any) -> Pipeline:
    """The fitted pipeline inside a calibrated model, or the pipeline itself."""
    if isinstance(model, CalibratedClassifierCV):
        return uncalibrated(model)
    if isinstance(model, Pipeline):
        return model
    raise ExplainError(f"Expected a pipeline or a calibrated model, got {type(model).__name__}.")


def bin_span(model: Any) -> int:
    """How many raw 3 Da bins one model column covers (6 when the pipeline coarsens, otherwise 1)."""
    steps = dict(inner_pipeline(model).steps)
    coarsen = steps.get("coarsen", "passthrough")
    return int(coarsen.factor) if isinstance(coarsen, CoarsenBins) else 1


def contributions(model: Any, X: np.ndarray, tolerance: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Exact TreeSHAP contributions and base value per row, on the model's own columns.

    Raises if the final estimator is not a gradient-boosted tree model, and if the contributions do not add
    up to the margin the model actually predicts: that identity is what makes them exact, so it is checked
    rather than trusted.
    """
    pipeline = inner_pipeline(model)
    estimator = pipeline.steps[-1][1]
    booster = getattr(estimator, "booster_", None)
    if booster is None:
        raise ExplainError(f"Exact TreeSHAP needs a fitted LightGBM model; this pipeline ends in "
                           f"{type(estimator).__name__}. Use permutation_importance for other models.")
    features = np.asarray(X, dtype=np.float32)
    if features.ndim == 1:
        features = features[None, :]
    transformed = pipeline[:-1].transform(features) if len(pipeline.steps) > 1 else features
    raw = np.asarray(booster.predict(transformed, pred_contrib=True), dtype=np.float64)
    contrib, base = raw[:, :-1], raw[:, -1]
    margin = np.asarray(booster.predict(transformed, raw_score=True), dtype=np.float64)
    gap = float(np.abs(contrib.sum(axis=1) + base - margin).max()) if len(margin) else 0.0
    if gap > tolerance:
        raise ExplainError(f"TreeSHAP contributions do not sum to the predicted margin (worst gap {gap:.2e} "
                           f"> {tolerance:.0e}); the explanation would not describe this model.")
    return contrib, base


def model_columns(model: Any, X: np.ndarray) -> np.ndarray:
    """The matrix the final estimator actually sees (raw bins, or coarsened ones), for the same rows."""
    pipeline = inner_pipeline(model)
    features = np.asarray(X, dtype=np.float32)
    if features.ndim == 1:
        features = features[None, :]
    return pipeline[:-1].transform(features) if len(pipeline.steps) > 1 else features


def _column_spearman(values: np.ndarray, contrib: np.ndarray) -> np.ndarray:
    """Spearman correlation per column between a column's value and its own contribution.

    This, not the mean signed contribution, is what gives a region its global direction: averaged over a
    whole part, a feature's signed contributions cancel out (it pushes up on resistant rows and down on
    susceptible ones), so the mean is near zero for even the strongest region. The correlation says which
    way the region points: positive means a higher intensity there pushes the prediction towards resistant.
    """
    from scipy.stats import rankdata

    a = rankdata(np.asarray(values, dtype=np.float64), axis=0)
    b = rankdata(np.asarray(contrib, dtype=np.float64), axis=0)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denominator = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    out = np.full(a.shape[1], np.nan, dtype=np.float64)
    np.divide((a * b).sum(axis=0), denominator, out=out, where=denominator > 0)
    return out


def column_frame(n_columns: int, span: int, pcfg: PreprocessingConfig) -> pd.DataFrame:
    """One row per model column with the raw bins and the m/z interval it covers."""
    edges = pcfg.bin_edges
    if n_columns * span > len(edges) - 1:
        raise ExplainError(f"{n_columns} columns of {span} bins do not fit {len(edges) - 1} bins of "
                           f"{pcfg.bin_width} Da; the model and the preprocessing settings disagree.")
    start = np.arange(n_columns, dtype=np.int64) * span
    return pd.DataFrame({"column": np.arange(n_columns, dtype=np.int64), "bin_start": start,
                         "bin_end": start + span - 1, "mz_start": edges[start].astype(np.float64),
                         "mz_end": edges[start + span].astype(np.float64)})


def global_importance(contrib: np.ndarray, span: int, pcfg: PreprocessingConfig,
                      columns: np.ndarray | None = None) -> pd.DataFrame:
    """Mean |contribution| per column over the rows given, strongest first.

    Pass `columns` (the matrix from `model_columns`) to get `value_correlation`, which is what a global
    direction has to be read from; `mean_signed` is kept because it is the right quantity for a single
    spectrum, but it is near zero over a whole part and must not be read as a direction there.
    """
    contrib = np.asarray(contrib, dtype=np.float64)
    if contrib.ndim != 2 or contrib.size == 0:
        raise ExplainError("Contributions must be a non-empty (rows x columns) matrix.")
    frame = column_frame(contrib.shape[1], span, pcfg)
    frame["mean_abs"] = np.abs(contrib).mean(axis=0)
    frame["mean_signed"] = contrib.mean(axis=0)
    frame["rows_used"] = int(contrib.shape[0])
    frame["nonzero_rows"] = (np.abs(contrib) > 0).sum(axis=0)
    if columns is None:
        frame["value_correlation"] = np.nan
    else:
        values = np.asarray(columns, dtype=np.float64)
        if values.shape != contrib.shape:
            raise ExplainError(f"Columns {values.shape} do not match the contributions {contrib.shape}.")
        frame["value_correlation"] = _column_spearman(values, contrib)
    return frame.sort_values("mean_abs", ascending=False).reset_index(drop=True)


def merge_regions(importance: pd.DataFrame, top_bins: int = 100, merge_gap: int = 2, keep: int | None = 20,
                  towards_from: str = "value_correlation") -> pd.DataFrame:
    """Merge the most important neighbouring columns into m/z regions, strongest region first.

    `merge_gap` is in columns: two important columns with at most that many unimportant columns between
    them describe the same peak and become one region.

    `towards_from` decides what the `towards` column means, and the two meanings are different:
    - `"value_correlation"` (for a whole part): *a higher intensity in this region pushes towards* R or S;
    - `"signed"` (for one spectrum): *this spectrum was pushed towards* R or S.
    """
    if top_bins < 1 or merge_gap < 0:
        raise ExplainError(f"top_bins must be >= 1 and merge_gap >= 0 (got {top_bins}, {merge_gap}).")
    if towards_from not in ("value_correlation", "signed"):
        raise ExplainError(f"towards_from must be 'value_correlation' or 'signed', got {towards_from!r}.")
    has_correlation = "value_correlation" in importance.columns
    chosen = importance.nlargest(int(top_bins), "mean_abs").sort_values("column")
    regions: list[dict[str, Any]] = []
    for row in chosen.itertuples(index=False):
        correlation = float(getattr(row, "value_correlation", np.nan)) if has_correlation else np.nan
        weighted = 0.0 if np.isnan(correlation) else correlation * row.mean_abs
        weight = 0.0 if np.isnan(correlation) else row.mean_abs
        if regions and row.column - regions[-1]["last_column"] <= merge_gap:
            region = regions[-1]
            region.update(last_column=row.column, mz_end=row.mz_end, n_columns=region["n_columns"] + 1,
                          total_abs=region["total_abs"] + row.mean_abs,
                          total_signed=region["total_signed"] + row.mean_signed,
                          _weighted=region["_weighted"] + weighted, _weight=region["_weight"] + weight)
            if abs(row.mean_signed) > abs(region["peak_signed"]):
                region.update(peak_mz=(row.mz_start + row.mz_end) / 2, peak_signed=row.mean_signed)
        else:
            regions.append({"first_column": row.column, "last_column": row.column, "mz_start": row.mz_start,
                            "mz_end": row.mz_end, "n_columns": 1, "total_abs": row.mean_abs,
                            "total_signed": row.mean_signed, "peak_mz": (row.mz_start + row.mz_end) / 2,
                            "peak_signed": row.mean_signed, "_weighted": weighted, "_weight": weight})
    frame = pd.DataFrame(regions)
    if frame.empty:
        return frame
    frame["value_correlation"] = np.where(frame["_weight"] > 0, frame["_weighted"] / frame["_weight"], np.nan)
    basis = frame["value_correlation"] if towards_from == "value_correlation" else frame["total_signed"]
    if towards_from == "value_correlation" and not np.isfinite(basis).any():
        raise ExplainError("No value correlations available: pass the model columns to global_importance, "
                           "or ask for towards_from='signed' (one spectrum).")
    frame["towards"] = np.where(basis > 0, "resistant", "susceptible")
    frame = frame.drop(columns=["_weighted", "_weight"]).sort_values("total_abs", ascending=False)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame if keep is None else frame.head(int(keep)).copy()


def contiguous_blocks(n_columns: int, block: int) -> list[tuple[int, int]]:
    """[(start, end), ...] column ranges of width `block`; the last one may be shorter."""
    if block < 1:
        raise ExplainError(f"Block width must be >= 1, got {block}.")
    return [(start, min(start + block, n_columns)) for start in range(0, n_columns, block)]


def permutation_importance(model: Any, X: np.ndarray, y: np.ndarray, blocks: list[tuple[int, int]], *,
                           repeats: int = 5, seed: int = 42, pcfg: PreprocessingConfig | None = None,
                           progress=None) -> pd.DataFrame:
    """AUROC drop when each block of input bins is permuted across rows, jointly.

    Model-agnostic: `model` only has to provide `predict_proba`, so the same call describes a tree model
    and a network. The columns are the *model input* (raw bins), not any coarsened representation, so
    results from different feature settings can be compared in m/z.
    """
    features = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y).astype(np.int64)
    if features.ndim != 2 or len(labels) != len(features):
        raise ExplainError("Need a (rows x bins) matrix and one label per row.")
    if np.unique(labels).size < 2:
        raise ExplainError("Permutation importance needs both classes in the rows given.")
    if repeats < 1:
        raise ExplainError(f"repeats must be >= 1, got {repeats}.")
    baseline = float(roc_auc_score(labels, model.predict_proba(features)[:, 1]))
    rng = np.random.default_rng(seed)
    working = features.copy()
    rows: list[dict[str, Any]] = []
    for number, (start, end) in enumerate(blocks, start=1):
        if not 0 <= start < end <= features.shape[1]:
            raise ExplainError(f"Block ({start}, {end}) is outside the {features.shape[1]} bins.")
        original = features[:, start:end]
        drops = np.empty(repeats, dtype=np.float64)
        for repeat in range(repeats):
            working[:, start:end] = original[rng.permutation(len(features))]
            drops[repeat] = baseline - float(roc_auc_score(labels, model.predict_proba(working)[:, 1]))
        working[:, start:end] = original                      # restore before moving to the next block
        rows.append({"bin_start": start, "bin_end": end - 1, "auroc_drop_mean": drops.mean(),
                     "auroc_drop_sd": drops.std(ddof=1) if repeats > 1 else float("nan"),
                     "auroc_drop_min": drops.min(), "auroc_drop_max": drops.max()})
        if progress is not None and number % progress == 0:
            print(f"    permuted {number}/{len(blocks)} blocks", flush=True)
    frame = pd.DataFrame(rows)
    frame.insert(0, "baseline_roc_auc", baseline)
    if pcfg is not None:
        edges = pcfg.bin_edges
        frame["mz_start"] = edges[frame["bin_start"].to_numpy()]
        frame["mz_end"] = edges[frame["bin_end"].to_numpy() + 1]
    return frame.sort_values("auroc_drop_mean", ascending=False).reset_index(drop=True)


def class_contrast(X: np.ndarray, y: np.ndarray, regions: pd.DataFrame) -> pd.DataFrame:
    """Standardised mean difference of summed region intensity, resistant minus susceptible.

    No model is involved: this says whether a region differs between the classes on its own. Positive
    means higher intensity in resistant spectra.
    """
    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y).astype(np.int64)
    if np.unique(labels).size < 2:
        raise ExplainError("The class contrast needs both classes.")
    out: list[dict[str, Any]] = []
    for row in regions.itertuples(index=False):
        first, last = int(row.first_column), int(row.last_column)
        total = features[:, first:last + 1].sum(axis=1)
        resistant, susceptible = total[labels == 1], total[labels == 0]
        pooled = np.sqrt((resistant.var(ddof=1) + susceptible.var(ddof=1)) / 2)
        difference = float(resistant.mean() - susceptible.mean())
        out.append({"rank": int(row.rank), "mz_start": float(row.mz_start), "mz_end": float(row.mz_end),
                    "mean_resistant": float(resistant.mean()), "mean_susceptible": float(susceptible.mean()),
                    "difference": difference,
                    "standardised_difference": difference / pooled if pooled > 0 else float("nan")})
    return pd.DataFrame(out)


def rank_agreement(first: np.ndarray, second: np.ndarray, top_k: int = 20) -> dict[str, float]:
    """How much two importance vectors agree: Spearman correlation and overlap of their top `top_k`."""
    from scipy.stats import spearmanr

    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ExplainError(f"Need two importance vectors of the same length, got {a.shape} and {b.shape}.")
    k = min(int(top_k), a.size)
    top_a, top_b = set(np.argsort(-a)[:k].tolist()), set(np.argsort(-b)[:k].tolist())
    correlation = float(spearmanr(a, b).statistic) if a.size > 2 else float("nan")
    return {"spearman": correlation, "top_k": k, "overlap": len(top_a & top_b),
            "overlap_fraction": len(top_a & top_b) / k if k else float("nan")}


def explain_one(model: Any, features: np.ndarray, pcfg: PreprocessingConfig, *, top_bins: int = 40,
                merge_gap: int = 2, keep: int = 5) -> pd.DataFrame:
    """The regions that moved one spectrum's prediction most, signed (positive = towards resistant)."""
    contrib, _ = contributions(model, features)
    if contrib.shape[0] != 1:
        raise ExplainError(f"Expected one spectrum, got {contrib.shape[0]} rows.")
    span = bin_span(model)
    frame = column_frame(contrib.shape[1], span, pcfg)
    frame["mean_abs"] = np.abs(contrib[0])
    frame["mean_signed"] = contrib[0]
    # One spectrum: the signed contribution *is* the direction, so no value correlation is involved.
    return merge_regions(frame, top_bins=top_bins, merge_gap=merge_gap, keep=keep, towards_from="signed")
