"""Version 1.4: excluded screening isolates as training-only data (docs/v1.4_screening_plan.md, amendment 11).

Fixed by the plan, recorded before this file existed. The clinical development pool is cross-validated exactly as
in Version 1.2; the screening (HospitalHygiene) spectra come from a dataset built with those samples kept, and may
only ever be *training* rows:

- `link_screening` maps that dataset onto the base one (every shared row must have identical features, label and
  patient grouping) and sorts each screening row: it joins a pool patient, it is a new patient, or it is excluded
  because its patient has a clinical row outside the pool (a spent test part, its patients, or later data).
- `training_screening` gives the screening rows one fold may train on: never one of a held-out patient.
- `corrected_cv_interval` is the plan's primary interval; `group_bootstrap` resamples whole patient groups.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.dataset import sample_keys

JOINS, NEW = "joins a pool patient", "new patient"


class ScreeningError(RuntimeError):
    """A Version 1.4 data, linkage or leakage check failed; nothing is reported as a result."""


@dataclass(frozen=True)
class ScreeningSet:
    """The usable screening rows, in the base dataset's patient-group numbering."""

    rows: np.ndarray          # row numbers in the screening-inclusive dataset
    groups: np.ndarray        # joined rows: their pool patient's group id; new patients: ids above the base's
    y: np.ndarray
    dates: np.ndarray         # datetime64[ns]
    kind: np.ndarray          # JOINS or NEW
    base_rows_in_full: np.ndarray = field(repr=False)   # full-dataset row of every base row
    counts: dict[str, Any] = field(default_factory=dict)


def _row_bytes(X: np.ndarray, rows: np.ndarray) -> list[bytes]:
    return [np.ascontiguousarray(X[i]).tobytes() for i in rows]


def link_screening(base_meta: pd.DataFrame, base_X: np.ndarray, pool: np.ndarray, full_meta: pd.DataFrame,
                   full_X: np.ndarray, workstation: str, before: str) -> ScreeningSet:
    """Link the screening-inclusive dataset to the base dataset and select the usable screening rows.

    Checked, or the run stops: every base row is in the full dataset with byte-identical features and the same
    label; the full dataset's patient groups restricted to those rows are exactly the base's; and every extra row
    is of `workstation`, so the only difference between the datasets is the screening samples.
    """
    full_index = pd.Series(np.arange(len(full_meta)), index=sample_keys(full_meta))
    base_keys = sample_keys(base_meta)
    if not pd.Index(base_keys).isin(full_index.index).all():
        raise ScreeningError("A row of the base dataset is missing from the screening-inclusive dataset.")
    in_full = full_index.loc[base_keys].to_numpy()
    if not np.array_equal(full_meta["label"].to_numpy()[in_full], base_meta["label"].to_numpy()):
        raise ScreeningError("A shared row has a different label in the screening-inclusive dataset.")
    for start in range(0, in_full.size, 512):                      # features, byte for byte, in chunks
        chunk = np.arange(start, min(start + 512, in_full.size))
        if _row_bytes(base_X, chunk) != _row_bytes(full_X, in_full[chunk]):
            raise ScreeningError("A shared row has different features in the screening-inclusive dataset.")
    base_groups, full_groups = base_meta["group_id"].to_numpy(), full_meta["group_id"].to_numpy()
    pairs = pd.DataFrame({"base": base_groups, "full": full_groups[in_full]})
    if pairs.groupby("base")["full"].nunique().max() != 1 or pairs.groupby("full")["base"].nunique().max() != 1:
        raise ScreeningError("The two datasets group the shared rows into patients differently.")
    to_base = pairs.drop_duplicates("full").set_index("full")["base"]

    extra = np.setdiff1d(np.arange(len(full_meta)), in_full)
    if (full_meta["workstation"].to_numpy()[extra] != workstation).any():
        raise ScreeningError(f"The screening-inclusive dataset adds rows that are not {workstation} samples.")
    dates = pd.to_datetime(full_meta["acquisition_date"]).to_numpy()
    if pd.isna(dates[extra]).any():
        raise ScreeningError("A screening row has no acquisition date; its time cannot be checked.")
    extra = extra[dates[extra] < np.datetime64(pd.Timestamp(before))]

    pool_full_groups = set(full_groups[in_full[np.asarray(pool, dtype=np.int64)]])
    outside = np.setdiff1d(np.arange(len(base_meta)), pool)
    outside_full_groups = set(full_groups[in_full[outside]])
    g = full_groups[extra]
    excluded = np.array([x in outside_full_groups for x in g], dtype=bool)
    joins = np.array([x in pool_full_groups for x in g], dtype=bool) & ~excluded
    rows, g, joins = extra[~excluded], g[~excluded], joins[~excluded]
    new_ids = {x: int(base_groups.max()) + 1 + i for i, x in enumerate(np.unique(g[~joins]))}
    groups = np.array([to_base[x] if j else new_ids[x] for x, j in zip(g, joins, strict=True)], dtype=np.int64)
    y = full_meta["label"].to_numpy().astype(np.int64)[rows]
    kind = np.where(joins, JOINS, NEW)
    counts = {"usable": int(rows.size), "usable_resistant": int(y.sum()), "patient_groups": int(np.unique(groups).size),
              "joining": int(joins.sum()), "joining_groups": int(np.unique(groups[joins]).size),
              "new": int((~joins).sum()), "new_resistant": int(y[~joins].sum()),
              "new_groups": int(np.unique(groups[~joins]).size),
              "new_resistant_groups": int(np.unique(groups[~joins & (y == 1)]).size),
              "excluded_outside_pool": int(excluded.sum())}
    return ScreeningSet(rows, groups, y, dates[rows], kind, in_full, counts)


def training_screening(screen: ScreeningSet, held_out_groups: np.ndarray, before: str | None = None) -> np.ndarray:
    """Positions (within `screen`) a fold may train on: never a held-out patient's, and, if given, dated before."""
    keep = ~np.isin(screen.groups, np.asarray(held_out_groups))
    if before is not None:
        keep &= screen.dates < np.datetime64(pd.Timestamp(before))
    return np.flatnonzero(keep)


# --- uncertainty ----------------------------------------------------------------------------------------------------

def corrected_cv_interval(differences: np.ndarray, folds: int, repeats: int, level: float = 0.95) -> dict[str, float]:
    """Mean of per-fold differences with the corrected repeated k-fold variance (Nadeau & Bengio 2003;
    Bouckaert & Frank 2004): var = (1 / (k r) + 1 / (k - 1)) s^2, t with k r - 1 degrees of freedom."""
    d = np.asarray(differences, dtype=np.float64)
    if d.size != folds * repeats or d.size < 2:
        raise ScreeningError(f"Expected {folds * repeats} per-fold differences, got {d.size}.")
    se = float(np.sqrt((1 / (folds * repeats) + 1 / (folds - 1)) * d.var(ddof=1)))
    t = float(stats.t.ppf(0.5 + level / 2, d.size - 1))
    mean = float(d.mean())
    return {"estimate": mean, "low": mean - t * se, "high": mean + t * se, "sd": float(d.std(ddof=1)), "se": se,
            "df": int(d.size - 1), "level": level, "folds": int(d.size)}


def group_bootstrap(groups: np.ndarray, statistic: Callable[[np.ndarray], dict[str, float] | None], *,
                    resamples: int, seed: int) -> tuple[dict[str, np.ndarray], int]:
    """Resample whole patient groups with replacement; `statistic(rows)` returns named values, or None for a
    replicate that cannot be scored (e.g. one class only), which is skipped and counted."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inverse))]
    members = [order[bounds[i]:bounds[i + 1]] for i in range(bounds.size - 1)]
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    skipped = 0
    for _ in range(resamples):
        rows = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        out = statistic(rows)
        if out is None:
            skipped += 1
            continue
        for k, v in out.items():
            values.setdefault(k, []).append(v)
    return {k: np.asarray(v) for k, v in values.items()}, skipped


def percentile_interval(samples: np.ndarray, estimate: float, level: float) -> dict[str, float]:
    alpha = (1 - level) / 2
    return {"estimate": float(estimate), "low": float(np.quantile(samples, alpha)),
            "high": float(np.quantile(samples, 1 - alpha))}


def verdict(low: float, high: float) -> str:
    """The plan's section 6, for A1 minus A0 on AUROC."""
    if low > 0:
        return "improved ranking"
    if high < 0:
        return "worse ranking"
    return "not demonstrated"
