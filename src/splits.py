"""Version 0.2 split infrastructure (no model training here).

Three split types, all returning row indices into a built dataset (metadata.csv / X.npy):

* random   - stratified by label, patient groups kept together (StratifiedGroupKFold)
* temporal - by acquisition_date only (never the year folder); patient groups that would appear on
             both sides of a date boundary are removed from the earlier part
* external - train on some sites, test on other sites

Known limitation: DRIAMS-A patient_no is re-hashed per year folder, so a person who appears in two
years has two different group IDs and cannot be kept together. Patients cannot be linked across
sites either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PARTS = ("train", "validation", "test")


class LeakageError(RuntimeError):
    """Raised when the same sample or patient group appears in more than one part of a split."""


class SplitError(ValueError):
    pass


@dataclass
class Split:
    name: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    description: str
    notes: list[str] = field(default_factory=list)

    def parts(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def summary(self, meta: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "description": self.description, "notes": self.notes}
        for part, idx in self.parts().items():
            m = meta.iloc[idx]
            dates = pd.to_datetime(m["acquisition_date"], errors="coerce")
            out[part] = {
                "samples": int(len(idx)),
                "resistant": int((m["label"] == 1).sum()),
                "susceptible": int((m["label"] == 0).sum()),
                "sites": {k: int(v) for k, v in m["site"].value_counts().sort_index().items()},
                "groups": int(m["group_id"].nunique()),
                "date_min": None if dates.isna().all() else dates.min().strftime("%Y-%m-%d"),
                "date_max": None if dates.isna().all() else dates.max().strftime("%Y-%m-%d"),
            }
        return out

    def save(self, path: Path, meta: pd.DataFrame) -> None:
        """Save indices (not identifiers) plus a summary as JSON."""
        payload = self.summary(meta)
        payload["indices"] = {part: idx.astype(int).tolist() for part, idx in self.parts().items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def load_split(path: Path) -> Split:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    idx = {p: np.asarray(payload["indices"][p], dtype=np.int64) for p in PARTS}
    return Split(payload["name"], idx["train"], idx["validation"], idx["test"],
                 payload["description"], payload.get("notes", []))


def check_split(meta: pd.DataFrame, split: Split) -> None:
    """Fail loudly if a row or a patient group is shared between parts."""
    parts = split.parts()
    for part, idx in parts.items():
        if idx.size and (idx.min() < 0 or idx.max() >= len(meta)):
            raise LeakageError(f"{split.name}: {part} has indices outside the dataset.")
        if np.unique(idx).size != idx.size:
            raise LeakageError(f"{split.name}: {part} contains duplicate indices.")
    groups = {part: set(meta["group_id"].to_numpy()[idx]) for part, idx in parts.items()}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.intersect1d(parts[a], parts[b]).size:
                raise LeakageError(f"{split.name}: samples shared between {a} and {b}.")
            shared = groups[a] & groups[b]
            if shared:
                raise LeakageError(f"{split.name}: {len(shared)} patient group(s) shared between {a} and {b}.")


def _rows_for_sites(meta: pd.DataFrame, sites: Iterable[str]) -> np.ndarray:
    sites = list(sites)
    unknown = sorted(set(sites) - set(meta["site"]))
    if unknown:
        raise SplitError(f"Site(s) {unknown} are not in this dataset (available: {sorted(meta['site'].unique())}).")
    return np.flatnonzero(meta["site"].isin(sites).to_numpy())


def _group_fold(meta: pd.DataFrame, rows: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split `rows` into (rest, held_out) with ~fraction held out, stratified, groups intact."""
    if not 0 < fraction < 1:
        raise SplitError(f"fraction must be between 0 and 1, got {fraction}")
    n_splits = max(2, int(round(1 / fraction)))
    y = meta["label"].to_numpy()[rows]
    groups = meta["group_id"].to_numpy()[rows]
    if np.unique(groups).size < n_splits:
        raise SplitError(f"Only {np.unique(groups).size} groups; need at least {n_splits}.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_pos, held_pos = next(splitter.split(np.zeros(rows.size), y, groups))
    return np.sort(rows[rest_pos]), np.sort(rows[held_pos])


def random_split(meta: pd.DataFrame, sites: Iterable[str], test_fraction: float, validation_fraction: float,
                 seed: int) -> Split:
    rows = _rows_for_sites(meta, sites)
    pool, test = _group_fold(meta, rows, test_fraction, seed)
    train, validation = _group_fold(meta, pool, validation_fraction / (1 - test_fraction), seed + 1)
    split = Split("random", train, validation, test,
                  f"Stratified, patient-grouped random split of {sorted(set(sites))} "
                  f"(test {test_fraction:.0%}, validation {validation_fraction:.0%}, seed {seed}).",
                  ["Group IDs are only valid within one DRIAMS-A year folder (patient_no is re-hashed per year)."])
    check_split(meta, split)
    return split


def temporal_split(meta: pd.DataFrame, sites: Iterable[str], validation_start: str, test_start: str,
                   date_column: str = "acquisition_date") -> Split:
    if date_column != "acquisition_date":
        raise SplitError("Temporal splits must use acquisition_date (never the year folder).")
    rows = _rows_for_sites(meta, sites)
    dates = pd.to_datetime(meta[date_column], errors="coerce").to_numpy()[rows]
    v_start, t_start = np.datetime64(pd.Timestamp(validation_start)), np.datetime64(pd.Timestamp(test_start))
    if not v_start < t_start:
        raise SplitError("validation_start must be earlier than test_start.")
    has_date = ~np.isnat(dates)
    if not has_date.any():
        raise SplitError(f"No acquisition dates for sites {sorted(set(sites))}; a temporal split is impossible.")
    notes = [f"{int((~has_date).sum())} sample(s) without acquisition_date left out."] if (~has_date).any() else []

    test = rows[has_date & (dates >= t_start)]
    validation = rows[has_date & (dates >= v_start) & (dates < t_start)]
    train = rows[has_date & (dates < v_start)]

    groups = meta["group_id"].to_numpy()
    test_groups = set(groups[test])
    before = validation.size
    validation = validation[~np.isin(groups[validation], list(test_groups))]
    later_groups = test_groups | set(groups[validation])
    before_train = train.size
    train = train[~np.isin(groups[train], list(later_groups))]
    notes.append(f"Removed {before - validation.size} validation and {before_train - train.size} training sample(s) "
                 "whose patient group also occurs in a later part.")
    notes.append("Patient groups are only valid within one year folder; cross-year overlap cannot be detected.")
    split = Split("temporal", np.sort(train), np.sort(validation), np.sort(test),
                  f"By acquisition_date at {sorted(set(sites))}: train < {validation_start} <= validation "
                  f"< {test_start} <= test.", notes)
    check_split(meta, split)
    return split


def external_split(meta: pd.DataFrame, train_sites: Iterable[str], test_sites: Iterable[str],
                   validation_fraction: float, seed: int) -> Split:
    train_sites, test_sites = list(train_sites), list(test_sites)
    if set(train_sites) & set(test_sites):
        raise SplitError("A site cannot be used for both training and external testing.")
    pool = _rows_for_sites(meta, train_sites)
    test = _rows_for_sites(meta, test_sites)
    train, validation = _group_fold(meta, pool, validation_fraction, seed)
    split = Split("external", train, validation, np.sort(test),
                  f"Train/validation on {train_sites}, external test on {test_sites} "
                  f"(validation {validation_fraction:.0%} of the training sites, seed {seed}).",
                  ["Patients cannot be linked across hospitals, so cross-site patient overlap is not checked."])
    check_split(meta, split)
    return split


def make_splits(meta: pd.DataFrame, config: dict[str, Any]) -> dict[str, Split]:
    s = config["splits"]
    seed = int(config["project"]["random_seed"])
    splits = {
        "random": random_split(meta, s["random"]["sites"], s["random"]["test_fraction"],
                               s["random"]["validation_fraction"], seed),
        "temporal": temporal_split(meta, s["temporal"]["sites"], s["temporal"]["validation_start"],
                                   s["temporal"]["test_start"], s["temporal"]["date_column"]),
    }
    available = set(meta["site"])
    test_sites = [site for site in s["external"]["test_sites"] if site in available]
    not_built = [site for site in s["external"]["test_sites"] if site not in available]
    if test_sites:
        external = external_split(meta, s["external"]["train_sites"], test_sites,
                                  s["external"]["validation_fraction"], seed)
        if not_built:
            external.notes.append(f"Configured test site(s) {not_built} are not in this dataset yet.")
        splits["external"] = external
    return splits
