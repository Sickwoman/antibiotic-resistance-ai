"""Version 0.2 split infrastructure (no model training here).

Four split types, all returning row indices into a built dataset (metadata.csv / X.npy):

* random      - stratified by label, patient groups kept together (StratifiedGroupKFold)
* within_year - the same, but inside one year folder, where DRIAMS-A patient IDs are consistent
* temporal    - by acquisition_date only (never the year folder); patient groups that would appear on
                both sides of a date boundary are removed from the earlier part
* external    - train on some sites, test on other sites

Every split carries the fingerprint of the dataset build it was made for and is checked against it
when loaded. Datasets other than the primary one (e.g. I excluded) reuse the primary dataset's saved
splits (`derive_split`), so a sample is always in the same part and results stay comparable.

Known limitation: DRIAMS-A patient_no is re-hashed per year folder, so a person who appears in two
years has two different group IDs and cannot be kept together. Patients cannot be linked across
sites either.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset import load_dataset, row_fingerprint, sample_keys

PARTS = ("train", "validation", "test")
SPLIT_ORDER = ("random", "within_year", "temporal", "external")   # order used in reports


class LeakageError(RuntimeError):
    """Raised when the same sample or patient group appears in more than one part of a split."""


class SplitError(ValueError):
    pass


class DatasetMismatchError(SplitError):
    """Raised when a split is used with a dataset build other than the one it was made for."""


@dataclass
class Split:
    name: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    description: str
    notes: list[str] = field(default_factory=list)
    dataset_fingerprint: str | None = None      # row_fingerprint of the dataset the indices refer to
    derived_from: dict[str, Any] | None = None  # reference dataset of a reused split

    def parts(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def summary(self, meta: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "description": self.description, "notes": self.notes,
                               "dataset_fingerprint": self.dataset_fingerprint, "derived_from": self.derived_from}
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
        """Save indices (not identifiers), the dataset fingerprint and a summary as JSON."""
        check_split(meta, self)
        self.dataset_fingerprint = row_fingerprint(meta)
        payload = self.summary(meta)
        payload["indices"] = {part: idx.astype(int).tolist() for part, idx in self.parts().items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def load_split(path: Path, meta: pd.DataFrame) -> Split:
    """Load a saved split and check that it belongs to `meta` and has no leakage."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not payload.get("dataset_fingerprint"):
        raise DatasetMismatchError(f"{Path(path).name} has no dataset fingerprint (made by an older version); "
                                   "rebuild the dataset with scripts/build_dataset.py.")
    idx = {p: np.asarray(payload["indices"][p], dtype=np.int64) for p in PARTS}
    split = Split(payload["name"], idx["train"], idx["validation"], idx["test"], payload["description"],
                  payload.get("notes", []), payload["dataset_fingerprint"], payload.get("derived_from"))
    check_split(meta, split)
    return split


def load_splits(folder: Path, meta: pd.DataFrame) -> dict[str, Split]:
    """All saved splits of a dataset, by name in SPLIT_ORDER (checked like `load_split`)."""
    splits = [load_split(p, meta) for p in Path(folder).glob("*.json")]
    rank = {name: i for i, name in enumerate(SPLIT_ORDER)}
    splits.sort(key=lambda s: (rank.get(s.name, len(rank)), s.name))
    return {s.name: s for s in splits}


def check_split(meta: pd.DataFrame, split: Split) -> None:
    """Fail loudly if the split belongs to another dataset or a row / patient group is shared between parts."""
    if split.dataset_fingerprint is not None and split.dataset_fingerprint != row_fingerprint(meta):
        raise DatasetMismatchError(f"{split.name}: the split was made for another dataset build (fingerprint "
                                   f"{split.dataset_fingerprint}, this dataset {row_fingerprint(meta)}). "
                                   "Rebuild the splits for this dataset.")
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


def assert_usable(meta: pd.DataFrame, split: Split, parts: Iterable[str] = ("train", "validation", "test"),
                  min_per_class: int = 1) -> None:
    """Refuse a split whose parts cannot carry the protocol's metrics.

    Building a split is a diagnostic step and may legitimately produce a thin part (a tiny dataset, or a
    date boundary that leaves little behind after overlapping patient groups are removed). Training or
    scoring on such a part is not: AUROC, PR-AUC and the sensitivity cut-off are undefined with one class,
    so the model scripts call this before they use a split and stop with a readable message instead of a
    confusing failure somewhere in scikit-learn.
    """
    labels = meta["label"].to_numpy()
    available = split.parts()
    for part in parts:
        idx = available[part]
        if idx.size == 0:
            raise SplitError(f"{split.name}: the {part} part is empty, so it cannot be used. See the split's "
                             "notes for how many samples were removed.")
        resistant = int(labels[idx].sum())
        if min(resistant, idx.size - resistant) < min_per_class:
            raise SplitError(f"{split.name}: the {part} part holds {resistant} resistant of {idx.size} samples, "
                             f"fewer than {min_per_class} of one class. AUROC, PR-AUC and the sensitivity "
                             "cut-off are undefined there.")


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
    if abs(1 / n_splits - fraction) > 0.02:  # the held-out part is one of n_splits folds
        raise SplitError(f"A held-out fraction of {fraction:.3f} cannot be produced with grouped folds "
                         f"(nearest is 1/{n_splits} = {1 / n_splits:.3f}); use a fraction such as 0.5, 0.25, "
                         "0.2 or 0.1 (for validation: validation_fraction / (1 - test_fraction)).")
    y = meta["label"].to_numpy()[rows]
    groups = meta["group_id"].to_numpy()[rows]
    if np.unique(groups).size < n_splits:
        raise SplitError(f"Only {np.unique(groups).size} groups; need at least {n_splits}.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_pos, held_pos = next(splitter.split(np.zeros(rows.size), y, groups))
    return np.sort(rows[rest_pos]), np.sort(rows[held_pos])


def _grouped_random_parts(meta: pd.DataFrame, rows: np.ndarray, test_fraction: float, validation_fraction: float,
                          seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pool, test = _group_fold(meta, rows, test_fraction, seed)
    train, validation = _group_fold(meta, pool, validation_fraction / (1 - test_fraction), seed + 1)
    return train, validation, test


def random_split(meta: pd.DataFrame, sites: Iterable[str], test_fraction: float, validation_fraction: float,
                 seed: int) -> Split:
    rows = _rows_for_sites(meta, sites)
    train, validation, test = _grouped_random_parts(meta, rows, test_fraction, validation_fraction, seed)
    split = Split("random", train, validation, test,
                  f"Stratified, patient-grouped random split of {sorted(set(sites))}: "
                  f"test {test.size / rows.size:.1%}, validation {validation.size / rows.size:.1%} of the samples "
                  f"(requested {test_fraction:.0%} / {validation_fraction:.0%}), seed {seed}.",
                  ["Group IDs are only valid within one DRIAMS-A year folder (patient_no is re-hashed per year), "
                   "so a patient seen in two years can be in two parts; compare with the within_year split."],
                  row_fingerprint(meta))
    check_split(meta, split)
    return split


def within_year_split(meta: pd.DataFrame, sites: Iterable[str], year_folder: str, test_fraction: float,
                      validation_fraction: float, seed: int) -> Split:
    """Random patient-grouped split restricted to one year folder.

    DRIAMS-A patient hashes are consistent within one metadata file (one year folder), so here no
    patient can end up in two parts. The year folder is used on purpose: this is not a temporal split.
    """
    sites, year_folder = list(sites), str(year_folder)
    rows = _rows_for_sites(meta, sites)
    rows = rows[meta["year_folder"].astype(str).to_numpy()[rows] == year_folder]
    if rows.size == 0:
        raise SplitError(f"No samples in year folder {year_folder} at {sorted(set(sites))}.")
    train, validation, test = _grouped_random_parts(meta, rows, test_fraction, validation_fraction, seed)
    without_id = int((meta["group_source"].to_numpy()[rows] != "patient_no").sum()) if "group_source" in meta else 0
    notes = [f"Only year folder {year_folder} ({rows.size} samples); patient IDs are consistent within it, so "
             "patient groups are complete. Differences to the random split estimate how much cross-year patient "
             "overlap affects the random split."]
    if without_id:
        notes.append(f"{without_id} sample(s) have no patient ID and are treated as their own group.")
    split = Split("within_year", train, validation, test,
                  f"Stratified, patient-grouped random split of {sorted(set(sites))}, year folder {year_folder} only: "
                  f"test {test.size / rows.size:.1%}, validation {validation.size / rows.size:.1%} of those samples "
                  f"(requested {test_fraction:.0%} / {validation_fraction:.0%}), seed {seed}.",
                  notes, row_fingerprint(meta))
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
                  f"< {test_start} <= test.", notes, row_fingerprint(meta))
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
                  f"(validation {validation.size / pool.size:.1%} of the training-site samples, "
                  f"requested {validation_fraction:.0%}, seed {seed}).",
                  ["Patients cannot be linked across hospitals, so cross-site patient overlap is not checked."],
                  row_fingerprint(meta))
    check_split(meta, split)
    return split


def derive_split(reference: Split, reference_meta: pd.DataFrame, meta: pd.DataFrame, reference_name: str) -> Split:
    """Reuse a reference dataset's split for another dataset built from the same data (e.g. I excluded).

    Every sample keeps the part it has in the reference split (matched by site, year folder and code).
    Samples that the reference split does not use, or that are not in the reference dataset, are left
    out, so no patient is placed differently from the reference.
    """
    check_split(reference_meta, reference)
    ref_part = np.full(len(reference_meta), "", dtype=object)
    for part, idx in reference.parts().items():
        ref_part[idx] = part
    lookup = dict(zip(sample_keys(reference_meta), ref_part, strict=True))
    mapped = np.array([lookup.get(key) for key in sample_keys(meta)], dtype=object)
    parts = {p: np.flatnonzero(mapped == p) for p in PARTS}

    notes = [f"Same partition as the '{reference.name}' split of dataset {reference_name}: every sample keeps "
             "its part."] + [f"[{reference_name}] {note}" for note in reference.notes]
    absent = int(np.equal(mapped, None).sum())
    unused = int((mapped == "").sum())
    if absent:
        notes.append(f"{absent} sample(s) are not in {reference_name} and are left out.")
    if unused:
        notes.append(f"{unused} sample(s) are not used by the reference split and are left out.")
    split = Split(reference.name, parts["train"], parts["validation"], parts["test"],
                  f"Reused from {reference_name}. {reference.description}", notes, row_fingerprint(meta),
                  {"dataset": reference_name, "dataset_fingerprint": reference.dataset_fingerprint})
    check_split(meta, split)
    return split


def make_splits(meta: pd.DataFrame, config: dict[str, Any], skipped: list[str] | None = None) -> dict[str, Split]:
    """Build every configured split that the dataset's sites allow; reasons for skipped ones go to `skipped`."""
    s = config["splits"]
    seed = int(config["project"]["random_seed"])
    available = set(meta["site"])
    skipped = skipped if skipped is not None else []

    def missing(sites: Iterable[str]) -> list[str]:
        return [site for site in sites if site not in available]

    splits: dict[str, Split] = {}
    if missing(s["random"]["sites"]):
        skipped.append(f"random: site(s) {missing(s['random']['sites'])} not in this dataset")
    else:
        splits["random"] = random_split(meta, s["random"]["sites"], s["random"]["test_fraction"],
                                        s["random"]["validation_fraction"], seed)
    w = s["within_year"]
    if missing(w["sites"]):
        skipped.append(f"within_year: site(s) {missing(w['sites'])} not in this dataset")
    elif str(w["year_folder"]) not in set(meta.loc[meta["site"].isin(w["sites"]), "year_folder"].astype(str)):
        skipped.append(f"within_year: year folder {w['year_folder']} not in this dataset")
    else:
        splits["within_year"] = within_year_split(meta, w["sites"], w["year_folder"], w["test_fraction"],
                                                  w["validation_fraction"], seed)
    if missing(s["temporal"]["sites"]):
        skipped.append(f"temporal: site(s) {missing(s['temporal']['sites'])} not in this dataset")
    else:
        splits["temporal"] = temporal_split(meta, s["temporal"]["sites"], s["temporal"]["validation_start"],
                                            s["temporal"]["test_start"], s["temporal"]["date_column"])
    test_sites = [site for site in s["external"]["test_sites"] if site in available]
    not_built = missing(s["external"]["test_sites"])
    if missing(s["external"]["train_sites"]) or not test_sites:
        skipped.append("external: training site(s) or all test sites are not in this dataset")
    else:
        external = external_split(meta, s["external"]["train_sites"], test_sites,
                                  s["external"]["validation_fraction"], seed)
        if not_built:
            external.notes.append(f"Configured test site(s) {not_built} are not in this dataset yet.")
        splits["external"] = external
    return splits


def build_splits(meta: pd.DataFrame, config: dict[str, Any], dataset_name: str, output_root: Path,
                 skipped: list[str] | None = None) -> dict[str, Split]:
    """Splits for a built dataset.

    The primary dataset (config dataset.name) gets the configured splits. Every other dataset reuses the
    primary dataset's saved splits, so comparisons (e.g. I counted as resistant vs. excluded) use the
    same partition of the samples.
    """
    skipped = skipped if skipped is not None else []
    primary = config["dataset"]["name"]
    if dataset_name == primary:
        return make_splits(meta, config, skipped)
    ref_dir = Path(output_root) / primary
    if not (ref_dir / "splits").is_dir():
        raise SplitError(f"{dataset_name} reuses the splits of the primary dataset {primary}, which has none yet. "
                         "Build it first: python scripts/build_dataset.py")
    _, ref_meta, _ = load_dataset(ref_dir)
    splits = {}
    for name, reference in load_splits(ref_dir / "splits", ref_meta).items():
        split = derive_split(reference, ref_meta, meta, primary)
        if split.train.size == 0 or split.test.size == 0:
            skipped.append(f"{name}: this dataset has no training or no test samples in the {primary} split")
            continue
        splits[name] = split
    return splits
