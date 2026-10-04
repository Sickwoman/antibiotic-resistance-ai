"""Version 2.0, steps 2 and 3, before any label: the MARISMa E. coli cohort, its frozen features, the spectra-only
checks, and the restricted schema check of AMR.csv.

docs/v2.0_marisma_plan.md, approved on 2026-10-03 for steps 1-3 only (with amendments A2 and A6).

The first run (no option) does, in order:
1. Record the protected state: both frozen bundles, the zone file, both logs, and the two DRIAMS datasets the models
   were trained on. The same state is checked again at the end.
2. Build the cohort from the archive's member names only (nothing is extracted): MARISMa's species field (genus
   folder "Escherichia", species folder "Coli"), years 2018-2024, and the exclusion of identifiers also filed under
   another species. The sample-source rule is reported as not applied: its field is not in the archive.
3. Select one spectrum per isolate (amendment A6.2) with the Bruker reader, reading each file straight from the
   archive. Every exclusion is counted by reason, and every failed attempt by reason and batch.
4. Compute the frozen features with the bundles' own preprocessing settings (fingerprint 347cbd6d5d956ff9).
5. Spectra-only checks: each spectrum's correlation with the mean DRIAMS-A E. coli training spectrum of each model.
6. List the antibiotic names in AMR.csv's header, through the restricted schema reader.

The second run (`--schema --columns '{...}'`) is step 2's schema check, with an explicit column per antibiotic, on the
isolates with a selected spectrum: matched and unmatched isolates, and per antibiotic the isolates with a non-missing
interpretation (src/marisma_schema.py). Nothing else is read from AMR.csv.

It never calls a model's predict or predict_proba (tests/test_v20_preparation.py checks this file for those names), and
fits nothing. Outputs:
- `--work`, outside the repository: the cohort's replicate folders, the per-isolate selection, every reader attempt,
  the features, and their hashes. They hold isolate identifiers and archive paths, so they are never committed.
- `results/metrics/v2.0/step3_preparation.json`: aggregate counts, rates and distributions only.

    python scripts/v20_prepare_marisma.py
    python scripts/v20_prepare_marisma.py --schema --columns '{"ciprofloxacin": "...", "ceftriaxone": "..."}'
    python scripts/v20_prepare_marisma.py --names      # refresh only the antibiotic names from the header
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bruker import BrukerReadError, BrukerSpectrum, _double, _text, convert  # noqa: E402
from src.dataset import file_sha256, load_dataset  # noqa: E402
from src.marisma_cohort import (  # noqa: E402
    Selection,
    exclusion_summary,
    layout_folders,
    parse_spectrum_member,
    pause_check,
    select_first_passing,
)
from src.marisma_schema import antibiotic_names, read_columns, schema_summary  # noqa: E402
from src.predict import load_bundle  # noqa: E402
from src.preprocessing import PreprocessingConfig, PreprocessingError, preprocess_arrays  # noqa: E402
from src.splits import load_splits  # noqa: E402
from src.utils import get_logger, git_commit, keep_awake, load_config  # noqa: E402
from src.zip_index import Member, ZipIndexError, iter_members, read_member  # noqa: E402

FEATURE_FINGERPRINT = "347cbd6d5d956ff9"
N_FEATURES = 6000
YEARS = tuple(str(y) for y in range(2018, 2025))
SPECIES = ("Escherichia", "Coli")   # MARISMa's genus and species folder names
AVAILABILITY_GATE = 100      # step 2: fewer isolates with a result -> the antibiotic is dropped (amendment A6.1)
MODELS = {   # label -> bundle, its training dataset, and the plan's SHA-256 prefix of the bundle
    "ciprofloxacin": ("models/v0.4/ecoli_ciprofloxacin/best_random.joblib", "ecoli_ciprofloxacin", "d59d6d7deafa1af4"),
    "ceftriaxone": ("models/v1.1/ecoli_ceftriaxone/best_random.joblib", "ecoli_ceftriaxone", "74c626903dc961fd"),
}
ZONE = ("results/metrics/v0.6/ecoli_ciprofloxacin/uncertainty.json", "8ee7b2bc20ba7ceb")
LOGS = {   # file -> (SHA-256, data rows), docs/reproduction_guide.md
    "results/experiments/test_evaluations.csv":
        ("6528eb2abcdfb611d65712354b8a05daed03febdb137a0910ed0eb7764f3ddba", 113),
    "results/experiments/development_runs.csv":
        ("dd5f4a74933a4cfc8382e9a4464ae774aab6944e280b68e8fe340fbbc2d9e894", 34),
}
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

log = get_logger("v20_prepare")


class StepError(RuntimeError):
    """A check failed: processing stops before anything further is computed."""


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- 1. Protected state ---

def protected_state() -> dict[str, Any]:
    """SHA-256 of the frozen artifacts and logs, and the DRIAMS datasets' fingerprints, checked against the records."""
    state: dict[str, Any] = {"bundles": {}, "zone": None, "logs": {}, "datasets": {}}
    for label, (bundle, dataset, prefix) in MODELS.items():
        digest = file_sha256(ROOT / bundle)
        if not digest.startswith(prefix):
            raise StepError(f"the {label} bundle does not have the SHA-256 recorded in the plan")
        card = json.loads((ROOT / bundle).with_suffix(".json").read_text(encoding="utf-8"))
        _, _, summary = load_dataset(ROOT / "data/processed" / dataset, verify_x=True)   # re-hashes X.npy
        if (summary["row_fingerprint"], summary["x_sha256"]) != (card["dataset"]["row_fingerprint"],
                                                                  card["dataset"]["x_sha256"]):
            raise StepError(f"the {dataset} dataset differs from the one the {label} model was trained on")
        state["bundles"][label] = digest
        state["datasets"][dataset] = {"row_fingerprint": summary["row_fingerprint"], "x_sha256": summary["x_sha256"],
                                      "feature_fingerprint": summary["feature_fingerprint"]}
    zone = file_sha256(ROOT / ZONE[0])
    if not zone.startswith(ZONE[1]):
        raise StepError("the zone file does not have the SHA-256 recorded in the plan")
    state["zone"] = zone
    for name, (expected, rows) in LOGS.items():
        digest = file_sha256(ROOT / name)
        data_rows = len((ROOT / name).read_text(encoding="utf-8").splitlines()) - 1
        if (digest, data_rows) != (expected, rows):
            raise StepError(f"{name} is not the recorded file ({rows} data rows)")
        state["logs"][name] = {"sha256": digest, "data_rows": data_rows}
    return state


def frozen_preprocessing() -> PreprocessingConfig:
    """The preprocessing settings stored in both bundles; they must be identical and match the fingerprint."""
    configs = {}
    for label, (bundle_path, _, _) in MODELS.items():
        bundle = load_bundle(ROOT / bundle_path)          # verifies the bundle against its .sha256 file
        pcfg = PreprocessingConfig.from_dict(bundle["preprocessing"])
        if (pcfg.fingerprint(), bundle["feature_fingerprint"], int(bundle["n_features"])) != (
                FEATURE_FINGERPRINT, FEATURE_FINGERPRINT, N_FEATURES):
            raise StepError(f"the {label} bundle's preprocessing does not match fingerprint {FEATURE_FINGERPRINT}")
        configs[label] = pcfg
        del bundle                                        # only the settings are kept; the model is never used
    first, second = configs.values()
    if first != second or first.n_bins != N_FEATURES:
        raise StepError("the two bundles' preprocessing settings differ")
    return first


# --- 2-3. Index and cohort rules ---

def build_cohort(names: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any], pd.DataFrame]:
    """The cohort from member names only: the species rule, the year rule, and one consistency rule.

    The consistency rule: an identifier whose isolate folders sit under more than one genus or species is excluded and
    counted, because MARISMa's species field does not then name one species for that isolate. Returns, per isolate,
    its year folder and replicate folders; the counts at each rule; and the cohort's replicate folders as a table
    (it holds identifiers, so it is written outside the repository only).
    """
    isolate_folders, replicate_folders = layout_folders(names)
    spectra = sum(1 for n in names if n.endswith("/fid"))
    taxa: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for _, genus, species, isolate in isolate_folders:
        taxa[isolate].add((genus, species))
    ecoli = {f for f in isolate_folders if (f[1], f[2]) == SPECIES}
    ecoli_ids = {f[3] for f in ecoli}
    inconsistent = {i for i in ecoli_ids if len(taxa[i]) > 1}
    consistent = {f for f in ecoli if f[3] not in inconsistent}
    in_years = {f for f in consistent if f[0] in YEARS}
    years_of: dict[str, set[str]] = defaultdict(set)
    for year, _, _, isolate in in_years:
        years_of[isolate].add(year)
    if any(len(y) > 1 for y in years_of.values()):
        raise StepError("an E. coli isolate has folders in more than one year folder; the plan has no rule for that")
    cohort = {isolate: {"year": next(iter(y)), "replicates": defaultdict(dict)} for isolate, y in years_of.items()}
    rows = []
    for r in sorted(replicate_folders):
        if (r.year, r.genus, r.species, r.isolate) in in_years:
            cohort[r.isolate]["replicates"][r.biological][r.technical] = r.folder
            rows.append(r._asdict())
    counts = {
        "archive": {"isolate_folders": len(isolate_folders), "identifiers": len(taxa), "files_named_fid": spectra,
                    "fid_files_outside_the_layout": sum(1 for n in names
                                                        if n.endswith("/fid") and parse_spectrum_member(n) is None)},
        "species_rule": {"rule": "genus folder 'Escherichia' and species folder 'Coli' (MARISMa's species field)",
                         "isolate_folders": len(ecoli), "identifiers": len(ecoli_ids)},
        "species_consistency_rule": {"excluded_identifiers": len(inconsistent),
                                     "rule": "identifier also filed under another genus or species: excluded"},
        "year_rule": {"years": list(YEARS), "excluded_identifiers": len({f[3] for f in consistent}) - len(cohort),
                      "identifiers": len(cohort)},
        "replicate_folders": {"technical_replicate_folders": len(rows),
                              "isolates_without_any": sum(1 for c in cohort.values() if not c["replicates"])},
        "isolates_per_year": dict(sorted(pd.Series([c["year"] for c in cohort.values()]).value_counts().items())),
    }
    return cohort, counts, pd.DataFrame(rows)


# --- 4-5. Selection and features ---

class ArchiveReader:
    """Reads one spectrum folder from the archive, and keeps, for every attempt, its batch and outcome.

    `members` maps member names to their place in the archive (src/zip_index.py); each read is checked against the
    member's CRC-32. As in readBrukerFlexData, `acqu` is used if present, otherwise `acqus`.
    """

    def __init__(self, fh: BinaryIO, members: dict[str, Member]):
        self.fh, self.members = fh, members
        self.attempts: list[dict[str, Any]] = []
        self.year = ""

    def __call__(self, folder: str) -> BrukerSpectrum:
        fid, acqu = f"{folder}/fid", f"{folder}/acqu" if f"{folder}/acqu" in self.members else f"{folder}/acqus"
        attempt: dict[str, Any] = {"year": self.year, "acquisition_month": "", "instrument": "", "reason": "",
                                   "acqu_file": acqu.rsplit("/", 1)[-1]}
        self.attempts.append(attempt)
        if fid not in self.members or acqu not in self.members:
            attempt["reason"] = "missing_files"
            raise BrukerReadError("missing_files")
        try:
            lines = read_member(self.fh, self.members[acqu]).decode("latin-1").splitlines()
            fid_bytes = read_member(self.fh, self.members[fid])
        except (OSError, ZipIndexError, zlib.error):
            attempt["reason"] = "unreadable"
            raise BrukerReadError("unreadable") from None
        attempt["acquisition_month"] = _text(lines, "AQ_DATE")[:7]   # "YYYY-MM": a batch, not an identifier
        attempt["instrument"] = _text(lines, "INSTRUM")
        for key in ("ML1", "ML2", "ML3", "TD", "DELAY", "DW"):
            attempt[key] = _double(lines, key)
        attempt["hpc_requested"] = _text(lines, "HPClUse") == "yes"
        try:
            spectrum = convert(lines, fid_bytes)
        except BrukerReadError as exc:
            attempt["reason"] = exc.reason
            raise
        attempt["reason"] = "passed"
        attempt["calibration"] = spectrum.calibration
        attempt["mz_first"], attempt["mz_last"], attempt["points"] = (float(spectrum.mz[0]), float(spectrum.mz[-1]),
                                                                      int(spectrum.mz.size))
        return spectrum


def select_and_featurise(cohort: dict[str, dict[str, Any]], fh: BinaryIO, members: dict[str, Member],
                         pcfg: PreprocessingConfig, work: Path
                         ) -> tuple[pd.DataFrame, list[Selection], list[dict[str, Any]], np.ndarray]:
    """Amendment A6.2 for every isolate, in isolate order; features of each selected spectrum go to X_all.npy."""
    isolates = sorted(cohort)
    X = np.lib.format.open_memmap(work / "X_all.npy", mode="w+", dtype=np.float32,
                                  shape=(len(isolates), N_FEATURES))
    records, outcomes = [], []
    started = time.perf_counter()
    reader = ArchiveReader(fh, members)
    for i, isolate in enumerate(isolates):
        year, replicates = cohort[isolate]["year"], cohort[isolate]["replicates"]
        reader.year = year
        selection = select_first_passing(replicates, reader)
        outcomes.append(Selection(None, selection.biological, selection.technical, selection.status,
                                  selection.failures))          # the spectrum itself is not kept
        record = {"isolate": isolate, "year": year, "status": selection.status,
                  "biological": selection.biological or "", "technical": selection.technical or "",
                  "failures": "|".join(selection.failures), "spectrum_folder": "", "calibration": "",
                  "acquisition_month": "", "instrument": "", "raw_points": 0, "nonzero_bins": 0}
        if selection.spectrum is not None:
            spectrum = selection.spectrum
            record.update(spectrum_folder=replicates[selection.biological][selection.technical],
                          calibration=spectrum.calibration, acquisition_month=spectrum.acquisition_date[:7],
                          instrument=spectrum.instrument)
            try:
                features, info = preprocess_arrays(spectrum.mz, spectrum.intensity, pcfg)
            except PreprocessingError:
                record["status"] = "feature_failure"     # counted and reviewed; no other replicate is tried
                features = None
            if features is not None:
                X[i] = features
                record["raw_points"], record["nonzero_bins"] = info.raw_points, info.nonzero_bins
        records.append(record)
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.perf_counter() - started)
            log.info("selected %d / %d isolates (%.0f per second)", i + 1, len(isolates), rate)
    X.flush()
    return pd.DataFrame(records), outcomes, reader.attempts, X


def batch_review(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Amendment A6.3: failed attempts by reason within each batch (year folder, acquisition month, instrument)."""
    frame = pd.DataFrame(attempts)
    out: dict[str, Any] = {"attempts": int(len(frame)),
                           "attempts_by_outcome": {k: int(v) for k, v in frame["reason"].value_counts().items()}}
    for batch in ("year", "acquisition_month", "instrument"):
        table = frame.groupby([batch, "reason"]).size().unstack(fill_value=0)
        out[f"by_{batch}"] = {str(b): {str(r): int(n) for r, n in row.items() if n} for b, row in table.iterrows()}
    passed = frame[frame["reason"] == "passed"]
    calibration = {}
    for key in ("ML1", "ML2", "ML3", "TD", "DELAY", "DW", "mz_first", "mz_last", "points"):
        values = pd.to_numeric(passed[key], errors="coerce").dropna().to_numpy()
        calibration[key] = quantiles(values)
    out["passed_spectra_parameters"] = calibration
    out["hpc_requested_among_attempts"] = int(frame["hpc_requested"].fillna(False).astype(bool).sum())
    out["calibration_among_passed"] = {k: int(v) for k, v in passed["calibration"].value_counts().items()}
    out["acqu_file_used"] = {k: int(v) for k, v in frame["acqu_file"].value_counts().items()}
    return out


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    out = {"n": int(values.size)}
    out.update({f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES})
    return out


# --- 6. Spectra-only checks ---

def mean_training_spectra() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per model: the mean DRIAMS-A E. coli training spectrum (random split, train part) and the validation part's
    feature matrix, the within-site reference for the correlation distribution."""
    out = {}
    for label, (_, dataset, _) in MODELS.items():
        X, meta, _ = load_dataset(ROOT / "data/processed" / dataset, verify_x=False)
        split = load_splits(ROOT / "data/processed" / dataset / "splits", meta)["random"]
        if set(meta.iloc[split.train]["site"]) != {"DRIAMS-A"}:
            raise StepError(f"the {dataset} random split's training part is not DRIAMS-A only")
        out[label] = (np.asarray(X[split.train], dtype=np.float64).mean(axis=0),
                      np.asarray(X[split.validation], dtype=np.float64))
    return out


def correlations(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    rc = reference - reference.mean()
    return (Xc @ rc) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(rc))


def peak_offsets(a: np.ndarray, b: np.ndarray, pcfg: PreprocessingConfig, top: int = 30) -> dict[str, Any]:
    """Calibration consistency across sites: the lag (in bins) that maximises the cross-correlation of two mean
    spectra, and, for the `top` strongest local maxima of `a`, the offset to the nearest maximum of `b` (parabolic
    interpolation, in Da). Descriptive: it is not proof of a correct calibration."""
    lags = range(-20, 21)
    ac, bc = a - a.mean(), b - b.mean()
    score = {lag: float(np.dot(np.roll(ac, lag), bc)) for lag in lags}
    best = max(score, key=score.get)

    def maxima(y):
        idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
        refined = []
        for i in idx:
            denom = y[i - 1] - 2 * y[i] + y[i + 1]
            refined.append(i + (0.5 * (y[i - 1] - y[i + 1]) / denom if denom != 0 else 0.0))
        return idx, np.asarray(refined)

    ia, ra = maxima(a)
    ib, rb = maxima(b)
    strongest = np.argsort(a[ia])[::-1][:top]
    offsets = []
    for k in strongest:
        j = np.argmin(np.abs(rb - ra[k]))
        offsets.append((rb[j] - ra[k]) * pcfg.bin_width)
    centres = pcfg.mz_min + (ra[strongest] + 0.5) * pcfg.bin_width
    return {"best_lag_bins": int(best), "lag_range_bins": [lags[0], lags[-1]],
            "top_peaks": int(len(offsets)), "peak_positions_da": [round(float(c), 1) for c in centres],
            "offset_to_nearest_peak_da": [round(float(o), 2) for o in offsets],
            "median_abs_offset_da": float(np.median(np.abs(offsets))) if offsets else None}


# --- 7. Schema check ---

def schema_check(amr: Path, isolates: list[str], columns: dict[str, str]) -> dict[str, Any]:
    summary = schema_summary(amr, isolates, columns)
    gate = {label: {"non_missing": n, "dropped": n < AVAILABILITY_GATE} for label, n in summary.non_missing.items()}
    return {"antibiotics": list(summary.antibiotics), "matched_isolates": summary.matched_isolates,
            "unmatched_isolates": summary.unmatched_isolates, "columns_used": columns,
            "availability_gate": AVAILABILITY_GATE, "per_antibiotic": gate,
            "both_dropped": all(g["dropped"] for g in gate.values())}


# --- main ---

def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Steps 2-3 without the schema check: cohort, selection, features, spectra-only checks."""
    result: dict[str, Any] = {"what": "Version 2.0 steps 2-3 before any label (docs/v2.0_marisma_plan.md)",
                              "started_utc": now(), "git_commit": git_commit()}
    before = protected_state()
    pcfg = frozen_preprocessing()
    result["feature_fingerprint"] = pcfg.fingerprint()

    log.info("indexing the archive's central directory")
    prefixes = tuple(f"MARISMa/{y}/{SPECIES[0]}/{SPECIES[1]}/" for y in YEARS)
    names, members = [], {}
    for name, member in iter_members(args.zip):
        names.append(name)
        if name.startswith(prefixes):
            members[name] = member            # only the E. coli members' places are kept
    result["archive"] = {"members": len(names)}
    cohort, counts, folders = build_cohort(names)
    del names
    folders.to_csv(args.work / "replicate_folders.csv", index=False, lineterminator="\n")
    result["cohort_rules"] = counts
    result["source_rule"] = {
        "applied": False,
        "status": "blocked: the source field is not in the archive, and amendment A2 lets no code read a "
                  "non-antibiotic field of AMR.csv, so the dated addendum cannot be written"}

    log.info("selecting one spectrum per isolate for %d isolates", len(cohort))
    with open(args.zip, "rb") as fh:
        selection, outcomes, attempts, X = select_and_featurise(cohort, fh, members, pcfg, args.work)
    selection.to_csv(args.work / "selection.csv", index=False, lineterminator="\n")
    pd.DataFrame(attempts).to_csv(args.work / "attempts.csv", index=False, lineterminator="\n")
    result["exclusions"] = exclusion_summary(outcomes)
    result["exclusions"]["feature_failures"] = int((selection["status"] == "feature_failure").sum())
    result["pause_check"] = {**pause_check(outcomes),
                             "note": "before the sample-source rule; recomputed once that rule is applied"}
    result["batch_review"] = batch_review(attempts)

    kept = selection.index[selection["status"] == "selected"].to_numpy()
    features = np.asarray(X[kept])
    del X
    np.save(args.work / "X.npy", features)
    (args.work / "X_all.npy").unlink()
    if features.shape[1] != N_FEATURES or not np.isfinite(features).all() or (features < 0).any():
        raise StepError("the features are not 6,000 finite, non-negative values per spectrum")
    result["features"] = {"spectra": int(features.shape[0]), "n_features": int(features.shape[1]),
                          "dtype": str(features.dtype), "fingerprint": pcfg.fingerprint(),
                          "nonzero_bins": quantiles(selection.loc[kept, "nonzero_bins"].to_numpy()),
                          "raw_points": quantiles(selection.loc[kept, "raw_points"].to_numpy())}

    diagnostics = {}
    years = selection.loc[kept, "year"].to_numpy()
    for label, (mean, validation) in mean_training_spectra().items():
        r = correlations(features.astype(np.float64), mean)
        diagnostics[label] = {"marisma": quantiles(r),
                              "driams_a_validation_reference": quantiles(correlations(validation, mean)),
                              "marisma_by_year": {y: quantiles(r[years == y]) for y in YEARS},
                              "mean_spectra_alignment": peak_offsets(mean, features.mean(axis=0), pcfg)}
    result["spectra_only_checks"] = diagnostics
    result["schema"] = {"antibiotics": list(antibiotic_names(read_columns(args.amr)))}

    if protected_state() != before:
        raise StepError("a protected artifact changed during the run")
    result["protected_state"] = {"unchanged": True, **before}
    result["finished_utc"] = now()
    hashes = {"replicate_folders_csv": file_sha256(args.work / "replicate_folders.csv"),
              "selection_csv": file_sha256(args.work / "selection.csv"),
              "attempts_csv": file_sha256(args.work / "attempts.csv"),
              "cohort_isolates_sha256": hashlib.sha256(
                  "\n".join(selection.loc[kept, "isolate"]).encode("utf-8")).hexdigest(),
              "x_npy": file_sha256(args.work / "X.npy")}
    (args.work / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    (args.work / "inputs.json").write_text(json.dumps({"zip": str(args.zip)}, indent=2) + "\n", encoding="utf-8")
    return result


def list_names(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    """The antibiotic names in AMR.csv's header, through the restricted reader; nothing else is read."""
    result["schema"] = {"antibiotics": list(antibiotic_names(read_columns(args.amr))), "listed_utc": now(),
                        "source": "AMR.csv header only, through src/marisma_schema.py"}
    return result


def run_schema(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    """Step 2's schema check, on the isolates with a selected spectrum, with an explicit column mapping.

    Refused while the aggregate pause (amendment A6.4) holds: processing pauses before any label is read, and only a
    dated owner decision can lift it.
    """
    if result["pause_check"]["pause"]:
        raise StepError("the aggregate pause (amendment A6.4) holds: the schema check waits for a dated owner decision")
    before = protected_state()
    selection = pd.read_csv(args.work / "selection.csv", dtype=str, keep_default_na=False)
    ids = selection.loc[selection["status"] == "selected", "isolate"].tolist()
    result["schema"] = {**schema_check(args.amr, ids, args.columns), "run_utc": now(),
                        "cohort": "isolates with a selected spectrum, before the sample-source rule"}
    if protected_state() != before:
        raise StepError("a protected artifact changed during the run")
    return result


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    data_root = Path(config["paths"]["driams_root"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=data_root / "MARISMa_v2.0.0" / "MARISMa.zip")
    parser.add_argument("--amr", type=Path, default=data_root / "MARISMa_v2.0.0_sealed" / "AMR.csv")
    parser.add_argument("--work", type=Path, default=data_root / "MARISMa_v2.0.0_work")
    parser.add_argument("--schema", action="store_true",
                        help="run only the schema check, on the cohort of an earlier run, with --columns")
    parser.add_argument("--names", action="store_true",
                        help="only list the antibiotic names in AMR.csv's header into the summary of an earlier run")
    parser.add_argument("--columns", type=json.loads, default=None,
                        help='explicit interpretation columns, e.g. {"ciprofloxacin": "...", "ceftriaxone": "..."}')
    parser.add_argument("--out", type=Path, default=ROOT / "results/metrics/v2.0/step3_preparation.json")
    args = parser.parse_args(argv)
    if args.work.resolve().is_relative_to(ROOT.resolve()):
        raise StepError("the work folder must be outside the repository")
    args.work.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((args.zip.parent / "provenance.json").read_text(encoding="utf-8"))
    for name, path in (("MARISMa.zip", args.zip), ("AMR.csv", args.amr)):
        entry = provenance["files"][name]
        if path.stat().st_size != entry["size_bytes"] or not entry.get("md5_published_and_verified"):
            raise StepError(f"{name} is not the verified download")

    with keep_awake():
        if args.schema:
            if not args.columns:
                raise StepError("--schema needs --columns")
            result = run_schema(args, json.loads(args.out.read_text(encoding="utf-8")))
        elif args.names:
            result = list_names(args, json.loads(args.out.read_text(encoding="utf-8")))
        else:
            result = prepare(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.info("written %s", args.out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
