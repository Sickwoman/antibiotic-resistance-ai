"""Version 0.2 dataset builder: raw DRIAMS spectra -> feature matrix, labels and an audit trail.

Output folder (git-ignored), e.g. data/processed/ecoli_ciprofloxacin/:
    X.npy            float32 matrix (samples x bins); open with mmap_mode="r"
    metadata.csv     one row per sample (row i describes X[i]); no patient/case/order IDs
    exclusions.csv   every target-species row that was not used, with its reason
    summary.json     counts, parameters, feature definition and known limitations

Every target-species metadata row ends up either in metadata.csv or in exclusions.csv.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import (
    CODE_COL,
    LABEL_AMBIGUOUS,
    LABEL_I_EXCLUDED,
    LABEL_MISSING,
    LABEL_UNSUPPORTED,
    SPECIES_COL,
    YEAR_COL,
    DataError,
    SpectrumFormatError,
    discover_sites,
    label_status,
    load_site_tables,
    read_binned_spectrum,
)
from src.preprocessing import PreprocessingConfig, PreprocessingError, preprocess_file
from src.utils import PROJECT_ROOT, driams_root, get_logger, project_path

# Exclusion reasons in the order they are checked; a row gets the first reason that applies.
EXCLUSION_REASONS: dict[str, str] = {
    "missing_code": "metadata row has no spectrum code",
    "antibiotic_not_reported": "the site's metadata has no column for the antibiotic",
    "no_ast_result": "no result for the antibiotic",
    "ambiguous_ast_result": "result that is not a single category, e.g. 'R(1), S(1)'",
    "unsupported_ast_result": "value other than R, I, S or a known ambiguous form",
    "intermediate_excluded": "I result while labels.intermediate_as is 'exclude'",
    "excluded_workstation": "sample type listed in dataset.exclude_workstations (e.g. HospitalHygiene)",
    "missing_acquisition_date": "no parseable acquisition_date while dataset.require_acquisition_date is true",
    "conflicting_duplicate_record": "spectrum code listed more than once at the site with different labels "
                                    "(all copies excluded)",
    "duplicate_record": "spectrum code listed more than once at the site with the same label "
                        "(first usable row kept)",
    "no_spectrum_file": "raw spectrum file not found on disk",
    "malformed_spectrum": "raw spectrum file unreadable or failed validation",
    "unusable_spectrum": "spectrum readable but preprocessing left no usable signal",
    "differs_from_driams_binned": "processed raw file does not match the published DRIAMS binned_6000 file for "
                                  "the same code (the two files hold different measurements)",
    "duplicate_spectrum": "processed features identical to an earlier sample (first kept)",
}
_STATUS_TO_REASON = {
    LABEL_MISSING: "no_ast_result",
    LABEL_AMBIGUOUS: "ambiguous_ast_result",
    LABEL_UNSUPPORTED: "unsupported_ast_result",
    LABEL_I_EXCLUDED: "intermediate_excluded",
}
# Identifiers that must never leave this module.
PRIVATE_COLUMNS = ("patient_no", "case_no", "order_no")
METADATA_COLUMNS = ["sample_index", "site", "year_folder", "code", "spectrum_relpath", "acquisition_date",
                    "workstation", "ast_value", "label", "group_id", "group_source", "raw_points", "nonzero_bins"]
EXCLUSION_COLUMNS = ["site", "year_folder", "code", "reason", "detail", "workstation", "ast_value"]
# A sample is identified by these columns; the fingerprint also covers its label and patient group.
SAMPLE_KEY_COLUMNS = ("site", "year_folder", "code")
FINGERPRINT_COLUMNS = (*SAMPLE_KEY_COLUMNS, "label", "group_id")

log = get_logger("dataset")


class DatasetError(DataError):
    pass


@dataclass(frozen=True)
class CohortSpec:
    name: str
    species: str
    antibiotic: str
    sites: tuple[str, ...]
    intermediate_as: str
    exclude_workstations: tuple[str, ...]
    require_acquisition_date: bool
    flag_duplicate_spectra: bool

    @classmethod
    def from_config(cls, config: dict[str, Any], *, intermediate_as: str | None = None,
                    sites: Iterable[str] | None = None, name: str | None = None) -> CohortSpec:
        d = config["dataset"]
        default_policy = config["labels"]["intermediate_as"]
        policy = intermediate_as or default_policy
        site_list = tuple(sites or d["sites"])
        if name is None:  # any non-default choice gets its own folder, so the primary dataset is never overwritten
            name = d["name"]
            if policy != default_policy:
                name += f"__intermediate-{policy}"
            if site_list != tuple(d["sites"]):
                name += "__sites-" + "-".join(s.replace("DRIAMS-", "") for s in site_list)
        return cls(
            name=name,
            species=config["target"]["species"],
            antibiotic=config["target"]["preferred_antibiotic"],
            sites=site_list,
            intermediate_as=policy,
            exclude_workstations=tuple(d.get("exclude_workstations") or ()),
            require_acquisition_date=bool(d.get("require_acquisition_date", False)),
            flag_duplicate_spectra=bool(d.get("flag_duplicate_spectra", True)),
        )


def spectrum_relpath(site: str, folder: str, year: str, code: str) -> str:
    """Portable (forward-slash) path of a spectrum relative to the DRIAMS root."""
    return str(PurePosixPath(site, folder, str(year), f"{code}.txt"))


def resolve_relpath(root: str | Path, relpath: str) -> Path:
    return Path(root).joinpath(*PurePosixPath(relpath).parts)


def _bool(values: Any) -> np.ndarray:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return series.fillna(False).to_numpy(dtype=bool)


def _assign_reason(reason: np.ndarray, mask: np.ndarray, name: str) -> None:
    """Give `name` to masked rows that have no reason yet (the first applicable reason wins)."""
    reason[mask & np.equal(reason, None)] = name


def select_cohort(tables: dict[str, pd.DataFrame], spec: CohortSpec, ambiguous_values: Iterable[str],
                  spectrum_folder: str, spectrum_exists: Callable[[str], bool],
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Metadata-level selection. Returns (candidates, exclusions, info).

    Only rows of the exact target species are considered; other species are not "exclusions".
    Internal column `group_key` holds the raw patient key and is dropped before anything is saved.
    """
    ambiguous_values = list(ambiguous_values)
    frames = []
    info: dict[str, Any] = {"target_species_rows": {}, "other_spellings_not_included": {},
                            "workstation_column_present": {}, "patient_id_column_present": {}}
    for site in spec.sites:
        table = tables[site]
        species = table[SPECIES_COL].astype("string")
        spellings = species[_bool(species.str.contains(spec.species, regex=False)) & _bool(species != spec.species)]
        info["other_spellings_not_included"][site] = {str(k): int(v) for k, v in spellings.value_counts().items()}
        t = table.loc[_bool(species == spec.species)]
        info["target_species_rows"][site] = len(t)
        n = len(t)
        reason = np.full(n, None, dtype=object)
        assign = partial(_assign_reason, reason)

        code = t[CODE_COL].astype("string")
        assign(_bool(code.isna()), "missing_code")

        if spec.antibiotic in t.columns:
            labels = label_status(t[spec.antibiotic], spec.intermediate_as, ambiguous_values)
            for status, name in _STATUS_TO_REASON.items():
                assign(labels["status"].to_numpy() == status, name)
            ast_value, label = labels["value"].to_numpy(dtype=object), labels["label"].to_numpy()
        else:
            assign(np.ones(n, dtype=bool), "antibiotic_not_reported")
            ast_value, label = np.full(n, None, dtype=object), np.full(n, np.nan)

        info["workstation_column_present"][site] = "workstation" in t.columns
        workstation = (t["workstation"].astype("string") if "workstation" in t.columns
                       else pd.Series(pd.NA, index=t.index, dtype="string"))
        if spec.exclude_workstations:
            assign(_bool(workstation.isin(list(spec.exclude_workstations))), "excluded_workstation")

        if "acquisition_date" in t.columns:
            dates = pd.to_datetime(t["acquisition_date"], errors="coerce")
        else:
            dates = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
        if spec.require_acquisition_date:
            assign(dates.isna().to_numpy(), "missing_acquisition_date")

        # Repeated codes, judged only among rows that are otherwise usable: conflicting labels exclude
        # every copy; identical labels keep the first copy.
        usable = pd.DataFrame({"code": code.to_numpy(dtype=object), "label": label})[np.equal(reason, None)]
        repeated = usable[usable["code"].duplicated(keep=False)]
        if len(repeated):
            n_labels = repeated.groupby("code")["label"].transform("nunique").to_numpy()
            conflict = np.zeros(n, dtype=bool)
            conflict[repeated.index[n_labels > 1]] = True
            assign(conflict, "conflicting_duplicate_record")
            later = np.zeros(n, dtype=bool)
            same = repeated[n_labels == 1]
            later[same.index[same["code"].duplicated(keep="first")]] = True
            assign(later, "duplicate_record")

        years = t[YEAR_COL].astype(str).to_numpy()
        codes = code.fillna("").to_numpy(dtype=object)
        relpaths = np.array([spectrum_relpath(site, spectrum_folder, y, c)
                             for y, c in zip(years, codes, strict=True)], dtype=object)
        pending = np.flatnonzero(np.equal(reason, None))
        exists = np.zeros(n, dtype=bool)
        exists[pending] = [spectrum_exists(relpaths[i]) for i in pending]
        assign(~exists, "no_spectrum_file")

        info["patient_id_column_present"][site] = "patient_no" in t.columns
        if "patient_no" in t.columns:
            # patient_no is re-hashed per year folder, so the key is only unique within (site, year).
            pid = t["patient_no"].astype("string")
            group_key = (site + "|" + t[YEAR_COL].astype(str) + "|" + pid).astype("string")
        else:
            group_key = pd.Series(pd.NA, index=t.index, dtype="string")

        frames.append(pd.DataFrame({
            "site": site, "year_folder": years, "code": codes, "spectrum_relpath": relpaths,
            "acquisition_date": dates.to_numpy(), "workstation": workstation.to_numpy(dtype=object),
            "ast_value": ast_value, "label": label, "group_key": group_key.to_numpy(dtype=object),
            "reason": reason,
        }))

    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
        "site", "year_folder", "code", "spectrum_relpath", "acquisition_date", "workstation",
        "ast_value", "label", "group_key", "reason"])
    keep = rows["reason"].isna().to_numpy()
    candidates = rows.loc[keep].drop(columns="reason").reset_index(drop=True)
    exclusions = rows.loc[~keep].drop(columns="group_key").assign(detail="").reset_index(drop=True)
    return candidates, exclusions, info


def assign_group_ids(group_keys: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Integer group IDs. Rows without a patient key each get their own group.

    The integers are assigned in order of first appearance and cannot be turned back into
    DRIAMS patient hashes without the original tables.
    """
    keys = pd.Series(group_keys, dtype="string").reset_index(drop=True)
    codes, _ = pd.factorize(keys, use_na_sentinel=True)
    codes = codes.astype(np.int64)
    missing = codes < 0
    next_id = codes.max() + 1 if (~missing).any() else 0
    codes[missing] = np.arange(next_id, next_id + missing.sum())
    source = np.where(missing, "sample", "patient_no")
    return codes, source


def sample_keys(meta: pd.DataFrame) -> np.ndarray:
    """'site|year_folder|code' per row; unique within a built dataset."""
    cols = [meta[c].astype(str) for c in SAMPLE_KEY_COLUMNS]
    keys = cols[0].str.cat(cols[1:], sep="|")
    if keys.duplicated().any():
        raise DatasetError("Sample keys (site, year_folder, code) are not unique.")
    return keys.to_numpy(dtype=object)


def row_fingerprint(meta: pd.DataFrame) -> str:
    """Short hash of which sample is in which row, with its label and patient group.

    Split files store row numbers only; this fingerprint ties them to the dataset they were made for, so
    a changed or different dataset can never be paired with them by mistake (an identical rebuild keeps
    the same fingerprint).
    """
    missing = [c for c in FINGERPRINT_COLUMNS if c not in meta.columns]
    if missing:
        raise DatasetError(f"Cannot fingerprint the metadata: column(s) {missing} missing.")
    cols = [meta[c].astype("int64").astype(str) if c in ("label", "group_id") else meta[c].astype(str)
            for c in FINGERPRINT_COLUMNS]
    text = "\n".join(cols[0].str.cat(cols[1:], sep="|"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path, chunk_bytes: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    """Short commit hash of the code that built the dataset; '-dirty' if tracked files were modified."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        return (head + ("-dirty" if changes else "")) or None
    except (OSError, subprocess.SubprocessError):
        return None


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}


def _close_memmap(array: Any) -> None:
    mm = getattr(array, "_mmap", None)
    if mm is not None:
        array.flush()
        mm.close()


def build_dataset(config: dict[str, Any], spec: CohortSpec | None = None, *, output_root: Path | None = None,
                  progress_every: int = 500) -> dict[str, Any]:
    """Build the processed dataset for `spec` and return its summary."""
    started = time.monotonic()
    spec = spec or CohortSpec.from_config(config)
    pcfg = PreprocessingConfig.from_config(config)
    d = config["driams"]
    root = driams_root(config)
    chunk = max(int(config["dataset"].get("chunk_size", 256)), 1)

    available = discover_sites(root, d["id_folder"])
    missing = [s for s in spec.sites if s not in available]
    if missing:
        raise DatasetError(f"Configured site(s) {missing} are not extracted under {root}. "
                           "Download and extract them first, or remove them from dataset.sites.")

    tables = {s: load_site_tables(root, s, d["id_suffixes"], config["labels"]["missing_values"], d["id_folder"])
              for s in spec.sites}
    candidates, exclusions, info = select_cohort(
        tables, spec, config["labels"]["ambiguous_values"], pcfg.spectrum_folder,
        lambda rel: resolve_relpath(root, rel).is_file())
    del tables
    n = len(candidates)
    if n == 0:
        raise DatasetError("No usable samples after metadata filtering; see the exclusion counts.")
    log.info("%s: %d candidate spectra after metadata filtering (%d rows excluded so far)",
             spec.name, n, len(exclusions))

    output_root = Path(output_root) if output_root else project_path(config["dataset"]["output_dir"])
    out_dir = output_root / spec.name
    tmp_dir = output_root / f"{spec.name}.building"
    if tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
        except OSError as exc:
            raise DatasetError(f"Cannot remove the unfinished build folder {tmp_dir} because a file in it is still "
                               "open. Close programs or Python sessions using it and run again.") from exc
    tmp_dir.mkdir(parents=True)

    staging = np.lib.format.open_memmap(tmp_dir / "X_unfiltered.npy", mode="w+", dtype=pcfg.dtype,
                                        shape=(n, pcfg.n_bins))
    matrix = None
    try:
        reasons, details, raw_points, nonzero, verification = _process_spectra(
            candidates, staging, root, pcfg, spec, config["dataset"], chunk, progress_every)
        valid = np.equal(reasons, None)
        keep_idx = np.flatnonzero(valid)
        if keep_idx.size == 0:
            raise DatasetError("Every candidate spectrum failed preprocessing; nothing to save.")
        matrix = np.lib.format.open_memmap(tmp_dir / "X.npy", mode="w+", dtype=pcfg.dtype,
                                           shape=(keep_idx.size, pcfg.n_bins))
        for start in range(0, keep_idx.size, chunk):
            part = keep_idx[start:start + chunk]
            matrix[start:start + part.size] = staging[part]
        x_bytes = int(matrix.nbytes)
    finally:  # always release the files, otherwise Windows keeps them locked
        if matrix is not None:
            _close_memmap(matrix)
        _close_memmap(staging)
        del matrix, staging
        gc.collect()
    (tmp_dir / "X_unfiltered.npy").unlink()

    spectrum_excluded = candidates.loc[~valid].drop(columns="group_key").assign(
        reason=reasons[~valid], detail=details[~valid])
    exclusions = pd.concat([exclusions, spectrum_excluded], ignore_index=True)
    exclusions["reason"] = pd.Categorical(exclusions["reason"], categories=list(EXCLUSION_REASONS))
    exclusions = exclusions.sort_values(["reason", "site", "year_folder"], kind="stable")

    meta = candidates.loc[valid].reset_index(drop=True)
    group_id, group_source = assign_group_ids(meta["group_key"])
    meta = meta.drop(columns="group_key")
    meta["group_id"] = group_id
    meta["group_source"] = group_source
    meta["label"] = meta["label"].astype(np.int8)
    meta["raw_points"] = raw_points[valid]
    meta["nonzero_bins"] = nonzero[valid]
    meta.insert(0, "sample_index", np.arange(len(meta)))
    meta["acquisition_date"] = pd.to_datetime(meta["acquisition_date"]).dt.strftime("%Y-%m-%d")
    meta = meta[METADATA_COLUMNS]
    assert not any(c in meta.columns or c in exclusions.columns for c in PRIVATE_COLUMNS)

    meta.to_csv(tmp_dir / "metadata.csv", index=False)
    exclusions[EXCLUSION_COLUMNS].to_csv(tmp_dir / "exclusions.csv", index=False)

    summary = summarize(meta, exclusions, info, spec, pcfg, x_bytes)
    summary["row_fingerprint"] = row_fingerprint(meta)
    summary["x_sha256"] = file_sha256(tmp_dir / "X.npy")
    summary["verification_against_driams_binned"] = verification
    summary["elapsed_seconds"] = round(time.monotonic() - started, 1)
    (tmp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    old_dir = output_root / f"{spec.name}.old"
    shutil.rmtree(old_dir, ignore_errors=True)
    try:
        if out_dir.exists():
            out_dir.rename(old_dir)
        tmp_dir.rename(out_dir)
    except PermissionError as exc:
        raise DatasetError(f"Could not replace {out_dir}; close any program (e.g. a notebook) that has the "
                           f"dataset open and run again. The new build is kept in {tmp_dir}.") from exc
    shutil.rmtree(old_dir, ignore_errors=True)
    log.info("%s: saved %d samples x %d features to %s", spec.name, len(meta), pcfg.n_bins, out_dir)
    return summary


def _process_spectra(candidates: pd.DataFrame, staging: np.ndarray, root: Path, pcfg: PreprocessingConfig,
                     spec: CohortSpec, dataset_cfg: dict[str, Any], chunk: int, progress_every: int,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Preprocess every candidate into `staging`; return per-row reasons/details/stats and check counts."""
    n = len(candidates)
    reasons = np.full(n, None, dtype=object)
    details = np.full(n, "", dtype=object)
    raw_points = np.zeros(n, dtype=np.int32)
    nonzero = np.zeros(n, dtype=np.int32)
    seen: dict[str, int] = {}
    codes = candidates["code"].to_numpy()
    sites_arr = candidates["site"].to_numpy()
    years_arr = candidates["year_folder"].to_numpy()
    tolerance = float(dataset_cfg.get("verification_tolerance", 1e-4))
    requested = bool(dataset_cfg.get("verify_against_driams_binned", False))
    verify = requested and pcfg.matches_driams_binning
    verification: dict[str, Any] = {
        "enabled": verify, "tolerance": tolerance, "verified": 0, "no_reference_file": 0,
        "unreadable_reference_file": 0, "failed": 0, "max_relative_difference_of_verified": 0.0,
        "skipped_reason": None if verify or not requested else
        "preprocessing settings differ from DRIAMS, so binned_6000 files are not comparable"}
    for i, rel in enumerate(candidates["spectrum_relpath"].to_numpy()):
        try:
            features, spec_info = preprocess_file(resolve_relpath(root, rel), pcfg)
        except PreprocessingError as exc:  # subclass of SpectrumFormatError: check first
            reasons[i], details[i] = "unusable_spectrum", str(exc)[:200]
            continue
        except (SpectrumFormatError, OSError, UnicodeDecodeError, ValueError) as exc:
            reasons[i], details[i] = "malformed_spectrum", f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        if verify:
            ref_path = resolve_relpath(root, spectrum_relpath(sites_arr[i], "binned_6000", years_arr[i], codes[i]))
            reference = None
            if not ref_path.is_file():
                verification["no_reference_file"] += 1
            else:
                try:
                    reference = read_binned_spectrum(ref_path, pcfg.n_bins)
                except SpectrumFormatError:
                    verification["unreadable_reference_file"] += 1
            if reference is not None:
                scale = reference.max()
                rel_diff = float(np.abs(features.astype(np.float64) - reference).max() / scale) if scale > 0 else np.inf
                if rel_diff > tolerance:
                    verification["failed"] += 1
                    reasons[i], details[i] = "differs_from_driams_binned", f"relative difference {rel_diff:.3g}"
                    continue
                verification["verified"] += 1
                verification["max_relative_difference_of_verified"] = max(
                    verification["max_relative_difference_of_verified"], rel_diff)
        if spec.flag_duplicate_spectra:
            digest = hashlib.sha1(features.tobytes()).hexdigest()
            if digest in seen:
                reasons[i], details[i] = "duplicate_spectrum", f"identical to code {codes[seen[digest]]}"
                continue
            seen[digest] = i
        staging[i] = features
        raw_points[i], nonzero[i] = spec_info.raw_points, spec_info.nonzero_bins
        if (i + 1) % chunk == 0:
            staging.flush()
        if progress_every and (i + 1) % progress_every == 0:
            log.info("%s: processed %d / %d spectra", spec.name, i + 1, n)
    return reasons, details, raw_points, nonzero, verification


def summarize(meta: pd.DataFrame, exclusions: pd.DataFrame, info: dict[str, Any], spec: CohortSpec,
              pcfg: PreprocessingConfig, x_bytes: int) -> dict[str, Any]:
    """Aggregate, identifier-free description of a built dataset."""
    reasons = exclusions["reason"].astype(str)
    by_reason = {r: {s: int(n) for s, n in exclusions.loc[reasons == r, "site"].value_counts().sort_index().items()}
                 for r in EXCLUSION_REASONS if (reasons == r).any()}
    ws = exclusions.loc[reasons == "excluded_workstation"]
    per_site = {}
    for site in spec.sites:
        m = meta[meta["site"] == site]
        dates = pd.to_datetime(m["acquisition_date"], errors="coerce")
        per_site[site] = {
            "target_species_rows": info["target_species_rows"].get(site, 0),
            "samples": len(m),
            "resistant": int((m["label"] == 1).sum()),
            "susceptible": int((m["label"] == 0).sum()),
            "ast_values": {k: int(v) for k, v in m["ast_value"].value_counts().sort_index().items()},
            "excluded": int((exclusions["site"] == site).sum()),
            "acquisition_date_min": None if dates.isna().all() else dates.min().strftime("%Y-%m-%d"),
            "acquisition_date_max": None if dates.isna().all() else dates.max().strftime("%Y-%m-%d"),
            "samples_without_acquisition_date": int(dates.isna().sum()),
            "year_folders": {k: int(v) for k, v in m["year_folder"].value_counts().sort_index().items()},
            "patient_groups": int(m.loc[m["group_source"] == "patient_no", "group_id"].nunique()),
            "samples_without_patient_id": int((m["group_source"] == "sample").sum()),
            "workstation_column_present": info["workstation_column_present"].get(site, False),
            "other_spellings_not_included": info["other_spellings_not_included"].get(site, {}),
        }
    notes = [
        "patient_no is re-hashed for every DRIAMS-A year folder: patient groups are only valid within a "
        "year, so the same person appearing in two years cannot be detected.",
        "Sites without patient IDs (DRIAMS-B) treat every spectrum as its own group.",
        "Sites without a workstation column (DRIAMS-B) cannot be filtered for hospital-hygiene samples.",
        f"Intermediate (I) results are handled as: {spec.intermediate_as}.",
        "Only stateless preprocessing is applied; no scaling, PCA or feature selection is fitted here.",
    ]
    return {
        "dataset": spec.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "species": spec.species,
        "antibiotic": spec.antibiotic,
        "sites": list(spec.sites),
        "intermediate_as": spec.intermediate_as,
        "exclude_workstations": list(spec.exclude_workstations),
        "label_definition": {"1": "resistant (R" + (" or I" if spec.intermediate_as == "resistant" else "") + ")",
                             "0": "susceptible (S" + (" or I" if spec.intermediate_as == "susceptible" else "") + ")"},
        "preprocessing": pcfg.to_dict(),
        "feature_fingerprint": pcfg.fingerprint(),
        "feature_definition": f"bin j = sum of processed intensities with {pcfg.mz_min} + j*{pcfg.bin_width} <= m/z "
                              f"< {pcfg.mz_min} + (j+1)*{pcfg.bin_width} (last bin includes {pcfg.mz_max})",
        "n_samples": len(meta),
        "n_features": pcfg.n_bins,
        "dtype": pcfg.dtype,
        "x_shape": [len(meta), pcfg.n_bins],
        "x_megabytes": round(x_bytes / 1e6, 2),
        "resistant": int((meta["label"] == 1).sum()),
        "susceptible": int((meta["label"] == 0).sum()),
        "per_site": per_site,
        "excluded_total": len(exclusions),
        "exclusions_by_reason": by_reason,
        "excluded_workstations_by_site": {s: {str(k): int(v) for k, v in g["workstation"].value_counts().items()}
                                          for s, g in ws.groupby("site")},
        "raw_points": _stats(meta["raw_points"].to_numpy()),
        "nonzero_bins": _stats(meta["nonzero_bins"].to_numpy()),
        "exclusion_reason_definitions": EXCLUSION_REASONS,
        "notes": notes,
    }


def load_dataset(path: str | Path, *, verify_x: bool = False) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Open a built dataset. X is memory-mapped read-only (no full copy in RAM).

    The metadata is always checked against the fingerprint in summary.json; `verify_x=True` also
    re-hashes X.npy (about a second for 100 MB) to detect a changed or replaced matrix.
    """
    path = Path(path)
    for name in ("X.npy", "metadata.csv", "summary.json"):
        if not (path / name).is_file():
            raise DatasetError(f"{path} is not a built dataset ({name} missing). Run scripts/build_dataset.py.")
    X = np.load(path / "X.npy", mmap_mode="r")
    meta = pd.read_csv(path / "metadata.csv", dtype={"code": str, "year_folder": str, "workstation": str,
                                                     "ast_value": str, "site": str})
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if X.shape[0] != len(meta) or X.shape[1] != summary["n_features"]:
        raise DatasetError(f"X shape {X.shape} does not match metadata ({len(meta)}) / summary.")
    if not np.array_equal(meta["sample_index"].to_numpy(), np.arange(len(meta))):
        raise DatasetError("metadata.csv sample_index is not 0..n-1 in order.")
    if "row_fingerprint" not in summary:
        raise DatasetError(f"{path} was built by an older version without fingerprints; rebuild it with "
                           "scripts/build_dataset.py.")
    if row_fingerprint(meta) != summary["row_fingerprint"]:
        raise DatasetError("metadata.csv does not match the fingerprint in summary.json (the file was changed "
                           "after the build); rebuild the dataset.")
    if verify_x and file_sha256(path / "X.npy") != summary["x_sha256"]:
        raise DatasetError("X.npy does not match the checksum in summary.json (the file was changed after the "
                           "build); rebuild the dataset.")
    return X, meta, summary
