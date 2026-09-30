"""Audit of possible limits to discrimination, before any Version 1.4 decision (docs/discrimination_audit.md).

Development data, saved outputs and metadata only. It fits nothing and scores nothing:

- the development pool's composition (patients, repeat spectra, time, sample type);
- who was left out of the ceftriaxone cohort at DRIAMS-A, and why (label-free except for aggregate counts);
- the excluded screening (HospitalHygiene) isolates: counts, dates, links to pool patients, and a label-free check of
  their spectra against the published DRIAMS binned files;
- the saved Version 1.2 held-out predictions: pooled against within-period and within-sample-type AUROC, and by
  patient; the noise of a paired per-fold AUROC difference (what a study on this pool could detect);
- the saved Version 0.4 / 1.1 search tables: class weights against none.

Patient keys are used in memory to link rows within a DRIAMS-A year folder, exactly as the dataset builder does;
they are never printed or written. Outputs are aggregate counts only.

    python scripts/discrimination_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import DataError, label_status, load_site_tables, read_binned_spectrum  # noqa: E402
from src.dataset import load_dataset  # noqa: E402
from src.development import development_pool, pool_fingerprint  # noqa: E402
from src.preprocessing import PreprocessingConfig, preprocess_file  # noqa: E402
from src.splits import load_splits  # noqa: E402
from src.tables import md_table  # noqa: E402
from src.utils import driams_root, git_commit, load_config, project_path  # noqa: E402


class AuditError(RuntimeError):
    pass


def half_year(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return (d.dt.year.astype(str) + np.where(d.dt.month <= 6, "H1", "H2")).to_numpy()


# --- 1. the development pool -----------------------------------------------------------------------------------------

def pool_composition(meta: pd.DataFrame, pool: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    m = meta.iloc[pool].copy()
    y = m["label"].astype(int)
    per_group = m.groupby("group_id").size()
    resistant = m[y == 1].groupby("group_id").size().sort_values(ascending=False)
    top = int(np.ceil(0.1 * resistant.size))
    summary = {
        "spectra": int(len(m)), "resistant_spectra": int(y.sum()), "patient_groups": int(per_group.size),
        "resistant_patient_groups": int(resistant.size),
        "spectra_per_group": {"mean": round(float(per_group.mean()), 2), "median": float(per_group.median()),
                              "p90": float(per_group.quantile(0.9)), "max": int(per_group.max())},
        "resistant_spectra_per_resistant_group": {
            "mean": round(float(resistant.mean()), 2), "median": float(resistant.median()),
            "max": int(resistant.max()), "groups_with_1": int((resistant == 1).sum()),
            "groups_with_2": int((resistant == 2).sum()), "groups_with_3_to_5": int(resistant.between(3, 5).sum()),
            "groups_with_6_or_more": int((resistant >= 6).sum())},
        "top_10pct_resistant_groups": {"groups": top, "resistant_spectra": int(resistant.head(top).sum()),
                                       "share": round(float(resistant.head(top).sum() / resistant.sum()), 3)},
        "groups_with_resistant_and_susceptible_spectra": int((m.groupby("group_id")["label"].nunique() > 1).sum()),
    }
    m["half"] = half_year(m["acquisition_date"])
    by_time = m.groupby("half").agg(spectra=("label", "size"), resistant=("label", "sum"),
                                    patient_groups=("group_id", "nunique"))
    by_time["resistant_patient_groups"] = m[y == 1].groupby("half")["group_id"].nunique()
    by_time["prevalence"] = (by_time["resistant"] / by_time["spectra"]).round(3)
    by_type = m.groupby("workstation").agg(spectra=("label", "size"), resistant=("label", "sum"))
    by_type["resistant_patient_groups"] = m[y == 1].groupby("workstation")["group_id"].nunique()
    by_type["prevalence"] = (by_type["resistant"] / by_type["spectra"]).round(3)
    mix = pd.crosstab(m["half"], m["workstation"], normalize="index").round(3)
    return summary, by_time.fillna(0).astype({"resistant_patient_groups": int}), by_type.join(mix.T, how="left")


# --- 2. who was left out ---------------------------------------------------------------------------------------------

def cohort_exclusions(data_dir: Path, meta: pd.DataFrame, splits: dict) -> dict[str, pd.DataFrame]:
    ex = pd.read_csv(data_dir / "exclusions.csv", dtype=str, keep_default_na=False)
    ex = ex[ex["site"] == "DRIAMS-A"]
    a = meta[meta["site"] == "DRIAMS-A"]
    no_result = ex[ex["reason"] == "no_ast_result"]
    missing = pd.concat([a.groupby("workstation").size().rename("with_result"),
                         no_result.groupby("workstation").size().rename("without_result")], axis=1).fillna(0)
    missing = missing.astype(int)
    missing["share_without_result"] = (missing["without_result"] / missing.sum(axis=1)).round(3)
    covered = np.unique(np.concatenate([np.concatenate([np.asarray(s.train), np.asarray(s.validation),
                                                        np.asarray(s.test)]) for s in splits.values()]))
    unsplit = meta.drop(index=covered)
    return {"reasons_by_year_folder": pd.crosstab(ex["reason"], ex["year_folder"]),
            "missing_result_by_sample_type": missing.sort_values("with_result", ascending=False),
            "unsplit_rows": pd.crosstab([unsplit["site"], unsplit["year_folder"]], unsplit["label"].astype(int))}


# --- 3. the excluded screening isolates -------------------------------------------------------------------------------

def screening_feasibility(config: dict, meta: pd.DataFrame, pool: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    sc = config["discrimination_audit"]["screening"]
    root = driams_root(config)
    t = load_site_tables(root, sc["site"], config["driams"]["id_suffixes"], config["labels"]["missing_values"],
                         config["driams"]["id_folder"])
    t = t[t["species"].astype("string") == config["target"]["species"]].copy()
    t["label"] = label_status(t[sc["antibiotic"]], config["labels"]["intermediate_as"],
                              config["labels"]["ambiguous_values"])["label"].to_numpy()
    t["year"] = t["driams_year"].astype(str)
    t["key"] = t["year"] + "|" + t["patient_no"].astype("string")     # the builder's within-year patient key
    t["date"] = pd.to_datetime(t["acquisition_date"], errors="coerce")

    screen = t[(t["workstation"].astype("string") == sc["workstation"]) & t["label"].notna()].copy()
    out: dict[str, Any] = {"with_single_category_result": {
        y: {"resistant": int(g["label"].sum()), "susceptible": int((g["label"] == 0).sum())}
        for y, g in screen.groupby("year")}}
    if screen["date"].isna().any() or screen["patient_no"].isna().any():
        raise AuditError("A screening row has no acquisition date or no patient key.")

    # link every screening row to the ceftriaxone dataset's patient groups, within the year folder
    a = meta[meta["site"] == sc["site"]].copy()
    key_of = t.drop_duplicates(["year", "code"]).set_index(["year", "code"])["key"]
    a["key"] = [key_of.get((str(y), c)) for y, c in zip(a["year_folder"].astype(str), a["code"], strict=True)]
    if a["key"].isna().any() or a.groupby("key")["group_id"].nunique().max() != 1 \
            or a.groupby("group_id")["key"].nunique().max() != 1:
        raise AuditError("The dataset's patient groups do not correspond one-to-one to within-year patient keys.")
    in_pool = a["group_id"].isin(set(meta.iloc[pool]["group_id"]))
    pool_keys, other_keys = set(a.loc[in_pool, "key"]), set(a.loc[~in_pool, "key"])
    pool_resistant_keys = set(a.loc[in_pool & (a["label"] == 1), "key"])

    dev = screen[screen["date"] < pd.Timestamp(sc["before"])].copy()
    dev["link"] = np.where(dev["key"].isin(other_keys), "excluded: patient has a clinical row outside the pool",
                           np.where(dev["key"].isin(pool_keys), "joins a pool patient (same year folder)",
                                    "screening only: a new patient"))
    links = dev.groupby("link").agg(rows=("label", "size"), resistant=("label", "sum"),
                                    patient_groups=("key", "nunique"))
    links["resistant_patient_groups"] = dev[dev["label"] == 1].groupby("link")["key"].nunique()
    out["before"] = sc["before"]
    out["dated_before"] = {"rows": int(len(dev)), "resistant": int(dev["label"].sum()),
                           "date_min": str(dev["date"].min().date()), "date_max": str(dev["date"].max().date()),
                           "patient_groups": int(dev["key"].nunique())}
    out["links"] = {k: {c: int(v) for c, v in row.items()} for k, row in links.fillna(0).iterrows()}
    joined = set(dev.loc[dev["link"].str.startswith("joins"), "key"])
    out["pool_resistant_groups_with_a_screening_isolate"] = len(pool_resistant_keys & joined)
    out["pool_susceptible_only_groups_with_a_screening_isolate"] = len((pool_keys - pool_resistant_keys) & joined)

    # label-free check of the eligible spectra: the builder's preprocessing against the published binned files
    eligible = dev[~dev["link"].str.startswith("excluded")]
    pcfg = PreprocessingConfig.from_config(config)
    tolerance = float(config["dataset"]["verification_tolerance"])
    clinical = {hashlib.sha256(np.ascontiguousarray(X[i]).tobytes()).hexdigest() for i in range(X.shape[0])}
    seen: set[str] = set()
    qa = {"checked": 0, "unreadable_or_malformed": 0, "differs_from_driams_binned": 0, "no_binned_file": 0,
          "identical_to_a_clinical_spectrum": 0, "identical_to_an_earlier_screening_spectrum": 0}
    keep = np.zeros(len(eligible), dtype=bool)
    for i, (year, code) in enumerate(zip(eligible["year"], eligible["code"], strict=True)):
        qa["checked"] += 1
        try:
            features, _ = preprocess_file(root / sc["site"] / "raw" / year / f"{code}.txt", pcfg)
        except (DataError, OSError, ValueError):
            qa["unreadable_or_malformed"] += 1
            continue
        binned = root / sc["site"] / "binned_6000" / year / f"{code}.txt"
        if not binned.is_file():
            qa["no_binned_file"] += 1
            continue
        reference = read_binned_spectrum(binned, pcfg.n_bins)
        if float(np.abs(features.astype(np.float64) - reference).max() / reference.max()) > tolerance:
            qa["differs_from_driams_binned"] += 1
            continue
        digest = hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
        if digest in clinical:
            qa["identical_to_a_clinical_spectrum"] += 1
            continue
        if digest in seen:
            qa["identical_to_an_earlier_screening_spectrum"] += 1
            continue
        seen.add(digest)
        keep[i] = True
    usable = eligible[keep]
    new_patients = usable[~usable["key"].isin(pool_keys)]
    out["spectral_check"] = qa
    out["usable"] = {"rows": int(len(usable)), "resistant": int(usable["label"].sum()),
                     "patient_groups": int(usable["key"].nunique()),
                     "joining_pool_patients": {"rows": int(usable["key"].isin(pool_keys).sum()),
                                               "patient_groups": int(usable.loc[usable["key"].isin(pool_keys),
                                                                                "key"].nunique())},
                     "new_patients": {"rows": int(len(new_patients)), "resistant": int(new_patients["label"].sum()),
                                      "patient_groups": int(new_patients["key"].nunique()),
                                      "resistant_patient_groups": int(new_patients.loc[new_patients["label"] == 1,
                                                                                       "key"].nunique())}}
    return out


# --- 4. what the saved predictions already say ------------------------------------------------------------------------

def stratified_auroc(meta: pd.DataFrame, saved: Any, minimum: int) -> pd.DataFrame:
    m = meta.iloc[saved["pool_rows"]].reset_index(drop=True)
    y = m["label"].to_numpy().astype(int)
    strata = {"half-year": half_year(m["acquisition_date"]), "sample type": m["workstation"].astype(str).to_numpy()}
    first = (m.assign(pos=np.arange(len(m))).sort_values(["acquisition_date", "pos"])
             .groupby("group_id").head(1)["pos"].to_numpy())
    rows = []
    for part in ("p42", "p43", "p44"):
        for arm in ("B", "C"):
            p = saved[f"{part}__{arm}"]
            row = {"partition": part[1:], "arm": arm, "pooled": roc_auc_score(y, p)}
            for name, s in strata.items():
                values, weights = [], []
                for level in np.unique(s):
                    k = s == level
                    if y[k].sum() >= minimum and (1 - y[k]).sum() >= minimum:
                        values.append(roc_auc_score(y[k], p[k]))
                        weights.append(y[k].sum())
                row[f"within {name} (weighted by resistant)"] = float(np.average(values, weights=weights))
                row[f"{name} strata used"] = len(values)
            g = pd.DataFrame({"g": m["group_id"], "y": y, "p": p}).groupby("g").agg(y=("y", "max"), p=("p", "mean"))
            row["patient mean score"] = roc_auc_score(g["y"], g["p"])
            row["first spectrum per patient"] = roc_auc_score(y[first], p[first])
            rows.append(row)
    table = pd.DataFrame(rows).round(3)
    prevalence_only = {name: round(float(roc_auc_score(y, pd.Series(y).groupby(s).transform("mean"))), 3)
                       for name, s in strata.items()}
    table.attrs["prevalence_only"] = prevalence_only
    return table


def paired_fold_noise(meta: pd.DataFrame, saved: Any) -> dict[str, Any]:
    """Per-fold AUROC difference C - B over 3 partitions x 5 folds, and its corrected repeated-CV interval."""
    y = meta["label"].to_numpy()[saved["pool_rows"]].astype(int)
    diffs = []
    for part in ("p42", "p43", "p44"):
        fold = saved[f"{part}__fold"]
        for k in np.unique(fold):
            m = fold == k
            diffs.append(roc_auc_score(y[m], saved[f"{part}__C"][m]) - roc_auc_score(y[m], saved[f"{part}__B"][m]))
    d = np.asarray(diffs)
    k, r = 5, 3
    se = float(np.sqrt((1 / (k * r) + 1 / (k - 1)) * d.var(ddof=1)))       # Nadeau-Bengio / Bouckaert-Frank
    t95, t80 = stats.t.ppf(0.975, k * r - 1), stats.t.ppf(0.80, k * r - 1)
    return {"folds": int(d.size), "mean": round(float(d.mean()), 4), "sd": round(float(d.std(ddof=1)), 4),
            "corrected_half_width_95": round(float(t95 * se), 4),
            "uncorrected_half_width_95": round(float(t95 * d.std(ddof=1) / np.sqrt(d.size)), 4),
            "detectable_at_80pct_power": round(float((t95 + t80) * se), 3),
            "note": "two LightGBM settings on the same rows; a change of training data may give a larger sd"}


def class_weight_evidence(config: dict) -> pd.DataFrame:
    rows = []
    for antibiotic, path in config["discrimination_audit"]["search_tables"].items():
        s = pd.read_csv(project_path(path))
        for weight, g in s.groupby("class_weight"):
            rows.append({"antibiotic": antibiotic, "class_weight": weight, "draws": len(g),
                         "mean_cv_auroc": round(float(g["cv_roc_auc_mean"].mean()), 4),
                         "best_cv_auroc": round(float(g["cv_roc_auc_mean"].max()), 4)})
    return pd.DataFrame(rows)


# --- report -----------------------------------------------------------------------------------------------------------

def markdown(results: dict[str, Any]) -> str:
    def md(df: pd.DataFrame) -> str:
        frame = df.reset_index() if df.index.name or any(df.index.names) else df
        frame.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in frame.columns]
        return md_table(frame.astype(object).where(frame.notna(), None).to_dict("records"))

    p, s = results["pool"], results["screening"]
    lines = ["# Discrimination audit: generated tables", "",
             f"Generated by `scripts/discrimination_audit.py` at `{results['git_commit']}`. Development data, saved "
             "outputs and metadata only; nothing was fitted or scored. Interpretation: `docs/discrimination_audit.md`.",
             "", "## 1. The development pool", "", "```json", json.dumps(p, indent=1), "```", "",
             md(results["pool_by_half_year"]), "", md(results["pool_by_sample_type"]), "",
             "## 2. Left out of the ceftriaxone cohort at DRIAMS-A", "",
             "Exclusion reasons by year folder (first applicable reason per metadata row):", "",
             md(results["exclusions"]["reasons_by_year_folder"]), "",
             "Rows with and without a ceftriaxone result, by sample type (label-free):", "",
             md(results["exclusions"]["missing_result_by_sample_type"]), "",
             "Rows of the dataset in no split part (ceftriaxone result but no ciprofloxacin result), by label:", "",
             md(results["exclusions"]["unsplit_rows"]), "",
             "## 3. The excluded screening (HospitalHygiene) isolates", "", "```json", json.dumps(s, indent=1), "```",
             "", "## 4. Saved Version 1.2 held-out predictions: pooled against within-stratum AUROC", "",
             md(results["stratified"]), "",
             f"AUROC of the stratum's prevalence alone: {results['stratified'].attrs['prevalence_only']}", "",
             "## 5. Noise of a paired per-fold AUROC difference (Version 1.2, C minus B)", "", "```json",
             json.dumps(results["noise"], indent=1), "```", "",
             "## 6. Class weights in the saved searches (cross-validated AUROC)", "", md(results["class_weights"]),
             ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ac = config["discrimination_audit"]
    data_dir = project_path(config["dataset"]["output_dir"]) / ac["dataset"]
    X, meta, _ = load_dataset(data_dir, verify_x=True)
    splits = load_splits(data_dir / "splits", meta)
    pool = development_pool(meta, splits, ac["pool_source_split"], list(ac["spent_splits"]))
    expected = ac.get("expected_pool")
    found = {"spectra": int(pool.size), "resistant": int(meta["label"].to_numpy()[pool].sum()),
             "fingerprint": pool_fingerprint(meta, pool)}
    if expected and {k: found[k] for k in expected} != expected:
        raise AuditError(f"The development pool differs from the recorded one: {found} against {expected}.")
    saved = np.load(project_path(ac["heldout_predictions"]), allow_pickle=False)
    if not np.array_equal(saved["pool_rows"], pool):
        raise AuditError("The saved Version 1.2 predictions do not cover exactly the development pool.")

    summary, by_time, by_type = pool_composition(meta, pool)
    results = {"git_commit": git_commit(), "pool": {**summary, "fingerprint": found["fingerprint"]},
               "pool_by_half_year": by_time, "pool_by_sample_type": by_type,
               "exclusions": cohort_exclusions(data_dir, meta, splits),
               "screening": screening_feasibility(config, meta, pool, X),
               "stratified": stratified_auroc(meta, saved, int(ac["min_per_class_in_stratum"])),
               "noise": paired_fold_noise(meta, saved), "class_weights": class_weight_evidence(config)}

    out = project_path(ac["report_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps({k: results[k] for k in ("git_commit", "pool", "screening", "noise")}
                                               | {"prevalence_only_auroc": results["stratified"].attrs[
                                                   "prevalence_only"]}, indent=2), encoding="utf-8")
    results["pool_by_half_year"].to_csv(out / "pool_by_half_year.csv")
    results["pool_by_sample_type"].to_csv(out / "pool_by_sample_type.csv")
    for name, table in results["exclusions"].items():
        table.to_csv(out / f"{name}.csv")
    results["stratified"].to_csv(out / "stratified_auroc.csv", index=False)
    results["class_weights"].to_csv(out / "class_weights.csv", index=False)
    (out / "audit_tables.md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
