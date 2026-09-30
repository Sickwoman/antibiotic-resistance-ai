"""Version 1.4 - excluded screening isolates as training-only data (docs/v1.4_screening_plan.md, amendment 11).

Run from the project root with the virtual environment active, after building the screening-inclusive dataset:

    python scripts/build_dataset.py --antibiotic Ceftriaxone --keep-workstation HospitalHygiene \
        --name ecoli_ceftriaxone_with_screening --report-version v1.4
    python scripts/v14_screening.py

**Development only, and exploratory.** Every evaluation is on the Version 1.2 development pool's clinical isolates;
the screening (HospitalHygiene) isolates are only ever training rows, joined to their patient's within-year group and
left out whenever that patient is held out. Before anything is reported, the baseline A0 must reproduce Version
1.2's arm C exactly, fold by fold. Nothing goes to the production test log or near the served model; the run
appends 9 rows to the separate development log through the strict pre-write gate.

Writes (paths from config.yaml -> v14_screening):
- results/metrics/v1.4/<dataset>/*   primary_result.json, secondary.json, fold_results.csv, tables.md (no identifiers)
- models/v1.4/<dataset>/heldout_predictions.npz   (git-ignored)
- results/experiments/development_runs.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import DataError  # noqa: E402
from src.dataset import load_dataset  # noqa: E402
from src.development import (  # noqa: E402
    DEV_LOG_COLUMNS,
    DevelopmentError,
    DevelopmentLog,
    assert_same_scale,
    cross_fitted_probabilities,
    development_pool,
    outer_folds,
    pool_fingerprint,
    response_method,
    spent_rows,
    stamp,
    supported_cutoff,
)
from src.evaluate import EvaluationError  # noqa: E402
from src.screening import (  # noqa: E402
    ScreeningError,
    corrected_cv_interval,
    group_bootstrap,
    link_screening,
    percentile_interval,
    training_screening,
    verdict,
)
from src.splits import SplitError, load_splits  # noqa: E402
from src.tables import md_table  # noqa: E402
from src.train import load_rows  # noqa: E402
from src.tuning import (  # noqa: E402
    FamilySpec,
    TuningError,
    base_pipeline,
    fit_calibrated,
    grouped_folds,
    pipeline_params,
    set_threads,
)
from src.utils import (  # noqa: E402
    ConfigError,
    get_logger,
    git_commit,
    keep_awake,
    load_config,
    project_path,
    show_path,
)

log = get_logger("v14")
ARMS = ("A0", "A1")
NO_ROWS = np.array([], dtype=np.int64)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def section_hash(section: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(section, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class Ledger:
    """Every fit and cut-off, checked against the plan's section 10, check 5, before it runs."""

    def __init__(self, run: Run) -> None:
        self.run = run
        self.entries: list[dict[str, Any]] = []

    def use(self, what: str, clinical: np.ndarray, screening: np.ndarray, held_out: np.ndarray, where: str,
            before: str) -> None:
        r, s = self.run, self.run.screen
        clinical, screening = np.asarray(clinical, dtype=np.int64), np.asarray(screening, dtype=np.int64)
        held_groups = np.unique(r.groups[held_out])
        limit = np.datetime64(pd.Timestamp(before))
        problems = []
        if np.intersect1d(clinical, held_out).size:
            problems.append("a held-out clinical row")
        if not np.isin(clinical, r.pool).all() or np.intersect1d(clinical, r.spent).size:
            problems.append("a clinical row outside the development pool")
        if np.isin(r.groups[clinical], held_groups).any():
            problems.append("a clinical row of a held-out patient")
        if np.isin(s.groups[screening], held_groups).any():
            problems.append("a screening row of a held-out patient")
        if (r.dates[clinical] >= limit).any() or (s.dates[screening] >= limit).any():
            problems.append(f"a row dated on or after {before}")
        if problems:
            raise DevelopmentError(f"{where}: {what} would use " + ", ".join(problems) + ".")
        self.entries.append({"where": where, "what": what, "clinical": int(clinical.size),
                             "screening": int(screening.size)})


class Run:
    """The configured study: both datasets, the pool, the screening link and provenance, checked before any fit."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config, self.sc, self.ev = config, config["v14_screening"], config["evaluation"]
        sc = self.sc
        self.dataset = str(sc["dataset"])
        self.commit = git_commit() or "unknown"
        dirty = self.commit.endswith("-dirty") or self.commit == "unknown"
        if sc.get("require_clean_tree", True) and dirty:
            raise DevelopmentError(f"The working tree is not clean ({self.commit}); a development run records "
                                   "the exact code it ran, so commit first.")
        self.run_id = f"{self.commit}-{time.strftime('%Y%m%d%H%M%S')}" if dirty else self.commit
        plan = project_path(sc["plan"])
        if not plan.is_file():
            raise ConfigError(f"The plan {show_path(plan)} does not exist; no development run without it.")
        self.plan_sha256, self.config_sha256 = sha256_file(plan), section_hash(sc)
        production = project_path(self.ev["test_log"])
        self.dev_log = DevelopmentLog(project_path(sc["development_log"]), production)
        self.production_log_sha = sha256_file(production) if production.is_file() else None
        self.started = time.monotonic()

    def prepare(self) -> None:
        """Load both datasets and the saved Version 1.2 predictions; checks 1-3. A failure here is logged."""
        config, sc = self.config, self.sc
        root = project_path(config["dataset"]["output_dir"])
        self.X, self.meta, self.summary = load_dataset(root / self.dataset, verify_x=True)
        self.splits = load_splits(root / self.dataset / "splits", self.meta)
        self.y = self.meta["label"].to_numpy().astype(np.int64)
        self.groups = self.meta["group_id"].to_numpy()
        self.dates = pd.to_datetime(self.meta["acquisition_date"]).to_numpy()
        spent = list(sc["spent_splits"])
        self.spent = spent_rows(self.splits, spent)
        self.pool = development_pool(self.meta, self.splits, str(sc["pool_source_split"]), spent)
        self.pool_fp = pool_fingerprint(self.meta, self.pool)
        found = {"spectra": int(self.pool.size), "resistant": int(self.y[self.pool].sum()), "fingerprint": self.pool_fp}
        expected = sc.get("expected_pool")
        if expected and {k: found[k] for k in expected} != expected:           # check 1
            raise DevelopmentError(f"The development pool differs from the plan's: {found} against {expected}.")

        self.full_X, self.full_meta, self.full_summary = load_dataset(root / str(sc["screening_dataset"]),
                                                                      verify_x=True)
        s = sc["screening"]
        self.screen = link_screening(self.meta, self.X, self.pool, self.full_meta, self.full_X,       # check 2
                                     str(s["workstation"]), str(s["before"]))
        if s.get("expected") and self.screen.counts != dict(s["expected"]):                          # check 3
            raise DevelopmentError(f"The screening rows differ from the plan's audit: {self.screen.counts} against "
                                   f"{dict(s['expected'])}.")
        self.ledger = Ledger(self)

        st = sc["setting"]
        card_path = project_path(st["card"])
        card = json.loads(card_path.read_text(encoding="utf-8"))
        self.spec = FamilySpec.from_config(st["family"], config[st["section"]]["families"][st["family"]])
        if card["model_kind"] != self.spec.kind:
            raise ConfigError(f"The card is a {card['model_kind']}, the family a {self.spec.kind}.")
        self.setting, self.card_sha256 = dict(card["params"]), sha256_file(card_path)
        th = sc["threshold"]
        self.rule = {"rule": th["rule"], "min_sensitivity": float(th["min_sensitivity"])}
        self.min_support = int(th["min_resistant_for_support"])

        rp = sc["reproduce"]
        reference = project_path(rp["predictions"])
        if not reference.is_file():
            raise DevelopmentError(f"The saved Version 1.2 predictions {show_path(reference)} are missing; A0 cannot "
                                   "be checked against arm C, so nothing may be reported.")
        self.reference = np.load(reference, allow_pickle=False)
        if not np.array_equal(self.reference["pool_rows"], self.pool):
            raise DevelopmentError("The saved Version 1.2 predictions do not cover exactly this development pool.")
        self.report_dir = project_path(sc["report_dir"]) / self.dataset
        self.model_dir = project_path(sc["model_dir"]) / self.dataset
        for folder in (self.report_dir, self.model_dir):
            folder.mkdir(parents=True, exist_ok=True)

    def budget(self) -> None:
        if time.monotonic() - self.started > float(self.sc["budget_seconds"]):
            raise DevelopmentError(f"The compute budget of {self.sc['budget_seconds']} s was exceeded (the plan's "
                                   "section 12); the run is recorded as failed.")

    def fit(self, arm: str, clinical: np.ndarray, screening: np.ndarray, held_out: np.ndarray, where: str,
            cutoff: bool, before: str) -> dict[str, Any]:
        """One arm on `clinical` (+ `screening` positions for A1): Version 1.2 arm C's calls, unchanged."""
        sc = self.sc
        self.ledger.use(f"{arm} fit and calibration", clinical, screening, held_out, where, before)
        started = time.perf_counter()
        rows = self.screen.rows[screening]
        X = np.concatenate([load_rows(self.X, clinical), load_rows(self.full_X, rows)]) if rows.size \
            else load_rows(self.X, clinical)
        y = np.concatenate([self.y[clinical], self.screen.y[screening]])
        g = np.concatenate([self.groups[clinical], self.screen.groups[screening]])
        seed = int(sc["model_seed"])
        inner = grouped_folds(y, g, int(sc["inner_folds"]), int(sc["inner_seed"]))
        model = fit_calibrated(self.spec, self.setting, X, y, inner, seed, str(sc["calibration"]))
        chosen = None
        if cutoff:
            crossfit = cross_fitted_probabilities(model, self.spec, self.setting, X, y, inner, seed)
            self.ledger.use(f"{arm} cut-off (clinical rows select)", clinical, NO_ROWS, held_out, where, before)
            chosen = supported_cutoff(y[: clinical.size], crossfit[: clinical.size], self.rule, self.min_support)
        set_threads(model, 1)
        assert_same_scale(model, X[: min(200, len(X))])
        return {"model": model, "cutoff": chosen, "fit_rows": int(len(X)), "screening_rows": int(rows.size),
                "fit_seconds": round(time.perf_counter() - started, 1)}

    def predict(self, fitted: dict[str, Any], rows: np.ndarray) -> np.ndarray:
        p = fitted["model"].predict_proba(load_rows(self.X, rows))[:, 1].astype(np.float64)
        if not np.isfinite(p).all():
            raise DevelopmentError("A model produced a non-finite probability.")
        return p

    def reproduce(self, p: np.ndarray, reference: np.ndarray, what: str) -> float:
        """Check 4: A0 is Version 1.2's arm C, or the run stops."""
        worst = float(np.max(np.abs(p - reference))) if p.size else 0.0
        if not worst <= float(self.sc["reproduce"]["tolerance"]):
            raise DevelopmentError(f"A0 does not reproduce Version 1.2's arm C on {what} (largest difference "
                                   f"{worst:.3g}); the baseline would not be the recorded one.")
        return worst


def auc_pr(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def operating(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    tp = int((flagged & (y == 1)).sum())
    return {"sensitivity": tp / int(y.sum()), "specificity": int((~flagged & (y == 0)).sum()) / int((y == 0).sum()),
            "flag_rate": float(flagged.mean()), "precision": tp / int(flagged.sum()) if flagged.any() else float("nan")}


def study(run: Run) -> dict[str, Any]:
    sc, ref = run.sc, run.reference
    y_pool, g_pool = run.y[run.pool], run.groups[run.pool]
    main_partition, before_all = int(sc["primary_partition_seed"]), str(sc["screening"]["before"])
    fold_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"pool_rows": run.pool}
    reproduction: dict[str, Any] = {}
    for partition in [int(s) for s in sc["partition_seeds"]]:
        folds = outer_folds(y_pool, g_pool, int(sc["outer_folds"]), partition,
                            int(sc["min_resistant_per_heldout_fold"]))
        probs = {a: np.full(run.pool.size, np.nan) for a in ARMS}
        decisions = {a: np.zeros(run.pool.size, dtype=bool) for a in ARMS}
        fold_id = np.full(run.pool.size, -1)
        worst = 0.0
        for k, (tr_pos, te_pos) in enumerate(folds):
            train, held_out = run.pool[tr_pos], run.pool[te_pos]
            where = f"partition {partition} fold {k}"
            screening = training_screening(run.screen, np.unique(run.groups[held_out]))
            cut = partition == main_partition
            fitted = {"A0": run.fit("A0", train, NO_ROWS, held_out, where, cut, before_all),
                      "A1": run.fit("A1", train, screening, held_out, where, cut, before_all)}
            for arm in ARMS:
                probs[arm][te_pos] = run.predict(fitted[arm], held_out)
            saved = ref[f"p{partition}__{sc['reproduce']['arm']}"][te_pos]
            worst = max(worst, run.reproduce(probs["A0"][te_pos], saved, where))
            y_te = run.y[held_out]
            for arm in ARMS:
                row = {"partition": partition, "fold": k, "arm": arm, "n": int(te_pos.size),
                       "n_resistant": int(y_te.sum()), **auc_pr(y_te, probs[arm][te_pos]),
                       "fit_rows": fitted[arm]["fit_rows"],
                       "screening_rows": fitted[arm]["screening_rows"], "fit_seconds": fitted[arm]["fit_seconds"]}
                if cut:
                    c = fitted[arm]["cutoff"]
                    decisions[arm][te_pos] = probs[arm][te_pos] >= c["threshold"]
                    row |= {"threshold": c["threshold"], "supported": c["supported"],
                            "selection_resistant": c["selection_resistant"],
                            **operating(y_te, decisions[arm][te_pos])}
                fold_rows.append(row)
            fold_id[te_pos] = k
            log.info("%s: AUROC A0 %.4f, A1 %.4f (A1 trained with %d screening spectra)", where,
                     fold_rows[-2]["roc_auc"], fold_rows[-1]["roc_auc"], screening.size)
            run.budget()
        if (fold_id < 0).any() or any(np.isnan(p).any() for p in probs.values()):
            raise DevelopmentError(f"Partition {partition}: not every pool spectrum was held out exactly once.")
        if not np.array_equal(fold_id, ref[f"p{partition}__fold"]):
            raise DevelopmentError(f"Partition {partition}: the outer folds are not Version 1.2's.")
        reproduction[str(partition)] = {"largest_probability_difference": worst}
        if partition == main_partition:
            same = np.array_equal(decisions["A0"], ref[f"p{partition}__{sc['reproduce']['arm']}__decision"])
            if not same:
                raise DevelopmentError("A0's cut-off decisions are not Version 1.2 arm C's.")
            reproduction[str(partition)]["decisions_identical"] = True
            for arm in ARMS:
                arrays[f"p{partition}__{arm}__decision"] = decisions[arm]
        arrays[f"p{partition}__fold"] = fold_id
        for arm in ARMS:
            arrays[f"p{partition}__{arm}"] = probs[arm]

    fw = sc["forward"]
    dates = run.dates[run.pool]
    train = run.pool[dates < np.datetime64(fw["train_before"])]
    later = run.pool[(dates >= np.datetime64(fw["evaluate_from"])) & (dates < np.datetime64(fw["evaluate_before"]))]
    if np.intersect1d(np.unique(run.groups[train]), np.unique(run.groups[later])).size:
        raise DevelopmentError("The forward-in-time parts share a patient group.")
    if not np.array_equal(later, ref["forward_rows"]):
        raise DevelopmentError("The forward part is not Version 1.2's.")
    screening = training_screening(run.screen, np.unique(run.groups[later]), before=str(fw["train_before"]))
    forward = {}
    for arm, rows in (("A0", NO_ROWS), ("A1", screening)):
        fitted = run.fit(arm, train, rows, later, "forward in time", False, str(fw["train_before"]))
        forward[arm] = run.predict(fitted, later)
        arrays[f"forward__{arm}"] = forward[arm]
    arrays["forward_rows"] = later
    reproduction["forward"] = {"largest_probability_difference": run.reproduce(
        forward["A0"], ref[f"forward__{sc['reproduce']['arm']}"], "the forward split")}
    run.budget()

    # the source diagnostic: resistant spectra only, clinical (0) against screening (1), grouped by patient
    clinical_r = run.pool[run.y[run.pool] == 1]
    screen_r = np.flatnonzero(run.screen.y == 1)
    run.ledger.use("source diagnostic", clinical_r, screen_r, NO_ROWS, "source diagnostic", before_all)
    X_src = np.concatenate([load_rows(run.X, clinical_r), load_rows(run.full_X, run.screen.rows[screen_r])])
    source = np.r_[np.zeros(clinical_r.size, dtype=np.int64), np.ones(screen_r.size, dtype=np.int64)]
    g_src = np.r_[run.groups[clinical_r], run.screen.groups[screen_r]]
    seed = int(sc["model_seed"])
    pipeline = base_pipeline(run.spec, seed).set_params(**pipeline_params(run.spec, run.setting, seed))
    folds = grouped_folds(source, g_src, int(sc["inner_folds"]), int(sc["inner_seed"]))
    oof = cross_val_predict(clone(pipeline), X_src, source, cv=folds, method=response_method(pipeline))
    oof = oof[:, 1] if oof.ndim == 2 else oof
    arrays["source_oof"], arrays["source_label"] = oof, source
    return {"folds": pd.DataFrame(fold_rows), "arrays": arrays, "forward": forward, "later": later,
            "train_forward": train, "forward_screening": int(screening.size), "reproduction": reproduction,
            "source": {"oof": oof, "label": source, "groups": g_src}}


# --- analysis -----------------------------------------------------------------------------------------------------

def analyse(run: Run, result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sc = run.sc
    b = sc["bootstrap"]
    kw = {"resamples": int(b["resamples"]), "seed": int(b["seed"])}
    level = float(b["level"])
    folds, arrays = result["folds"], result["arrays"]
    auc = folds.pivot_table(index=["partition", "fold"], columns="arm", values="roc_auc")
    d = (auc["A1"] - auc["A0"]).to_numpy()                                            # A1 minus A0, per fold
    primary = {"endpoint": "AUROC on held-out clinical spectra, A1 minus A0, per outer fold (3 partitions x 5 folds)",
               **corrected_cv_interval(d, int(sc["outer_folds"]), len(sc["partition_seeds"]), float(sc["level"])),
               "per_fold": [round(float(x), 6) for x in d],
               "per_partition_mean": {str(p): float((auc["A1"] - auc["A0"]).loc[p].mean())
                                      for p in auc.index.get_level_values(0).unique()},
               "exploratory": True}
    primary["verdict"] = verdict(primary["low"], primary["high"])

    y_pool, g_pool = run.y[run.pool], run.groups[run.pool]
    main = int(sc["primary_partition_seed"])
    p = {a: arrays[f"p{main}__{a}"] for a in ARMS}
    dec = {a: arrays[f"p{main}__{a}__decision"] for a in ARMS}

    def pooled_stats(rows: np.ndarray) -> dict[str, float] | None:
        y = y_pool[rows]
        if y.min() == y.max():
            return None
        out: dict[str, float] = {}
        for a in ARMS:
            out |= {f"{a}_{k}": v for k, v in auc_pr(y, p[a][rows]).items()}
            out |= {f"{a}_{k}": v for k, v in operating(y, dec[a][rows]).items()}
        for k in ("roc_auc", "pr_auc", "sensitivity", "specificity", "flag_rate", "precision"):
            out[f"diff_{k}"] = out[f"A1_{k}"] - out[f"A0_{k}"]
        return out

    point = pooled_stats(np.arange(run.pool.size))
    samples, skipped = group_bootstrap(g_pool, pooled_stats, **kw)
    partition_42 = {k: percentile_interval(samples[k], v, level) for k, v in point.items()}

    first = (pd.DataFrame({"g": g_pool, "d": run.dates[run.pool], "i": np.arange(run.pool.size)})
             .sort_values(["d", "i"]).groupby("g").head(1)["i"].to_numpy())
    first = np.sort(first)

    def first_stats(rows: np.ndarray) -> dict[str, float] | None:
        idx = first[rows]
        y = y_pool[idx]
        if y.min() == y.max():
            return None
        a0, a1 = roc_auc_score(y, p["A0"][idx]), roc_auc_score(y, p["A1"][idx])
        return {"A0_roc_auc": a0, "A1_roc_auc": a1, "diff_roc_auc": a1 - a0}

    first_point = first_stats(np.arange(first.size))
    first_samples, first_skipped = group_bootstrap(g_pool[first], first_stats, **kw)

    later = result["later"]
    y_l, g_l = run.y[later], run.groups[later]

    def forward_stats(rows: np.ndarray) -> dict[str, float] | None:
        y = y_l[rows]
        if y.min() == y.max():
            return None
        a0, a1 = roc_auc_score(y, result["forward"]["A0"][rows]), roc_auc_score(y, result["forward"]["A1"][rows])
        return {"A0_roc_auc": a0, "A1_roc_auc": a1, "diff_roc_auc": a1 - a0}

    fw_point = forward_stats(np.arange(later.size))
    fw_samples, fw_skipped = group_bootstrap(g_l, forward_stats, **kw)

    src = result["source"]

    def source_stats(rows: np.ndarray) -> dict[str, float] | None:
        lab = src["label"][rows]
        return None if lab.min() == lab.max() else {"roc_auc": roc_auc_score(lab, src["oof"][rows])}

    src_point = source_stats(np.arange(src["label"].size))
    src_samples, src_skipped = group_bootstrap(src["groups"], source_stats, **kw)

    useless = float(sc["useless_specificity"])
    secondary = {
        "partition_pooled": {str(s): {a: auc_pr(y_pool, arrays[f"p{s}__{a}"]) for a in ARMS}
                             for s in sc["partition_seeds"]},
        f"partition_{main}_bootstrap": {"intervals": partition_42, "skipped_single_class": skipped,
                                        "flags_nearly_everyone": {a: bool(point[f"{a}_specificity"] < useless)
                                                                  for a in ARMS},
                                        "supported_cutoffs": {a: int(folds[(folds["partition"] == main)
                                                                           & (folds["arm"] == a)]["supported"].sum())
                                                              for a in ARMS}},
        "first_spectrum_per_patient": {"patients": int(first.size), "resistant_patients": int(y_pool[first].sum()),
                                       "intervals": {k: percentile_interval(first_samples[k], v, level)
                                                     for k, v in first_point.items()},
                                       "skipped_single_class": first_skipped},
        "forward": {"train_clinical": int(result["train_forward"].size),
                    "train_clinical_resistant": int(run.y[result["train_forward"]].sum()),
                    "train_screening_A1": result["forward_screening"], "evaluated": int(later.size),
                    "evaluated_resistant": int(y_l.sum()),
                    "intervals": {k: percentile_interval(fw_samples[k], v, level) for k, v in fw_point.items()},
                    "skipped_single_class": fw_skipped},
        "source_diagnostic": {"clinical_resistant": int((src["label"] == 0).sum()),
                              "screening_resistant": int((src["label"] == 1).sum()),
                              "roc_auc": percentile_interval(src_samples["roc_auc"], src_point["roc_auc"], level),
                              "skipped_single_class": src_skipped},
        "bootstrap": {"unit": "patient group", "resamples": int(b["resamples"]), "seed": int(b["seed"]),
                      "level": level, "fitting_variability": "not included (fitted models held fixed)"},
    }
    return primary, secondary


# --- report -------------------------------------------------------------------------------------------------------

def signed(e: dict[str, float], digits: int = 3) -> str:
    return f"{e['estimate']:+.{digits}f} [{e['low']:+.{digits}f}, {e['high']:+.{digits}f}]"


def plain(e: dict[str, float], digits: int = 3) -> str:
    return f"{e['estimate']:.{digits}f} ({e['low']:.{digits}f}–{e['high']:.{digits}f})"


def tables(run: Run, primary: dict[str, Any], secondary: dict[str, Any]) -> str:
    sc, counts = run.sc, run.screen.counts
    main = int(sc["primary_partition_seed"])
    boot = secondary[f"partition_{main}_bootstrap"]
    iv = boot["intervals"]
    fw, first, src = secondary["forward"], secondary["first_spectrum_per_patient"], secondary["source_diagnostic"]
    lines = [
        f"# Version 1.4 development tables (`{run.dataset}`) — exploratory", "",
        f"Generated by `scripts/v14_screening.py`. **Development only; every number here is exploratory.** Evaluation "
        "is on the clinical isolates of the Version 1.2 development pool only; the screening isolates were training "
        "rows only. No test part was read and DRIAMS-C was not opened. The clinical labels have informed earlier "
        f"decisions, so nothing here is independent evidence. Run `{run.run_id}`, plan SHA-256 "
        f"`{run.plan_sha256[:16]}…`. Research only; no clinical claim.", "",
        "## The added training data", "",
        f"Screening spectra usable: {counts['usable']} ({counts['usable_resistant']} resistant) from "
        f"{counts['patient_groups']} patient groups — {counts['joining']} spectra of {counts['joining_groups']} pool "
        f"patients, {counts['new']} of {counts['new_groups']} new patients ({counts['new_resistant_groups']} "
        f"resistant); {counts['excluded_outside_pool']} excluded because their patient has a clinical row outside the "
        "pool. A screening spectrum trains only when its patient is not held out, and is never evaluated.", "",
        "## Primary endpoint (exploratory)", "",
        f"AUROC on held-out clinical spectra, **A1 minus A0**, mean over {primary['folds']} outer folds: "
        f"**{signed(primary)}** (corrected repeated-cross-validation interval, {primary['df']} df; it approximates "
        "training-set variability and holds the model seed fixed).", "",
        f"**Verdict under the plan: {primary['verdict']}.** Improved ranking would not be useful sensitivity and "
        "specificity; no usefulness requirement exists.", "",
        md_table([{"partition": k, "mean A1 minus A0": f"{v:+.4f}"} for k, v in primary["per_partition_mean"].items()]),
        "",
        "Baseline check: A0 reproduced Version 1.2's arm C — largest probability difference "
        + ", ".join(f"{k}: {v['largest_probability_difference']:.1e}" for k, v in primary["reproduction"].items())
        + f"; partition {main}'s cut-off decisions identical.", "",
        "## Secondary endpoints (exploratory)", "",
        "Pooled out-of-fold ranking per partition:", "",
        md_table([{"partition": s, "arm": a, "AUROC": f"{v['roc_auc']:.3f}", "PR-AUC": f"{v['pr_auc']:.3f}"}
                  for s, arms in secondary["partition_pooled"].items() for a, v in arms.items()]), "",
        f"Partition {main}, paired patient-group bootstrap ({secondary['bootstrap']['resamples']} replicates; fitted "
        "models held fixed, so no fitting variability):", "",
        md_table([{"quantity": q, "A0": plain(iv[f"A0_{k}"]), "A1": plain(iv[f"A1_{k}"]),
                   "A1 minus A0": signed(iv[f"diff_{k}"])}
                  for q, k in (("AUROC", "roc_auc"), ("PR-AUC", "pr_auc"), ("delivered sensitivity", "sensitivity"),
                               ("delivered specificity", "specificity"), ("flag rate", "flag_rate"),
                               ("precision", "precision"))]), "",
        f"Operating point: Version 1.2 arm C's cut-off procedure (highest cut-off with sensitivity ≥ 0.90 on the "
        f"cross-fitted clinical training spectra); supported cut-offs A0 {boot['supported_cutoffs']['A0']} / 5, A1 "
        f"{boot['supported_cutoffs']['A1']} / 5. Flags nearly everyone (specificity below "
        f"{sc['useless_specificity']:.2f}, a research line): A0 "
        f"{'yes' if boot['flags_nearly_everyone']['A0'] else 'no'}, A1 "
        f"{'yes' if boot['flags_nearly_everyone']['A1'] else 'no'}.", "",
        f"One spectrum per patient (the earliest; {first['patients']} patients, {first['resistant_patients']} "
        f"resistant): AUROC A0 {plain(first['intervals']['A0_roc_auc'])}, "
        f"A1 {plain(first['intervals']['A1_roc_auc'])}; A1 minus A0 {signed(first['intervals']['diff_roc_auc'])}.", "",
        f"Forward in time (Version 1.2's split: fitted on {fw['train_clinical']} clinical spectra, "
        f"{fw['train_clinical_resistant']} resistant, plus {fw['train_screening_A1']} screening spectra for A1; "
        f"evaluated on {fw['evaluated']} spectra of 2017, {fw['evaluated_resistant']} resistant): AUROC A0 "
        f"{plain(fw['intervals']['A0_roc_auc'])}, A1 {plain(fw['intervals']['A1_roc_auc'])}; A1 minus A0 "
        f"{signed(fw['intervals']['diff_roc_auc'])}. No label-availability gap, as in Version 1.2.", "",
        f"Source diagnostic (resistant spectra only: {src['clinical_resistant']} clinical, "
        f"{src['screening_resistant']} screening; 5 patient-grouped folds): AUROC for telling screening from clinical "
        f"{plain(src['roc_auc'])}. "
        "Descriptive only: a high value means the two sources are distinguishable (culture, lineage mix or both).", "",
    ]
    return "\n".join(lines)


def report(run: Run, result: dict[str, Any]) -> dict[str, Any]:
    sc = run.sc
    primary, secondary = analyse(run, result)
    primary["reproduction"] = result["reproduction"]
    provenance = {"run_id": run.run_id, "git_commit": run.commit, "plan": sc["plan"], "plan_sha256": run.plan_sha256,
                  "config_section_sha256": run.config_sha256, "dataset": run.dataset,
                  "dataset_row_fingerprint": run.summary["row_fingerprint"], "x_sha256": run.summary["x_sha256"],
                  "screening_dataset": sc["screening_dataset"],
                  "screening_dataset_row_fingerprint": run.full_summary["row_fingerprint"],
                  "screening_x_sha256": run.full_summary["x_sha256"], "pool_fingerprint": run.pool_fp,
                  "pool_spectra": int(run.pool.size), "pool_resistant": int(run.y[run.pool].sum()),
                  "screening_counts": run.screen.counts, "card_sha256": run.card_sha256, "setting": run.setting,
                  "production_log_sha256_at_start": run.production_log_sha,
                  "ledger_entries": len(run.ledger.entries), "seconds": round(time.monotonic() - run.started, 1),
                  "finished": stamp()}
    primary["provenance"] = provenance
    result["folds"].to_csv(run.report_dir / "fold_results.csv", index=False, lineterminator="\n")
    (run.report_dir / "secondary.json").write_text(json.dumps(secondary, indent=2, default=float), encoding="utf-8",
                                                   newline="\n")
    np.savez_compressed(run.model_dir / "heldout_predictions.npz", **result["arrays"])

    base = {"logged_at": stamp(), "git_commit": run.commit, "run_id": run.run_id, "plan_sha256": run.plan_sha256,
            "config_sha256": run.config_sha256, "dataset": run.dataset,
            "dataset_fingerprint": run.summary["row_fingerprint"], "pool_fingerprint": run.pool_fp, "status": "ok",
            **dict.fromkeys(("brier", "log_loss", "calibration_slope", "calibration_intercept"), "")}
    folds, main = result["folds"], int(sc["primary_partition_seed"])
    iv = secondary[f"partition_{main}_bootstrap"]["intervals"]
    y_pool = run.y[run.pool]
    rows = []
    for s, arms in secondary["partition_pooled"].items():
        for arm, m in arms.items():
            at_main = int(s) == main
            rows.append({**base, "experiment": f"v1.4/{run.run_id}/cv", "model": arm, "seed": int(s),
                         "n": int(run.pool.size), "n_resistant": int(y_pool.sum()), **m,
                         "delivered_sensitivity": iv[f"{arm}_sensitivity"]["estimate"] if at_main else "",
                         "delivered_specificity": iv[f"{arm}_specificity"]["estimate"] if at_main else "",
                         "supported_cutoffs": int(folds[(folds["partition"] == main) & (folds["arm"] == arm)]
                                                  ["supported"].sum()) if at_main else ""})
    fw = secondary["forward"]
    for arm in ARMS:
        rows.append({**base, "experiment": f"v1.4/{run.run_id}/forward", "model": arm, "seed": main,
                     "n": fw["evaluated"], "n_resistant": fw["evaluated_resistant"],
                     "roc_auc": fw["intervals"][f"{arm}_roc_auc"]["estimate"],
                     "pr_auc": float(average_precision_score(run.y[result["later"]], result["forward"][arm])),
                     "delivered_sensitivity": "", "delivered_specificity": "", "supported_cutoffs": ""})
    src = secondary["source_diagnostic"]
    rows.append({**base, "experiment": f"v1.4/{run.run_id}/source", "model": "source", "seed": main,
                 "n": src["clinical_resistant"] + src["screening_resistant"], "n_resistant": src["screening_resistant"],
                 "roc_auc": src["roc_auc"]["estimate"], "pr_auc": "", "delivered_sensitivity": "",
                 "delivered_specificity": "", "supported_cutoffs": ""})
    primary["provenance"]["development_log_pre_write"] = run.dev_log.append(rows)
    (run.report_dir / "primary_result.json").write_text(json.dumps(primary, indent=2, default=float),
                                                      encoding="utf-8", newline="\n")
    (run.report_dir / "tables.md").write_text(tables(run, primary, secondary), encoding="utf-8", newline="\n")
    production = project_path(run.ev["test_log"])
    if run.production_log_sha and sha256_file(production) != run.production_log_sha:
        raise DevelopmentError("The production test log changed during a development run.")
    return primary


def failure_row(run: Run | None, config: dict[str, Any], exc: Exception) -> dict[str, Any]:
    run_id = run.run_id if run else "not-started"
    return dict.fromkeys(DEV_LOG_COLUMNS, "") | {
        "logged_at": stamp(), "git_commit": run.commit if run else "", "run_id": run_id,
        "plan_sha256": run.plan_sha256 if run else "", "config_sha256": run.config_sha256 if run else "",
        "dataset": config["v14_screening"]["dataset"], "experiment": f"v1.4/{run_id}/failed", "model": "run",
        "seed": 0, "status": f"failed: {exc}"[:500]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Version 1.4: screening isolates as training-only data.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    run, config = None, None
    try:
        config = load_config(args.config)
        run = Run(config)
        run.prepare()
        log.info("pool %d spectra (%d resistant), fingerprint %s; screening %s; run %s", run.pool.size,
                 int(run.y[run.pool].sum()), run.pool_fp, run.screen.counts, run.run_id)
        with keep_awake():
            primary = report(run, study(run))
        log.info("primary (exploratory): AUROC A1 minus A0 %+.4f [%+.4f, %+.4f] -> %s", primary["estimate"],
                 primary["low"], primary["high"], primary["verdict"])
        log.info("reports in %s; development log %s", show_path(run.report_dir), show_path(run.dev_log.path))
        return 0
    except (DataError, ConfigError, SplitError, DevelopmentError, EvaluationError, TuningError, ScreeningError) as exc:
        log.error("%s", exc)
        if run is not None:                            # a failed run is recorded, never reported as a result
            try:
                run.dev_log.append([failure_row(run, config, exc)])
            except (DevelopmentError, EvaluationError, OSError) as inner:
                log.error("the failure could not be recorded in the development log: %s", inner)
        return 1


if __name__ == "__main__":
    sys.exit(main())
