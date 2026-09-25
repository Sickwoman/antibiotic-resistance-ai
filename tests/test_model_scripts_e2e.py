"""End-to-end tests of the model scripts on a small synthetic DRIAMS folder.

scripts/train_baselines.py (Version 0.3) and scripts/tune_models.py (Version 0.4) are run through their
main() functions with command-line arguments, on about 230 synthetic raw spectra. Every configured location
(DRIAMS root, processed data, reports, plots, models, the tuning cache and the test log) points into pytest's
temporary folder, so the real data, results, models and results/experiments/test_evaluations.csv are never
touched.

Work-around for a known issue (reported, not fixed here): both scripts print saved files with
`path.relative_to(project_path('.'))`, which raises ValueError when a configured folder is outside the
project. The runs below therefore let `project_path('.')` resolve to the temporary folder; every other
path is resolved by the real `project_path`.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import hashlib
import importlib
import io
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from src.dataset import CohortSpec, build_dataset, load_dataset, resolve_relpath
from src.predict import load_bundle, predict_features, predict_spectrum_file
from src.splits import Split, build_splits, load_splits
from src.train import load_rows
from src.utils import load_config, project_path
from tests.test_preprocessing import synthetic_spectrum, write_spectrum

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DATASET = "e2e_synthetic"           # not the real dataset name, so a stray write would be easy to spot
SITE = "DRIAMS-Y"                   # dated site with patient IDs (like DRIAMS-A)
EXTERNAL_SITE = "DRIAMS-Z"          # no patient IDs, no dates (like DRIAMS-B/D); used by the external split only
# A second external site, so the three-site shape of `external_ab` exists here too: Z is a *test* site in
# `external` and a *training* site in `external_ab`, exactly as DRIAMS-B is in the real configuration.
EXTERNAL_SITE_2 = "DRIAMS-W"
YEAR_SIZES = {"2016": 50, "2017": 96, "2018": 54}
EXTERNAL_SIZE = 30
EXTERNAL_SIZE_2 = 40
MARKER_MZ = 7000.0                  # resistant spectra carry a stronger peak here (a learnable signal)
SEEDS = [42, 43]
# Locked until Version 0.7 (docs/evaluation_protocol.md). `external_ab` was created for Version 0.7 and so
# was never available to the Version 0.3-0.6 scripts either; it belongs in the same set.
NEVER_SCORED = {"temporal", "external", "external_ab"}
SPIED = ("run_search", "fit_calibrated")  # tune_models functions that do the expensive work
TEST_REPORTS = ("test_metrics.csv", "test_intervals.json", "test_intervals.csv", "seed_variation.csv",
                "patient_overlap_check.csv")

# One to four settings per family: every code path of the search plan, at a small cost. The tree models run
# single-threaded (`fixed` overrides the scripts' all-threads default), which keeps the test fast on a busy machine.
SMALL_LIGHTGBM = {"min_child_samples": 5, "max_bin": 15}
FAMILIES: dict[str, dict[str, Any]] = {
    "logistic_regression": {"kind": "logistic_regression", "n_jobs": 1,
                            "grid": [{"features": ["bins_3da", "bins_18da"], "penalty": ["l2"], "C": [0.1, 1.0]}]},
    "random_forest": {"kind": "random_forest", "n_jobs": 1, "stochastic": True,
                      "fixed": {"n_estimators": 20, "n_jobs": 1},
                      "grid": [{"max_features": ["sqrt"], "min_samples_leaf": [1]}]},
    "lightgbm": {"kind": "lightgbm", "n_jobs": 1, "stochastic": True, "random_draws": 2,
                 "fixed": {"subsample_freq": 1, "n_jobs": 1, **SMALL_LIGHTGBM},
                 "space": {"n_estimators": [5, 10], "learning_rate": [0.1], "num_leaves": [4],
                           "colsample_bytree": [0.3, 1.0], "subsample": [0.7, 1.0],
                           "class_weight": [None, "balanced"]}},
    "svm_rbf": {"kind": "svm_rbf", "n_jobs": 1, "grid": [{"C": [1.0, 10.0], "gamma": [1.0e-4]}]},
}
CANDIDATES = {"logistic_regression": 4, "random_forest": 1, "lightgbm": 2, "svm_rbf": 2}

# Version 0.5: tiny networks on the coarse bins, three epochs each. Enough to exercise the section, not to learn.
DEEP_FAMILIES: dict[str, dict[str, Any]] = {
    "mlp": {"kind": "mlp", "stochastic": True,
            "grid": [{"features": ["bins_18da"], "hidden": ["8"], "dropout": [0.0, 0.3]}]},
    "cnn": {"kind": "cnn", "stochastic": True,
            "grid": [{"features": ["bins_18da"], "channels": ["4"], "kernel_size": [5]}]},
}
DEEP_CANDIDATES = {"mlp": 2, "cnn": 1}
DEEP_TRAINING = {"max_epochs": 3, "batch_size": 32, "learning_rate": 1.0e-3, "weight_decay": 1.0e-4,
                 "patience": 2, "inner_validation_fraction": 0.2, "threads": 1}


# ------------------------------------------------------------------------------------------------ synthetic data

def identifier(scope: str, patient: int) -> str:
    """32-character hex hash, like DRIAMS-A patient_no / case_no / order_no."""
    return hashlib.md5(f"e2e|{scope}|{patient}".encode()).hexdigest()


def write_raw(path: Path, seed: int, resistant: bool) -> None:
    rng = np.random.default_rng(seed + 1_000_000)
    mz, intensity = synthetic_spectrum(n=1500, seed=seed)
    height = rng.uniform(2000, 8000) if resistant else rng.uniform(0, 5000)     # overlapping: AUROC below 1
    intensity = intensity + np.round(height * np.exp(-0.5 * ((mz - MARKER_MZ) / 25) ** 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_spectrum(path, mz, intensity)


def write_driams(root: Path) -> set[str]:
    """Synthetic DRIAMS tree with a raw spectrum for every row; returns every identifier hash written."""
    secrets: set[str] = set()
    for year, n in YEAR_SIZES.items():
        rows = []
        for k in range(n):
            patient = k // 2                                           # two spectra per patient (same year)
            value = "R" if patient % 10 in (1, 4) else "I" if patient % 10 == 7 else "S"   # I counts as resistant
            ids = {col: identifier(f"{year}|{col}", patient) for col in ("patient_no", "case_no", "order_no")}
            secrets.update(ids.values())
            date = pd.Timestamp(f"{year}-01-10") + pd.Timedelta(days=k * 340 // n)
            code = f"y{year}_{k:03d}"
            rows.append({"code": code, "species": "Escherichia coli", "laboratory_species": "Escherichia coli",
                         "Ciprofloxacin": value, "acquisition_date": date.strftime("%Y-%m-%d"),
                         "acquisition_time": "10:00:00", "workstation": "Blood" if k % 3 == 0 else "Urine", **ids})
            write_raw(root / SITE / "raw" / year / f"{code}.txt", int(year) * 1000 + k, value != "S")
        id_dir = root / SITE / "id" / year
        id_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(id_dir / f"{year}_strat.csv", index=False)

    for site, size, letter, offset in ((EXTERNAL_SITE, EXTERNAL_SIZE, "z", 900_000),
                                       (EXTERNAL_SITE_2, EXTERNAL_SIZE_2, "w", 950_000)):
        rows = []
        for k in range(size):
            value = "R" if k % 4 == 0 else "S"
            rows.append({"species": "Escherichia coli", "code": f"{letter}{k:03d}", "combined_code": None,
                         "Ciprofloxacin": value})
            write_raw(root / site / "raw" / "2018" / f"{letter}{k:03d}.txt", offset + k, value == "R")
        id_dir = root / site / "id" / "2018"
        id_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(id_dir / "2018_clean.csv", index=False)
    return secrets


def make_config(tmp: Path, root: Path) -> dict[str, Any]:
    """The project configuration with every data/output path in `tmp` and small, fast model settings."""
    config = copy.deepcopy(load_config())
    config["paths"]["driams_root"] = str(root)
    config["dataset"].update(name=DATASET, output_dir=str(tmp / "processed"),
                             sites=[SITE, EXTERNAL_SITE, EXTERNAL_SITE_2],
                             verify_against_driams_binned=False)
    s = config["splits"]
    s["random"]["sites"] = [SITE]
    s["within_year"].update(sites=[SITE], year_folder="2017")
    s["temporal"]["sites"] = [SITE]
    s["external"].update(train_sites=[SITE], test_sites=[EXTERNAL_SITE, EXTERNAL_SITE_2])
    s["external_ab"].update(train_sites=[SITE, EXTERNAL_SITE], test_sites=[EXTERNAL_SITE_2])

    ev = config["evaluation"]
    ev.update(test_log=str(tmp / "results" / "experiments" / "test_evaluations.csv"), timing_samples=5)
    ev["bootstrap"]["resamples"] = 20

    bl = config["baselines"]
    bl.update(seeds=SEEDS, model_dir=str(tmp / "models" / "v0.3"),
              report_dir=str(tmp / "results" / "metrics" / "v0.3"), plot_dir=str(tmp / "results" / "plots" / "v0.3"))
    for model in bl["models"].values():
        if model["kind"] == "random_forest":
            model["params"] = {**(model.get("params") or {}), "n_estimators": 20}
        elif model["kind"] == "lightgbm":
            # src/train.py passes n_jobs=-1; LightGBM's own name num_threads takes precedence over it
            model["params"] = {**(model.get("params") or {}), "n_estimators": 10, "num_leaves": 4,
                               "num_threads": 1, **SMALL_LIGHTGBM}

    tc = config["tuning"]
    tc.update(cv_folds=3, seeds=SEEDS, families=copy.deepcopy(FAMILIES), model_dir=str(tmp / "models" / "v0.4"),
              report_dir=str(tmp / "results" / "metrics" / "v0.4"), plot_dir=str(tmp / "results" / "plots" / "v0.4"))

    dp = config["deep"]
    dp.update(cv_folds=3, seeds=SEEDS, families=copy.deepcopy(DEEP_FAMILIES), training=dict(DEEP_TRAINING),
              model_dir=str(tmp / "models" / "v0.5"), report_dir=str(tmp / "results" / "metrics" / "v0.5"),
              plot_dir=str(tmp / "results" / "plots" / "v0.5"), compare_with=tc["model_dir"])

    # Version 0.6 describes the model Version 0.4 saved; small settings keep the run to a few seconds.
    xp = config["explain"]
    xp.update(model_dir=tc["model_dir"], split=tc["split"], seeds=SEEDS, cross_model=dp["model_dir"],
              report_dir=str(tmp / "results" / "metrics" / "v0.6"),
              plot_dir=str(tmp / "results" / "plots" / "v0.6"),
              regions={"top_bins": 20, "merge_gap_bins": 2, "report_regions": 5},
              permutation={"block_bins": 300, "repeats": 2, "seed": 42, "top_single_bins": 5})

    # Version 0.7 refits the setting Version 0.4 chose and spends the previously locked test parts.
    gn = config["generalisation"]
    gn.update(source_model=tc["model_dir"], source_split=tc["split"], seeds=SEEDS,
              model_dir=str(tmp / "models" / "v0.7"),
              report_dir=str(tmp / "results" / "metrics" / "v0.7"),
              plot_dir=str(tmp / "results" / "plots" / "v0.7"))
    gn["zone_transfer"] = {**gn["zone_transfer"], "zones": xp["report_dir"]}
    gn["shift_diagnostic"] = {**gn["shift_diagnostic"], "regions_from": xp["report_dir"], "low_mz_bins": 20}

    written = [config["dataset"]["output_dir"], ev["test_log"],
               *(section[key] for section in (bl, tc, dp, xp) for key in ("model_dir", "report_dir", "plot_dir")),
               *(gn[key] for key in ("model_dir", "report_dir", "plot_dir"))]
    assert all(Path(p).resolve().is_relative_to(tmp.resolve()) for p in written), written
    return config


# ------------------------------------------------------------------------------------------------ helpers

@dataclass
class Workspace:
    root: Path
    config: dict[str, Any]
    config_path: Path
    data_dir: Path
    secrets: set[str]

    def folder(self, section: str, key: str) -> Path:
        path = Path(self.config[section][key])
        return path / DATASET if key in ("report_dir", "model_dir") else path

    @property
    def cache_dir(self) -> Path:
        return self.cache_dir_of("tuning")

    def cache_dir_of(self, section: str) -> Path:
        return Path(self.config[section]["model_dir"]) / "cache" / DATASET

    def write_config(self, config: dict[str, Any], name: str) -> Path:
        path = self.root / name
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def read_json(self, section: str, name: str) -> dict[str, Any]:
        return json.loads((self.folder(section, "report_dir") / name).read_text(encoding="utf-8"))

    def test_log(self) -> pd.DataFrame:
        path = Path(self.config["evaluation"]["test_log"])
        return pd.read_csv(path) if path.is_file() and path.stat().st_size else pd.DataFrame()

    def outputs(self) -> dict[str, int]:
        """Every file under the temporary results/ and models/ folders, with its modification time.

        run_status.json is left out on purpose: a refused run is meant to record why it stopped there.
        """
        return {str(p.relative_to(self.root)): p.stat().st_mtime_ns
                for top in ("results", "models") if (self.root / top).is_dir()
                for p in (self.root / top).rglob("*") if p.is_file() and p.name != "run_status.json"}

    def run_status(self, section: str = "tuning") -> dict[str, Any]:
        path = self.folder(section, "report_dir") / "run_status.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def report_files(self, section: str = "tuning") -> list[str]:
        folder = self.folder(section, "report_dir")
        return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []

    def cache_files(self, section: str = "tuning") -> dict[str, int]:
        folder = self.cache_dir_of(section)
        if not folder.is_dir():
            return {}
        return {p.relative_to(folder).as_posix(): p.stat().st_mtime_ns for p in folder.rglob("*.joblib")}

    def splits(self) -> tuple[pd.DataFrame, dict[str, Split]]:
        X, meta, _ = load_dataset(self.data_dir)
        del X
        return meta, load_splits(self.data_dir / "splits", meta)

    def chosen_family(self) -> str:
        return self.read_json("tuning", "best_model_card.json")["model_kind"]

    def finalists(self) -> tuple[int, int]:
        """(finalists on the main split, finalists of the re-tuned experiments) the search plan produces."""
        tc = self.config["tuning"]
        runs = {name: len(tc["seeds"]) if f.get("stochastic") else 1 for name, f in tc["families"].items()}
        return sum(runs.values()), len(tc["retune_experiments"]) * runs[self.chosen_family()]

    def baseline_runs(self) -> int:
        bl = self.config["baselines"]
        per_split = sum(len(bl["seeds"]) if m.get("stochastic") else 1 for m in bl["models"].values())
        return per_split * (len(bl["splits"]) + (1 if bl.get("size_matched") else 0))


def is_test_report(name: str) -> bool:
    return name in TEST_REPORTS or name.startswith("test_")


class Messages(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def has(self, text: str) -> bool:
        return any(text in m for m in self.messages)


@dataclass
class Run:
    code: int
    stdout: str
    log: Messages
    seconds: float
    calls: list[str] = field(default_factory=list)


def run_script(module: ModuleType, ws: Workspace, *args: Any, spy: tuple[str, ...] = ()) -> Run:
    """Call `module.main()` as if run from the command line; `spy` names module functions to count."""
    handler, out, calls = Messages(), io.StringIO(), []
    loggers = [logging.getLogger(name) for name in ("baselines", "tuning", "explain", "generalise")]

    def counted(name: str, fn):
        def wrapper(*a, **kw):
            calls.append(name)
            return fn(*a, **kw)
        return wrapper

    def project_path_in_tmp(relative):                 # see the module docstring
        return ws.root if Path(relative) == Path(".") else project_path(relative)

    started = time.perf_counter()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", [f"{module.__name__}.py", *(str(a) for a in args)])
        mp.delenv("DRIAMS_ROOT", raising=False)         # would override the synthetic paths.driams_root
        mp.setenv("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED", "0"))   # set_seed() changes it; restored after
        mp.setattr(module, "project_path", project_path_in_tmp)
        for name in spy:
            mp.setattr(module, name, counted(name, getattr(module, name)))
        for logger in loggers:
            logger.addHandler(handler)
        try:
            with contextlib.redirect_stdout(out):
                code = module.main()
        finally:
            for logger in loggers:
                logger.removeHandler(handler)
    gc.collect()                                         # release memory-mapped X.npy files (Windows locks them)
    return Run(code, out.getvalue(), handler, time.perf_counter() - started, calls)


# ------------------------------------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def scripts() -> tuple[ModuleType, ModuleType]:
    return importlib.import_module("train_baselines"), importlib.import_module("tune_models")


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Workspace:
    tmp = tmp_path_factory.mktemp("model_scripts_e2e")
    root = tmp / "DRIAMS"
    secrets = write_driams(root)
    config = make_config(tmp, root)
    build_dataset(config, CohortSpec.from_config(config))
    data_dir = Path(config["dataset"]["output_dir"]) / DATASET
    X, meta, _ = load_dataset(data_dir)
    del X
    skipped: list[str] = []
    splits = build_splits(meta, config, DATASET, data_dir.parent, skipped)   # as scripts/build_dataset.py does
    assert skipped == [] and set(splits) == {"random", "within_year", "temporal", "external", "external_ab"}
    for name, split in splits.items():
        split.save(data_dir / "splits" / f"{name}.json", meta)
    ws = Workspace(tmp, config, tmp / "config.yaml", data_dir, secrets)
    ws.write_config(config, "config.yaml")
    yield ws
    gc.collect()


@dataclass
class Pipeline:
    runs: dict[str, Run]
    logs: dict[str, pd.DataFrame]          # test log after each run
    reports: dict[str, list[str]]          # files in the Version 0.4 report folder after each run
    caches: dict[str, dict[str, int]]      # cached searches/fits (with modification times) after each run
    run_configs: dict[str, dict[str, Any]]


STEPS = [  # (name, script, extra arguments), run in this order on the same folders
    ("baselines", "train_baselines", ["--evaluate-test"]),
    ("test_without_cache", "tune_models", ["--evaluate-test"]),
    ("development", "tune_models", []),
    ("test", "tune_models", ["--evaluate-test"]),
    ("test_again", "tune_models", ["--evaluate-test"]),
    ("rescore", "tune_models", ["--evaluate-test", "--allow-rescore"]),
    # Version 0.5 reuses the same script and safeguards through --section deep, comparing with Version 0.4
    ("deep_test_without_cache", "tune_models", ["--section", "deep", "--evaluate-test"]),
    ("deep_development", "tune_models", ["--section", "deep"]),
    ("deep_test", "tune_models", ["--section", "deep", "--evaluate-test"]),
]


@pytest.fixture(scope="module")
def pipeline(workspace, scripts) -> Pipeline:
    modules = dict(zip(("train_baselines", "tune_models"), scripts, strict=True))
    result = Pipeline({}, {}, {}, {}, {})
    for name, script, extra in STEPS:
        section = "deep" if name.startswith("deep") else "tuning"
        spy = SPIED if script == "tune_models" else ()
        run = run_script(modules[script], workspace, "--config", workspace.config_path, *extra, spy=spy)
        print(f"{name}: exit code {run.code}, {run.seconds:.1f} s")
        run_config = workspace.folder(section, "report_dir") / "run_config.json"
        result.runs[name] = run
        result.logs[name] = workspace.test_log()
        result.reports[name] = workspace.report_files(section)
        result.caches[name] = workspace.cache_files(section)
        result.run_configs[name] = json.loads(run_config.read_text(encoding="utf-8")) if run_config.is_file() else {}
    return result


# ------------------------------------------------------------------------------------------------ Version 0.3

def test_baselines_score_only_the_allowed_test_parts_and_save_the_model(workspace, pipeline):
    run = pipeline.runs["baselines"]
    assert run.code == 0, run.log.messages
    bl, main = workspace.config["baselines"], workspace.config["baselines"]["splits"][0]
    reports = workspace.folder("baselines", "report_dir")
    for name in ("validation_metrics.csv", *TEST_REPORTS, "best_model_card.json", "inference_timing.json",
                 "run_config.json"):
        assert (reports / name).is_file(), name

    runs = workspace.baseline_runs()
    matched = f"{bl['size_matched']['split']}_size_matched"
    validation = pd.read_csv(reports / "validation_metrics.csv")
    test = pd.read_csv(reports / "test_metrics.csv")
    assert len(validation) == len(test) == runs
    assert set(test["experiment"]) == {*bl["splits"], matched}
    assert set(test.loc[test["model"] == "random_forest", "seed"]) == set(SEEDS)
    assert test["roc_auc"].notna().all()
    assert (test.loc[test["model"] == "prevalence", "roc_auc"] == 0.5).all()

    meta, splits = workspace.splits()
    for _, row in test.iterrows():
        split = splits[row["split"]]
        assert row["n"] == len(split.test) and row["n_resistant"] == int(meta["label"].iloc[split.test].sum())
        if row["experiment"] == matched:
            assert 0 < row["train_size"] <= len(splits[bl["size_matched"]["size_of"]].train)
        else:
            assert row["train_size"] == len(split.train)
    assert set(workspace.read_json("baselines", "test_intervals.json")) == {*bl["splits"], matched}

    log = pipeline.logs["baselines"]
    assert len(log) == runs
    assert set(log["stage"]) == {"v0.3-baselines"} and set(log["dataset"]) == {DATASET}
    assert set(log["split"]) == set(bl["splits"])
    assert not set(log["split"]) & (NEVER_SCORED | set(workspace.config["evaluation"]["locked_test_splits"]))
    assert np.allclose(log["roc_auc"].to_numpy(), test["roc_auc"].to_numpy())

    card = workspace.read_json("baselines", "best_model_card.json")
    candidates = card["selection"]["candidates"]
    assert card["selection"]["part"] == "validation" and card["model"] == max(candidates, key=candidates.get)
    assert candidates[card["model"]] > 0.5
    assert card["split"]["name"] == main and card["dataset"]["name"] == DATASET and card["seed"] == SEEDS[0]
    model_path = workspace.folder("baselines", "model_dir") / f"best_{main}.joblib"
    assert model_path.is_file() and model_path.with_suffix(".json").is_file()

    timing = workspace.read_json("baselines", "inference_timing.json")
    assert timing["matches_stored_features"] is True
    assert timing["samples"] == workspace.config["evaluation"]["timing_samples"]
    assert workspace.read_json("baselines", "run_config.json")["test_parts_scored"] is True

    plots = workspace.folder("baselines", "plot_dir")
    for name in ("roc_pr", "calibration", f"confusion_{card['model']}"):
        assert (plots / f"{DATASET}_{main}_{name}.png").stat().st_size > 0


# ------------------------------------------------------------------------------------------------ Version 0.4

def test_test_run_refuses_to_fit_models_that_are_not_cached(pipeline):
    run = pipeline.runs["test_without_cache"]
    assert run.code == 1
    assert run.log.has("is not cached for these settings/code")
    assert run.calls == []
    pd.testing.assert_frame_equal(pipeline.logs["test_without_cache"], pipeline.logs["baselines"])
    assert pipeline.caches["test_without_cache"] == {}
    assert not [n for n in pipeline.reports["test_without_cache"] if is_test_report(n) or n.startswith("search_")]


def test_development_run_searches_and_validates_without_touching_the_test_parts(workspace, pipeline):
    run = pipeline.runs["development"]
    assert run.code == 0, run.log.messages
    tc = workspace.config["tuning"]
    run_config = pipeline.run_configs["development"]
    chosen = run_config["chosen_family"]
    assert chosen == workspace.chosen_family()
    assert run_config["status"] == "finished" and run_config["test_parts_scored"] is False
    assert run_config["config"]["dataset"]["name"] == DATASET
    required = ({f"search_{tc['split']}_{family}.csv" for family in FAMILIES}
                | {f"search_{name}_{chosen}.csv" for name in tc["retune_experiments"]}
                | {"search_summary.csv", "validation_metrics.csv", "best_model_card.json", "inference_timing.json",
                   "run_config.json"})
    assert required <= set(pipeline.reports["development"])
    assert not [name for name in pipeline.reports["development"] if is_test_report(name)]
    pd.testing.assert_frame_equal(pipeline.logs["development"], pipeline.logs["baselines"])

    # every search and fit was computed once and cached
    main, retune = workspace.finalists()
    searches = len(FAMILIES) + len(tc["retune_experiments"])
    assert sorted(run.calls) == sorted(["run_search"] * searches + ["fit_calibrated"] * (main + retune))
    cache = pipeline.caches["development"]
    assert len(cache) == searches + main + retune
    assert {f"{tc['split']}/{family}/search.joblib" for family in FAMILIES} <= set(cache)
    assert not run.log.has("using cached")
    info = json.loads((workspace.cache_dir / "cache_info.json").read_text(encoding="utf-8"))
    _, _, summary = load_dataset(workspace.data_dir)
    gc.collect()
    assert (info["x_sha256"], info["feature_fingerprint"]) == (summary["x_sha256"], summary["feature_fingerprint"])

    reports = workspace.folder("tuning", "report_dir")         # later runs rewrite these from the cache
    for family, n in CANDIDATES.items():
        table = pd.read_csv(reports / f"search_{tc['split']}_{family}.csv")
        assert len(table) == n and table["cv_roc_auc_mean"].notna().all()
        assert table["rank"].min() == 1
    validation = pd.read_csv(reports / "validation_metrics.csv")
    reference = validation["model"] == "v0.3_prevalence"
    assert int(validation["calibrated"].sum()) == main + retune
    assert set(validation.loc[reference, "experiment"]) == {tc["split"], *tc["retune_experiments"]}
    assert reference.sum() == 1 + len(tc["retune_experiments"])
    assert set(validation["experiment"]) == {tc["split"], *tc["retune_experiments"]}
    card = workspace.read_json("tuning", "best_model_card.json")
    assert card["selection"]["part"] == "validation" and card["model"] == f"tuned_{chosen}"


def test_test_run_reuses_the_cache_and_logs_each_evaluation_once(workspace, pipeline):
    run = pipeline.runs["test"]
    assert run.code == 0, run.log.messages
    assert run.calls == []
    assert pipeline.caches["test"] == pipeline.caches["development"]          # same files, not rewritten
    assert sum("using cached" in m for m in run.log.messages) == len(pipeline.caches["development"])
    assert "validation rows identical to the development run" in run.stdout
    assert "Version 0.3 model re-scored" in run.stdout
    assert pipeline.run_configs["test"]["status"] == "finished"
    assert pipeline.run_configs["test"]["test_parts_scored"] is True

    tc, bl = workspace.config["tuning"], workspace.config["baselines"]
    main_name, experiments = tc["split"], {tc["split"], *tc["retune_experiments"]}
    chosen = workspace.chosen_family()
    main, retune = workspace.finalists()
    assert set(TEST_REPORTS) <= set(pipeline.reports["test"])

    # test_metrics.csv: the finalists, the re-scored Version 0.3 model, then the Version 0.3 no-skill rows
    test = pd.read_csv(workspace.folder("tuning", "report_dir") / "test_metrics.csv")
    v03_card = workspace.read_json("baselines", "best_model_card.json")
    v03_name = f"v0.3_{v03_card['model']}"
    rescored = test["reproduces_logged_auroc_within"].notna()
    prevalence = test["model"] == "v0.3_prevalence"
    assert len(test) == main + retune + 1 + len(experiments)
    assert rescored.sum() == 1 and test.loc[rescored, "model"].iloc[0] == v03_name
    assert float(test.loc[rescored, "reproduces_logged_auroc_within"].iloc[0]) <= 1e-4
    assert set(test.loc[prevalence, "experiment"]) == experiments and (test.loc[prevalence, "roc_auc"] == 0.5).all()
    tuned = test.iloc[:main + retune]
    assert set(tuned["model"]) == {f"tuned_{family}" for family in FAMILIES}
    assert set(tuned.loc[tuned["experiment"] != main_name, "model"]) == {f"tuned_{chosen}"}
    assert tuned["roc_auc"].notna().all()
    v03 = pd.read_csv(workspace.folder("baselines", "report_dir") / "test_metrics.csv")
    v03 = v03[(v03["experiment"] == main_name) & (v03["model"] == v03_card["model"])
              & (v03["seed"] == v03_card["seed"])]
    assert abs(float(test.loc[rescored, "roc_auc"].iloc[0]) - float(v03["roc_auc"].iloc[0])) <= 1e-4

    # the test log got exactly those rows (the no-skill rows are copies and are not logged again)
    new = pipeline.logs["test"].iloc[len(pipeline.logs["development"]):]
    assert len(new) == main + retune + 1
    assert (new["stage"] == "v0.4-tuned").sum() == main + retune
    assert new["stage"].iloc[-1] == "v0.4-reference" and new["model"].iloc[-1] == v03_name
    assert np.allclose(new["roc_auc"].to_numpy(), test["roc_auc"].iloc[:len(new)].to_numpy())
    assert set(new["experiment"]) == experiments and set(new["dataset"]) == {DATASET}
    locked = NEVER_SCORED | set(workspace.config["evaluation"]["locked_test_splits"])
    for column in ("split", "experiment"):
        assert not set(pipeline.logs["rescore"][column]) & locked

    meta, splits = workspace.splits()
    matched = f"{bl['size_matched']['split']}_size_matched"
    for _, row in new.iterrows():
        assert row["n"] == len(splits[row["split"]].test)
        if row["experiment"] == matched:
            assert 0 < row["train_size"] <= len(splits[bl["size_matched"]["size_of"]].train)
        else:
            assert row["train_size"] == len(splits[row["split"]].train)

    # saved test probabilities reproduce the reported AUROCs
    labels = meta["label"].to_numpy()
    with np.load(workspace.folder("tuning", "model_dir") / "test_probabilities.npz") as npz:
        stored = {key: npz[key] for key in npz.files}
    keys = {f"{r.experiment}__{r.model}__seed{r.seed}": r.roc_auc for r in tuned.itertuples()}
    assert set(stored) == {*keys, f"{main_name}__{v03_name}", *(f"{name}__rows" for name in experiments)}
    for name in experiments:
        split = bl["size_matched"]["split"] if name == matched else name
        assert np.array_equal(stored[f"{name}__rows"], splits[split].test)
    for key, auroc in keys.items():
        rows = stored[f"{key.split('__')[0]}__rows"]
        assert roc_auc_score(labels[rows], stored[key]) == pytest.approx(auroc, abs=1e-12)

    intervals = workspace.read_json("tuning", "test_intervals.json")
    assert set(intervals) == experiments
    assert set(intervals[main_name]["intervals"]) == {*(f"tuned_{f}" for f in FAMILIES), v03_name}
    assert set(intervals[main_name]["differences_to_reference_model"]) == {f"tuned_{f}" for f in FAMILIES}
    ci = pd.read_csv(workspace.folder("tuning", "report_dir") / "test_intervals.csv")
    assert {f"AUROC tuned_{chosen} minus this", "AUROC this minus Version 0.3"} <= set(ci.columns)
    overlap = pd.read_csv(workspace.folder("tuning", "report_dir") / "patient_overlap_check.csv")
    size_of = bl["size_matched"]["size_of"]
    assert overlap["comparison"].tolist() == [f"{matched} minus {size_of}", f"{main_name} minus {size_of}"]
    assert set(overlap["model"]) == {f"tuned_{chosen}"}

    plots = workspace.folder("tuning", "plot_dir")
    for name in ("search_overview", f"{main_name}_roc_pr", f"{main_name}_calibration",
                 f"{main_name}_confusion_tuned_{chosen}"):
        assert (plots / f"{DATASET}_{name}.png").stat().st_size > 0


def test_second_test_run_needs_allow_rescore_and_is_logged_again(workspace, pipeline):
    again = pipeline.runs["test_again"]
    assert again.code == 1 and again.calls == []
    assert again.log.has("already logged")
    pd.testing.assert_frame_equal(pipeline.logs["test_again"], pipeline.logs["test"])
    assert pipeline.run_configs["test_again"] == pipeline.run_configs["test"]      # stopped before writing

    rescore = pipeline.runs["rescore"]
    assert rescore.code == 0, rescore.log.messages
    assert rescore.calls == [] and pipeline.caches["rescore"] == pipeline.caches["development"]
    main, retune = workspace.finalists()
    first = pipeline.logs["test"].iloc[len(pipeline.logs["development"]):].reset_index(drop=True)
    second = pipeline.logs["rescore"].iloc[len(pipeline.logs["test"]):].reset_index(drop=True)
    assert len(second) == len(first) == main + retune + 1
    columns = ["stage", "experiment", "split", "model", "seed", "n", "threshold", "roc_auc", "pr_auc", "tp", "fp"]
    pd.testing.assert_frame_equal(second[columns], first[columns])      # same models, same numbers


def test_version_03_mismatch_is_logged_and_stops_the_run(workspace, scripts, pipeline):
    """A copy of the development cache and reports; the logged Version 0.3 AUROC is changed."""
    _, tune_models = scripts
    assert pipeline.runs["development"].code == 0
    base = workspace.root / "tampered"
    config = copy.deepcopy(workspace.config)
    tc = config["tuning"]
    tc.update(model_dir=str(base / "models" / "v0.4"), report_dir=str(base / "metrics" / "v0.4"),
              plot_dir=str(base / "plots" / "v0.4"))
    config["baselines"]["report_dir"] = str(base / "metrics" / "v0.3")
    config["evaluation"]["test_log"] = str(base / "test_evaluations.csv")
    shutil.copytree(workspace.cache_dir, base / "models" / "v0.4" / "cache" / DATASET)
    tuned_reports = Path(tc["report_dir"]) / DATASET          # the development run's validation results
    tuned_reports.mkdir(parents=True)
    shutil.copy2(workspace.folder("tuning", "report_dir") / "validation_metrics.csv", tuned_reports)
    v03_reports = base / "metrics" / "v0.3" / DATASET
    v03_reports.mkdir(parents=True)
    source = workspace.folder("baselines", "report_dir")
    shutil.copy2(source / "validation_metrics.csv", v03_reports)
    logged = pd.read_csv(source / "test_metrics.csv")
    logged["roc_auc"] = logged["roc_auc"] + 0.01
    logged.to_csv(v03_reports / "test_metrics.csv", index=False)
    outputs = workspace.outputs()

    run = run_script(tune_models, workspace, "--config", workspace.write_config(config, "config_tampered.yaml"),
                     "--evaluate-test", spy=SPIED)
    assert run.code == 1 and run.calls == []
    assert run.log.has("logged above; check the model file")
    log = pd.read_csv(config["evaluation"]["test_log"])                  # scored rows are never lost
    main, retune = workspace.finalists()
    assert len(log) == main + retune + 1 and log["stage"].iloc[-1] == "v0.4-reference"
    reports = Path(tc["report_dir"]) / DATASET
    assert not [p.name for p in reports.iterdir() if is_test_report(p.name)]
    models = Path(tc["model_dir"]) / DATASET
    assert not (models / "test_probabilities.npz").exists()
    assert json.loads((models / "best_random.json").read_text(encoding="utf-8"))["metrics"]["test"] is None
    assert json.loads((reports / "run_status.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert not (reports / "run_config.json").is_file()          # a failed run is not recorded as finished
    assert workspace.outputs() == outputs                       # the main folders are untouched


def test_test_run_refuses_when_there_is_nothing_to_compare_with(workspace, scripts, pipeline):
    """Cached models but no development-run validation results: the run must stop before scoring."""
    _, tune_models = scripts
    assert pipeline.runs["development"].code == 0
    base = workspace.root / "no_validation"
    config = copy.deepcopy(workspace.config)
    config["tuning"].update(model_dir=str(base / "models" / "v0.4"), report_dir=str(base / "metrics" / "v0.4"),
                            plot_dir=str(base / "plots" / "v0.4"))
    config["evaluation"]["test_log"] = str(base / "test_evaluations.csv")
    shutil.copytree(workspace.cache_dir, base / "models" / "v0.4" / "cache" / DATASET)

    run = run_script(tune_models, workspace, "--config", workspace.write_config(config, "config_no_validation.yaml"),
                     "--evaluate-test", spy=SPIED)
    assert run.code == 1 and run.calls == []
    assert run.log.has("No earlier validation results to compare with")
    assert not Path(config["evaluation"]["test_log"]).exists()          # nothing scored, nothing logged


def test_saved_tuned_model_predicts_from_a_raw_spectrum(workspace, pipeline):
    assert pipeline.runs["rescore"].code == 0
    tc = workspace.config["tuning"]
    bundle = load_bundle(workspace.folder("tuning", "model_dir") / f"best_{tc['split']}.joblib")
    chosen = workspace.chosen_family()
    assert bundle["model"] == f"tuned_{chosen}" and bundle["calibration"]["method"] == tc["calibration"]
    assert bundle["selection"]["part"] == "validation" and bundle["split"]["name"] == tc["split"]
    assert bundle["metrics"]["validation"]["roc_auc"] > 0.5 and bundle["metrics"]["test"] is not None

    X, meta, summary = load_dataset(workspace.data_dir)
    try:
        assert bundle["dataset"]["name"] == DATASET and bundle["n_features"] == summary["n_features"] == X.shape[1]
        assert bundle["feature_fingerprint"] == summary["feature_fingerprint"]
        row = int(load_splits(workspace.data_dir / "splits", meta)[tc["split"]].validation[0])
        path = resolve_relpath(workspace.config["paths"]["driams_root"], meta["spectrum_relpath"].iloc[row])
        result = predict_spectrum_file(bundle, path).to_dict()
        expected = float(predict_features(bundle, load_rows(X, np.array([row])))[0][0])
    finally:
        del X
        gc.collect()
    assert result["prediction"] in ("Resistant", "Susceptible")
    assert result["resistance_probability"] == pytest.approx(expected, abs=1e-4)
    assert (result["prediction"] == "Resistant") == (expected >= bundle["threshold"])
    assert result["model_version"] == bundle["model_version"]
    assert workspace.read_json("tuning", "inference_timing.json")["matches_stored_features"] is True


def test_outputs_hold_no_patient_identifiers(workspace, pipeline):
    files = [p for p in (workspace.root / "results").rglob("*") if p.suffix in (".csv", ".json")]
    files += list((workspace.root / "models").rglob("*.json"))
    assert len(files) > 20
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not any(secret in text for secret in workspace.secrets), path.name
    for run in pipeline.runs.values():
        assert not any(secret in run.stdout for secret in workspace.secrets)


# ------------------------------------------------------------------------------------------------ Version 0.5

def test_deep_test_run_refuses_to_fit_networks_that_are_not_cached(workspace, pipeline):
    """The same lockbox as Version 0.4: no cached network, no test scoring."""
    run = pipeline.runs["deep_test_without_cache"]
    assert run.code == 1
    assert run.log.has("is not cached for these settings/code")
    assert run.calls == []
    pd.testing.assert_frame_equal(pipeline.logs["deep_test_without_cache"], pipeline.logs["rescore"])
    assert pipeline.caches["deep_test_without_cache"] == {}
    assert not [n for n in pipeline.reports["deep_test_without_cache"] if is_test_report(n)]


def test_deep_development_run_trains_networks_and_leaves_the_test_parts_alone(workspace, pipeline):
    run = pipeline.runs["deep_development"]
    assert run.code == 0, run.log.messages
    dp = workspace.config["deep"]
    run_config = pipeline.run_configs["deep_development"]
    assert run_config["stage"] == "v0.5-deep" and run_config["status"] == "finished"
    assert run_config["test_parts_scored"] is False
    assert run_config["families"] == list(DEEP_FAMILIES)
    required = {f"search_{dp['split']}_{family}.csv" for family in DEEP_FAMILIES} | {
        "search_summary.csv", "validation_metrics.csv", "best_model_card.json", "inference_timing.json",
        "training_history.json", "run_config.json"}
    assert required <= set(pipeline.reports["deep_development"])
    assert not [name for name in pipeline.reports["deep_development"] if is_test_report(name)]
    pd.testing.assert_frame_equal(pipeline.logs["deep_development"], pipeline.logs["rescore"])

    # the patient-overlap check is not repeated here, so only the main experiment is searched
    fits = len(DEEP_FAMILIES) * len(dp["seeds"])
    assert sorted(run.calls) == sorted(["run_search"] * len(DEEP_FAMILIES) + ["fit_calibrated"] * fits)
    assert len(pipeline.caches["deep_development"]) == len(DEEP_FAMILIES) + fits
    reports = workspace.folder("deep", "report_dir")
    for family, n in DEEP_CANDIDATES.items():
        table = pd.read_csv(reports / f"search_{dp['split']}_{family}.csv")
        assert len(table) == n and table["cv_roc_auc_mean"].notna().all()
    validation = pd.read_csv(reports / "validation_metrics.csv")
    assert set(validation["experiment"]) == {dp["split"]}                  # no re-tuned experiments
    assert set(validation.loc[validation["model"] != "v0.3_prevalence", "model"]) == {"tuned_mlp", "tuned_cnn"}
    assert (validation["model"] == "v0.3_prevalence").sum() == 1           # carried over from Version 0.4
    card = workspace.read_json("deep", "best_model_card.json")
    # The stamp is the project version at the time the model was saved, not the section it came from.
    assert card["model_kind"] in DEEP_FAMILIES
    assert card["model_version"].startswith(f"v{workspace.config['project']['version']}-")

    # the loss curves come from the fitted networks, with one entry per family
    history = workspace.read_json("deep", "training_history.json")
    assert len(history) == len(DEEP_FAMILIES)
    for entry in history.values():
        assert entry["epochs"] == len(entry["train_loss"]) == len(entry["validation_loss"])
        assert entry["epochs"] <= DEEP_TRAINING["max_epochs"] and entry["parameters"] > 0
        assert 0 <= entry["best_epoch"] < entry["epochs"]
    curves = workspace.folder("deep", "plot_dir") / f"{DATASET}_{dp['split']}_training_curves.png"
    assert curves.stat().st_size > 0


def test_deep_test_run_reuses_the_cache_and_compares_with_version_04(workspace, pipeline):
    run = pipeline.runs["deep_test"]
    assert run.code == 0, run.log.messages
    assert run.calls == []                                                  # nothing refitted
    assert pipeline.caches["deep_test"] == pipeline.caches["deep_development"]
    assert "validation rows identical to the development run" in run.stdout
    assert "Version 0.4 model re-scored" in run.stdout
    assert pipeline.run_configs["deep_test"]["test_parts_scored"] is True

    dp = workspace.config["deep"]
    fits = len(DEEP_FAMILIES) * len(dp["seeds"])
    v04_card = workspace.read_json("tuning", "best_model_card.json")
    v04_name = f"v0.4_{v04_card['model']}"
    test = pd.read_csv(workspace.folder("deep", "report_dir") / "test_metrics.csv")
    rescored = test["reproduces_logged_auroc_within"].notna()
    assert len(test) == fits + 1 + 1                                        # networks, Version 0.4, no-skill
    assert rescored.sum() == 1 and test.loc[rescored, "model"].iloc[0] == v04_name
    assert float(test.loc[rescored, "reproduces_logged_auroc_within"].iloc[0]) <= 1e-4
    assert (test["model"] == "v0.3_prevalence").sum() == 1

    new = pipeline.logs["deep_test"].iloc[len(pipeline.logs["deep_development"]):]
    assert len(new) == fits + 1
    assert (new["stage"] == "v0.5-deep").sum() == fits
    assert new["stage"].iloc[-1] == "v0.5-reference" and new["model"].iloc[-1] == v04_name
    assert not set(new["split"]) & (NEVER_SCORED | set(workspace.config["evaluation"]["locked_test_splits"]))

    intervals = workspace.read_json("deep", "test_intervals.json")
    assert set(intervals) == {dp["split"]}
    assert set(intervals[dp["split"]]["differences_to_reference_model"]) == {"tuned_mlp", "tuned_cnn"}


def test_deep_section_refuses_a_comparison_path_that_does_not_match(workspace, scripts):
    """`deep.compare_with` is pre-registered; it must name the model this run is really compared with."""
    _, tune_models = scripts
    config = copy.deepcopy(workspace.config)
    config["deep"]["compare_with"] = str(Path(config["deep"]["compare_with"]).parent / "v0.2")
    path = workspace.write_config(config, "config_wrong_compare.yaml")
    run = run_script(tune_models, workspace, "--config", path, "--section", "deep", spy=SPIED)
    assert run.code == 1 and run.calls == []
    assert run.log.has("deep.compare_with")


# ------------------------------------------------------------------------------------------------ refusals

def test_partial_family_run_cannot_score_test_parts(workspace, scripts, pipeline):
    _, tune_models = scripts
    log, outputs = workspace.test_log(), workspace.outputs()
    run = run_script(tune_models, workspace, "--config", workspace.config_path, "--families", "logistic_regression",
                     "--evaluate-test", spy=SPIED)
    assert run.code == 1 and run.calls == []
    assert run.log.has("--evaluate-test needs the full search plan")
    pd.testing.assert_frame_equal(workspace.test_log(), log)
    assert workspace.outputs() == outputs


@pytest.mark.parametrize(("script", "locked"), [("tune_models", "random"), ("tune_models", "within_year"),
                                                ("train_baselines", "within_year")])
def test_locked_test_parts_are_refused(workspace, scripts, pipeline, script, locked):
    module = dict(zip(("train_baselines", "tune_models"), scripts, strict=True))[script]
    config = copy.deepcopy(workspace.config)
    config["evaluation"]["locked_test_splits"] = [*config["evaluation"]["locked_test_splits"], locked]
    path = workspace.write_config(config, f"config_locked_{locked}.yaml")
    # --allow-rescore: Version 0.4 rows are already logged; only the lock may stop this run
    extra, spy = (["--allow-rescore"], SPIED) if script == "tune_models" else ([], ())
    log, outputs = workspace.test_log(), workspace.outputs()
    run = run_script(module, workspace, "--config", path, "--evaluate-test", *extra, spy=spy)
    assert run.code == 1 and run.calls == []
    assert run.log.has("locked by the evaluation protocol")
    pd.testing.assert_frame_equal(workspace.test_log(), log)
    assert workspace.outputs() == outputs                    # nothing trained, saved or scored
    if script == "tune_models":                              # the refusal is recorded, not silent
        status = workspace.run_status()
        assert status["status"] == "failed" and "locked by the evaluation protocol" in status["error"]


# ------------------------------------------------------------------------------------------------ Version 0.6

@pytest.fixture(scope="module")
def explain_scripts() -> tuple[ModuleType, ModuleType]:
    return importlib.import_module("explain_model"), importlib.import_module("explain_tables")


@pytest.fixture(scope="module")
def explained(workspace, pipeline, explain_scripts) -> Run:
    """One explanation run on the model Version 0.4 saved, after the whole pipeline has run."""
    return run_script(explain_scripts[0], workspace, "--config", workspace.config_path)


def test_explaining_the_saved_model_never_touches_the_test_log(workspace, explained):
    """The one rule of Version 0.6: it describes a scored model, it does not score anything."""
    assert explained.code == 0, explained.log.messages
    record = workspace.read_json("explain", "run_config.json")
    assert record["status"] == "finished" and record["test_parts_scored"] is False
    assert record["rows_explained"] == "validation"
    # a leftover "running" status must always mean the run really stopped early
    assert workspace.read_json("explain", "run_status.json")["status"] == "finished"
    assert record["seconds"] > 0                               # the runtime is recorded, not just logged
    assert record["n_rows_explained"] == len(workspace.splits()[1][workspace.config["explain"]["split"]].validation)
    # the run records the hash it checked, and the file still has it
    log_path = Path(workspace.config["evaluation"]["test_log"])
    digest = hashlib.sha256(log_path.read_bytes()).hexdigest()
    assert record["test_log_sha256"] == digest


def test_the_explanation_run_writes_its_reports_and_a_figure(workspace, explained):
    assert explained.code == 0
    folder = workspace.folder("explain", "report_dir")
    written = {p.name for p in folder.iterdir()}
    assert {"permutation_blocks.csv", "uncertainty.json", "uncertainty_curve.csv", "zone_metrics.csv",
            "run_config.json", "run_status.json"} <= written, sorted(written)
    plots = list(Path(workspace.config["explain"]["plot_dir"]).glob("*.png"))
    assert any("uncertainty_zones" in p.name for p in plots), [p.name for p in plots]
    record = workspace.read_json("explain", "run_config.json")
    if record["exact_contributions"]:                 # a tree model: the full analysis ran
        assert {"regions.csv", "global_importance.csv", "region_contrast.csv", "examples.csv",
                "seed_stability.csv", "null_control.json"} <= written
    else:                                             # another model: the parts that need TreeSHAP are left out
        assert explained.log.has("no exact contributions")
        assert "regions.csv" not in written


def test_the_confidence_zones_are_fitted_on_validation_and_keep_the_models_cut_off(workspace, explained):
    assert explained.code == 0
    zones = workspace.read_json("explain", "uncertainty.json")
    card = workspace.read_json("tuning", "best_model_card.json")
    assert zones["threshold"] == card["threshold"]              # the cut-off itself never moves
    assert zones["fitted_on"] == "validation"
    assert zones["rule"]["target_npv"] == workspace.config["explain"]["uncertainty"]["target_npv"]
    for side in zones["sides"].values():
        if side["edge"] is None:                                # a zone that cannot meet its target
            assert side["exists"] is False
        else:
            assert side["coverage"] >= zones["rule"]["min_coverage"]
            assert side["achieved"] >= side["target"]
    metrics = pd.read_csv(workspace.folder("explain", "report_dir") / "zone_metrics.csv")
    counted = metrics[["n_zone_susceptible", "n_zone_uncertain", "n_zone_resistant"]].sum(axis=1)
    assert (counted == metrics["n"]).all()                      # every row lands in exactly one zone


def test_explanations_on_anything_but_validation_are_refused(workspace, explain_scripts):
    config = copy.deepcopy(workspace.config)
    config["explain"]["rows"] = "test"
    path = workspace.write_config(config, "config_explain_test_rows.yaml")
    log, outputs = workspace.test_log(), workspace.outputs()
    run = run_script(explain_scripts[0], workspace, "--config", path)
    assert run.code == 1
    assert run.log.has("validation part only")
    pd.testing.assert_frame_equal(workspace.test_log(), log)
    assert workspace.outputs() == outputs


def test_version_06_tables_are_generated_and_claim_no_protein_identity(workspace, explained, explain_scripts):
    assert explained.code == 0
    run = run_script(explain_scripts[1], workspace, "--config", workspace.config_path)
    assert run.code == 0, run.stdout
    text = (workspace.folder("explain", "report_dir") / "tables.md").read_text(encoding="utf-8")
    assert "generated by scripts/explain_tables.py" in text
    assert "No m/z region is given a protein or peptide identity" in text or \
           "regions.csv" not in {p.name for p in workspace.folder("explain", "report_dir").iterdir()}
    assert not any(secret in text for secret in workspace.secrets)


def test_version_06_outputs_hold_no_patient_identifiers(workspace, explained):
    folder = workspace.folder("explain", "report_dir")
    files = [p for p in folder.rglob("*") if p.suffix in (".csv", ".json", ".md")]
    assert files
    for path in files:
        assert not any(secret in path.read_text(encoding="utf-8") for secret in workspace.secrets), path.name
    assert not any(secret in explained.stdout for secret in workspace.secrets)


def test_a_locked_split_is_refused_before_anything_is_explained(workspace, explain_scripts):
    """Protocol point 5: a locked part is refused by the script too, not only by the test log."""
    config = copy.deepcopy(workspace.config)
    config["evaluation"]["locked_test_splits"] = [*config["evaluation"]["locked_test_splits"],
                                                  config["explain"]["split"]]
    path = workspace.write_config(config, "config_explain_locked.yaml")
    log, outputs = workspace.test_log(), workspace.outputs()
    run = run_script(explain_scripts[0], workspace, "--config", path)
    assert run.code == 1
    assert run.log.has("locked by the evaluation protocol")
    pd.testing.assert_frame_equal(workspace.test_log(), log)
    assert workspace.outputs() == outputs                    # nothing was written before the refusal


def test_rescaled_test_probabilities_are_refused_although_their_auroc_matches(workspace, explained,
                                                                              explain_scripts):
    """AUROC cannot police this on its own: it is unchanged by any monotone rescaling, while every zone
    number depends on the absolute probabilities. The Brier check is what catches it."""
    assert explained.code == 0
    path = workspace.folder("explain", "model_dir") / "test_probabilities.npz"
    if not path.is_file():
        pytest.skip("no stored test probabilities in this workspace")
    original = dict(np.load(path))
    card = workspace.read_json("tuning", "best_model_card.json")
    key = f"{workspace.config['explain']['split']}__{card['model']}__seed{card['seed']}"
    if key not in original:
        pytest.skip(f"{key} was not saved")
    try:
        tampered = {**original, key: np.sqrt(original[key])}     # same ranking, different probabilities
        np.savez(path, **tampered)
        before = roc_auc_score(*_labels_and_probabilities(workspace, original, key))
        after = roc_auc_score(*_labels_and_probabilities(workspace, tampered, key))
        assert after == pytest.approx(before, abs=1e-12)         # the rescaling is invisible to AUROC
        run = run_script(explain_scripts[0], workspace, "--config", workspace.config_path)
        assert run.code == 1
        assert run.log.has("Brier score")
    finally:
        np.savez(path, **original)
        run_script(explain_scripts[0], workspace, "--config", workspace.config_path)   # restore the reports


def _labels_and_probabilities(workspace, stored: dict[str, np.ndarray], key: str):
    meta, _ = workspace.splits()
    rows = stored[f"{workspace.config['explain']['split']}__rows"]
    return meta["label"].to_numpy().astype(int)[rows], stored[key]


# ------------------------------------------------------------------------------------------------ Version 0.7

@pytest.fixture(scope="module")
def generalise_script() -> ModuleType:
    return importlib.import_module("measure_generalisation")


@pytest.fixture(scope="module")
def generalisation_tables_script() -> ModuleType:
    return importlib.import_module("generalisation_tables")


@pytest.fixture(scope="module")
def generalised(workspace, pipeline, explained, generalise_script) -> Run:
    """One Version 0.7 run. It needs Version 0.4 (the setting and the saved model), and Version 0.6 for the
    confidence zones and the regions the shift diagnostic looks at."""
    assert pipeline.runs["test"].code == 0 and explained.code == 0
    return run_script(generalise_script, workspace, "--config", workspace.config_path)


def test_generalisation_refits_the_saved_setting_and_scores_the_released_parts(workspace, generalised):
    run = generalised
    assert run.code == 0, run.log.messages
    gn = workspace.config["generalisation"]
    reports = workspace.folder("generalisation", "report_dir")
    for name in ("test_metrics.csv", "validation_metrics.csv", "generalisation_gaps.csv",
                 "shift_diagnostic.csv", "run_config.json", "run_status.json"):
        assert (reports / name).is_file(), name

    record = workspace.read_json("generalisation", "run_config.json")
    assert record["status"] == "finished" and record["seconds"] > 0
    assert workspace.read_json("generalisation", "run_status.json")["status"] == "finished"
    # the setting is the one Version 0.4 chose, not a new search
    card = workspace.read_json("tuning", "best_model_card.json")
    assert record["setting"] == card["params"]
    assert record["reused_setting_from"] == card["model_version"]
    # and the refit was proved to reproduce that model before anything was scored
    faithful = record["refit_faithful"]
    assert faithful["difference"] <= 1e-9
    assert faithful["refit_validation_roc_auc"] == pytest.approx(
        card["metrics"]["validation"]["roc_auc"], abs=1e-9)
    assert run.log.has("refit proved faithful")

    test = pd.read_csv(reports / "test_metrics.csv")
    assert set(test["split"]) == set(gn["experiments"]) | set(gn["score_saved_model_on"])
    # every experiment is reported per site, never pooled
    meta, splits = workspace.splits()
    for name in gn["experiments"]:
        sites = sorted(set(meta["site"].to_numpy()[splits[name].test].tolist()))
        rows = test[(test["split"] == name) & (test["model"] != "prevalence")]
        assert sorted(set(rows["site"])) == sites, name


def test_the_released_test_parts_are_scored_once_per_model_and_only_appended(workspace, generalised,
                                                                             explained):
    """The log is append-only: Version 0.7 adds rows and changes none of the earlier ones."""
    assert generalised.code == 0
    log = workspace.test_log()
    v07 = log[log["stage"].str.startswith("v0.7")]
    assert len(v07) > 0
    # the previously locked parts appear for the first time here and nowhere earlier
    earlier = log[~log["stage"].str.startswith("v0.7")]
    assert not set(earlier["split"]) & NEVER_SCORED
    assert set(v07["split"]) & NEVER_SCORED
    # one row per (experiment, model, seed), never twice
    keys = v07[["experiment", "model", "seed"]]
    assert not keys.duplicated().any()
    record = workspace.read_json("generalisation", "run_config.json")
    assert record["n_test_evaluations"] == len(v07)


def test_the_saved_model_is_scored_on_the_external_part_but_never_on_the_temporal_one(workspace, generalised):
    """Protocol amendment 3, point 3: the saved model has seen most of the temporal test rows."""
    assert generalised.code == 0
    log = workspace.test_log()
    saved = log[log["stage"] == "v0.7-reference"]
    saved = saved[saved["experiment"].str.contains("saved_model")]
    assert len(saved) > 0
    assert set(saved["split"]) == set(workspace.config["generalisation"]["score_saved_model_on"])
    assert "temporal" not in set(saved["split"])
    # it keeps its own threshold, chosen on the source split's validation part
    card = workspace.read_json("tuning", "best_model_card.json")
    assert np.allclose(saved["threshold"].to_numpy(), float(card["threshold"]))


def test_scoring_the_saved_model_on_rows_it_trained_on_is_refused_as_leakage(workspace, generalise_script):
    """The guard, not the configuration, is what keeps this honest.

    The real case is the temporal part: it is date-separated from its *own* training part, but the saved
    model was trained on the `random` training part, which reaches into the later year. Scoring it there
    would measure memory, not generalisation.
    """
    gn = workspace.config["generalisation"]
    meta, splits = workspace.splits()
    source_train = splits[gn["source_split"]].train
    overlapping = [name for name in splits
                   if len(np.intersect1d(source_train, splits[name].test))
                   or len(np.intersect1d(np.unique(meta["group_id"].to_numpy()[source_train]),
                                         np.unique(meta["group_id"].to_numpy()[splits[name].test])))]
    overlapping = [n for n in overlapping if n != gn["source_split"]]
    if not overlapping:
        pytest.skip("no split in this workspace overlaps the saved model's training rows")
    config = copy.deepcopy(workspace.config)
    config["generalisation"]["score_saved_model_on"] = [overlapping[0]]
    path = workspace.write_config(config, "config_gn_leak.yaml")
    log = workspace.test_log()
    run = run_script(generalise_script, workspace, "--config", path)
    assert run.code == 1
    assert run.log.has("would be leakage")
    pd.testing.assert_frame_equal(workspace.test_log(), log)      # nothing was appended


def test_a_refit_that_does_not_reproduce_the_saved_model_stops_the_run(workspace, generalise_script):
    """If the reused setting is not the saved model's setting, no gap measured against it means anything."""
    config = copy.deepcopy(workspace.config)
    reports = Path(config["tuning"]["report_dir"]) / DATASET
    card_path = reports / "best_model_card.json"
    original = card_path.read_text(encoding="utf-8")
    card = json.loads(original)
    try:
        card["params"] = {**card["params"], "n_estimators": int(card["params"].get("n_estimators", 10)) + 7}
        card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
        log = workspace.test_log()
        run = run_script(generalise_script, workspace, "--config", workspace.config_path)
        assert run.code == 1
        assert run.log.has("The refit is not the saved model")
        assert run.log.has("Nothing was scored")
        pd.testing.assert_frame_equal(workspace.test_log(), log)
    finally:
        card_path.write_text(original, encoding="utf-8")


def test_a_locked_split_is_refused_before_anything_is_generalised(workspace, generalise_script):
    """Releasing the parts is a configuration act; putting one back must still stop the run."""
    config = copy.deepcopy(workspace.config)
    config["evaluation"]["locked_test_splits"] = [config["generalisation"]["experiments"][0]]
    path = workspace.write_config(config, "config_gn_locked.yaml")
    log = workspace.test_log()
    run = run_script(generalise_script, workspace, "--config", path)
    assert run.code == 1
    assert run.log.has("locked by the evaluation protocol")
    pd.testing.assert_frame_equal(workspace.test_log(), log)


def test_the_two_training_regimes_are_compared_on_the_same_rows(workspace, generalised):
    """The specification's 'train two sites, test a third' comparison must be paired, and must happen.

    The pair is found from the rows, not from experiment names, so this also checks that the comparison is
    not silently missing when a site is named differently.
    """
    assert generalised.code == 0
    path = workspace.folder("generalisation", "report_dir") / "two_sites_versus_one.json"
    assert path.is_file(), "the paired comparison is missing; the run warned instead of producing it"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    for paired in (loaded if isinstance(loaded, list) else [loaded]):
        assert paired["kind"] == "paired (same rows)"
        assert isinstance(paired["demonstrated"], bool)
        assert paired["low"] <= paired["high"]
        # the side called "more sites" really does train on more of them
        assert len(paired["more_sites"]["train_sites"]) > len(paired["fewer_sites"]["train_sites"])
        meta, splits = workspace.splits()
        assert paired["n"] == len(splits[paired["more_sites"]["experiment"]].test)


def test_the_generalisation_gaps_are_reported_with_intervals_and_an_honest_verdict(workspace, generalised):
    assert generalised.code == 0
    gaps = pd.read_csv(workspace.folder("generalisation", "report_dir") / "generalisation_gaps.csv")
    assert len(gaps) > 0
    assert set(gaps["kind"]) == {"unpaired (different rows)"}
    for row in gaps.itertuples(index=False):
        assert row.low <= row.high
        # a gap counts as shown only when the interval excludes zero
        assert bool(row.demonstrated) == bool(row.low > 0 or row.high < 0)


def test_the_confidence_zone_is_carried_across_unchanged_and_never_refitted(workspace, generalised):
    assert generalised.code == 0
    reports = workspace.folder("generalisation", "report_dir")
    path = reports / "zone_transfer.csv"
    if not path.is_file():
        pytest.skip("no Version 0.6 zones in this workspace")
    transfer = pd.read_csv(path)
    zones = workspace.read_json("explain", "uncertainty.json")
    if zones.get("lower") is None:
        pytest.skip("Version 0.6 found no susceptible zone here")
    # the edge is the Version 0.6 edge, everywhere: it is applied, not refitted
    assert np.allclose(transfer["edge"].to_numpy(), float(zones["lower"]))
    assert transfer["pure_site_test"].any()          # the saved model gives at least one clean test


def test_the_shift_diagnostic_uses_no_label_and_runs_after_the_scoring(workspace, generalised):
    assert generalised.code == 0
    shift = pd.read_csv(workspace.folder("generalisation", "report_dir") / "shift_diagnostic.csv")
    assert len(shift) > 0
    for column in ("median_abs_smd", "share_bins_abs_smd_over_half", "median_abs_smd_lowest_bins"):
        assert shift[column].notna().all() and (shift[column] >= 0).all()
    assert (shift["share_bins_abs_smd_over_half"] <= 1).all()
    # it describes inputs only: no metric that needs a label may appear here
    assert not {"roc_auc", "pr_auc", "brier", "sensitivity"} & set(shift.columns)


def test_version_07_outputs_hold_no_patient_identifiers(workspace, generalised):
    assert generalised.code == 0
    folder = workspace.folder("generalisation", "report_dir")
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix in (".csv", ".json", ".md")]
    assert files
    for path in files:
        assert not any(secret in path.read_text(encoding="utf-8") for secret in workspace.secrets), path.name
    assert not any(secret in generalised.stdout for secret in workspace.secrets)


def test_version_07_tables_are_generated_and_carry_the_required_caveats(workspace, generalised,
                                                                        generalisation_tables_script):
    """Every number in the tables comes from the saved reports; no metric is typed by hand."""
    assert generalised.code == 0
    run = run_script(generalisation_tables_script, workspace, "--config", workspace.config_path)
    assert run.code == 0, run.log.messages
    path = workspace.folder("generalisation", "report_dir") / "tables.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")

    assert "generated by scripts/generalisation_tables.py" in text
    # the caveats the protocol amendments require are generated, not typed, so they cannot be dropped
    assert "date-separated evaluation with incomplete patient linkage" in text
    assert "sample-level with unknown within-patient dependence" in text
    assert "do not show that the problem is solved" in text
    # a literal pipe inside a cell splits the row: every table row must have a constant column count
    for block in text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln.startswith("|")]
        if len(lines) < 2:
            continue
        widths = {ln.count("|") for ln in lines}
        assert len(widths) == 1, f"ragged table ({widths}):\n{block}"
    assert not any(secret in text for secret in workspace.secrets)


def test_version_07_draws_its_figures(workspace, generalised):
    """The figures are drawn from the written reports, and a figure failure never loses a scored run."""
    assert generalised.code == 0
    record = workspace.read_json("generalisation", "run_config.json")
    plots = list(Path(workspace.config["generalisation"]["plot_dir"]).glob("*.png"))
    assert plots, "no figure was written"
    assert sorted(p.name for p in plots) == sorted(record["plots"])
    for path in plots:
        assert path.stat().st_size > 5_000, f"{path.name} looks empty"


def test_already_scored_regimes_are_reused_from_the_log_not_scored_again(workspace, generalised):
    """The plan reports the earlier regimes beside the new ones; reusing them must add no test-log row."""
    assert generalised.code == 0
    reports = workspace.folder("generalisation", "report_dir")
    path = reports / "reused_from_log.csv"
    assert path.is_file(), "the reused reference rows are missing"
    reused = pd.read_csv(path)
    gn = workspace.config["generalisation"]
    assert set(reused["split"]) <= set(gn["reuse_logged"])
    # every reused row must already exist in the log under an earlier stage, with the same numbers
    log = workspace.test_log()
    for r in reused.itertuples(index=False):
        match = log[(log["experiment"] == r.experiment) & (log["model"] == r.model) & (log["seed"] == r.seed)]
        assert len(match) >= 1, f"{r.experiment} is not in the log at all"
        assert not str(r.logged_stage).startswith("v0.7"), "a reused row must come from an earlier version"
        assert float(match["roc_auc"].iloc[0]) == pytest.approx(float(r.roc_auc), abs=1e-12)
    # and none of those splits may appear among this version's own scorings
    v07 = log[log["stage"].str.startswith("v0.7")]
    assert not set(v07["split"]) & set(gn["reuse_logged"])


def test_a_split_cannot_be_both_reused_and_rescored(workspace, generalise_script):
    """Asking for both would spend an already-scored test part a second time."""
    config = copy.deepcopy(workspace.config)
    gn = config["generalisation"]
    gn["experiments"] = [*gn["experiments"], gn["reuse_logged"][0]]
    path = workspace.write_config(config, "config_gn_double.yaml")
    log = workspace.test_log()
    run = run_script(generalise_script, workspace, "--config", path)
    assert run.code == 1
    assert run.log.has("would spend them twice")
    pd.testing.assert_frame_equal(workspace.test_log(), log)
