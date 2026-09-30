"""Version 1.1 - the report: the tables, and the one comparison no earlier script makes.

Run from the project root with the virtual environment active, after both Version 1.1 test runs:

    python scripts/tune_models.py --section second_antibiotic --dataset ecoli_ceftriaxone --evaluate-test
    python scripts/measure_generalisation.py --section second_antibiotic_generalisation --dataset ecoli_ceftriaxone
    python scripts/second_antibiotic_report.py

**It scores nothing and fits nothing.** It reads the reports those two runs wrote, the append-only test log
and the stored test probabilities, and writes (under config.yaml -> second_antibiotic.report_dir/<dataset>):

- antibiotic_comparison.json  endpoint 4 of docs/v1.1_ceftriaxone_plan.md
- tables.md                   every table of the Version 1.1 section, generated, never typed

Endpoint 4 compares the two antibiotics on the same 856 `random` test spectra: arm T's ceftriaxone AUROC
minus the saved Version 0.4 model's ciprofloxacin AUROC. The ciprofloxacin side is **not re-scored**: it is
the stored Version 0.4 test probabilities, used only after they cover exactly that test part's rows and
reproduce the logged AUROC *and* Brier score (protocol amendment 2, point 3; amendment 8, point 3). The same
check is applied to the ceftriaxone side. The interval is a paired patient-group bootstrap in which both
labels are scored on every resample, drawn exactly as src/evaluate.py draws them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import DataError  # noqa: E402
from src.dataset import load_dataset, sample_keys  # noqa: E402
from src.evaluate import EvaluationError, group_resample_indices, interval  # noqa: E402
from src.splits import SplitError, load_splits  # noqa: E402
from src.utils import ConfigError, get_logger, load_config, project_path, show_path  # noqa: E402

log = get_logger("second_antibiotic")

MODEL = "tuned_lightgbm"             # arm T, and the saved Version 0.4 model: same family, same name
FIXED = "tuned_lightgbm_cipro_setting"
SEED = 42
TOLERANCE = 1e-6                     # the log stores six decimals
# Checked on 2026-09-30 before any Version 1.1 result was written (the plan's endpoint 2): the abstract
# (NCBI E-utilities, PMID 35013613) gives "areas under the receiver operating characteristic curve of 0.80,
# 0.74 and 0.74" for S. aureus, E. coli and K. pneumoniae, and the authors' code
# (BorgwardtLab/maldi_amr, plot_sliding_window_validation_major_scenarios.py) defines those three scenarios as
# DRIAMS-A S. aureus + oxacillin, E. coli + ceftriaxone and K. pneumoniae + ceftriaxone.
PUBLISHED = {"roc_auc": 0.74, "citation": "Weis C, Cuénod A, Rieck B, et al. Direct antimicrobial resistance "
             "prediction from clinical MALDI-TOF mass spectra using machine learning. Nat Med 2022;28:164-174. "
             "doi:10.1038/s41591-021-01619-9",
             "scenario": "E. coli + ceftriaxone at DRIAMS-A (LightGBM)"}


# --- the stored probabilities, verified before anything is derived from them ----------------------------------

def verified_probabilities(npz_path: Path, log_rows: pd.DataFrame, *, dataset: str, split: str, model: str,
                           seed: int, rows: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stored test probabilities, refused unless they are exactly what was scored and logged.

    The rows must be the test part's own rows in its own order, and the probabilities must reproduce both the
    logged AUROC and the logged Brier score: AUROC alone survives any monotone rescaling. When a model was
    scored more than once (an additional run), the latest logged row is the one checked, as in Version 0.7.
    """
    if not npz_path.is_file():
        raise EvaluationError(f"{show_path(npz_path)} does not exist; refusing to re-score instead.")
    stored = np.load(npz_path)
    key = f"{split}__{model}__seed{seed}"
    if key not in stored.files or f"{split}__rows" not in stored.files:
        raise EvaluationError(f"{npz_path.name} holds no {key}; nothing to compare.")
    if not np.array_equal(stored[f"{split}__rows"], rows):
        raise EvaluationError(f"The rows stored in {npz_path.name} are not the {split} test part of {dataset} "
                              "(order included); refusing to derive anything from them.")
    prob = np.asarray(stored[key], dtype=np.float64)
    logged = log_rows[(log_rows["dataset"] == dataset) & (log_rows["split"] == split)
                      & (log_rows["experiment"] == split) & (log_rows["model"] == model)
                      & (log_rows["seed"] == seed)]
    if logged.empty:
        raise EvaluationError(f"The test log has no {split} row for {model} seed {seed} on {dataset}, so the "
                              "stored probabilities cannot be checked against anything.")
    got = {"roc_auc": float(roc_auc_score(y, prob)), "brier": float(np.mean((prob - y) ** 2))}
    for metric, value in got.items():
        want = float(logged[metric].iloc[-1])
        if abs(value - want) > TOLERANCE:
            raise EvaluationError(f"The stored {dataset} probabilities give {metric} {value:.12f} but {want:.12f} "
                                  "was logged. They are not the scored predictions; refusing to use them.")
    return prob


def paired_label_difference(y_a: np.ndarray, p_a: np.ndarray, y_b: np.ndarray, p_b: np.ndarray,
                            groups: np.ndarray, *, resamples: int, seed: int, level: float) -> dict[str, Any]:
    """AUROC(a) minus AUROC(b) on the same spectra with two different labels, paired over patient groups.

    src.evaluate.bootstrap takes one label vector, and it skips a resample that holds a single class, which
    depends on the labels: called twice, the two antibiotics could skip different resamples and fall out of
    step. Here each resample is drawn once, exactly as that function draws it (same group order, same random
    stream), both labels are scored on it, and it is skipped only if either label is single-class there.
    """
    groups = np.asarray(groups)
    if not (y_a.size == p_a.size == y_b.size == p_b.size == groups.size):
        raise EvaluationError("Both labels, both probability vectors and the groups must cover the same spectra.")
    order = np.argsort(groups, kind="stable")
    _, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    a, b, skipped = [], [], 0
    for _ in range(resamples):
        idx = group_resample_indices(order, starts, counts, rng.integers(0, starts.size, starts.size))
        ya, yb = y_a[idx], y_b[idx]
        if ya.min() == ya.max() or yb.min() == yb.max():
            skipped += 1
            continue
        a.append(roc_auc_score(ya, p_a[idx]))
        b.append(roc_auc_score(yb, p_b[idx]))
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    point_a, point_b = float(roc_auc_score(y_a, p_a)), float(roc_auc_score(y_b, p_b))
    low, high = interval(a_arr - b_arr, level)
    return {"a_roc_auc": point_a, "a_interval": list(interval(a_arr, level)),
            "b_roc_auc": point_b, "b_interval": list(interval(b_arr, level)),
            "difference": point_a - point_b, "low": low, "high": high,
            "demonstrated": bool(low > 0 or high < 0), "resamples": resamples, "used": resamples - skipped,
            "skipped": skipped, "level": level, "seed": seed, "kind": "paired (same spectra, two labels)"}


def compare_antibiotics(config: dict[str, Any], dataset: str, reference: str) -> dict[str, Any]:
    """Endpoint 4: arm T on ceftriaxone against the saved Version 0.4 model on ciprofloxacin, same spectra."""
    out_root = project_path(config["dataset"]["output_dir"])
    split = str(config["second_antibiotic"]["split"])
    X_a, meta_a, _ = load_dataset(out_root / dataset)
    X_b, meta_b, _ = load_dataset(out_root / reference)
    del X_a, X_b                                        # labels and keys only; no spectrum is read
    rows_a = np.asarray(load_splits(out_root / dataset / "splits", meta_a)[split].test, dtype=np.int64)
    rows_b = np.asarray(load_splits(out_root / reference / "splits", meta_b)[split].test, dtype=np.int64)
    keys_a, keys_b = sample_keys(meta_a)[rows_a], sample_keys(meta_b)[rows_b]
    if set(keys_a) != set(keys_b):
        raise SplitError(f"The {split} test parts of {dataset} and {reference} are not the same spectra, so a "
                         "paired comparison would be wrong.")
    position = {key: i for i, key in enumerate(keys_b)}
    b_in_a_order = np.array([position[key] for key in keys_a], dtype=np.int64)

    test_log = pd.read_csv(project_path(config["evaluation"]["test_log"]))
    y_a = meta_a["label"].to_numpy().astype(np.int64)[rows_a]
    y_b = meta_b["label"].to_numpy().astype(np.int64)[rows_b]
    p_a = verified_probabilities(project_path(config["second_antibiotic"]["model_dir"]) / dataset /
                                 "test_probabilities.npz", test_log, dataset=dataset, split=split, model=MODEL,
                                 seed=SEED, rows=rows_a, y=y_a)
    p_b = verified_probabilities(project_path(config["tuning"]["model_dir"]) / reference /
                                 "test_probabilities.npz", test_log, dataset=reference, split=split, model=MODEL,
                                 seed=SEED, rows=rows_b, y=y_b)
    b = config["evaluation"]["bootstrap"]
    result = paired_label_difference(y_a, p_a, y_b[b_in_a_order], p_b[b_in_a_order],
                                     meta_a["group_id"].to_numpy()[rows_a], resamples=int(b["resamples"]),
                                     seed=int(b["seed"]), level=float(b["level"]))
    return {"comparison": f"{dataset} (arm T) minus {reference} (saved Version 0.4 model), {split} test part",
            "a": {"dataset": dataset, "model": MODEL, "seed": SEED, "n": int(rows_a.size),
                  "n_resistant": int(y_a.sum())},
            "b": {"dataset": reference, "model": MODEL, "seed": SEED, "n": int(rows_b.size),
                  "n_resistant": int(y_b.sum()), "source": "stored Version 0.4 test probabilities, not re-scored"},
            "same_spectra": True, **result,
            "reading": "how hard the two labels are for one method on the same spectra; not a comparison of models"}


def split_counts(config: dict[str, Any], dataset: str) -> pd.DataFrame:
    """Samples and resistant spectra per split part, counted from the dataset's own split files and labels."""
    folder = project_path(config["dataset"]["output_dir"]) / dataset
    X, meta, _ = load_dataset(folder)
    del X
    y = meta["label"].to_numpy().astype(np.int64)
    return pd.DataFrame([{"split": name, "part": part, "samples": int(idx.size), "resistant": int(y[idx].sum())}
                         for name, s in load_splits(folder / "splits", meta).items()
                         for part, idx in s.parts().items()])


def left_out(config: dict[str, Any], dataset: str, reference: str) -> dict[str, int]:
    """The dataset's spectra the reference dataset does not have (in no split), against the ones it does."""
    out_root = project_path(config["dataset"]["output_dir"])
    X_a, meta_a, _ = load_dataset(out_root / dataset)
    X_b, meta_b, _ = load_dataset(out_root / reference)
    del X_a, X_b
    outside = ~np.isin(sample_keys(meta_a), sample_keys(meta_b))
    y = meta_a["label"].to_numpy().astype(np.int64)
    return {"left_out": int(outside.sum()), "left_out_resistant": int(y[outside].sum()),
            "cohort": int((~outside).sum()), "cohort_resistant": int(y[~outside].sum())}


# --- tables -----------------------------------------------------------------------------------------------------

def ci(estimate: float, low: float, high: float) -> str:
    return f"{estimate:.3f} ({low:.3f}–{high:.3f})"


def signed(estimate: float, low: float, high: float) -> str:
    """A difference, which may be negative: signs shown, bracketed, so no minus sign meets a dash."""
    return f"{estimate:+.3f} [{low:+.3f}, {high:+.3f}]"


def md_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    head = list(rows[0])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(r[h]) for h in head) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def verdict(low: float, high: float) -> str:
    return "demonstrated" if (low > 0 or high < 0) else "not demonstrated (interval includes 0)"


def build_tables(config: dict[str, Any], dataset: str, reference: str, comparison: dict[str, Any]) -> str:
    sa, sg = config["second_antibiotic"], config["second_antibiotic_generalisation"]
    report = project_path(sa["report_dir"]) / dataset
    general = project_path(sg["report_dir"]) / dataset
    split = str(sa["split"])
    splits = split_counts(config, dataset)
    test = pd.read_csv(report / "test_metrics.csv")
    intervals = json.loads((report / "test_intervals.json").read_text(encoding="utf-8"))[split]
    seeds = pd.read_csv(report / "seed_variation.csv")
    g_test = pd.read_csv(general / "test_metrics.csv")
    gaps = pd.read_csv(general / "generalisation_gaps.csv")
    level = int(round(100 * float(config["evaluation"]["bootstrap"]["level"])))
    rule = config["pair_selection"]
    sites = ", ".join(config["splits"][split]["sites"])
    lo = left_out(config, dataset, reference)

    random_parts = splits[splits["split"] == split]
    dev_resistant = int(random_parts["resistant"].sum())
    validation_resistant = int(random_parts.loc[random_parts["part"] == "validation", "resistant"].iloc[0])
    out: list[str] = [
        f"# Version 1.1 tables: *E. coli* + ceftriaxone (`{dataset}`)",
        "",
        "Generated by `scripts/second_antibiotic_report.py` from the reports of the Version 1.1 runs, the "
        "append-only test log and the stored test probabilities; nothing here is typed by hand, and nothing was "
        "scored to make it. Plan: `docs/v1.1_ceftriaxone_plan.md` (protocol amendment 8). Research prototype, "
        "not a diagnostic: these are statements about a laboratory label, not advice about any patient.",
        "",
        "## What the reader must know first",
        "",
        f"- **The pair does not meet the project's pair-selection rule on its cohort:** {dev_resistant} resistant "
        f"spectra at {sites} against a minimum of {rule['min_resistant_dev']}. It was run as the benchmark "
        "antibiotic named in Version 0.1, not as a pair the rule selected.",
        f"- **The cohort is isolates with both a ciprofloxacin and a ceftriaxone result.** {lo['left_out']} "
        f"isolates with only a ceftriaxone result are in no split: {lo['left_out_resistant']} of them resistant "
        f"({100 * lo['left_out_resistant'] / max(lo['left_out'], 1):.1f} %, against "
        f"{100 * lo['cohort_resistant'] / lo['cohort']:.1f} % in the cohort). Nothing here is claimed for them.",
        "- **These spectra were scored before, for ciprofloxacin.** Their ceftriaxone labels had never been used "
        "by any model, threshold or choice (amendment 8, point 2).",
        f"- **The threshold is coarse:** it was set on a validation part with {validation_resistant} resistant "
        f"spectra, so one spectrum moves validation sensitivity by {100 / validation_resistant:.1f} points. "
        "Threshold-dependent numbers are secondary.",
        f"- Intervals are {level} % patient-group bootstrap intervals. \"Not demonstrated\" is never \"no "
        "difference\".",
        "",
        "## Split sizes (n / resistant)",
        "",
    ]
    rows = []
    for name in (split, "temporal", "external"):
        part = splits[splits["split"] == name].set_index("part")
        rows.append({"Split": f"`{name}`", **{p.title(): f"{int(part.loc[p, 'samples']):,} / "
                                                         f"{int(part.loc[p, 'resistant'])}"
                                              for p in ("train", "validation", "test")}})
    out += [md_table(rows), ""]

    iv, diff = intervals["intervals"], intervals["differences_to_reference"]
    t_row = test[(test["model"] == MODEL) & (test["seed"] == SEED)].iloc[0]
    better = iv[MODEL]["roc_auc"]["low"] > 0.5
    out += [f"## Primary endpoint: arm T on the `{split}` test part (seed {SEED})", ""]
    rows = []
    for model, label in ((MODEL, "T: Version 0.4 search, rerun on ceftriaxone"),
                         (FIXED, "F: Version 0.4 ciprofloxacin setting, refitted")):
        r = test[(test["model"] == model) & (test["seed"] == SEED)].iloc[0]
        m = iv[model]
        rows.append({"Arm": label, "n / resistant": f"{int(r['n'])} / {int(r['n_resistant'])}",
                     "AUROC": ci(m["roc_auc"]["estimate"], m["roc_auc"]["low"], m["roc_auc"]["high"]),
                     "PR-AUC": ci(m["pr_auc"]["estimate"], m["pr_auc"]["low"], m["pr_auc"]["high"]),
                     "Brier": f"{r['brier']:.3f}", "Threshold": f"{r['threshold']:.3f}",
                     "Sensitivity*": f"{r['sensitivity']:.3f}", "Specificity*": f"{r['specificity']:.3f}"})
    out += [md_table(rows),
            f"\\* at the validation threshold; secondary (see above). PR-AUC's no-skill level is the test "
            f"prevalence, {t_row['n_resistant'] / t_row['n']:.3f}.",
            "",
            f"**Better than chance** (the plan's rule: the AUROC interval's lower bound above 0.5): "
            f"**{'yes' if better else 'not demonstrated'}** — lower bound {iv[MODEL]['roc_auc']['low']:.3f}.",
            ""]
    d = diff[FIXED]["roc_auc"]
    out += ["## T against F: does a second antibiotic need its own search?", "",
            f"AUROC, T minus F, paired on the same {int(t_row['n'])} spectra: "
            f"**{signed(d['estimate'], d['low'], d['high'])}** — {verdict(d['low'], d['high'])}. An interval "
            "containing 0 is not \"equivalent\".", ""]

    s = seeds[seeds["model"] == MODEL].iloc[0]
    out += ["## Seed variation of arm T on the test part", "",
            f"{int(s['seeds'])} seeds: AUROC mean {s['roc_auc_mean']:.3f} (range {s['roc_auc_min']:.3f}–"
            f"{s['roc_auc_max']:.3f}); PR-AUC mean {s['pr_auc_mean']:.3f} (range {s['pr_auc_min']:.3f}–"
            f"{s['pr_auc_max']:.3f}).", ""]

    c = comparison
    out += ["## Ceftriaxone against ciprofloxacin on the same spectra (endpoint 4)", "",
            md_table([{"Label": "ceftriaxone (arm T)", "n / resistant": f"{c['a']['n']} / {c['a']['n_resistant']}",
                       "AUROC": ci(c["a_roc_auc"], *c["a_interval"])},
                      {"Label": "ciprofloxacin (saved Version 0.4 model, stored, not re-scored)",
                       "n / resistant": f"{c['b']['n']} / {c['b']['n_resistant']}",
                       "AUROC": ci(c["b_roc_auc"], *c["b_interval"])}]),
            f"Difference, ceftriaxone minus ciprofloxacin, paired over the same spectra: "
            f"**{signed(c['difference'], c['low'], c['high'])}** — {verdict(c['low'], c['high'])}. This describes how "
            "hard the two labels are for one method; it compares no models.", ""]

    ref = PUBLISHED["roc_auc"]
    low, high = iv[MODEL]["roc_auc"]["low"], iv[MODEL]["roc_auc"]["high"]
    where = "inside" if low <= ref <= high else ("above" if ref > high else "below")
    out += ["## The published reference point (descriptive only)", "",
            f"Weis et al. (2022) report AUROC {ref:.2f} for {PUBLISHED['scenario']}. It lies **{where}** arm T's "
            f"interval ({low:.3f}–{high:.3f}). Their cohort, split and exclusions differ from this project's, so "
            "this is a reference point, not a comparison: nothing here \"matches\" or \"beats\" it.",
            "",
            f"Reference: {PUBLISHED['citation']}. Checked against the abstract (PMID 35013613) and the authors' "
            "code (BorgwardtLab/maldi_amr) before this table was first generated.",
            ""]

    out += ["## Where the model was not trained: `temporal` and `external`", "",
            "Arm T's setting refitted unchanged on each split's own training part, thresholded on its own "
            "validation part (seed 42 shown; the refit was first proved to reproduce T's validation AUROC).", ""]
    rows = []
    for g in gaps.itertuples(index=False):
        part = g.comparison.split(" minus ", 1)[1]
        seeds_part = g_test[(g_test["experiment"] == part) & (g_test["model"] == MODEL)]
        rows.append({"Test part": f"`{part}`", "n / resistant": f"{g.n} / {g.n_resistant}",
                     "AUROC (seed 42)": ci(g.other_roc_auc, g.other_low, g.other_high),
                     "Seeds: mean (range)": f"{seeds_part['roc_auc'].mean():.3f} ({seeds_part['roc_auc'].min():.3f}"
                                            f"–{seeds_part['roc_auc'].max():.3f})",
                     f"Gap: `{split}` minus this": signed(g.gap, g.low, g.high),
                     "Gap": "demonstrated" if g.demonstrated else "not demonstrated"})
    out += [md_table(rows),
            "Gaps use independent (unpaired) bootstraps because the test parts are different spectra; each interval "
            "is wide, and \"not demonstrated\" is not \"no gap\". No conclusion rests on DRIAMS-B alone.", ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Version 1.1: generated tables and the antibiotic comparison.")
    parser.add_argument("--dataset", default="ecoli_ceftriaxone")
    parser.add_argument("--reference-dataset", default=None, help="default: dataset.name (the primary)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        reference = args.reference_dataset or config["dataset"]["name"]
        report = project_path(config["second_antibiotic"]["report_dir"]) / args.dataset
        comparison = compare_antibiotics(config, args.dataset, reference)
        (report / "antibiotic_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        log.info("ceftriaxone minus ciprofloxacin AUROC, same %d spectra: %.4f [%.4f, %.4f]", comparison["a"]["n"],
                 comparison["difference"], comparison["low"], comparison["high"])
        (report / "tables.md").write_text(build_tables(config, args.dataset, reference, comparison),
                                          encoding="utf-8", newline="\n")
        log.info("wrote %s", show_path(report / "tables.md"))
        return 0
    except (DataError, ConfigError, SplitError, EvaluationError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
