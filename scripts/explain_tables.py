"""Markdown tables of the Version 0.6 explanation and confidence zones, generated from the saved reports.

Run from the project root with the virtual environment active, after scripts/explain_model.py:

    python scripts/explain_tables.py

Writes <explain.report_dir>/<dataset>/tables.md and prints it. Nothing here computes a result: every
number comes from the files the explanation run wrote, as the evaluation protocol requires.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tables import md_table, number  # noqa: E402
from src.utils import ConfigError, load_config, project_path  # noqa: E402

# Printed under every region table. The project has no MS/MS confirmation and no independent panel, so a
# region is an m/z interval and nothing more; this sentence is generated, not typed, so it cannot be lost.
NO_IDENTITY = ("No m/z region is given a protein or peptide identity: this project has no MS/MS "
               "confirmation and no independent panel, so a region is named by its m/z interval only.")


def mz_range(row: Any) -> str:
    return f"{row.mz_start:,.0f} – {row.mz_end:,.0f}"


def read(folder: Path, name: str) -> pd.DataFrame | None:
    path = folder / name
    return pd.read_csv(path) if path.is_file() else None


def read_json(folder: Path, name: str) -> dict[str, Any] | None:
    path = folder / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def regions_section(regions: pd.DataFrame, contrast: pd.DataFrame | None) -> str:
    """The reported regions: how much they moved predictions, which way, and the model-free difference."""
    by_rank = {int(r.rank): r for r in contrast.itertuples()} if contrast is not None else {}
    rows = []
    for r in regions.itertuples():
        reference = by_rank.get(int(r.rank))
        rows.append({"#": int(r.rank), "m/z region": mz_range(r), "Bins": int(r.n_columns),
                     "Mean absolute contribution": number(float(r.total_abs), 4),
                     "Higher intensity points to": r.towards,
                     "Value–contribution correlation": number(float(r.value_correlation), 2),
                     "R vs S difference (SD)": number(float(reference.standardised_difference), 2)
                     if reference is not None else "–"})
    return md_table(rows)


def permutation_section(blocks: pd.DataFrame, keep: int = 10) -> str:
    rows = [{"m/z block": f"{r.mz_start:,.0f} – {r.mz_end:,.0f}",
             "AUROC lost when permuted": f"{r.auroc_drop_mean:+.4f} ± {r.auroc_drop_sd:.4f}"}
            for r in blocks.head(keep).itertuples()]
    return md_table(rows)


def agreement_section(run_config: dict[str, Any], pairs: pd.DataFrame | None) -> str:
    rows = []
    method = run_config.get("method_agreement") or {}
    if method:
        rows.append({"Comparison": "TreeSHAP against permutation importance (same model)",
                     "Spearman": number(method.get("spearman"), 2),
                     "Top-20 blocks shared": f"{method.get('overlap', 0)} of {method.get('top_k', 0)}"})
    cross = run_config.get("cross_model_agreement") or {}
    if cross:
        rows.append({"Comparison": f"Permutation importance against {cross.get('model', 'the other model')}",
                     "Spearman": number(cross.get("spearman"), 2),
                     "Top-20 blocks shared": f"{cross.get('overlap', 0)} of {cross.get('top_k', 0)}"})
    if pairs is not None and not pairs.empty:
        rows.append({"Comparison": f"Between the {len(pairs)} seed pairs of the same setting (mean)",
                     "Spearman": number(float(pairs["spearman"].mean()), 2),
                     "Top-20 blocks shared": f"{pairs['overlap'].mean():.1f} of {int(pairs['top_k'].iloc[0])}"})
    return md_table(rows)


def control_section(control: dict[str, Any] | None) -> str:
    if not control:
        return ""
    rows = [{"Model": "The fitted model, strongest region",
             "Mean absolute contribution": number(control.get("real_strongest_region"), 4)},
            {"Model": f"Shuffled training labels, same setting and size ({control.get('n_train', 0):,} rows)",
             "Mean absolute contribution": number(control.get("null_top_region_equivalent"), 4)}]
    return md_table(rows)


def zone_section(zones: dict[str, Any] | None, metrics: pd.DataFrame | None,
                 intervals: dict[str, Any] | None) -> str:
    """What the three-way output does, and honestly what it fails to do."""
    if not zones:
        return ""
    lines = []
    for side, label in (("susceptible", "High-confidence susceptible"), ("resistant", "High-confidence resistant")):
        info = (zones.get("sides") or {}).get(side) or {}
        target = info.get("target")
        if info.get("exists"):
            lines.append(f"- **{label}:** probability {'below' if side == 'susceptible' else 'above'} "
                         f"{info['edge']:.4f}. It covers {100 * info['coverage']:.1f} % of the validation part "
                         f"and is correct for {100 * info['achieved']:.1f} % of them (target "
                         f"{100 * target:.0f} %).")
        else:
            best, coverage = info.get("best_achieved"), info.get("best_coverage") or 0.0
            lines.append(f"- **{label}: does not exist.** No cut reaches the pre-registered "
                         f"{100 * target:.0f} % at the required coverage. The most any cut reaches is "
                         f"{100 * best:.1f} % (covering {100 * coverage:.1f} % of the validation part), so "
                         f"every one of those isolates is reported as uncertain instead.")
    rows = []
    if metrics is not None:
        for r in metrics.itertuples():
            rows.append({"Part": r.part, "n": f"{int(r.n):,}",
                         "High-confidence susceptible": f"{int(r.n_zone_susceptible):,} "
                                                        f"({100 * r.share_susceptible:.1f} %)",
                         "Uncertain": f"{int(r.n_zone_uncertain):,} ({100 * r.share_uncertain:.1f} %)",
                         "High-confidence resistant": f"{int(r.n_zone_resistant):,} "
                                                      f"({100 * r.share_resistant:.1f} %)",
                         "Correct among confident": number(float(r.accuracy_confident), 3)})
    table = md_table(rows)
    if intervals:
        share = intervals.get("confident_share") or {}
        if share:
            table += (f"\n\nTest-part interval for the confident share: "
                      f"{share['value']:.3f} [{share['low']:.3f}, {share['high']:.3f}] "
                      f"(patient-group bootstrap; derived from the stored test predictions, not a new scoring).")
    return "\n\n".join(x for x in ["\n".join(lines), table] if x)


def examples_section(examples: pd.DataFrame | None, keep_regions: int = 3) -> str:
    if examples is None or examples.empty:
        return ""
    rows = []
    for (example, position), group in examples.groupby(["example", "validation_position"], sort=False):
        first = group.iloc[0]
        regions = "; ".join(f"{r.mz_start:,.0f}–{r.mz_end:,.0f} ({r.contribution:+.2f})"
                            for r in group.head(keep_regions).itertuples())
        rows.append({"Validation spectrum": f"{example} (row {int(position)})",
                     "Probability": number(float(first.resistance_probability), 3),
                     "True label": "resistant" if int(first.true_label) == 1 else "susceptible",
                     "Output": first.output, "Strongest regions (signed contribution)": regions})
    return md_table(rows)


def build_tables(report_dir: Path) -> str:
    regions = read(report_dir, "regions.csv")
    run_config = read_json(report_dir, "run_config.json")
    if regions is None or run_config is None:
        raise ConfigError(f"{report_dir} has no regions.csv / run_config.json. Run scripts/explain_model.py "
                          "first.")
    zones = read_json(report_dir, "uncertainty.json")
    blocks = read(report_dir, "permutation_blocks.csv")
    try:
        shown = report_dir.resolve().relative_to(project_path(".").resolve()).as_posix()
    except ValueError:                                  # a report folder outside the project (tests)
        shown = report_dir.as_posix()
    parts = [f"<!-- generated by scripts/explain_tables.py from {shown} -->",
             f"**What is explained:** {run_config.get('explained_model')}, on the "
             f"{run_config.get('n_rows_explained', 0):,} rows of the "
             f"`{run_config.get('split')}` validation part. No test row was scored.",
             "**Influential m/z regions (exact TreeSHAP)**", regions_section(regions, read(report_dir,
                                                                                           "region_contrast.csv")),
             NO_IDENTITY]
    if blocks is not None:
        parts += ["**What the model's AUROC depends on (block permutation importance)**",
                  permutation_section(blocks)]
    agreement = agreement_section(run_config, read(report_dir, "seed_agreement.csv"))
    if agreement:
        parts += ["**Do the methods, the seeds and the two model families agree?**", agreement]
    control = control_section(run_config.get("null_control"))
    if control:
        parts += ["**Against chance (the same setting refitted on shuffled labels)**", control]
    zone_text = zone_section(zones, read(report_dir, "zone_metrics.csv"), read_json(report_dir,
                                                                                    "zone_intervals.json"))
    if zone_text:
        parts += ["**Confidence zones**", zone_text]
    examples = examples_section(read(report_dir, "examples.csv"))
    if examples:
        parts += ["**Individual explanations**", examples]
    return "\n\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Version 0.6 result tables from the saved reports.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if "explain" not in config:
            raise ConfigError("config.yaml has no 'explain' section (Version 0.6 settings).")
        dataset = args.dataset or config["dataset"]["name"]
        report_dir = project_path(config["explain"]["report_dir"]) / dataset
        text = build_tables(report_dir)
    except (ConfigError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    (report_dir / "tables.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
