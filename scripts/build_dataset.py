"""Version 0.2 - build the processed E. coli + ciprofloxacin dataset from raw DRIAMS spectra.

No model is trained. Run from the project root with the virtual environment active:

    python scripts/build_dataset.py                          # primary dataset (I counted as resistant)
    python scripts/build_dataset.py --intermediate-as exclude  # sensitivity dataset (I removed)

Needs the raw E. coli spectra, extracted with:
    python scripts/extract_driams.py --site A --folders raw preprocessed --species "Escherichia coli"

Writes (git-ignored)  data/processed/<name>/{X.npy, metadata.csv, exclusions.csv, summary.json, splits/}
Writes (committed)    results/metrics/v0.2/<name>/*  (aggregate counts only, no identifiers)
                      results/plots/v0.2/<name>_example_preprocessing.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import exploration as ex  # noqa: E402  (plot style only)
from src.data_loader import INTERMEDIATE_POLICIES, DataError, read_binned_spectrum, read_raw_spectrum  # noqa: E402
from src.dataset import EXCLUSION_REASONS, CohortSpec, build_dataset, load_dataset, resolve_relpath  # noqa: E402
from src.preprocessing import PreprocessingConfig, preprocess_file  # noqa: E402
from src.splits import LeakageError, SplitError, check_split, make_splits  # noqa: E402
from src.utils import (  # noqa: E402
    ConfigError,
    driams_root,
    get_logger,
    keep_awake,
    load_config,
    project_path,
    set_seed,
)

log = get_logger("build")


def section(title: str) -> None:
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")


def compare_with_driams(config: dict, X: np.ndarray, meta: pd.DataFrame, n: int, seed: int) -> dict:
    """Compare our features with the published DRIAMS binned_6000 files for a random subset."""
    root = driams_root(config)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(meta), size=min(n, len(meta)), replace=False))
    diffs, missing = [], 0
    for i in rows:
        r = meta.iloc[i]
        path = resolve_relpath(root, f"{r['site']}/binned_6000/{r['year_folder']}/{r['code']}.txt")
        if not path.is_file():
            missing += 1
            continue
        reference = read_binned_spectrum(path, X.shape[1])
        diffs.append(float(np.abs(X[i].astype(np.float64) - reference).max() / reference.max()))
    arr = np.asarray(diffs)
    return {"compared": int(arr.size), "reference_file_missing": missing,
            "max_relative_difference": float(arr.max()) if arr.size else None,
            "median_relative_difference": float(np.median(arr)) if arr.size else None,
            "note": "relative difference = max |ours - DRIAMS| / max(DRIAMS); float32 storage alone gives ~1e-7"}


def plot_example(config: dict, X: np.ndarray, meta: pd.DataFrame, row: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    ex.apply_style()
    pcfg = PreprocessingConfig.from_config(config)
    r = meta.iloc[row]
    raw = read_raw_spectrum(resolve_relpath(driams_root(config), r["spectrum_relpath"]), pcfg.min_raw_points)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2))
    axes[0].plot(raw[:, 0], raw[:, 1], lw=0.8, color=ex.SERIES[0])
    axes[0].set_title(f"Raw spectrum ({len(raw):,} points)", pad=8)
    axes[0].set_ylabel("Intensity (counts)")
    axes[1].plot(pcfg.bin_centers, X[row], lw=0.8, color=ex.SERIES[0])
    axes[1].set_title(f"After preprocessing ({pcfg.n_bins:,} bins of {pcfg.bin_width:g} Da, "
                      f"{int(np.count_nonzero(X[row])):,} non-zero)", pad=8)
    axes[1].set_ylabel("Binned intensity (TIC-normalised)")
    axes[1].set_xlabel("m/z (Da)")
    for ax in axes:
        ax.set_xlim(pcfg.mz_min - 200, raw[-1, 0] + 200)
    fig.suptitle(f"Example: {r['site']} sample, label {'resistant' if r['label'] == 1 else 'susceptible'}",
                 x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Version 0.2 processed dataset (no model training).")
    parser.add_argument("--intermediate-as", choices=INTERMEDIATE_POLICIES, default=None,
                        help="override labels.intermediate_as (default: config value)")
    parser.add_argument("--sites", nargs="+", default=None, help="override dataset.sites")
    parser.add_argument("--name", default=None, help="output dataset name")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--compare", type=int, default=300,
                        help="number of samples compared with DRIAMS binned_6000 files (0 = skip)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        seed = int(config["project"]["random_seed"])
        set_seed(seed)
        spec = CohortSpec.from_config(config, intermediate_as=args.intermediate_as, sites=args.sites, name=args.name)
        with keep_awake():
            summary = build_dataset(config, spec)
            out_dir = project_path(config["dataset"]["output_dir"]) / spec.name
            X, meta, summary = load_dataset(out_dir)
            report_dir = project_path("results/metrics/v0.2") / spec.name
            report_dir.mkdir(parents=True, exist_ok=True)

            section(f"Dataset {spec.name}: {spec.species} + {spec.antibiotic}, I -> {spec.intermediate_as}")
            print(f"X: shape {tuple(X.shape)}, dtype {X.dtype}, {summary['x_megabytes']} MB on disk "
                  f"(memory-mapped when loaded); metadata.csv {(out_dir / 'metadata.csv').stat().st_size / 1e6:.2f} MB")
            print(f"Samples: {summary['n_samples']:,}  resistant: {summary['resistant']:,}  "
                  f"susceptible: {summary['susceptible']:,}  (built in {summary['elapsed_seconds']} s)")
            per_site = pd.DataFrame(summary["per_site"]).T[
                ["target_species_rows", "samples", "resistant", "susceptible", "excluded",
                 "patient_groups", "samples_without_patient_id", "acquisition_date_min", "acquisition_date_max"]]
            print(per_site.to_string())
            print("\nAST values used:", {s: v["ast_values"] for s, v in summary["per_site"].items()})
            print("Other spellings of the species (not included):",
                  {s: v["other_spellings_not_included"] for s, v in summary["per_site"].items()})

            section("Exclusions (first applicable reason per metadata row)")
            rows = [{"reason": r, "site": s, "count": c, "meaning": EXCLUSION_REASONS[r]}
                    for r, by_site in summary["exclusions_by_reason"].items() for s, c in by_site.items()]
            excl = pd.DataFrame(rows, columns=["reason", "site", "count", "meaning"])
            print(excl.to_string(index=False) if len(excl) else "(none)")
            print(f"Total excluded: {summary['excluded_total']:,}; workstation detail: "
                  f"{summary['excluded_workstations_by_site']}")
            print("Check against published DRIAMS binned_6000 files during the build:",
                  json.dumps(summary["verification_against_driams_binned"]))
            excl.to_csv(report_dir / "exclusion_summary.csv", index=False)
            (report_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

            if args.compare:
                section(f"Check against published DRIAMS binned_6000 files ({args.compare} random samples)")
                comparison = compare_with_driams(config, X, meta, args.compare, seed)
                print(json.dumps(comparison, indent=2))
                (report_dir / "comparison_with_driams_binned.json").write_text(json.dumps(comparison, indent=2),
                                                                                encoding="utf-8")

            if not args.skip_splits:
                section("Splits (row indices only; patient groups kept together)")
                skipped: list[str] = []
                splits = make_splits(meta, config, skipped)
                for message in skipped:
                    print(f"skipped split - {message}")
                split_rows, split_json = [], {}
                for name, split in splits.items():
                    check_split(meta, split)
                    split.save(out_dir / "splits" / f"{name}.json", meta)
                    s = split.summary(meta)
                    split_json[name] = s
                    for part in ("train", "validation", "test"):
                        p = s[part]
                        split_rows.append({"split": name, "part": part, "samples": p["samples"],
                                           "resistant": p["resistant"], "susceptible": p["susceptible"],
                                           "groups": p["groups"], "sites": p["sites"],
                                           "dates": f"{p['date_min']} .. {p['date_max']}"})
                table = pd.DataFrame(split_rows)
                print(table.to_string(index=False))
                for name, s in split_json.items():
                    print(f"{name}: {s['description']}")
                    for note in s["notes"]:
                        print(f"   note: {note}")
                table.to_csv(report_dir / "split_summary.csv", index=False)
                (report_dir / "split_summary.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")
                print("All splits passed the sample and patient-group overlap checks.")

            section("Example preprocessing (privacy-safe)")
            row = 0
            r = meta.iloc[row]
            pcfg = PreprocessingConfig.from_config(config)
            t0 = time.perf_counter()
            features, info = preprocess_file(resolve_relpath(driams_root(config), r["spectrum_relpath"]), pcfg)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            example = {"sample_index": int(r["sample_index"]), "site": r["site"], "raw_points": info.raw_points,
                       "points_in_range": info.points_in_range, "features": int(features.size),
                       "nonzero_bins": info.nonzero_bins, "dtype": str(features.dtype),
                       "ast_value": r["ast_value"], "label": int(r["label"]),
                       "matches_stored_row": bool(np.array_equal(features, X[row])),
                       "read_and_preprocess_ms": round(elapsed_ms, 1)}
            print(json.dumps(example, indent=2))
            (report_dir / "example_preprocessing.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
            plot_path = project_path("results/plots/v0.2") / f"{spec.name}_example_preprocessing.png"
            plot_example(config, X, meta, row, plot_path)
            print(f"\nReports: {report_dir}")
            print("No model was trained.")
        return 0
    except (DataError, ConfigError, SplitError, LeakageError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
