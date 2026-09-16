"""Version 0.1 - explore the DRIAMS data that has been downloaded and extracted. No model training.

Run from the project root with the virtual environment active:
    python scripts/explore_dataset.py

Writes:
    results/metrics/eda/*.csv       tables (the "table view" of every figure)
    results/metrics/eda/summary.json
    results/plots/eda/*.png         figures
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import exploration as ex  # noqa: E402
from src.data_loader import DataError  # noqa: E402
from src.utils import ConfigError, driams_root, get_logger, load_config, project_path, set_seed  # noqa: E402

log = get_logger("explore")


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def show(df: pd.DataFrame, max_rows: int = 40) -> None:
    if df is None or df.empty:
        print("(none)")
        return
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 50, "display.width", 250,
                           "display.max_colwidth", 70):
        print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"... {len(df) - max_rows} more rows in the CSV")


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore extracted DRIAMS data (Version 0.1).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-spectrum-hash", action="store_true",
                        help="skip hashing target-species binned files for exact duplicates")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        log.error("%s", exc)
        return 1
    set_seed(config["project"]["random_seed"])
    started = time.monotonic()
    metrics_dir = project_path(config["paths"]["eda_metrics_dir"])
    plots_dir = project_path(config["paths"]["eda_plots_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    d = config["driams"]
    species = config["target"]["species"]
    preferred = config["target"]["preferred_antibiotic"]
    ambiguous = config["labels"]["ambiguous_values"]

    try:
        sites, missing = ex.load_sites(config)
    except DataError as exc:
        log.error("Could not load metadata: %s", exc)
        return 1
    if not sites:
        log.error("No extracted DRIAMS site found under %s. Run download_driams.py and extract_driams.py first.",
                  driams_root(config))
        return 1

    ex.apply_style()
    written: list[Path] = []

    def save_csv(df: pd.DataFrame, name: str) -> None:
        path = metrics_dir / name
        df.to_csv(path, index=False)
        written.append(path)

    section("DRIAMS exploration - Version 0.1 (descriptive only, no model)")
    print(f"DRIAMS root      : {driams_root(config)}")
    print(f"Sites available  : {', '.join(sites)}")
    print(f"Not available yet: {', '.join(missing) or 'none'}")
    for site, sd in sites.items():
        manifest = "yes" if sd.manifest is not None else "NO (run extract_driams.py to create it)"
        years = sorted(sd.table["driams_year"].unique())
        print(f"  {site}: {len(sd.table):,} metadata rows, years {years}, manifest: {manifest}")

    # ---------------------------------------------------------------- inventory
    inventory = ex.inventory_table(sites, d["spectra_folders"])
    save_csv(inventory, "inventory.csv")
    section("1. Inventory: metadata rows vs spectrum files (per site and year folder)")
    show(inventory)

    # ---------------------------------------------------------------- columns
    presence, categories = ex.column_tables(sites, d["metadata_columns"])
    save_csv(presence, "metadata_column_presence.csv")
    save_csv(categories, "column_categories.csv")
    section("2. Metadata columns (non-null counts; 'absent' = the column does not exist at that site)")
    show(presence)
    print()
    for site, sd in sites.items():
        s = sd.split
        files = sorted(sd.table["driams_id_file"].unique())
        print(f"{site}: metadata files used {files}")
        print(f"   {len(s.antibiotics)} R/I/S antibiotic columns | 0/1 marker columns: {s.binary_markers or 'none'}"
              f" | all-empty columns: {s.empty or 'none'} | UNKNOWN columns: {s.unknown or 'none'}")
        dropped = sd.table.attrs.get("dropped_columns", {})
        if dropped:
            print(f"   ignored leftover index columns: {dropped}")
    unknown = categories[categories["category"] == "unknown"] if not categories.empty else categories
    if not unknown.empty:
        print("\nColumns that are neither metadata nor R/I/S nor 0/1 (review before using):")
        show(unknown)

    # ---------------------------------------------------------------- missing values
    missing_tbl = ex.missing_value_table(sites)
    save_csv(missing_tbl, "missing_values.csv")
    section("3. Missing values")
    meta_missing = missing_tbl[missing_tbl["category"] == "metadata"]
    show(meta_missing)
    abx_missing = missing_tbl[missing_tbl["category"] == "antibiotic_RIS"]
    if not abx_missing.empty:
        print("\nAntibiotic columns: share of rows WITHOUT a result (per site):")
        show(abx_missing.groupby("site")["missing_pct"].describe().round(1).reset_index())

    # ---------------------------------------------------------------- species
    species_tbl, species_qc = ex.species_tables(sites, d["failed_identification_species"], d["mixed_species_prefix"])
    save_csv(species_tbl, "species_counts.csv")
    save_csv(species_qc, "species_quality.csv")
    section("4. Species")
    show(species_qc)
    top_species = (species_tbl.groupby("species")[["n_rows", "n_labelled"]].sum()
                   .sort_values("n_rows", ascending=False).head(15).reset_index())
    print("\nTop 15 species (all available sites):")
    show(top_species)
    related = species_tbl[species_tbl["species"].str.contains(species.split()[0], case=False, na=False)]
    print(f"\nSpecies strings containing '{species.split()[0]}' (only the exact name '{species}' is used as target):")
    show(related)

    # ---------------------------------------------------------------- antibiotics
    labels_all = ex.antibiotic_label_table(sites, ambiguous)
    labels_target = ex.antibiotic_label_table(sites, ambiguous, species)
    save_csv(labels_all, "antibiotic_label_counts_all_species.csv")
    save_csv(labels_target, "antibiotic_label_counts_target_species.csv")
    section("5. Antibiotic results (R / I / S / ambiguous / missing), all species")
    top_abx = (labels_all.groupby("antibiotic")[["R", "I", "S", "ambiguous", "labelled_RIS"]].sum()
               .sort_values("labelled_RIS", ascending=False).head(15).reset_index())
    show(top_abx)
    print(f"Distinct antibiotic columns across sites: {labels_all['antibiotic'].nunique()}")

    # ---------------------------------------------------------------- duplicates
    group_cols = d["group_columns"]
    dups = ex.duplicate_table(sites, species, group_cols)
    save_csv(dups, "duplicates.csv")
    section("6. Duplicates and repeated patients (leakage checks; 'group' = patient_no, else case_no)")
    show(dups.T.reset_index().rename(columns={"index": "check"}))
    concentration = ex.group_concentration_table(sites, species, group_cols)
    save_csv(concentration, "target_species_largest_patient_groups.csv")
    if not concentration.empty:
        print(f"\nLargest {species} patient groups (IDs not shown). Many spectra under one case/order spread over "
              "months and sample types suggest a placeholder ID, not one real patient:")
        show(concentration)
    acq_vs_folder = ex.acquisition_vs_folder_table(sites)
    save_csv(acq_vs_folder, "acquisition_year_vs_folder_year.csv")
    mismatch = acq_vs_folder[acq_vs_folder["folder_year"] != acq_vs_folder["acquisition_year"]]
    if not mismatch.empty:
        print("\nSpectra whose acquisition year differs from their year folder:")
        show(mismatch)
    if not args.skip_spectrum_hash:
        dup_spectra = ex.duplicate_spectra_table(config, sites, species)
        save_csv(dup_spectra, "duplicate_binned_spectra_target_species.csv")
        n_groups = dup_spectra["md5"].nunique() if not dup_spectra.empty else 0
        print(f"\nByte-identical binned spectra among {species}: {n_groups} group(s), "
              f"{len(dup_spectra)} codes involved")
        if n_groups:
            show(dup_spectra, 20)

    # ---------------------------------------------------------------- pair selection
    candidates, decision = ex.pair_candidates(sites, config)
    save_csv(candidates, "pair_candidates.csv")
    section(f"7. {species}: antibiotic candidates (samples with a binned spectrum, I counted as "
            f"{config['labels']['intermediate_as']})")
    ext_cols = [c for c in candidates.columns if c.startswith("DRIAMS-") and c.endswith(("_class1", "_class0"))]
    base_cols = [c for c in ["antibiotic", "status", "dev_R", "dev_I", "dev_S", "dev_minority_frac",
                             "dev_years_with_both_classes", "dev_class1_test_year",
                             "dev_groups_class1", "dev_groups_class0"] if c in candidates.columns]
    tested = candidates[candidates["pooled_available_class1"] + candidates["pooled_available_class0"] > 0]
    untested = candidates.loc[candidates["pooled_available_class1"] + candidates["pooled_available_class0"] == 0,
                              "antibiotic"].tolist()
    show(tested[base_cols + ext_cols + ["reasons"]], 20)
    print(f"\nAntibiotics with no {species} results at the available sites ({len(untested)}): "
          f"{', '.join(untested) or 'none'}")
    print(f"\nPreferred antibiotic : {preferred} -> status: {decision['preferred_status']}"
          f"{' (' + decision['preferred_reasons'] + ')' if decision['preferred_reasons'] else ''}")
    print(f"Selected antibiotic  : {decision['selected_antibiotic']}")
    print(f"Decision             : {decision['message']}")

    focus = decision["selected_antibiotic"] or preferred
    per_year, per_ws = ex.pair_detail_tables(sites, config, focus)
    save_csv(per_year, "focus_pair_by_site_year.csv")
    save_csv(per_ws, "focus_pair_by_workstation.csv")
    section(f"8. {species} + {focus}: labels by site/year, and by workstation (label-leak check)")
    show(per_year)
    if per_ws.empty:
        print("\nNo 'workstation' column at the available sites -> workstation leak check not possible yet.")
    else:
        print("\nResistance rate by workstation (large differences mean sample type could act as a shortcut):")
        show(per_ws)

    benchmark = config["target"].get("benchmark_antibiotic")
    bench_year = pd.DataFrame()
    bench_info = None
    if benchmark:
        bench_row = candidates[candidates["antibiotic"] == benchmark]
        bench_info = {"antibiotic": benchmark,
                      "status": bench_row["status"].iloc[0] if len(bench_row) else "absent",
                      "reasons": bench_row["reasons"].iloc[0] if len(bench_row) else f"{benchmark} not found"}
        if benchmark != focus:
            bench_year, _ = ex.pair_detail_tables(sites, config, benchmark)
            save_csv(bench_year, "benchmark_pair_by_site_year.csv")
            section(f"8b. Benchmark pair {species} + {benchmark} (published AUROC 0.74 on DRIAMS-A)")
            print(f"Status under the same rules: {bench_info['status']}"
                  f"{' (' + bench_info['reasons'] + ')' if bench_info['reasons'] else ''}")
            show(bench_year)

    months = ex.acquisition_months(sites)
    save_csv(months, "acquisition_months.csv")
    if months.empty:
        print("\nNo parseable 'acquisition_date' at the available sites -> temporal analysis uses year folders only.")

    # ---------------------------------------------------------------- figures
    section("9. Figures")
    failed = d["failed_identification_species"]
    figures = [
        ex.plot_samples_per_site(inventory, plots_dir / "01_samples_per_site.png"),
        ex.plot_samples_per_site_year(inventory, plots_dir / "02_samples_per_site_year.png"),
        ex.plot_monthly(months, plots_dir / "03_acquisitions_per_month.png"),
        ex.plot_top_species(species_tbl, failed, plots_dir / "04_top_species.png"),
        ex.plot_top_antibiotics(labels_all, plots_dir / "05_top_antibiotics.png"),
        ex.plot_label_distribution(labels_all, plots_dir / "06_label_distribution_all_species.png",
                                   "R / I / S results for the most-tested antibiotics", "All species and sites pooled"),
    ]
    target_binned = pd.concat(
        [ex.antibiotic_label_table({s: ex.SiteData(s, ex.target_rows(sd, species), sd.split, sd.manifest)}, ambiguous)
         for s, sd in sites.items()], ignore_index=True)
    save_csv(target_binned, "target_species_label_counts_with_binned_spectrum.csv")
    figures.append(ex.plot_label_distribution(
        target_binned, plots_dir / "07_target_species_class_balance.png",
        f"{species}: class balance per antibiotic", "Samples with a binned spectrum, all available sites pooled",
        top=20, annotate_rate=True, highlight=focus))
    figures.append(ex.plot_pair_by_site_year(per_year, focus, species, plots_dir / "08_focus_pair_by_site_year.png"))
    if not bench_year.empty:
        figures.append(ex.plot_pair_by_site_year(bench_year, benchmark, species,
                                                 plots_dir / "08b_benchmark_pair_by_site_year.png"))
    coverage_fig, coverage = ex.plot_label_coverage(sites, species, plots_dir / "09_target_species_label_coverage.png")
    save_csv(coverage, "target_species_label_coverage.csv")
    figures.append(coverage_fig)
    figures.append(ex.plot_example_spectra(config, sites, species, plots_dir / "10_example_spectrum.png"))
    for fig in figures:
        if fig is not None:
            written.append(fig)
            print(f"  saved {fig.relative_to(project_path('.'))}")
    if figures[2] is None:
        print("  skipped 03_acquisitions_per_month.png (no acquisition dates available)")

    # ---------------------------------------------------------------- summary
    summary = {
        "version": config["project"]["version"],
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "driams_root": str(driams_root(config)),
        "sites_available": list(sites),
        "sites_not_available": missing,
        "rows_per_site": {s: len(sd.table) for s, sd in sites.items()},
        "years_per_site": {s: sorted(sd.table["driams_year"].unique()) for s, sd in sites.items()},
        "metadata_files_used": {s: sorted(sd.table["driams_id_file"].unique()) for s, sd in sites.items()},
        "ignored_index_columns": {s: sd.table.attrs.get("dropped_columns", {}) for s, sd in sites.items()},
        "absent_metadata_columns": {s: [c for c in d["metadata_columns"] if c not in sd.table.columns]
                                    for s, sd in sites.items()},
        "patient_group_column": {s: ex.group_column(sd.table, group_cols) for s, sd in sites.items()},
        "antibiotic_columns_per_site": {s: len(sd.split.antibiotics) for s, sd in sites.items()},
        "binary_marker_columns": {s: sd.split.binary_markers for s, sd in sites.items()},
        "unknown_columns": {s: sd.split.unknown for s, sd in sites.items()},
        "target_species": species,
        "target_rows_with_binned_spectrum": {s: len(ex.target_rows(sd, species)) for s, sd in sites.items()},
        "pair_decision": decision,
        "benchmark_pair": bench_info,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    summary_path = metrics_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written.append(summary_path)
    section("Done")
    print(f"{len(written)} files written to {metrics_dir.parent.parent} ({summary['elapsed_seconds']} s)")
    print("No model was trained. Paste this console output back to continue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
