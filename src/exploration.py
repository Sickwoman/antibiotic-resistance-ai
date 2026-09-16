"""Version 0.1 dataset exploration: descriptive tables and figures for DRIAMS. No modelling here.

Everything is computed from the files actually present under DRIAMS_ROOT. Columns are only used
when they exist (DRIAMS-B, for example, has no case_no, acquisition_date or workstation column).
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

if "ipykernel" not in sys.modules:  # headless for scripts; notebooks keep their inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src.data_loader import (  # noqa: E402
    ADDED_COLS,
    CLEAR_LABELS,
    CODE_COL,
    SPECIES_COL,
    YEAR_COL,
    ColumnSplit,
    discover_sites,
    label_counts,
    load_site_tables,
    read_binned_spectrum,
    read_raw_spectrum,
    spectrum_path,
    split_metadata_antibiotic_columns,
)
from src.utils import driams_root, manifests_dir  # noqa: E402

# ------------------------------------------------------------------------------------------------
# Palette (dataviz reference palette, light mode). Validated with validate_palette.js:
# sites = first four categorical slots (adjacent pairs pass); S/R/I = first three slots (all pairs
# pass). Aqua and yellow are below 3:1 contrast, so every figure has a legend or axis labels and a
# CSV table twin.
# ------------------------------------------------------------------------------------------------
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SITE_COLORS = {"DRIAMS-A": SERIES[0], "DRIAMS-B": SERIES[1], "DRIAMS-C": SERIES[2], "DRIAMS-D": SERIES[3]}
LABEL_COLORS = {"S": SERIES[0], "R": SERIES[1], "I": SERIES[2]}
LABEL_NAMES = {"S": "Susceptible (S)", "I": "Intermediate (I)", "R": "Resistant (R)"}
BLUE_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def site_color(site: str) -> str:
    return SITE_COLORS.get(site, MUTED)


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


SAVE_DPI = 150
BAR_PX = 20  # target bar thickness in the saved image (dataviz spec: <= 24 px)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _bar_size(fig, ax, n_slots: int, horizontal: bool, cap: float = 0.8) -> float:
    """Bar width/height in data units so a bar is about BAR_PX pixels thick, never filling its slot."""
    pos = ax.get_position()
    inches = pos.height * fig.get_figheight() if horizontal else pos.width * fig.get_figwidth()
    px_per_slot = inches * SAVE_DPI / max(n_slots, 1)
    return min(cap, BAR_PX / px_per_slot)


def _slot_limits(n: int, min_slots: int = 3) -> tuple[float, float]:
    """Axis limits for n categorical slots, padded so a lone bar sits centred and thin."""
    pad = max(min_slots - n, 0) / 2
    return -0.5 - pad, n - 0.5 + pad


def _legend_outside(ax) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------

@dataclass
class SiteData:
    site: str
    table: pd.DataFrame
    split: ColumnSplit
    manifest: pd.DataFrame | None


def load_manifest(config: dict, site: str) -> pd.DataFrame | None:
    path = manifests_dir(config) / f"{site}_manifest.csv"
    if not path.is_file():
        return None
    m = pd.read_csv(path, dtype=str, usecols=["folder", "year", "filename", "size_bytes"])
    m["year"] = m["year"].fillna("")
    m["code"] = m["filename"].str.replace(r"\.txt$", "", regex=True)
    m["size_bytes"] = m["size_bytes"].astype("int64")
    return m


def load_sites(config: dict) -> tuple[dict[str, SiteData], list[str]]:
    """Load every extracted site. Returns (sites, expected sites that are not available)."""
    d = config["driams"]
    root = driams_root(config)
    sites: dict[str, SiteData] = {}
    for site in discover_sites(root, d["id_folder"]):
        table = load_site_tables(root, site, d["id_suffixes"], config["labels"]["missing_values"], d["id_folder"])
        split = split_metadata_antibiotic_columns(table, d["metadata_columns"], config["labels"]["ambiguous_values"],
                                                  d["binary_marker_values"])
        sites[site] = SiteData(site, table, split, load_manifest(config, site))
    expected = [a["site"] for a in d["archives"].values()]
    return sites, [s for s in expected if s not in sites]


def clear_label_mask(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(False, index=df.index)
    return df[columns].isin(list(CLEAR_LABELS)).any(axis=1)


def has_spectrum_file(sd: SiteData, folder: str) -> pd.Series | None:
    """True where the id row has a <folder>/<year>/<code>.txt file in the archive manifest."""
    if sd.manifest is None:
        return None
    m = sd.manifest[sd.manifest["folder"] == folder]
    keys = set(m["year"] + "/" + m["code"])
    return (sd.table[YEAR_COL] + "/" + sd.table[CODE_COL].fillna("")).isin(keys)


def target_rows(sd: SiteData, species: str, require_binned: bool = True) -> pd.DataFrame:
    """Rows of the target species, one per code, optionally only those with a binned spectrum."""
    t = sd.table
    mask = t[SPECIES_COL] == species
    binned = has_spectrum_file(sd, "binned_6000")
    if require_binned and binned is not None:
        mask &= binned
    return t.loc[mask].drop_duplicates(subset=[CODE_COL], keep="first")


def group_column(df: pd.DataFrame, group_columns: list[str]) -> str | None:
    """First patient-level grouping column present (patient_no, then case_no)."""
    return next((c for c in group_columns if c in df.columns), None)


def class_counts(values: pd.Series, intermediate_as: str) -> dict[str, int]:
    v = values.dropna()
    r, i, s = int((v == "R").sum()), int((v == "I").sum()), int((v == "S").sum())
    class1 = r + (i if intermediate_as == "resistant" else 0)
    class0 = s + (i if intermediate_as == "susceptible" else 0)
    return {"R": r, "I": i, "S": s, "class1": class1, "class0": class0}


# ------------------------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------------------

def inventory_table(sites: dict[str, SiteData], spectra_folders: list[str]) -> pd.DataFrame:
    rows = []
    for site, sd in sites.items():
        t = sd.table
        labelled = clear_label_mask(t, sd.split.antibiotics)
        has = {f: has_spectrum_file(sd, f) for f in spectra_folders}
        years = set(t[YEAR_COL])
        if sd.manifest is not None:
            years |= set(sd.manifest.loc[sd.manifest["folder"].isin(spectra_folders), "year"]) - {""}
        for year in sorted(years):
            idx = t.index[t[YEAR_COL] == year]
            codes = set(t.loc[idx, CODE_COL].dropna())
            row: dict[str, Any] = {"site": site, "year": year, "id_rows": len(idx),
                                   "unique_codes": len(codes),
                                   "rows_with_RIS_label": int(labelled.loc[idx].sum())}
            for f in spectra_folders:
                if sd.manifest is None:
                    row[f"{f}_files"] = pd.NA
                    row[f"id_rows_with_{f}"] = pd.NA
                    row[f"{f}_files_without_id_row"] = pd.NA
                    continue
                m = sd.manifest[(sd.manifest["folder"] == f) & (sd.manifest["year"] == year)]
                row[f"{f}_files"] = len(m)
                row[f"id_rows_with_{f}"] = int(has[f].loc[idx].sum())
                row[f"{f}_files_without_id_row"] = len(set(m["code"]) - codes)
            if has.get("binned_6000") is not None:
                row["labelled_rows_without_binned"] = int((labelled.loc[idx] & ~has["binned_6000"].loc[idx]).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def column_tables(sites: dict[str, SiteData], metadata_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(metadata presence table, per-column category table)."""
    meta_cols = list(dict.fromkeys(
        list(metadata_columns)
        + [c for sd in sites.values() for c in sd.split.metadata if c not in ADDED_COLS]))
    presence = []
    for col in meta_cols:
        row: dict[str, Any] = {"column": col}
        for site, sd in sites.items():
            row[f"{site}_non_null"] = int(sd.table[col].notna().sum()) if col in sd.table.columns else "absent"
        presence.append(row)

    categories = []
    for site, sd in sites.items():
        groups = {"metadata": sd.split.metadata, "antibiotic_RIS": sd.split.antibiotics,
                  "binary_marker_0_1": sd.split.binary_markers, "empty": sd.split.empty,
                  "unknown": sd.split.unknown}
        for category, cols in groups.items():
            for col in cols:
                if col in ADDED_COLS:
                    continue
                values = ""
                if category in ("binary_marker_0_1", "unknown"):
                    values = ", ".join(sorted(sd.table[col].dropna().astype(str).unique())[:8])
                categories.append({"site": site, "column": col, "category": category,
                                   "non_null": int(sd.table[col].notna().sum()), "distinct_values": values})
    return pd.DataFrame(presence), pd.DataFrame(categories)


def species_tables(sites: dict[str, SiteData], failed: list[str], mixed_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, qc = [], []
    for site, sd in sites.items():
        t = sd.table
        labelled = clear_label_mask(t, sd.split.antibiotics)
        g = pd.DataFrame({"species": t[SPECIES_COL].fillna("<missing>"), "labelled": labelled})
        agg = g.groupby("species").agg(n_rows=("labelled", "size"), n_labelled=("labelled", "sum")).reset_index()
        agg.insert(0, "site", site)
        rows.append(agg)
        sp = t[SPECIES_COL].fillna("<missing>")
        qc.append({"site": site, "id_rows": len(t), "distinct_species_strings": sp.nunique(),
                   "failed_identification_rows": int(sp.isin(failed).sum()),
                   "mixed_culture_rows": int(sp.str.startswith(mixed_prefix).sum()),
                   "missing_species_rows": int(t[SPECIES_COL].isna().sum())})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(qc)


def antibiotic_label_table(sites: dict[str, SiteData], ambiguous: list[str], species: str | None = None) -> pd.DataFrame:
    rows = []
    for site, sd in sites.items():
        t = sd.table if species is None else sd.table[sd.table[SPECIES_COL] == species]
        for col in sd.split.antibiotics:
            c = label_counts(t[col], ambiguous)
            rows.append({"site": site, "antibiotic": col, **c, "labelled_RIS": c["R"] + c["I"] + c["S"]})
    return pd.DataFrame(rows)


def missing_value_table(sites: dict[str, SiteData]) -> pd.DataFrame:
    rows = []
    for site, sd in sites.items():
        n = len(sd.table)
        groups = [("metadata", sd.split.metadata), ("antibiotic_RIS", sd.split.antibiotics),
                  ("binary_marker_0_1", sd.split.binary_markers), ("empty", sd.split.empty),
                  ("unknown", sd.split.unknown)]
        for category, cols in groups:
            for col in cols:
                if col in ADDED_COLS:
                    continue
                miss = int(sd.table[col].isna().sum())
                rows.append({"site": site, "column": col, "category": category, "rows": n,
                             "missing": miss, "missing_pct": round(100 * miss / n, 2)})
    return pd.DataFrame(rows)


def duplicate_table(sites: dict[str, SiteData], species: str, group_columns: list[str]) -> pd.DataFrame:
    """Duplicate codes/rows and repeated patients (the main leakage risks). Never outputs IDs."""
    code_sites: dict[str, set[str]] = {}
    for site, sd in sites.items():
        for code in sd.table[CODE_COL].dropna().unique():
            code_sites.setdefault(code, set()).add(site)
    rows = []
    for site, sd in sites.items():
        t = sd.table
        years_per_code = t.groupby(CODE_COL)[YEAR_COL].nunique()
        gcol = group_column(t, group_columns)
        row: dict[str, Any] = {
            "site": site, "id_rows": len(t),
            "rows_with_missing_code": int(t[CODE_COL].isna().sum()),
            "duplicated_code_rows": int(t[CODE_COL].dropna().duplicated().sum()),
            "fully_duplicated_rows": int(t.drop(columns=[c for c in ADDED_COLS if c in t.columns]).duplicated().sum()),
            "codes_in_more_than_one_year": int((years_per_code > 1).sum()),
            "codes_shared_with_other_sites": int(sum(1 for c in t[CODE_COL].dropna().unique() if len(code_sites[c]) > 1)),
            "group_column": gcol or "none available",
        }
        if gcol:
            per_group = t[gcol].dropna().value_counts()
            tgt_rows = t[t[SPECIES_COL] == species]
            tgt = tgt_rows[gcol].dropna().value_counts()
            years_per_group = t.dropna(subset=[gcol]).groupby(gcol)[YEAR_COL].nunique()
            row.update({
                "groups": len(per_group),
                "groups_with_more_than_one_row": int((per_group > 1).sum()),
                "max_rows_per_group": int(per_group.max()) if len(per_group) else 0,
                "rows_missing_group": int(t[gcol].isna().sum()),
                "groups_in_more_than_one_year": int((years_per_group > 1).sum()),
                "target_rows": len(tgt_rows),
                "target_groups": len(tgt),
                "target_groups_with_more_than_one_row": int((tgt > 1).sum()),
                "target_max_rows_per_group": int(tgt.max()) if len(tgt) else 0,
                "target_share_in_top10_groups": round(float(tgt.head(10).sum() / max(len(tgt_rows), 1)), 4),
            })
            if "order_no" in t.columns:
                per_order = tgt_rows["order_no"].dropna().value_counts()
                row["target_orders_with_more_than_one_spectrum"] = int((per_order > 1).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def group_concentration_table(sites: dict[str, SiteData], species: str, group_columns: list[str],
                              top: int = 15) -> pd.DataFrame:
    """Largest patient groups of the target species, described without their IDs.

    In DRIAMS-A some 'patients' have hundreds of spectra under one case and one order, spread over
    months and all sample types, which suggests placeholder identifiers rather than real patients.
    """
    rows = []
    for site, sd in sites.items():
        t = target_rows(sd, species)
        gcol = group_column(t, group_columns)
        if gcol is None or t.empty:
            continue
        counts = t[gcol].value_counts().head(top)
        for rank, (gid, n) in enumerate(counts.items(), 1):
            g = t[t[gcol] == gid]
            row: dict[str, Any] = {"site": site, "rank": rank, "spectra": int(n),
                                   "share_of_target_spectra": round(n / len(t), 4),
                                   "years": ",".join(sorted(g[YEAR_COL].unique()))}
            for col in ("case_no", "order_no"):
                if col in g.columns and col != gcol:
                    row[f"distinct_{col}"] = g[col].nunique()
            if "acquisition_date" in g.columns:
                row["distinct_acquisition_days"] = pd.to_datetime(g["acquisition_date"], errors="coerce").dt.date.nunique()
            if "workstation" in g.columns:
                row["distinct_workstations"] = g["workstation"].nunique()
                row["workstations"] = ", ".join(f"{k}:{v}" for k, v in g["workstation"].value_counts().head(4).items())
            rows.append(row)
    return pd.DataFrame(rows)


def acquisition_vs_folder_table(sites: dict[str, SiteData]) -> pd.DataFrame:
    """Rows per (year folder, acquisition year): shows spectra filed under a different year."""
    frames = []
    for site, sd in sites.items():
        if "acquisition_date" not in sd.table.columns:
            continue
        years = pd.to_datetime(sd.table["acquisition_date"], errors="coerce").dt.year
        ct = (pd.DataFrame({"folder_year": sd.table[YEAR_COL],
                            "acquisition_year": years.astype("Int64").astype("string").fillna("missing")})
              .value_counts().rename("rows").reset_index())
        ct.insert(0, "site", site)
        frames.append(ct)
    if not frames:
        return pd.DataFrame(columns=["site", "folder_year", "acquisition_year", "rows"])
    return pd.concat(frames, ignore_index=True).sort_values(["site", "folder_year", "acquisition_year"])


def duplicate_spectra_table(config: dict, sites: dict[str, SiteData], species: str) -> pd.DataFrame:
    """Groups of target-species codes whose binned spectrum files are byte-identical."""
    root = driams_root(config)
    records = []
    for site, sd in sites.items():
        for _, r in target_rows(sd, species).iterrows():
            path = spectrum_path(root, site, "binned_6000", r[YEAR_COL], r[CODE_COL])
            if path.is_file():
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                records.append({"site": site, "year": r[YEAR_COL], "code": r[CODE_COL], "md5": digest})
    df = pd.DataFrame(records, columns=["site", "year", "code", "md5"])
    if df.empty:
        return df.assign(group_size=pd.Series(dtype="int64"))
    df["group_size"] = df.groupby("md5")["code"].transform("size")
    return df[df["group_size"] > 1].sort_values(["md5", "site", "year"]).reset_index(drop=True)


def pair_candidates(sites: dict[str, SiteData], config: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the pre-registered eligibility rules (config.pair_selection) to every antibiotic."""
    ps = config["pair_selection"]
    species = config["target"]["species"]
    preferred = config["target"]["preferred_antibiotic"]
    inter = config["labels"]["intermediate_as"]
    dev = ps["development_site"]
    externals = ps["external_sites"]
    min_ext = ps["min_per_class_external"]

    targets = {site: target_rows(sd, species) for site, sd in sites.items()}
    antibiotics = sorted({a for sd in sites.values() for a in sd.split.antibiotics})
    rows = []
    for abx in antibiotics:
        row: dict[str, Any] = {"antibiotic": abx}
        dev_ok = dev in sites and abx in targets[dev].columns
        if dev_ok:
            t = targets[dev]
            c = class_counts(t[abx], inter)
            n = c["class1"] + c["class0"]
            by_year = {y: class_counts(g[abx], inter) for y, g in t.groupby(YEAR_COL)}
            years_ok = sum(1 for v in by_year.values() if v["class1"] > 0 and v["class0"] > 0)
            test = by_year.get(str(ps["temporal_test_year"]), {"class1": 0})
            row.update({"dev_R": c["R"], "dev_I": c["I"], "dev_S": c["S"], "dev_class1": c["class1"],
                        "dev_class0": c["class0"], "dev_minority_frac": round(min(c["class1"], c["class0"]) / n, 4) if n else 0.0,
                        "dev_years_with_both_classes": years_ok, "dev_years_total": len(by_year) if by_year else 0,
                        "dev_class1_test_year": test["class1"]})
            gcol = group_column(t, config.get("driams", {}).get("group_columns", []))
            if gcol:  # informational: spectra per class are not independent patients
                class1_values = ["R"] + (["I"] if inter == "resistant" else [])
                class0_values = ["S"] + (["I"] if inter == "susceptible" else [])
                row["dev_groups_class1"] = t.loc[t[abx].isin(class1_values), gcol].nunique()
                row["dev_groups_class0"] = t.loc[t[abx].isin(class0_values), gcol].nunique()
        meeting, missing = 0, []
        for ext in externals:
            if ext not in sites:
                row[f"{ext}_class1"] = row[f"{ext}_class0"] = pd.NA
                missing.append(ext)
                continue
            if abx in targets[ext].columns:
                c = class_counts(targets[ext][abx], inter)
            else:
                c = {"class1": 0, "class0": 0}
            row[f"{ext}_class1"], row[f"{ext}_class0"] = c["class1"], c["class0"]
            meeting += int(c["class1"] >= min_ext and c["class0"] >= min_ext)
        row["external_sites_meeting_min"] = meeting
        row["external_sites_not_available"] = ", ".join(missing)

        reasons, pending = [], []
        if not dev_ok:
            pending.append(f"{dev} not available" if dev not in sites else f"no {abx} column in {dev}")
        else:
            if row["dev_class1"] < ps["min_resistant_dev"]:
                reasons.append(f"resistant {row['dev_class1']} < {ps['min_resistant_dev']}")
            if row["dev_class0"] < ps["min_susceptible_dev"]:
                reasons.append(f"susceptible {row['dev_class0']} < {ps['min_susceptible_dev']}")
            if row["dev_minority_frac"] < ps["min_minority_fraction"]:
                reasons.append(f"minority fraction {row['dev_minority_frac']:.3f} < {ps['min_minority_fraction']}")
            if row["dev_years_with_both_classes"] < ps["min_years_with_labels"]:
                reasons.append(f"years with both classes {row['dev_years_with_both_classes']} < {ps['min_years_with_labels']}")
            if row["dev_class1_test_year"] < ps["min_resistant_test_year"]:
                reasons.append(f"resistant in {ps['temporal_test_year']} {row['dev_class1_test_year']} < {ps['min_resistant_test_year']}")
        if meeting < ps["min_external_sites"]:
            if meeting + len(missing) >= ps["min_external_sites"]:
                pending.append(f"external sites not yet available: {', '.join(missing)}")
            else:
                reasons.append(f"only {meeting} external site(s) with >= {min_ext} per class")
        if reasons:
            row["status"] = "not eligible"
        elif pending:
            row["status"] = "pending"
        else:
            row["status"] = "eligible"
        row["reasons"] = "; ".join(reasons + pending)
        row["minority_count_dev"] = min(row.get("dev_class1", 0), row.get("dev_class0", 0)) if dev_ok else 0
        pooled1 = sum(class_counts(targets[s][abx], inter)["class1"] for s in sites if abx in targets[s].columns)
        pooled0 = sum(class_counts(targets[s][abx], inter)["class0"] for s in sites if abx in targets[s].columns)
        row["pooled_available_class1"], row["pooled_available_class0"] = pooled1, pooled0
        row["minority_count_available"] = min(pooled1, pooled0)
        rows.append(row)

    table = pd.DataFrame(rows)
    order = {"eligible": 0, "pending": 1, "not eligible": 2}
    table["_status_rank"] = table["status"].map(order)
    table = (table.sort_values(["_status_rank", "minority_count_dev", "minority_count_available"],
                               ascending=[True, False, False], kind="stable")
             .drop(columns="_status_rank").reset_index(drop=True))

    decision: dict[str, Any] = {"species": species, "preferred_antibiotic": preferred,
                                "intermediate_as": inter, "selected_antibiotic": None}
    pref = table[table["antibiotic"] == preferred]
    pref_status = pref["status"].iloc[0] if len(pref) else "absent"
    decision["preferred_status"] = pref_status
    decision["preferred_reasons"] = pref["reasons"].iloc[0] if len(pref) else f"{preferred} not found"
    dev_available = dev in sites
    if not dev_available:
        decision["message"] = (f"No decision yet: the development site {dev} is not available. "
                               "Counts above are preliminary and come only from the sites present.")
    elif pref_status == "eligible":
        decision["selected_antibiotic"] = preferred
        decision["message"] = f"{preferred} meets every pre-registered rule."
    elif pref_status == "pending":
        decision["selected_antibiotic"] = preferred
        decision["message"] = (f"{preferred} meets every rule that can be checked now; "
                               f"final confirmation after: {decision['preferred_reasons']}.")
    else:
        fallback = table[table["status"].isin(["eligible", "pending"])]
        if len(fallback):
            decision["selected_antibiotic"] = fallback["antibiotic"].iloc[0]
            decision["message"] = (f"{preferred} is not eligible ({decision['preferred_reasons']}); "
                                   f"{decision['selected_antibiotic']} has the largest minority class among eligible antibiotics.")
        else:
            decision["message"] = "No antibiotic meets the pre-registered rules; thresholds must be reconsidered explicitly."
    return table, decision


def pair_detail_tables(sites: dict[str, SiteData], config: dict, antibiotic: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(counts per site-year, resistance rate per workstation where that column exists)."""
    species = config["target"]["species"]
    inter = config["labels"]["intermediate_as"]
    per_year, per_ws = [], []
    for site, sd in sites.items():
        t = target_rows(sd, species)
        if antibiotic not in t.columns:
            continue
        for year, g in t.groupby(YEAR_COL):
            c = class_counts(g[antibiotic], inter)
            n = c["class1"] + c["class0"]
            per_year.append({"site": site, "year": year, **c, "labelled": n,
                             "class1_rate": round(c["class1"] / n, 4) if n else np.nan})
        if "workstation" in t.columns:
            for ws, g in t.groupby(t["workstation"].fillna("<missing>")):
                c = class_counts(g[antibiotic], inter)
                n = c["class1"] + c["class0"]
                if n:
                    per_ws.append({"site": site, "workstation": ws, "labelled": n, "class1": c["class1"],
                                   "class1_rate": round(c["class1"] / n, 4)})
    return pd.DataFrame(per_year), pd.DataFrame(per_ws)


def acquisition_months(sites: dict[str, SiteData]) -> pd.DataFrame:
    rows = []
    for site, sd in sites.items():
        if "acquisition_date" not in sd.table.columns:
            continue
        dates = pd.to_datetime(sd.table["acquisition_date"], errors="coerce")
        valid = dates.dropna()
        if valid.empty:
            continue
        counts = valid.dt.to_period("M").value_counts().sort_index()
        for period, n in counts.items():
            rows.append({"site": site, "month": str(period), "spectra": int(n)})
        rows.append({"site": site, "month": "unparseable_or_missing", "spectra": int(dates.isna().sum())})
    return pd.DataFrame(rows, columns=["site", "month", "spectra"])


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------

def _hbar_value_labels(ax, bars, values, fmt="{:,}") -> None:
    xmax = ax.get_xlim()[1]
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                fmt.format(v), va="center", ha="left", fontsize=8.5, color=INK_2)


def plot_samples_per_site(inventory: pd.DataFrame, path: Path) -> Path | None:
    if inventory.empty:
        return None
    agg = inventory.groupby("site")[["id_rows", "rows_with_RIS_label"]].sum()
    if "id_rows_with_binned_6000" in inventory.columns:
        agg["id_rows_with_binned_6000"] = inventory.groupby("site")["id_rows_with_binned_6000"].sum()
    titles = {"id_rows": "Metadata rows", "rows_with_RIS_label": "Rows with ≥1 R/I/S result",
              "id_rows_with_binned_6000": "Rows with a binned spectrum"}
    cols = [c for c in titles if c in agg.columns]
    sites = list(agg.index)[::-1]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 0.45 * len(sites) + 1.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        vals = [int(agg.loc[s, col]) for s in sites]
        y = np.arange(len(sites))
        bars = ax.barh(y, vals, height=_bar_size(fig, ax, len(sites), horizontal=True),
                       color=[site_color(s) for s in sites])
        ax.set_yticks(y, sites)
        ax.set_ylim(-0.6, len(sites) - 0.4)
        ax.set_xlim(0, max(vals + [1]) * 1.3)
        ax.set_title(titles[col], fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        _hbar_value_labels(ax, bars, vals)
    fig.suptitle("Samples per site", x=0.01, y=1.12, ha="left", fontsize=13, fontweight="semibold")
    return _save(fig, path)


def plot_samples_per_site_year(inventory: pd.DataFrame, path: Path) -> Path | None:
    if inventory.empty:
        return None
    measures = [("id_rows", "Metadata rows"), ("rows_with_RIS_label", "Rows with ≥1 R/I/S result")]
    years = sorted(inventory["year"].unique())
    sites = sorted(inventory["site"].unique())
    fig, axes = plt.subplots(len(measures), 1, figsize=(max(6, 1.3 * len(years) + 2), 6.2), sharex=True)
    width = _bar_size(fig, axes[0], max(len(years), 3), horizontal=False, cap=0.8 / len(sites))
    x = np.arange(len(years))
    for ax, (col, title) in zip(axes, measures):
        ax.set_xlim(*_slot_limits(len(years)))
        for k, site in enumerate(sites):
            sub = inventory[inventory["site"] == site].set_index("year")[col]
            vals = [int(sub.get(y, 0)) for y in years]
            ax.bar(x + (k - (len(sites) - 1) / 2) * width, vals, width=width * 0.92,
                   color=site_color(site), label=site)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    title = "Samples per site and year"
    if len(sites) > 1:
        axes[0].legend(ncol=len(sites), loc="upper left", bbox_to_anchor=(0, 1.32))
    else:
        title += f" ({sites[0]})"
    axes[-1].set_xticks(x, years)
    axes[-1].set_xlabel("Year folder in DRIAMS")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold", y=1.04)
    return _save(fig, path)


def plot_monthly(months: pd.DataFrame, path: Path) -> Path | None:
    m = months[months["month"] != "unparseable_or_missing"]
    if m.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.6))
    sites = sorted(m["site"].unique())
    for site, g in m.groupby("site"):
        ax.plot(pd.PeriodIndex(g["month"], freq="M").to_timestamp(), g["spectra"], lw=2,
                color=site_color(site), label=site, solid_capstyle="round")
    ax.set_ylabel("Spectra per month")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    if len(sites) > 1:
        ax.set_title("Spectra per month (acquisition date)")
        ax.legend(loc="upper left")
    else:
        ax.set_title(f"Spectra per month (acquisition date), {sites[0]}")
    return _save(fig, path)


def plot_top_species(species: pd.DataFrame, failed: list[str], path: Path, top: int = 20) -> Path | None:
    s = species[~species["species"].isin(failed)]
    if s.empty:
        return None
    pivot = s.pivot_table(index="species", columns="site", values="n_rows", aggfunc="sum", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:top]].iloc[::-1]
    return _stacked_hbar(pivot, {c: site_color(c) for c in pivot.columns}, path,
                         f"Top {len(pivot)} species by number of spectra",
                         "All metadata rows; failed identifications ('no peaks found') excluded", "Spectra (metadata rows)")


def plot_top_antibiotics(labels: pd.DataFrame, path: Path, top: int = 25) -> Path | None:
    if labels.empty:
        return None
    pivot = labels.pivot_table(index="antibiotic", columns="site", values="labelled_RIS", aggfunc="sum", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:top]].iloc[::-1]
    return _stacked_hbar(pivot, {c: site_color(c) for c in pivot.columns}, path,
                         f"Top {len(pivot)} antibiotics by number of R/I/S results",
                         "All species pooled", "Samples with an R, I or S result")


def plot_label_distribution(labels: pd.DataFrame, path: Path, title: str, subtitle: str,
                            top: int = 25, annotate_rate: bool = False, highlight: str | None = None) -> Path | None:
    if labels.empty:
        return None
    agg = labels.groupby("antibiotic")[["S", "I", "R"]].sum()
    agg = agg[agg.sum(axis=1) > 0]
    if agg.empty:
        return None
    agg = agg.loc[agg.sum(axis=1).sort_values(ascending=False).index[:top]].iloc[::-1]
    fig = _stacked_hbar(agg, LABEL_COLORS, None, title, subtitle, "Samples", legend_names=LABEL_NAMES)
    ax = fig.axes[0]
    if annotate_rate:
        xmax = ax.get_xlim()[1]
        for i, (abx, r) in enumerate(agg.iterrows()):
            total = r.sum()
            rate = (r["R"] + r["I"]) / total if total else 0
            ax.text(total + xmax * 0.01, i, f"{rate:.0%} R+I", va="center", fontsize=8, color=INK_2)
    if highlight is not None:
        for tick in ax.get_yticklabels():
            if tick.get_text() == highlight:
                tick.set_fontweight("bold")
                tick.set_color(INK)
    return _save(fig, path)


def _stacked_hbar(pivot: pd.DataFrame, colors: dict[str, str], path: Path | None, title: str, subtitle: str,
                  xlabel: str, legend_names: dict[str, str] | None = None):
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(pivot) + 1.6))
    left = np.zeros(len(pivot))
    height = _bar_size(fig, ax, len(pivot), horizontal=True)
    for col in pivot.columns:
        vals = pivot[col].to_numpy(dtype=float)
        ax.barh(pivot.index, vals, left=left, height=height, color=colors.get(col, MUTED),
                edgecolor=SURFACE, linewidth=1.5, label=(legend_names or {}).get(col, col))
        left += vals
    ax.set_xlim(0, max(left.max(), 1) * 1.14)
    ax.set_ylim(-0.6, len(pivot) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_title(title, pad=22)
    if len(pivot.columns) > 1:
        _legend_outside(ax)
    else:  # one series: no legend box, the subtitle names it
        name = (legend_names or {}).get(pivot.columns[0], pivot.columns[0])
        subtitle = f"{subtitle} · {name}"
    _subtitle(ax, subtitle)
    if path is None:
        return fig
    return _save(fig, path)


def plot_pair_by_site_year(per_year: pd.DataFrame, antibiotic: str, species: str, path: Path) -> Path | None:
    if per_year.empty:
        return None
    df = per_year.copy()
    df.index = df["site"].str.replace("DRIAMS-", "", regex=False) + " " + df["year"]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * len(df) + 2), 4))
    bottom = np.zeros(len(df))
    x = np.arange(len(df))
    width = _bar_size(fig, ax, max(len(df), 3), horizontal=False)
    ax.set_xlim(*_slot_limits(len(df)))
    for lab in ["S", "I", "R"]:
        vals = df[lab].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, width=width, color=LABEL_COLORS[lab], edgecolor=SURFACE,
               linewidth=1.5, label=LABEL_NAMES[lab])
        bottom += vals
    ymax = max(bottom.max(), 1)
    for xi, total, rate in zip(x, bottom, df["class1_rate"]):
        ax.text(xi, total + ymax * 0.015, f"n={int(total):,}\n{rate:.0%} R+I" if total else "n=0",
                ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_ylim(0, ymax * 1.25)
    ax.set_xticks(x, df.index)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Samples with a binned spectrum")
    ax.set_title(f"{species} – {antibiotic}: labels by site and year", pad=22)
    _subtitle(ax, "Site letter + year folder; one row per spectrum code")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, path)


def plot_label_coverage(sites: dict[str, SiteData], species: str, path: Path, top: int = 25) -> tuple[Path | None, pd.DataFrame]:
    cols = {}
    for site, sd in sites.items():
        t = target_rows(sd, species)
        for year, g in t.groupby(YEAR_COL):
            key = f"{site.replace('DRIAMS-', '')} {year}"
            cols[key] = {abx: g[abx].isin(list(CLEAR_LABELS)).mean() for abx in sd.split.antibiotics} | {"_n": len(g)}
    cov = pd.DataFrame(cols)
    if cov.empty:
        return None, cov
    n_row = cov.loc["_n"]
    cov = cov.drop(index="_n").astype(float).fillna(0.0)
    cov = cov.loc[cov.mean(axis=1).sort_values(ascending=False).index[:top]]
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(1.0 * cov.shape[1] + 4, 0.3 * len(cov) + 2))
    im = ax.imshow(cov.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(cov.shape[1]), [f"{c}\n(n={int(n_row[c]):,})" for c in cov.columns])
    ax.set_yticks(range(len(cov)), cov.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Fraction of samples with an R/I/S result")
    cbar.outline.set_visible(False)
    ax.set_title(f"{species}: which antibiotics were tested, by site and year", pad=30)
    ax.text(0, 1.01, "Samples with a binned spectrum; darker = larger share tested",
            transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    return _save(fig, path), cov.reset_index(names="antibiotic")


def plot_example_spectra(config: dict, sites: dict[str, SiteData], species: str, path: Path) -> Path | None:
    """Format sanity check: one binned spectrum (and a raw one if raw files were extracted)."""
    root = driams_root(config)
    order = [config["pair_selection"]["development_site"]] + [s for s in sites if s != config["pair_selection"]["development_site"]]
    for site in [s for s in order if s in sites]:
        t = target_rows(sites[site], species)
        for _, r in t.iterrows():
            binned_path = spectrum_path(root, site, "binned_6000", r[YEAR_COL], r[CODE_COL])
            if not binned_path.is_file():
                continue
            binned = read_binned_spectrum(binned_path, config["driams"]["n_bins"])
            raw_path = spectrum_path(root, site, "raw", r[YEAR_COL], r[CODE_COL])
            raw = read_raw_spectrum(raw_path) if raw_path.is_file() else None
            n_panels = 2 if raw is not None else 1
            fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.2 * n_panels))
            axes = np.atleast_1d(axes)
            if raw is not None:
                axes[0].plot(raw[:, 0], raw[:, 1], lw=1, color=SERIES[0])
                axes[0].set_title("Raw spectrum")
                axes[0].set_ylabel("Intensity (counts)")
            ax = axes[-1]
            centres = 2000 + 3 * np.arange(len(binned)) + 1.5
            ax.plot(centres, binned, lw=1, color=SERIES[0])
            ax.set_title(f"One {species} spectrum from {site} {r[YEAR_COL]} (binned_6000)", pad=22)
            _subtitle(ax, "x-axis assumes the documented 2,000–20,000 Da range in 3 Da bins (6,000 bins)")
            ax.set_xlabel("m/z (Da), bin centre")
            ax.set_ylabel("Binned intensity")
            return _save(fig, path)
    return None
