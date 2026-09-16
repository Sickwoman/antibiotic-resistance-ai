"""Loading DRIAMS metadata tables and spectrum files, with explicit validation.

Folder layout (DRIAMS README, confirmed by inspecting the archives):
    <DRIAMS_ROOT>/DRIAMS-<X>/id/<year>/<year>_clean.csv
    <DRIAMS_ROOT>/DRIAMS-<X>/{raw,preprocessed,binned_6000}/<year>/<code>.txt
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SITE_PATTERN = re.compile(r"^DRIAMS-[A-Z]$")
YEAR_PATTERN = re.compile(r"^\d{4}$")
CLEAR_LABELS = ("R", "I", "S")
SITE_COL = "driams_site"
YEAR_COL = "driams_year"
SOURCE_COL = "driams_id_file"
ADDED_COLS = (SITE_COL, YEAR_COL, SOURCE_COL)
DEFAULT_SUFFIXES = ("strat", "clean")
INDEX_ARTIFACT_PREFIX = "Unnamed:"
# Column names used by maldi-learn for DRIAMS id tables (checked on the real files by the explorer).
CODE_COL = "code"
SPECIES_COL = "species"


class DataError(Exception):
    """Base class for data problems that should be shown to users as plain messages."""


class MetadataError(DataError):
    pass


class SpectrumFormatError(DataError):
    pass


# ----------------------------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------------------------

def discover_sites(root: str | Path, id_folder: str = "id") -> list[str]:
    """Site folders (DRIAMS-A, ...) under `root` that contain an id folder."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and SITE_PATTERN.match(p.name) and (p / id_folder).is_dir())


def discover_years(root: str | Path, site: str, id_folder: str = "id") -> list[str]:
    id_dir = Path(root) / site / id_folder
    if not id_dir.is_dir():
        return []
    return sorted(p.name for p in id_dir.iterdir() if p.is_dir() and YEAR_PATTERN.match(p.name))


def id_file_path(root: str | Path, site: str, year: str, suffix: str = "clean", id_folder: str = "id") -> Path:
    return Path(root) / site / id_folder / str(year) / f"{year}_{suffix}.csv"


def spectrum_path(root: str | Path, site: str, folder: str, year: str, code: str) -> Path:
    return Path(root) / site / folder / str(year) / f"{code}.txt"


# ----------------------------------------------------------------------------------------------
# Metadata tables
# ----------------------------------------------------------------------------------------------

def find_id_file(root: str | Path, site: str, year: str, suffixes: str | Sequence[str] = DEFAULT_SUFFIXES,
                 id_folder: str = "id") -> Path:
    """First existing id/<year>/<year>_<suffix>.csv in the order given (DRIAMS-A: strat, then clean)."""
    suffixes = [suffixes] if isinstance(suffixes, str) else list(suffixes)
    tried = [id_file_path(root, site, year, s, id_folder) for s in suffixes]
    for path in tried:
        if path.is_file():
            return path
    raise MetadataError(f"Metadata file not found; tried: {', '.join(str(p) for p in tried)}")


def load_id_table(root: str | Path, site: str, year: str, suffixes: str | Sequence[str] = DEFAULT_SUFFIXES,
                  missing_values: Iterable[str] = ("-",), id_folder: str = "id") -> pd.DataFrame:
    """Read one year's metadata table as strings, marking '-' as missing.

    Leftover pandas index columns ('Unnamed: 0', found in DRIAMS-A 2018) are dropped and listed in
    df.attrs['dropped_columns']. Added columns: driams_site, driams_year, driams_id_file.
    """
    path = find_id_file(root, site, year, suffixes, id_folder)
    if path.stat().st_size == 0:
        raise MetadataError(f"Metadata file is empty: {path}")
    na_values = [v for v in missing_values if v != ""]
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False, na_values=na_values, keep_default_na=True)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MetadataError(f"Could not parse {path}: {exc}") from exc
    if df.empty:
        raise MetadataError(f"Metadata file has no rows: {path}")
    for col in ADDED_COLS:
        if col in df.columns:
            raise MetadataError(f"{path} already has a column named {col!r}; refusing to overwrite it.")
    dropped = [c for c in df.columns if str(c).startswith(INDEX_ARTIFACT_PREFIX)]
    df = df.drop(columns=dropped)
    df[SITE_COL] = site
    df[YEAR_COL] = str(year)
    df[SOURCE_COL] = path.name
    df.attrs["dropped_columns"] = dropped
    return df


def load_site_tables(root: str | Path, site: str, suffixes: str | Sequence[str] = DEFAULT_SUFFIXES,
                     missing_values: Iterable[str] = ("-",), id_folder: str = "id") -> pd.DataFrame:
    """All years of one site concatenated (columns are the union across years)."""
    years = discover_years(root, site, id_folder)
    if not years:
        raise MetadataError(f"No year folders found in {Path(root) / site / id_folder}")
    frames = [load_id_table(root, site, y, suffixes, missing_values, id_folder) for y in years]
    dropped = {f"{site}/{y}": f.attrs.get("dropped_columns", []) for y, f in zip(years, frames)}
    table = pd.concat(frames, ignore_index=True, sort=False)
    table.attrs = {"dropped_columns": {k: v for k, v in dropped.items() if v}}
    return table


@dataclass
class ColumnSplit:
    metadata: list[str] = field(default_factory=list)
    antibiotics: list[str] = field(default_factory=list)      # R / I / S phenotypes
    binary_markers: list[str] = field(default_factory=list)   # 0 / 1 screens, e.g. ESBL, MRSA
    empty: list[str] = field(default_factory=list)            # all values missing
    unknown: list[str] = field(default_factory=list)          # none of the above


def split_metadata_antibiotic_columns(df: pd.DataFrame, metadata_columns: Iterable[str],
                                      ambiguous_values: Iterable[str],
                                      binary_marker_values: Iterable[str] = ("0", "1")) -> ColumnSplit:
    """Classify columns using the data itself.

    A column counts as an antibiotic only if every non-missing value is a known phenotype label
    (R, I, S or one of the documented ambiguous entries). Columns holding only 0/1 are screening
    markers (DRIAMS-B has ESBL, MRSA, Cefoxitin_screen, Clindamycin_induced). Anything else is
    reported as unknown instead of being silently used.
    """
    meta = set(metadata_columns) | set(ADDED_COLS)
    vocab = set(CLEAR_LABELS) | {v.strip() for v in ambiguous_values}
    markers = {v.strip() for v in binary_marker_values}
    split = ColumnSplit()
    for col in df.columns:
        if col in meta:
            split.metadata.append(col)
            continue
        values = df[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        uniques = set(values.unique())
        if values.empty:
            split.empty.append(col)
        elif uniques <= vocab:
            split.antibiotics.append(col)
        elif uniques <= markers:
            split.binary_markers.append(col)
        else:
            split.unknown.append(col)
    return split


def label_counts(series: pd.Series, ambiguous_values: Iterable[str]) -> dict[str, int]:
    """Count R / I / S / ambiguous / missing / other entries of one antibiotic column."""
    ambiguous = {v.strip() for v in ambiguous_values}
    values = series.astype("string").str.strip()
    missing = values.isna() | (values == "")
    present = values[~missing]
    counts = {"R": int((present == "R").sum()), "I": int((present == "I").sum()),
              "S": int((present == "S").sum()), "ambiguous": int(present.isin(ambiguous).sum()),
              "missing": int(missing.sum())}
    counts["other"] = int(len(present) - counts["R"] - counts["I"] - counts["S"] - counts["ambiguous"])
    return counts


def encode_labels(series: pd.Series, intermediate_as: str = "resistant",
                  ambiguous_values: Iterable[str] = ()) -> pd.Series:
    """Map phenotype strings to 1 (resistant) / 0 (susceptible) / NaN (unusable).

    intermediate_as: 'resistant' (Weis et al. 2022 convention), 'susceptible' or 'exclude'.
    Unknown strings raise an error rather than being dropped silently.
    """
    i_value = {"resistant": 1.0, "susceptible": 0.0, "exclude": np.nan}
    if intermediate_as not in i_value:
        raise ValueError(f"intermediate_as must be one of {sorted(i_value)}, got {intermediate_as!r}")
    mapping = {"R": 1.0, "S": 0.0, "I": i_value[intermediate_as]}
    mapping.update({v.strip(): np.nan for v in ambiguous_values})
    values = series.astype("string").str.strip()
    unknown = values.dropna()
    unknown = unknown[(unknown != "") & ~unknown.isin(list(mapping))]
    if not unknown.empty:
        examples = sorted(unknown.unique())[:5]
        raise MetadataError(f"Column {series.name!r} has unrecognised label values: {examples}")
    out = values.map(mapping, na_action="ignore").astype("float64")
    out.name = series.name
    return out


# ----------------------------------------------------------------------------------------------
# Spectrum files
# ----------------------------------------------------------------------------------------------

def read_spectrum_table(path: str | Path, allowed_suffixes: Iterable[str] = (".txt",)) -> np.ndarray:
    """Read a two-column whitespace-separated spectrum file into an (n, 2) float array.

    Handles the DRIAMS layout: '#' comment lines, then an optional header line of column names,
    then numeric rows. Raises SpectrumFormatError with a readable message on any problem.
    """
    path = Path(path)
    if path.suffix.lower() not in {s.lower() for s in allowed_suffixes}:
        raise SpectrumFormatError(f"Unsupported file type '{path.suffix}' for {path.name}; expected {sorted(allowed_suffixes)}.")
    if not path.is_file():
        raise SpectrumFormatError(f"Spectrum file not found: {path}")
    if path.stat().st_size == 0:
        raise SpectrumFormatError(f"Spectrum file is empty: {path.name}")
    try:
        table = pd.read_csv(path, sep=r"\s+", comment="#", header=None, dtype=str, engine="c")
    except pd.errors.EmptyDataError as exc:
        raise SpectrumFormatError(f"Spectrum file has no data rows: {path.name}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise SpectrumFormatError(f"Spectrum file {path.name} is corrupted or not a text table: {exc}") from exc
    if table.shape[1] != 2:
        raise SpectrumFormatError(f"{path.name}: expected 2 columns (m/z, intensity), found {table.shape[1]}.")
    numeric = table.apply(pd.to_numeric, errors="coerce")
    if len(numeric) and numeric.iloc[0].isna().all():  # header line with column names
        numeric = numeric.iloc[1:]
    if numeric.empty:
        raise SpectrumFormatError(f"Spectrum file has no numeric rows: {path.name}")
    if numeric.isna().any().any():
        bad = int(numeric.isna().any(axis=1).sum())
        raise SpectrumFormatError(f"{path.name}: {bad} row(s) contain non-numeric values.")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise SpectrumFormatError(f"{path.name}: contains infinite values.")
    return values


def read_raw_spectrum(path: str | Path, min_points: int = 100) -> np.ndarray:
    """Raw DRIAMS spectrum -> (n, 2) array of [m/z, intensity], validated."""
    values = read_spectrum_table(path)
    if len(values) < min_points:
        raise SpectrumFormatError(f"{Path(path).name}: only {len(values)} points; expected at least {min_points}.")
    mz, intensity = values[:, 0], values[:, 1]
    if np.any(np.diff(mz) <= 0):
        raise SpectrumFormatError(f"{Path(path).name}: m/z values are not strictly increasing.")
    if np.any(mz <= 0):
        raise SpectrumFormatError(f"{Path(path).name}: m/z values must be positive.")
    if np.any(intensity < 0):
        raise SpectrumFormatError(f"{Path(path).name}: raw intensities must not be negative.")
    return values


def read_binned_spectrum(path: str | Path, expected_bins: int = 6000) -> np.ndarray:
    """DRIAMS binned_6000 file -> 1-D float array of length `expected_bins`.

    Verified format (DRIAMS-B): header 'bin_index binned_intensity', then rows '0 <value>' ...
    '5999 <value>'.
    """
    values = read_spectrum_table(path)
    name = Path(path).name
    if len(values) != expected_bins:
        raise SpectrumFormatError(f"{name}: expected {expected_bins} bins, found {len(values)}.")
    if not np.array_equal(values[:, 0], np.arange(expected_bins)):
        raise SpectrumFormatError(f"{name}: bin indices are not 0..{expected_bins - 1} in order.")
    return values[:, 1].astype(np.float64)
