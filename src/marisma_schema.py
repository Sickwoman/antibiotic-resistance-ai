"""The restricted schema reader for MARISMa's AMR.csv (docs/v2.0_marisma_plan.md, amendment A2).

Before scoring, this is the only code that may open AMR.csv. It may only:
- match records to cohort isolates by identifier;
- discover antibiotic names;
- count, per requested antibiotic, the cohort isolates with a non-missing interpretation.

It returns exactly the outputs amendment A2 permits: the antibiotic names, those counts, and the numbers of cohort
isolates matched and unmatched.

It never exposes an interpretation category, a count or share by category, an MIC value, a raw row, any other field
value, or a per-isolate result. Nothing here prints or logs, and error messages carry a reason and a line number only.

The layout follows MARISMa's published pipeline code (4_AMR_labeler.py, 4.1_clean_AMR.py.py): semicolon-separated,
UTF-8 with a byte-order mark, an "Identifier" column, and, for each antibiotic X, an MIC column "CMI_X" beside the
interpretation column "X".
"""

from __future__ import annotations

import csv
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

DELIMITER = ";"
ENCODING = "utf-8-sig"
IDENTIFIER = "Identifier"
MIC_PREFIX = "CMI_"
MISSING_TOKENS = frozenset({"", "NA", "N/A", "NAN", "-", "ND"})   # compared after strip() and upper()


class RestrictedReaderError(Exception):
    """A refusal or a parsing problem. The message holds a reason and, at most, a line number: never content."""


@dataclass(frozen=True)
class SchemaSummary:
    antibiotics: tuple[str, ...]          # every X with both "CMI_X" and "X" columns, in header order
    matched_isolates: int                 # cohort isolates with at least one record
    unmatched_isolates: int               # cohort isolates with no record
    non_missing: dict[str, int]           # requested antibiotic -> cohort isolates with a non-missing interpretation


def read_columns(path: str | Path) -> list[str]:
    """The header row's column names, and nothing else from the file."""
    try:
        with open(path, encoding=ENCODING, newline="") as fh:
            header = next(csv.reader(fh, delimiter=DELIMITER))
    except (OSError, StopIteration, csv.Error, UnicodeDecodeError):
        raise RestrictedReaderError("header unreadable") from None
    return [c.strip() for c in header]


def antibiotic_names(columns: list[str]) -> tuple[str, ...]:
    present = set(columns)
    return tuple(c[len(MIC_PREFIX):] for c in columns if c.startswith(MIC_PREFIX) and c[len(MIC_PREFIX):] in present)


def is_non_missing(value: str) -> bool:
    return value.strip().upper() not in MISSING_TOKENS


def schema_summary(path: str | Path, cohort_ids: Collection[str],
                   interpretation_columns: Mapping[str, str]) -> SchemaSummary:
    """Counts for the requested antibiotics among the cohort's isolates.

    `interpretation_columns` maps a label (e.g. "ciprofloxacin") to the exact interpretation column name. The mapping is
    explicit, so no column is found by fuzzy matching and no antibiotic is substituted for another.
    """
    columns = read_columns(path)
    if IDENTIFIER not in columns:
        raise RestrictedReaderError("no Identifier column")
    names = antibiotic_names(columns)
    for label, column in interpretation_columns.items():
        if column not in names:
            raise RestrictedReaderError(f"requested antibiotic {label!r} has no interpretation/MIC column pair")
    id_index = columns.index(IDENTIFIER)
    col_index = {label: columns.index(column) for label, column in interpretation_columns.items()}
    cohort = set(cohort_ids)
    matched: set[str] = set()
    available: dict[str, set[str]] = {label: set() for label in interpretation_columns}
    line = 1
    try:
        with open(path, encoding=ENCODING, newline="") as fh:
            reader = csv.reader(fh, delimiter=DELIMITER)
            next(reader)
            for line, row in enumerate(reader, start=2):
                if len(row) != len(columns):
                    raise RestrictedReaderError(f"row length differs from the header at line {line}")
                identifier = row[id_index].strip()
                if identifier not in cohort:
                    continue
                matched.add(identifier)
                for label, j in col_index.items():
                    if is_non_missing(row[j]):        # reduced to a boolean at once; the value is not kept
                        available[label].add(identifier)
    except RestrictedReaderError:
        raise
    except (OSError, csv.Error, UnicodeDecodeError):
        raise RestrictedReaderError(f"unreadable at line {line}") from None
    return SchemaSummary(antibiotics=names, matched_isolates=len(matched), unmatched_isolates=len(cohort - matched),
                         non_missing={label: len(ids) for label, ids in available.items()})
