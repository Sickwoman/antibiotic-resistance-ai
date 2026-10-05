"""Step 5's label reader for MARISMa's AMR.csv (docs/v2.0_marisma_plan.md, step 5; amendment F5).

This is the only code that reads interpretation categories, and it reads the sealed AMR.csv only through
`read_interpretations`, which first verifies the owner's recorded scoring authorisation (src/v20_scoring_guard.py).
Until that authorisation is recorded, every call on the sealed file refuses.

For each cohort isolate with a record, the interpretation in the requested column is returned as text, stripped. An
isolate whose records give different non-missing interpretations is "conflicting"; one whose records give none is
"missing". Both are counted, and both are excluded later by the label rule (F5.1). The column name must be exact: no
antibiotic is found by fuzzy matching or substituted.

`parse_interpretations` does the same for synthetic files in tests, and refuses the sealed file.
"""

from __future__ import annotations

import csv
from collections.abc import Collection
from pathlib import Path

from src.marisma_schema import ENCODING, IDENTIFIER, RestrictedReaderError, is_non_missing, read_header
from src.v20_scoring_guard import ROOT, GuardError, verify

MISSING, CONFLICTING = "missing", "conflicting"


def _is_sealed(path: str | Path) -> bool:
    path = Path(path)
    return path.name == "AMR.csv" and path.resolve().parent.name.endswith("_sealed")


def _parse(path: str | Path, cohort_ids: Collection[str], column: str) -> tuple[dict[str, str], dict[str, int]]:
    columns, delimiter = read_header(path)
    if IDENTIFIER not in columns or column not in columns:
        raise RestrictedReaderError("no Identifier column, or no column with the requested exact name")
    id_index, col_index = columns.index(IDENTIFIER), columns.index(column)
    cohort = set(cohort_ids)
    seen: dict[str, set[str]] = {}
    line = 1
    try:
        with open(path, encoding=ENCODING, newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            next(reader)
            for line, row in enumerate(reader, start=2):
                if len(row) != len(columns):
                    raise RestrictedReaderError(f"row length differs from the header at line {line}")
                identifier = row[id_index].strip()
                if identifier not in cohort:
                    continue
                values = seen.setdefault(identifier, set())
                if is_non_missing(row[col_index]):
                    values.add(row[col_index].strip())
    except RestrictedReaderError:
        raise
    except (OSError, csv.Error, UnicodeDecodeError):
        raise RestrictedReaderError(f"unreadable at line {line}") from None
    out = {i: (MISSING if not v else next(iter(v)) if len(v) == 1 else CONFLICTING) for i, v in seen.items()}
    counts = {"cohort": len(cohort), "matched": len(out), "unmatched": len(cohort) - len(out),
              "missing": sum(1 for v in out.values() if v == MISSING),
              "conflicting": sum(1 for v in out.values() if v == CONFLICTING)}
    return out, counts


def read_interpretations(path: str | Path, cohort_ids: Collection[str], column: str, *,
                         root: Path = ROOT) -> tuple[dict[str, str], dict[str, int]]:
    """Step 5: the interpretations of the cohort's isolates, after the scoring authorisation is verified."""
    verify(root)                                           # raises GuardError without a recorded authorisation
    return _parse(path, cohort_ids, column)


def parse_interpretations(path: str | Path, cohort_ids: Collection[str],
                          column: str) -> tuple[dict[str, str], dict[str, int]]:
    """The same parser, for synthetic files only: the sealed AMR.csv is refused."""
    if _is_sealed(path):
        raise GuardError("the sealed AMR.csv is read only by read_interpretations, under a verified authorisation")
    return _parse(path, cohort_ids, column)
