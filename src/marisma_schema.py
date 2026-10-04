"""The restricted schema reader for MARISMa's AMR.csv (docs/v2.0_marisma_plan.md, amendment A2).

Before scoring, this is the only code that may open AMR.csv. It may only:
- match records to cohort isolates by identifier;
- discover antibiotic names;
- count, per requested antibiotic, the cohort isolates with a non-missing interpretation.

It returns exactly the outputs amendment A2 permits: the antibiotic names, those counts, and the numbers of cohort
isolates matched and unmatched.

It never exposes an interpretation category, a count or share by category, an MIC value, a raw row, any other field
value, or a per-isolate result. Nothing here prints or logs, and error messages carry a reason and a line number only.

The layout: an "Identifier" column and, for each antibiotic X, an MIC column beside the interpretation column "X".
MARISMa 2.0.0's AMR.csv is comma-separated, UTF-8 without a byte-order mark, with MIC columns named "MIC_X" (read
from its header on 2026-10-03: 166 columns, 79 MIC/interpretation pairs). MARISMa's older pipeline code
(4_AMR_labeler.py) wrote semicolon-separated files with "CMI_X" columns, so both prefixes are accepted. The delimiter
is detected from the header line alone.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

DELIMITERS = (",", ";", "\t")
ENCODING = "utf-8-sig"                   # also reads UTF-8 without a byte-order mark
IDENTIFIER = "Identifier"
MIC_PREFIXES = ("MIC_", "CMI_")
MISSING_TOKENS = frozenset({"", "NA", "N/A", "NAN", "-", "ND"})   # compared after strip() and upper()


class RestrictedReaderError(Exception):
    """A refusal or a parsing problem. The message holds a reason and, at most, a line number: never content."""


@dataclass(frozen=True)
class SchemaSummary:
    antibiotics: tuple[str, ...]          # every X with an MIC column ("MIC_X" or "CMI_X") and an "X" column
    matched_isolates: int                 # cohort isolates with at least one record
    unmatched_isolates: int               # cohort isolates with no record
    non_missing: dict[str, int]           # requested antibiotic -> cohort isolates with a non-missing interpretation


def detect_delimiter(header_line: str) -> str:
    """The delimiter that occurs most often in the header line; refused if none occurs or two tie."""
    counts = {d: header_line.count(d) for d in DELIMITERS}
    best = max(counts.values())
    if best == 0 or list(counts.values()).count(best) > 1:
        raise RestrictedReaderError("delimiter not recognised in the header")
    return next(d for d, n in counts.items() if n == best)


def read_header(path: str | Path) -> tuple[list[str], str]:
    """The header row's column names and the delimiter, and nothing else from the file."""
    try:
        with open(path, encoding=ENCODING, newline="") as fh:
            line = fh.readline()
    except (OSError, UnicodeDecodeError):
        raise RestrictedReaderError("header unreadable") from None
    delimiter = detect_delimiter(line)
    try:
        header = next(csv.reader([line], delimiter=delimiter))
    except (StopIteration, csv.Error):
        raise RestrictedReaderError("header unreadable") from None
    return [c.strip() for c in header], delimiter


def read_columns(path: str | Path) -> list[str]:
    """The header row's column names, and nothing else from the file."""
    return read_header(path)[0]


def antibiotic_names(columns: list[str]) -> tuple[str, ...]:
    present, names = set(columns), []
    for c in columns:
        for prefix in MIC_PREFIXES:
            if c.startswith(prefix) and c[len(prefix):] in present and c[len(prefix):] not in names:
                names.append(c[len(prefix):])
    return tuple(names)


def is_non_missing(value: str) -> bool:
    return value.strip().upper() not in MISSING_TOKENS


def schema_summary(path: str | Path, cohort_ids: Collection[str],
                   interpretation_columns: Mapping[str, str]) -> SchemaSummary:
    """Counts for the requested antibiotics among the cohort's isolates.

    `interpretation_columns` maps a label (e.g. "ciprofloxacin") to the exact interpretation column name. The mapping is
    explicit, so no column is found by fuzzy matching and no antibiotic is substituted for another.
    """
    columns, delimiter = read_header(path)
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
            reader = csv.reader(fh, delimiter=delimiter)
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


# --- Amendment C4 (2026-10-04): the Sample field, for cohort matching and source classification ---

SAMPLE = "Sample"
MISSING = "missing"                      # no Sample value in any of the isolate's records
CONFLICTING = "conflicting"              # the isolate's records give different Sample values
MIN_CELL = 5                             # a category is named only if at least 5 cohort isolates hold it
MAX_CATEGORY_CHARS = 60
MAX_CATEGORY_WORDS = 6
MAX_CATEGORIES = 100                     # more distinct categories than this: the field is treated as free text
MAX_SUPPRESSED_SHARE = 0.05              # suppressed categories holding more than this share: also free text
CLASSES = ("clinical", "screening", "ambiguous")


@dataclass(frozen=True)
class SampleSummary:
    """Amendment C4's permitted outputs for the Sample field, and nothing else."""
    categories: dict[str, int]           # category -> cohort isolates holding it (named categories only)
    missing: int
    conflicting: int
    suppressed_categories: int
    suppressed_isolates: int
    matched_isolates: int
    unmatched_isolates: int
    schema_problem: str | None           # why the field was treated as free text, or None


def nameable(category: str) -> bool:
    """C4's content rules: short, few words, no run of 4 digits, no "@" or "http", printable."""
    return (len(category) <= MAX_CATEGORY_CHARS and len(category.split()) <= MAX_CATEGORY_WORDS
            and not re.search(r"\d{4}", category) and "@" not in category and "http" not in category.lower()
            and category.isprintable())


def _isolate_samples(path: str | Path, cohort_ids: Collection[str]) -> tuple[dict[str, str], int]:
    """Each matched cohort isolate's single Sample value, MISSING or CONFLICTING, and the cohort's size.

    Internal: the result never leaves this module. Only the Identifier and Sample fields of a row are looked at.
    """
    columns, delimiter = read_header(path)
    if IDENTIFIER not in columns or SAMPLE not in columns:
        raise RestrictedReaderError("no Identifier or no Sample column")
    id_index, sample_index = columns.index(IDENTIFIER), columns.index(SAMPLE)
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
                if is_non_missing(row[sample_index]):
                    values.add(row[sample_index].strip())
    except RestrictedReaderError:
        raise
    except (OSError, csv.Error, UnicodeDecodeError):
        raise RestrictedReaderError(f"unreadable at line {line}") from None
    single = {i: (MISSING if not v else next(iter(v)) if len(v) == 1 else CONFLICTING) for i, v in seen.items()}
    return single, len(cohort)


def _named(single: dict[str, str]) -> tuple[dict[str, int], dict[str, int], str | None]:
    counts = Counter(v for v in single.values() if v not in (MISSING, CONFLICTING))
    named = {c: n for c, n in counts.items() if n >= MIN_CELL and nameable(c)}
    suppressed = {c: n for c, n in counts.items() if c not in named}
    problem = None
    if len(counts) > MAX_CATEGORIES:
        problem = f"more than {MAX_CATEGORIES} distinct categories: the field is treated as free text"
    elif single and sum(suppressed.values()) > MAX_SUPPRESSED_SHARE * len(single):
        problem = "suppressed categories hold more than 5 % of matched isolates: the field is treated as free text"
    if problem:
        named, suppressed = {}, dict(counts)
    return named, suppressed, problem


def sample_categories(path: str | Path, cohort_ids: Collection[str]) -> SampleSummary:
    """The distinct Sample categories among the cohort's isolates, with isolate counts (amendment C4)."""
    single, cohort_size = _isolate_samples(path, cohort_ids)
    named, suppressed, problem = _named(single)
    values = Counter(single.values())
    return SampleSummary(categories=dict(sorted(named.items(), key=lambda kv: (-kv[1], kv[0]))),
                         missing=values[MISSING], conflicting=values[CONFLICTING],
                         suppressed_categories=len(suppressed), suppressed_isolates=sum(suppressed.values()),
                         matched_isolates=len(single), unmatched_isolates=cohort_size - len(single),
                         schema_problem=problem)


def source_classes(path: str | Path, cohort_ids: Collection[str], mapping: Mapping[str, str]) -> dict[str, str]:
    """Each matched cohort isolate's class under a fixed mapping (amendment C5): "clinical", "screening" or "ambiguous".

    Only named categories can be mapped. A category the mapping does not list, a suppressed category, "missing" and
    "conflicting" are all "ambiguous". No Sample value is returned; with a schema problem, nothing is mapped.
    """
    if any(c not in CLASSES for c in mapping.values()):
        raise RestrictedReaderError("a mapping class is not clinical, screening or ambiguous")
    single, _ = _isolate_samples(path, cohort_ids)
    named, _, problem = _named(single)
    if problem:
        raise RestrictedReaderError("the Sample field has a schema problem; no mapping is applied")
    unknown = [c for c in mapping if c not in named]
    if unknown:
        raise RestrictedReaderError(f"the mapping lists {len(unknown)} categories that are not named categories")
    return {i: mapping.get(v, "ambiguous") if v in named else "ambiguous" for i, v in single.items()}


def match_counts(path: str | Path, cohort_ids: Collection[str]) -> tuple[int, int]:
    """The numbers of cohort isolates matched to at least one record by Identifier, and unmatched (amendments A2, C4).

    Only the Identifier field is looked at, and only these two counts leave the function.
    """
    columns, delimiter = read_header(path)
    if IDENTIFIER not in columns:
        raise RestrictedReaderError("no Identifier column")
    id_index = columns.index(IDENTIFIER)
    cohort, matched = set(cohort_ids), set()
    line = 1
    try:
        with open(path, encoding=ENCODING, newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            next(reader)
            for line, row in enumerate(reader, start=2):
                if len(row) != len(columns):
                    raise RestrictedReaderError(f"row length differs from the header at line {line}")
                if row[id_index].strip() in cohort:
                    matched.add(row[id_index].strip())
    except RestrictedReaderError:
        raise
    except (OSError, csv.Error, UnicodeDecodeError):
        raise RestrictedReaderError(f"unreadable at line {line}") from None
    return len(matched), len(cohort - matched)


def identifier_profile(path: str | Path) -> dict[str, object]:
    """Aggregate shape of AMR.csv's Identifier field, to diagnose matching: the numbers of rows and distinct
    identifiers, and distinct identifiers by length and by character class. No identifier leaves the function."""
    columns, delimiter = read_header(path)
    if IDENTIFIER not in columns:
        raise RestrictedReaderError("no Identifier column")
    id_index, rows, distinct = columns.index(IDENTIFIER), 0, set()
    line = 1
    try:
        with open(path, encoding=ENCODING, newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            next(reader)
            for line, row in enumerate(reader, start=2):
                if len(row) != len(columns):
                    raise RestrictedReaderError(f"row length differs from the header at line {line}")
                rows += 1
                distinct.add(row[id_index].strip())
    except RestrictedReaderError:
        raise
    except (OSError, csv.Error, UnicodeDecodeError):
        raise RestrictedReaderError(f"unreadable at line {line}") from None

    def kind(i: str) -> str:
        if i.isdigit():
            return "digits only"
        return "letters and digits" if i.isascii() and i.isalnum() else "other characters"
    return {"rows": rows, "distinct_identifiers": len(distinct),
            "by_length": dict(sorted(Counter(len(i) for i in distinct).items())),
            "by_character_class": dict(sorted(Counter(kind(i) for i in distinct).items()))}


def suppression_profile(path: str | Path, cohort_ids: Collection[str]) -> dict[str, int]:
    """Why Sample categories are suppressed, in aggregate (amendment C4: "report the schema problem"). Counts of
    categories and of isolates by reason; no category name and no identifier leaves the function."""
    single, _ = _isolate_samples(path, cohort_ids)
    counts = Counter(v for v in single.values() if v not in (MISSING, CONFLICTING))
    big = {c: n for c, n in counts.items() if n >= MIN_CELL}
    content_ok = {c for c in counts if nameable(c)}
    return {"categories": len(counts), "matched_isolates": len(single),
            "categories_eligible_by_size_and_content": sum(1 for c in big if c in content_ok),
            "isolates_in_eligible_categories": sum(n for c, n in big.items() if c in content_ok),
            "categories_below_min_isolates": len(counts) - len(big),
            "isolates_in_categories_below_min": sum(n for c, n in counts.items() if c not in big),
            "categories_failing_content_rules": sum(1 for c in counts if c not in content_ok),
            "isolates_in_categories_failing_content_rules": sum(n for c, n in counts.items() if c not in content_ok),
            "missing": sum(1 for v in single.values() if v == MISSING),
            "conflicting": sum(1 for v in single.values() if v == CONFLICTING)}
