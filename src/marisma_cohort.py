"""Deterministic replicate selection for the Version 2.0 cohort (docs/v2.0_marisma_plan.md, amendment A6.2).

For each isolate:
- biological-replicate folders are examined in ascending order, and within each, technical-replicate folders in
  ascending order;
- folder names are ordered numerically when all of them are integers, and otherwise by Unicode code point;
- the first spectrum that passes the reader's checks is used;
- if none passes, the isolate is excluded, and every failure reason is counted.

An isolate without any replicate folder is counted separately from reader failures. No label is used anywhere here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from src.bruker import BrukerReadError

NO_REPLICATE_FOLDER = "no_replicate_folder"
NO_PASSING_REPLICATE = "no_passing_replicate"


def folder_order(names: Iterable[str]) -> list[str]:
    """Ascending: numerically if every name is an integer, otherwise by Unicode code point."""
    names = list(names)
    if names and all(n.isascii() and n.isdigit() for n in names):
        return sorted(names, key=lambda n: (int(n), n))   # "01" and "1" stay in a fixed order
    return sorted(names)


@dataclass(frozen=True)
class Selection:
    spectrum: Any                       # the first passing spectrum, or None
    biological: str | None              # its folders
    technical: str | None
    status: str                         # "selected", NO_REPLICATE_FOLDER or NO_PASSING_REPLICATE
    failures: tuple[str, ...] = field(default=())   # reader reasons of the replicates tried before (or instead)


def select_first_passing(replicates: Mapping[str, Mapping[str, Any]], read: Callable[[Any], Any]) -> Selection:
    """Try replicates in the registered order; `read` raises BrukerReadError for a spectrum that fails a check."""
    tried = [(b, t) for b in folder_order(replicates) for t in folder_order(replicates[b])]
    if not tried:
        return Selection(None, None, None, NO_REPLICATE_FOLDER)
    failures: list[str] = []
    for b, t in tried:
        try:
            spectrum = read(replicates[b][t])
        except BrukerReadError as exc:
            failures.append(exc.reason)
            continue
        return Selection(spectrum, b, t, "selected", tuple(failures))
    return Selection(None, None, None, NO_PASSING_REPLICATE, tuple(failures))


def exclusion_summary(selections: Iterable[Selection]) -> dict[str, object]:
    """Aggregate counts only: isolates by status, and failed replicate attempts by reader reason."""
    selections = list(selections)
    status = Counter(s.status for s in selections)
    reasons = Counter(r for s in selections for r in s.failures)
    excluded_reasons = Counter(r for s in selections if s.status == NO_PASSING_REPLICATE for r in s.failures)
    return {"isolates": len(selections), "by_status": dict(sorted(status.items())),
            "failed_replicate_attempts_by_reason": dict(sorted(reasons.items())),
            "reasons_among_excluded_isolates": dict(sorted(excluded_reasons.items()))}


def pause_check(selections: Iterable[Selection], threshold: float = 0.05) -> dict[str, object]:
    """Amendment A6.4: pause if isolates with no passing replicate exceed 5 % of isolates with a replicate folder.

    Denominator: isolates (after the species, year and source rules) with at least one replicate folder.
    Numerator: those excluded because no replicate passes the reader's checks.
    """
    selections = list(selections)
    denominator = sum(1 for s in selections if s.status != NO_REPLICATE_FOLDER)
    numerator = sum(1 for s in selections if s.status == NO_PASSING_REPLICATE)
    share = numerator / denominator if denominator else float("nan")
    return {"numerator": numerator, "denominator": denominator, "share": share, "threshold": threshold,
            "pause": bool(denominator) and share > threshold}
