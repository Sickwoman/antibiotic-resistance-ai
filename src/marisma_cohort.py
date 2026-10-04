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
from typing import Any, NamedTuple

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


class SpectrumFolder(NamedTuple):
    year: str
    genus: str
    species: str
    isolate: str
    biological: str
    technical: str
    folder: str                         # the archive folder holding `fid` (no trailing slash)


ROOT_FOLDER = "MARISMa"
SPECTRUM_FOLDER = "1SLin"


def parse_spectrum_member(name: str) -> SpectrumFolder | None:
    """Place a `fid` member in MARISMa 2.0.0's layout, or return None if it lies outside it.

    The layout, read from the archive's member names on 2026-10-03 (every one of its 241,980 `fid` files follows it):
    MARISMa/<year>/<genus>/<species>/<isolate>/<biological replicate>/<technical replicate>/1SLin/fid, where the
    isolate folder is MARISMa's identifier, a biological replicate is a target position such as "0_A1", and technical
    replicates are numbered. Only a file named exactly "fid" counts, as in readBrukerFlexData, so copies named
    "fid (2)" are not spectra here.
    """
    parts = name.split("/")
    if len(parts) != 9 or parts[0] != ROOT_FOLDER or parts[7] != SPECTRUM_FOLDER or parts[8] != "fid":
        return None
    if any(not p for p in parts):
        return None
    year, genus, species, isolate, biological, technical = parts[1:7]
    return SpectrumFolder(year, genus, species, isolate, biological, technical, "/".join(parts[:8]))


def layout_folders(names: Iterable[str]) -> tuple[set[tuple[str, str, str, str]], set[SpectrumFolder]]:
    """Isolate folders, as (year, genus, species, isolate), and technical-replicate folders, from member names.

    Only directory components count, so a file such as ".DS_Store" is never taken for a folder. A technical-replicate
    folder's `folder` is where its spectrum belongs (<...>/<technical>/1SLin), whether or not a `fid` is there: a
    missing spectrum is then a reader failure ("missing_files"), and an isolate folder with no technical-replicate
    folder at all is counted as having no replicate folder (amendment A6.4).
    """
    isolates: set[tuple[str, str, str, str]] = set()
    replicates: set[SpectrumFolder] = set()
    for name in names:
        dirs = name.split("/")[:-1]          # "a/b/" -> ["a", "b"]; "a/b/file" -> ["a", "b"]
        if len(dirs) < 5 or dirs[0] != ROOT_FOLDER or not all(dirs):
            continue
        isolates.add((dirs[1], dirs[2], dirs[3], dirs[4]))
        if len(dirs) >= 7:
            year, genus, species, isolate, biological, technical = dirs[1:7]
            replicates.add(SpectrumFolder(year, genus, species, isolate, biological, technical,
                                          "/".join([*dirs[:7], SPECTRUM_FOLDER])))
    return isolates, replicates


# --- Coverage rules (amendment C7's investigation), applied with the replicate order of amendment A6.2 --------------

COVERAGE_RULES = ("approved: first <= 2000 Da and last >= 20000 Da", "every feature bin has acquired data")


def coverage_passes(windows, rule: str):
    """Which decoded replicates pass `rule`. `windows` holds, per replicate, the reader's verdict (a spectrum refused
    for any reason other than its m/z range never passes), its first and last m/z, and its number of empty bins."""
    decoded = windows["verdict"].isin(["passed", "range_not_covered"])
    if rule == COVERAGE_RULES[0]:
        return decoded & (windows["first_mz"] <= 2000) & (windows["last_mz"] >= 20000)
    if rule == COVERAGE_RULES[1]:
        return decoded & (windows["empty_bins"] == 0)
    raise ValueError(f"unknown coverage rule {rule!r}")


def coverage_outcomes(windows, rule: str) -> list[dict[str, Any]]:
    """Per isolate under `rule`: whether a replicate passes, taking replicates in the registered order (A6.2), and the
    year and instrument of the chosen replicate, or of the first replicate when none passes."""
    windows = windows.assign(ok=coverage_passes(windows, rule))
    out = []
    for isolate, g in windows.groupby("isolate", sort=True):
        ranked = [(b, t) for b in folder_order(g["biological"].unique())
                  for t in folder_order(g.loc[g["biological"] == b, "technical"].unique())]
        rank = {key: k for k, key in enumerate(ranked)}
        g = g.assign(rank=[rank[(b, t)] for b, t in zip(g["biological"], g["technical"], strict=True)]).sort_values(
            "rank")
        chosen = g[g["ok"]]
        row = chosen.iloc[0] if len(chosen) else g.iloc[0]
        out.append({"isolate": isolate, "year": row["year"], "instrument": row.get("instrument", ""),
                    "kept": bool(len(chosen))})
    return out
