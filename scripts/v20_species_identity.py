"""Version 2.0, amendment C6: the identifiers filed under E. coli and under another genus or species.

Label-blind and model-free; archive metadata and spectra only. For each such identifier it asks whether its folders
look like one organism identified in two ways (conflicting identification) or like two organisms that share an
identifier (an identifier reused across the archive's year or species folders):
- **Folders:** the year, genus and species of each isolate folder, and its replicate folders.
- **Acquisition:** the acquisition date and instrument of each spectrum (from its acqu file).
- **Spectra:** the highest correlation between a spectrum filed under E. coli and one filed under the other species
  (frozen features; settings from the bundle's JSON card). Three references calibrate it:
  - two replicates of one E. coli isolate (one organism);
  - two different E. coli isolates;
  - the E. coli spectrum against spectra of the other species from other identifiers.

The output holds aggregate counts only: no identifier and no path. Per-identifier details go to the work folder
outside the repository.

    python scripts/v20_species_identity.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bruker import BrukerReadError, _text, decode  # noqa: E402
from src.marisma_cohort import layout_folders  # noqa: E402
from src.preprocessing import PreprocessingConfig, PreprocessingError, preprocess_arrays  # noqa: E402
from src.utils import get_logger, git_commit, load_config  # noqa: E402
from src.zip_index import iter_members, read_member  # noqa: E402

FINGERPRINT = "347cbd6d5d956ff9"
SPECIES = ("Escherichia", "Coli")
YEARS = tuple(str(y) for y in range(2018, 2025))
log = get_logger("v20_species_identity")


def frozen_config() -> PreprocessingConfig:
    card = json.loads((ROOT / "models/v0.4/ecoli_ciprofloxacin/best_random.json").read_text(encoding="utf-8"))
    cfg = PreprocessingConfig.from_dict(card["preprocessing"])
    if cfg.fingerprint() != FINGERPRINT:
        raise SystemExit("the frozen preprocessing settings do not match the fingerprint")
    return cfg


class Spectra:
    """Decoded spectra and their frozen features, read straight from the archive."""

    def __init__(self, zip_path: Path, folders: set[str], cfg: PreprocessingConfig):
        self.zip_path, self.cfg = zip_path, cfg
        self.members = {n: m for n, m in iter_members(zip_path) if n.rsplit("/", 1)[0] in folders}

    def read(self, fh, folder: str) -> dict[str, Any] | None:
        acqu = f"{folder}/acqu" if f"{folder}/acqu" in self.members else f"{folder}/acqus"
        if acqu not in self.members or f"{folder}/fid" not in self.members:
            return None
        lines = read_member(fh, self.members[acqu]).decode("latin-1").splitlines()
        try:
            s = decode(lines, read_member(fh, self.members[f"{folder}/fid"]))
            features, _ = preprocess_arrays(s.mz, s.intensity, self.cfg)
        except (BrukerReadError, PreprocessingError):
            return None
        return {"date": _text(lines, "AQ_DATE")[:10], "instrument": _text(lines, "INSTRUM"),
                "features": features.astype(np.float64)}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def quantiles(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), **{f"q{int(q * 100):02d}": float(np.quantile(values, q))
                                      for q in (0.0, 0.05, 0.5, 0.95, 1.0)}}


def main(argv: list[str] | None = None) -> int:
    data_root = Path(load_config()["paths"]["driams_root"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=data_root / "MARISMa_v2.0.0" / "MARISMa.zip")
    parser.add_argument("--work", type=Path, default=data_root / "MARISMa_v2.0.0_work" / "investigation_2026-10-04")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--references", type=int, default=150)
    parser.add_argument("--out", type=Path, default=ROOT / "results/metrics/v2.0/species_identity.json")
    args = parser.parse_args(argv)
    if args.work.resolve().is_relative_to(ROOT.resolve()):
        raise SystemExit("the work folder must be outside the repository")
    args.work.mkdir(parents=True, exist_ok=True)
    cfg, rng = frozen_config(), np.random.default_rng(args.seed)

    isolate_folders, replicate_folders = layout_folders(name for name, _ in iter_members(args.zip))
    taxa: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for year, genus, species, isolate in isolate_folders:
        taxa[isolate].add((year, genus, species))
    ecoli = {i for i, t in taxa.items() if any((g, s) == SPECIES and y in YEARS for y, g, s in t)}
    flagged = sorted(i for i in ecoli if len({(g, s) for _, g, s in taxa[i]}) > 1)
    reps: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for r in sorted(replicate_folders):
        reps[(r.year, r.genus, r.species, r.isolate)].append(r.folder)

    # references: E. coli isolates with two replicates; pairs of different E. coli isolates; the other species
    multi = [k for k, v in reps.items() if (k[1], k[2]) == SPECIES and len(v) >= 2 and k[3] not in flagged]
    pairs_same = [reps[multi[i]][:2] for i in rng.choice(len(multi), size=args.references, replace=False)]
    singles = [v[0] for k, v in reps.items() if (k[1], k[2]) == SPECIES and k[3] not in flagged]
    pairs_diff = [[singles[i], singles[j]] for i, j in rng.choice(len(singles), size=(args.references, 2),
                                                                     replace=False)]
    others = sorted({(g, s) for i in flagged for _, g, s in taxa[i] if (g, s) != SPECIES})
    other_refs = {}
    for g, s in others:
        pool = [v[0] for k, v in reps.items() if (k[1], k[2]) == (g, s) and k[3] not in flagged]
        other_refs[(g, s)] = [pool[i] for i in rng.choice(len(pool), size=min(10, len(pool)), replace=False)]

    needed = {f for i in flagged for k, v in reps.items() if k[3] == i for f in v}
    needed |= {f for p in pairs_same + pairs_diff for f in p} | {f for v in other_refs.values() for f in v}
    spectra = Spectra(args.zip, needed, cfg)
    details, same, diff = [], [], []
    with open(args.zip, "rb") as fh:
        for a, b in pairs_same:
            sa, sb = spectra.read(fh, a), spectra.read(fh, b)
            if sa and sb:
                same.append(corr(sa["features"], sb["features"]))
        for a, b in pairs_diff:
            sa, sb = spectra.read(fh, a), spectra.read(fh, b)
            if sa and sb:
                diff.append(corr(sa["features"], sb["features"]))
        for identifier in flagged:
            folders = {k: v for k, v in reps.items() if k[3] == identifier}
            read = {k: [s for s in (spectra.read(fh, f) for f in v) if s] for k, v in folders.items()}
            ec = [s for k, v in read.items() if (k[1], k[2]) == SPECIES for s in v]
            ot = {k: v for k, v in read.items() if (k[1], k[2]) != SPECIES}
            for k, v in ot.items():
                cross = [corr(a["features"], b["features"]) for a in ec for b in v]
                refs = [spectra.read(fh, f) for f in other_refs[(k[1], k[2])]]
                typical = [corr(a["features"], r["features"]) for a in ec for r in refs if r]
                ec_years = sorted({kk[0] for kk in folders if (kk[1], kk[2]) == SPECIES})
                details.append({
                    "identifier": identifier, "ecoli_years": ec_years, "other": f"{k[1]}/{k[2]}",
                    "other_year": k[0], "same_year": k[0] in ec_years,
                    "ecoli_dates": sorted({s["date"] for s in ec}), "other_dates": sorted({s["date"] for s in v}),
                    "same_acquisition_date": bool({s["date"] for s in ec} & {s["date"] for s in v}),
                    "ecoli_instruments": sorted({s["instrument"] for s in ec}),
                    "other_instruments": sorted({s["instrument"] for s in v}),
                    "max_cross_correlation": max(cross) if cross else None,
                    "median_correlation_with_other_species_references": float(np.median(typical)) if typical
                    else None})
    frame = pd.DataFrame(details)
    frame.to_csv(args.work / "species_identity_details.csv", index=False, lineterminator="\n")

    same_q05 = float(np.quantile(same, 0.05))
    diff_q95 = float(np.quantile(diff, 0.95))

    def pattern(row) -> str:
        r = row["max_cross_correlation"]
        if r is None or pd.isna(r):
            return "no readable spectrum pair"
        if r >= same_q05:
            return "spectra as similar as two replicates of one isolate"
        if r <= row["median_correlation_with_other_species_references"]:
            return "spectra no closer than typical spectra of the other species"
        return "in between"

    frame["pattern"] = frame.apply(pattern, axis=1) if len(frame) else []
    result = {
        "what": "Version 2.0, amendment C6: identifiers filed under E. coli and another genus or species "
                "(archive metadata and spectra only; aggregate counts)",
        "run_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": git_commit(),
        "identifiers": len(flagged), "folder_pairs": int(len(frame)),
        "other_genus_species": {k: int(v) for k, v in frame["other"].value_counts().items()} if len(frame) else {},
        "same_year_folder": int(frame["same_year"].sum()) if len(frame) else 0,
        "same_acquisition_date": int(frame["same_acquisition_date"].sum()) if len(frame) else 0,
        "patterns": {k: int(v) for k, v in frame["pattern"].value_counts().items()} if len(frame) else {},
        "references": {"two_replicates_of_one_ecoli_isolate": quantiles(same),
                       "two_different_ecoli_isolates": quantiles(diff),
                       "thresholds": {"same_isolate_q05": same_q05, "different_isolates_q95": diff_q95}},
        "max_cross_correlation": quantiles(frame["max_cross_correlation"].dropna()) if len(frame) else {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.info("written %s", args.out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
