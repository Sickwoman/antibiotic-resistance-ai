"""Version 2.0, amendment C7: how the frozen pipeline treats the edges of a spectrum. Label-blind and model-free.

The coverage gate stays as approved while this runs; nothing here changes it, the reader or the pipeline. The script
answers three questions, without any label and without loading a model (the frozen preprocessing settings are read
from the bundle's JSON card and checked against the fingerprint 347cbd6d5d956ff9):

1. **Support per feature bin.** For every replicate folder of the cohort, the m/z grid each spectrum actually has
   (from its own acquisition parameters and the length of its fid) is binned exactly as the pipeline bins it. This
   counts, for every one of the 6,000 bins, the acquired points it receives. It shows directly, not from the nominal
   bin width, which spectra leave a bin without data.
2. **Controlled truncation.** Fully covering spectra are cut at chosen m/z values, and their frozen features are
   compared before and after: synthetic spectra, DRIAMS-A raw spectra (about 1,960-20,130 Da) and MARISMa spectra that
   passed the check (about 1,999-21,000 Da). The comparison separates the global effect of total-ion-current
   normalisation from local effects near the edges.
3. **Cohort consequences.** For the approved check and for the candidate rules, the replicate-selection rule of
   amendment A6.2 is applied to every isolate, using every replicate's own decoded spectrum. Exclusions are reported by
   year folder and instrument.

Outputs: results/metrics/v2.0/coverage_investigation.json and results/plots/v2.0/coverage_edge_effects.png (aggregate
only). Per-replicate windows go to the work folder outside the repository.

    python scripts/v20_coverage_investigation.py
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

from src.bruker import BrukerReadError, _double, _text, convert, decode  # noqa: E402
from src.dataset import load_dataset, resolve_relpath  # noqa: E402
from src.marisma_cohort import folder_order  # noqa: E402
from src.preprocessing import PreprocessingConfig, preprocess_arrays, read_raw_spectrum  # noqa: E402
from src.utils import get_logger, git_commit, keep_awake, load_config  # noqa: E402
from src.zip_index import iter_members, read_member  # noqa: E402

FINGERPRINT = "347cbd6d5d956ff9"
QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)
log = get_logger("v20_coverage")

# Truncation scenarios: (name, low cut, high cut). A cut keeps lo <= m/z <= hi; None keeps that end as acquired.
LOW_CUTS = [("start 1999.9 Da (no margin below 2,000)", 1999.9), ("start 2000.05 Da", 2000.05),
            ("start 2000.5 Da", 2000.5), ("start 2002.9 Da (bin 0 partly covered)", 2002.9),
            ("start 2003.1 Da (bin 0 empty)", 2003.1), ("start 2006.1 Da (bins 0-1 empty)", 2006.1),
            ("start 2013.2 Da (latest start observed)", 2013.2)]
HIGH_CUTS = [("end 20130 Da (DRIAMS-A's end)", 20130.0), ("end 20010 Da", 20010.0), ("end 20000.5 Da", 20000.5),
             ("end 19999.9 Da", 19999.9), ("end 19997.1 Da (bin 5999 partly covered)", 19997.1),
             ("end 19996.9 Da (bin 5999 empty)", 19996.9), ("end 19990 Da", 19990.0),
             ("end 19960.3 Da (earliest end observed)", 19960.3)]


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def frozen_config() -> PreprocessingConfig:
    card = json.loads((ROOT / "models/v0.4/ecoli_ciprofloxacin/best_random.json").read_text(encoding="utf-8"))
    cfg = PreprocessingConfig.from_dict(card["preprocessing"])
    if cfg.fingerprint() != FINGERPRINT or card["feature_fingerprint"] != FINGERPRINT:
        raise SystemExit("the frozen preprocessing settings do not match the fingerprint")
    return cfg


def counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def quantiles(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), **{f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES}}


# --- 1. Support per feature bin -------------------------------------------------------------------------------------

def bin_support(mz: np.ndarray, cfg: PreprocessingConfig) -> np.ndarray:
    """Acquired points per feature bin, binned exactly as src/preprocessing.trim and bin_intensities bin them."""
    inside = mz[(mz >= cfg.mz_min) & (mz <= cfg.mz_max)]
    counts, _ = np.histogram(inside, bins=cfg.bin_edges)
    return counts


def replicate_windows(zip_path: Path, folders: pd.DataFrame, cfg: PreprocessingConfig) -> pd.DataFrame:
    """Every replicate folder's decoded spectrum: window, sampling, bin support, and the reader's verdict."""
    wanted = {f"{f}/{p}" for f in folders["folder"] for p in ("fid", "acqu", "acqus")}
    members = {name: member for name, member in iter_members(zip_path) if name in wanted}
    rows = []
    with open(zip_path, "rb") as fh:
        for r in folders.itertuples(index=False):
            acqu = f"{r.folder}/acqu" if f"{r.folder}/acqu" in members else f"{r.folder}/acqus"
            row: dict[str, Any] = {"folder": r.folder, "isolate": r.isolate, "year": r.year,
                                   "biological": r.biological, "technical": r.technical}
            if f"{r.folder}/fid" not in members or acqu not in members:
                rows.append({**row, "verdict": "missing_files"})
                continue
            lines = read_member(fh, members[acqu]).decode("latin-1").splitlines()
            fid = read_member(fh, members[f"{r.folder}/fid"])
            row["instrument"] = _text(lines, "INSTRUM")
            try:
                spectrum = decode(lines, fid)
            except BrukerReadError as exc:
                rows.append({**row, "verdict": exc.reason})
                continue
            try:
                convert(lines, fid)
                row["verdict"] = "passed"
            except BrukerReadError as exc:
                row["verdict"] = exc.reason
            mz = spectrum.mz
            support = bin_support(mz, cfg)
            nonempty = np.flatnonzero(support)
            row.update(first_mz=float(mz[0]), last_mz=float(mz[-1]), points=int(mz.size),
                       spacing_at_start=float(mz[1] - mz[0]), spacing_at_end=float(mz[-1] - mz[-2]),
                       dw=_double(lines, "DW"), empty_bins=int((support == 0).sum()),
                       leading_empty_bins=int(nonempty[0]) if nonempty.size else cfg.n_bins,
                       trailing_empty_bins=int(cfg.n_bins - 1 - nonempty[-1]) if nonempty.size else cfg.n_bins,
                       support_bin_first=int(support[0]), support_bin_last=int(support[-1]),
                       support_interior_min=int(support[1:-1].min()))
            rows.append(row)
    return pd.DataFrame(rows)


# --- 2. Controlled truncation ---------------------------------------------------------------------------------------

def truncate(mz: np.ndarray, y: np.ndarray, lo: float | None, hi: float | None) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(mz.size, dtype=bool)
    if lo is not None:
        keep &= mz >= lo
    if hi is not None:
        keep &= mz <= hi
    return mz[keep], y[keep]


def edge_extent(local: np.ndarray, threshold: float) -> tuple[int, int]:
    """Bins from the low and the high edge out to the last bin whose local change exceeds `threshold`."""
    big = np.flatnonzero(np.abs(local) > threshold)
    low = big[big < local.size // 2]
    high = big[big >= local.size // 2]
    return (int(low.max()) + 1 if low.size else 0), (int(local.size - high.min()) if high.size else 0)


def compare_truncations(spectra: list[tuple[np.ndarray, np.ndarray]], scenarios: list[tuple[str, Any, Any]],
                        cfg: PreprocessingConfig) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """For each scenario, how the frozen features change when the raw spectrum is cut. `local` removes the global
    effect of total-ion-current normalisation (features scale with 1 / TIC), leaving what changes near the edges."""
    per: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    curves: dict[str, list[np.ndarray]] = defaultdict(list)
    for mz, y in spectra:
        full, info = preprocess_arrays(mz, y, cfg)
        full = full.astype(np.float64)
        scale = float(full[100:5900].mean())
        for name, lo, hi in scenarios:
            if (lo is not None and lo <= mz[0]) or (hi is not None and hi >= mz[-1]):
                continue                                   # the spectrum does not reach this cut: nothing to cut
            cmz, cy = truncate(mz, y, lo, hi)
            cut, cinfo = preprocess_arrays(cmz, cy, cfg)
            cut = cut.astype(np.float64)
            tic_ratio = cinfo.total_ion_current / info.total_ion_current
            raw = (cut - full) / scale
            local = (cut * tic_ratio - full) / scale
            low1, high1 = edge_extent(local, 0.01)
            low01, high01 = edge_extent(local, 0.001)
            p = per[name]
            p["tic_ratio"].append(tic_ratio)
            p["bins_from_low_edge_changed_over_1pct"].append(low1)
            p["bins_from_high_edge_changed_over_1pct"].append(high1)
            p["bins_from_low_edge_changed_over_0.1pct"].append(low01)
            p["bins_from_high_edge_changed_over_0.1pct"].append(high01)
            p["interior_max_abs_local_change"].append(float(np.abs(local[200:5800]).max()))
            p["max_abs_raw_change"].append(float(np.abs(raw).max()))
            p["first_bin_local_change"].append(float(local[0]))
            p["last_bin_local_change"].append(float(local[-1]))
            p["correlation_with_uncut"].append(float(np.corrcoef(full, cut)[0, 1]))
            curves[name].append(np.abs(local))
    summary = {name: {k: quantiles(v) for k, v in metrics.items()} for name, metrics in per.items()}
    return summary, {name: np.median(np.vstack(c), axis=0) for name, c in curves.items()}


def synthetic_spectra(n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Smooth synthetic spectra on a linear-TOF grid (1,950-21,050 Da), with a decaying baseline, peaks of m/z-
    proportional width and Poisson noise. Peaks are random: no real organism is imitated."""
    rng = np.random.default_rng(seed)
    ml1, ml2, ml3 = 5.42e6, 440.0, -0.012
    b = np.sqrt(1e12 / ml1)
    def mass(t):
        return ((-b + np.sqrt(b * b - 4 * ml3 * (ml2 - t))) / (2 * ml3)) ** 2
    t0 = 19350.0
    tof = t0 + 2.0 * np.arange(24000)
    mz = mass(tof)
    mz = mz[(mz >= 1950) & (mz <= 21050)]
    out = []
    for _ in range(n):
        y = 4000 * np.exp(-(mz - 1950) / 2500) + 300
        for c in rng.uniform(2000, 20500, size=60):
            y += rng.lognormal(7, 1) * np.exp(-0.5 * ((mz - c) / (c * 6e-4)) ** 2)
        out.append((mz, rng.poisson(y).astype(np.float64)))
    return out


# --- 3. Cohort consequences -----------------------------------------------------------------------------------------

def rule_passes(rows: pd.DataFrame, rule: str) -> pd.Series:
    decoded = rows["verdict"].isin(["passed", "range_not_covered"])     # every other check passed
    first, last = rows["first_mz"], rows["last_mz"]
    if rule == "approved: first <= 2000 Da and last >= 20000 Da":
        return decoded & (first <= 2000) & (last >= 20000)
    if rule == "every feature bin has acquired data":
        return decoded & (rows["empty_bins"] == 0)
    raise ValueError(rule)


RULES = ("approved: first <= 2000 Da and last >= 20000 Da", "every feature bin has acquired data")


def consequences(windows: pd.DataFrame) -> dict[str, Any]:
    """Amendment A6.2 per isolate under each rule: the first replicate, in the registered order, that passes."""
    out: dict[str, Any] = {}
    windows = windows.copy()
    for rule in RULES:
        windows["ok"] = rule_passes(windows, rule)
        records = []
        for _isolate, g in windows.groupby("isolate", sort=True):
            order = {(b, t): k for k, (b, t) in enumerate(
                (b, t) for b in folder_order(g["biological"].unique())
                for t in folder_order(g.loc[g["biological"] == b, "technical"].unique()))}
            g = g.assign(rank=[order[(b, t)] for b, t in zip(g["biological"], g["technical"], strict=True)])
            g = g.sort_values("rank")
            chosen = g[g["ok"]]
            first = chosen.iloc[0] if len(chosen) else g.iloc[0]
            records.append({"year": first["year"], "instrument": first.get("instrument", ""),
                            "kept": bool(len(chosen))})
        frame = pd.DataFrame(records)
        table = frame.groupby(["year", "instrument"])["kept"].agg(isolates="size", kept="sum")
        table["excluded"] = table["isolates"] - table["kept"]
        out[rule] = {"isolates": int(len(frame)), "excluded": int((~frame["kept"]).sum()),
                     "excluded_share": float((~frame["kept"]).mean()),
                     "by_year_and_instrument": {f"{y} {i}": {k: int(v) for k, v in row.items()}
                                                for (y, i), row in table.iterrows()}}
    return out


def plot_edges(curves: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = ["-", "--", ":", "-."]
    for (source, cs), style in zip(curves.items(), styles, strict=False):
        for name, curve in cs.items():
            if name.startswith("start 2000.05") or name.startswith("start 1999.9"):
                axes[0].plot(np.arange(60), 100 * curve[:60], style, label=f"{source}: {name}")
            if name.startswith("end 20000.5") or name.startswith("end 20130"):
                axes[1].plot(np.arange(5900, 6000), 100 * curve[5900:], style, label=f"{source}: {name}")
    axes[0].set(title="Low edge: features after cutting the start", xlabel="feature bin (bin 0 = 2,000-2,003 Da)",
                ylabel="median |local change|, % of mean bin")
    axes[1].set(title="High edge: features after cutting the end", xlabel="feature bin (bin 5999 = 19,997-20,000 Da)")
    for ax in axes:
        ax.set_yscale("symlog", linthresh=0.1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    data_root = Path(config["paths"]["driams_root"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=data_root / "MARISMa_v2.0.0" / "MARISMa.zip")
    parser.add_argument("--run1", type=Path, default=data_root / "MARISMa_v2.0.0_work" / "run1_blocked_2026-10-03")
    parser.add_argument("--work", type=Path, default=data_root / "MARISMa_v2.0.0_work" / "investigation_2026-10-04")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "results/metrics/v2.0/coverage_investigation.json")
    parser.add_argument("--plot", type=Path, default=ROOT / "results/plots/v2.0/coverage_edge_effects.png")
    args = parser.parse_args(argv)
    if args.work.resolve().is_relative_to(ROOT.resolve()):
        raise SystemExit("the work folder must be outside the repository")
    args.work.mkdir(parents=True, exist_ok=True)
    cfg = frozen_config()
    rng = np.random.default_rng(args.seed)
    result: dict[str, Any] = {"what": "Version 2.0, amendment C7: the frozen pipeline at the edges of a spectrum "
                                      "(label-blind, model-free; the coverage gate unchanged)",
                              "run_utc": now(), "git_commit": git_commit(), "feature_fingerprint": cfg.fingerprint(),
                              "seed": args.seed}
    with keep_awake():
        log.info("decoding every replicate folder of the run 1 cohort")
        folders = pd.read_csv(args.run1 / "replicate_folders.csv", dtype=str, keep_default_na=False)
        windows = replicate_windows(args.zip, folders, cfg)
        windows.to_csv(args.work / "replicate_windows.csv", index=False, lineterminator="\n")
        decoded = windows[windows["verdict"].isin(["passed", "range_not_covered"])]
        result["support"] = {
            "replicate_folders": int(len(windows)), "verdicts": counts(windows["verdict"]),
            "by_instrument": {inst: {
                "spectra": int(len(g)), "first_mz": quantiles(g["first_mz"]), "last_mz": quantiles(g["last_mz"]),
                "spacing_at_start_da": quantiles(g["spacing_at_start"]),
                "spacing_at_end_da": quantiles(g["spacing_at_end"]), "dw_ns": counts(g["dw"]),
                "with_an_empty_bin": int((g["empty_bins"] > 0).sum()),
                "leading_empty_bins": counts(g["leading_empty_bins"]),
                "trailing_empty_bins": counts(g["trailing_empty_bins"]),
                "support_in_bin_0": quantiles(g["support_bin_first"]),
                "support_in_bin_5999": quantiles(g["support_bin_last"]),
                "support_in_interior_bins_min": int(g["support_interior_min"].min())}
                for inst, g in decoded.groupby("instrument")},
            "three_da_claim": {
                "spectra_with_first_below_2003_and_last_at_least_19997": int(
                    ((decoded["first_mz"] < 2003) & (decoded["last_mz"] >= 19997)).sum()),
                "of_which_with_an_empty_bin": int(((decoded["first_mz"] < 2003) & (decoded["last_mz"] >= 19997)
                                                   & (decoded["empty_bins"] > 0)).sum()),
                "spectra_with_no_empty_bin": int((decoded["empty_bins"] == 0).sum()),
                "of_which_outside_the_3_da_window": int(((decoded["empty_bins"] == 0)
                                                         & ~((decoded["first_mz"] < 2003)
                                                             & (decoded["last_mz"] >= 19997))).sum())},
        }

        log.info("controlled truncation")
        scenarios = [(n, lo, None) for n, lo in LOW_CUTS] + [(n, None, hi) for n, hi in HIGH_CUTS] + [
            ("window 1999.9-19998.8 Da (a FLEX-PC spectrum that fails)", 1999.9, 19998.8),
            ("window 1999.4-20004.6 Da (a FLEX-PC spectrum that passes)", 1999.4, 20004.6)]
        synthetic = synthetic_spectra(50, args.seed)
        X, meta, _ = load_dataset(ROOT / "data/processed/ecoli_ciprofloxacin", verify_x=False)
        rows_a = meta[meta["site"] == "DRIAMS-A"]
        pick = rows_a.iloc[np.sort(rng.choice(len(rows_a), size=args.n, replace=False))]
        driams = []
        for rel in pick["spectrum_relpath"]:
            values = read_raw_spectrum(resolve_relpath(data_root, rel), min_points=cfg.min_raw_points)
            driams.append((values[:, 0], values[:, 1]))
        mbt = decoded[(decoded["instrument"] == "MBT-WIN10") & (decoded["verdict"] == "passed")]
        chosen = mbt.iloc[np.sort(rng.choice(len(mbt), size=args.n, replace=False))]["folder"].tolist()
        chosen_set = set(chosen)
        members = {n: m for n, m in iter_members(args.zip) if n.rsplit("/", 1)[0] in chosen_set}
        marisma = []
        with open(args.zip, "rb") as fh:
            for f in chosen:
                acqu = f"{f}/acqu" if f"{f}/acqu" in members else f"{f}/acqus"
                s = convert(read_member(fh, members[acqu]).decode("latin-1").splitlines(),
                            read_member(fh, members[f"{f}/fid"]))
                marisma.append((s.mz, s.intensity))
        truncation, curves = {}, {}
        for source, spectra in (("synthetic", synthetic), ("DRIAMS-A raw", driams), ("MARISMa MBT-WIN10", marisma)):
            truncation[source], curves[source] = compare_truncations(spectra, scenarios, cfg)
            truncation[source]["spectra"] = len(spectra)
            truncation[source]["window"] = {"first_mz": quantiles([s[0][0] for s in spectra]),
                                            "last_mz": quantiles([s[0][-1] for s in spectra])}
        result["truncation"] = truncation
        plot_edges(curves, args.plot)

        log.info("cohort consequences")
        result["consequences"] = consequences(windows)
    result["finished_utc"] = now()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.info("written %s", args.out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
