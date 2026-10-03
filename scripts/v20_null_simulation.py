"""Version 2.0, before any data access: how candidate primary tests behave, on synthetic scores only.

No research data is read, here or anywhere this script imports from. Every number comes from simulated scores
with fixed seeds, to answer three pre-data questions for amendment A of docs/v2.0_marisma_plan.md:

1. Null rejection rates of candidate one-sided tests of "AUROC > 0.5" at the plan's class counts, including
   heavy ties, unequal score spread with AUROC exactly 0.5, and repeat isolates from one patient (clustering),
   which isolate-level data cannot detect.
2. Power at the minimum of 50 resistant and 50 susceptible isolates, under stated binormal assumptions.
3. Which direction scipy's Brunner-Munzel `alternative` argument takes, so the procedure is stated exactly.

AUROC throughout is the Mann-Whitney statistic with ties counted one half (Bamber 1975), which is what
`sklearn.metrics.roc_auc_score` computes.

    python scripts/v20_null_simulation.py --out results/metrics/v2.0/null_simulation.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

ALPHAS = (0.025, 0.05)          # Holm's first and second step for a family of two, one-sided
SETTINGS = ((50, 50), (50, 200), (100, 400))   # (resistant, susceptible)
SCENARIOS = ("identical", "heavy_ties", "unequal_spread", "clustered")
SEED = 42


# ------------------------------------------------------------------------------------------------
# Scores under the null hypothesis (scores carry no information about the label)
# ------------------------------------------------------------------------------------------------

def null_sample(scenario: str, n_pos: int, n_neg: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Scores and 0/1 labels with no association between them. Positives (resistant) come first."""
    if scenario == "identical":          # one continuous distribution for both classes
        s = rng.beta(2.0, 5.0, size=n_pos + n_neg)
    elif scenario == "heavy_ties":       # 8 distinct values, as a small tree ensemble can produce
        s = rng.choice(np.linspace(0.05, 0.40, 8), size=n_pos + n_neg)
    elif scenario == "unequal_spread":   # different distributions, P(pos > neg) exactly 0.5
        s = np.concatenate([rng.normal(0.0, 2.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
    elif scenario == "clustered":
        return clustered_null(n_pos, n_neg, rng)
    else:
        raise ValueError(scenario)
    return s, np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]


def clustered_null(n_pos: int, n_neg: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Patients with 1 + Poisson(1) isolates each (mean 2), one resistance status per patient, and a patient
    effect shared by that patient's scores (intraclass correlation 0.5). Scores are independent of the label,
    but isolates of one patient are not independent units. Patients are drawn until both class counts are
    reached; the last patient of a class is truncated to hit the count exactly. These are assumptions for
    illustration, not estimates for MARISMa."""
    scores, labels, need = [], [], {1: n_pos, 0: n_neg}
    p_pos = n_pos / (n_pos + n_neg)
    while need[1] > 0 or need[0] > 0:
        label = int(rng.random() < p_pos)
        if need[label] == 0:
            continue
        m = min(1 + rng.poisson(1.0), need[label])
        scores.append(rng.normal(0.0, 1.0) + rng.normal(0.0, 1.0, m))
        labels.append(np.full(m, label))
        need[label] -= m
    s, y = np.concatenate(scores), np.concatenate(labels)
    order = np.argsort(-y, kind="stable")          # positives first
    return s[order], y[order]


# ------------------------------------------------------------------------------------------------
# Candidate tests of H0: AUROC <= 0.5 against H1: AUROC > 0.5
# ------------------------------------------------------------------------------------------------

def auc(s: np.ndarray, y: np.ndarray) -> float:
    r = stats.rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_tail_p(s: np.ndarray, y: np.ndarray, b: int, rng: np.random.Generator,
                     stratified: bool) -> tuple[float, int]:
    """The plan's proposal: p = (1 + #{AUROC* <= 0.5}) / (B' + 1) over resamples of the observed data.
    Unstratified resampling can draw a single class; those resamples are undefined, dropped and counted."""
    n = len(s)
    if stratified:
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        idx = np.concatenate([rng.choice(pos, (b, len(pos))), rng.choice(neg, (b, len(neg)))], axis=1)
    else:
        idx = rng.integers(0, n, size=(b, n))
    ss, yy = s[idx], y[idx]
    r = stats.rankdata(ss, axis=1)
    n1 = yy.sum(axis=1)
    n0 = n - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        a = ((r * yy).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    defined = (n1 > 0) & (n0 > 0)
    a = a[defined]
    return float((1 + np.sum(a <= 0.5)) / (len(a) + 1)), int((~defined).sum())


def mann_whitney_p(s: np.ndarray, y: np.ndarray) -> float:
    """One-sided Mann-Whitney test, normal approximation with tie correction (tests equal distributions)."""
    return float(stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="greater", method="asymptotic").pvalue)


BM_ALTERNATIVE = None   # set by orient_brunner_munzel(): the scipy argument meaning "resistant scores larger"


def orient_brunner_munzel() -> str:
    """Return the scipy `alternative` that rejects when the first sample (resistant) tends to be larger."""
    rng = np.random.default_rng(SEED)
    hi, lo = rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 200)
    small = [alt for alt in ("greater", "less")
             if stats.brunnermunzel(hi, lo, alternative=alt, distribution="t").pvalue < 1e-6]
    if len(small) != 1:
        raise RuntimeError(f"cannot orient scipy.stats.brunnermunzel: {small}")
    return small[0]


def brunner_munzel_p(s: np.ndarray, y: np.ndarray) -> float:
    """One-sided Brunner-Munzel test (t approximation): H0 theta = P(R > S) + P(R = S)/2 <= 0.5."""
    return float(stats.brunnermunzel(s[y == 1], s[y == 0], alternative=BM_ALTERNATIVE, distribution="t").pvalue)


def delong_p(s: np.ndarray, y: np.ndarray) -> float:
    """One-sided z-test with DeLong's placement-value variance for a single AUROC."""
    pos, neg = s[y == 1], s[y == 0]
    n1, n0 = len(pos), len(neg)
    r_all = stats.rankdata(np.r_[pos, neg])
    v10 = (r_all[:n1] - stats.rankdata(pos)) / n0                 # placements of resistant scores
    v01 = 1.0 - (r_all[n1:] - stats.rankdata(neg)) / n1           # placements of susceptible scores
    var = v10.var(ddof=1) / n1 + v01.var(ddof=1) / n0
    if var <= 0:
        return float("nan")
    return float(stats.norm.sf((v10.mean() - 0.5) / math.sqrt(var)))


# ------------------------------------------------------------------------------------------------
# Experiments
# ------------------------------------------------------------------------------------------------

def rejection_rates(pvals: list[float]) -> dict[str, object]:
    p = np.asarray(pvals, float)
    ok = p[np.isfinite(p)]
    out: dict[str, object] = {"reps": int(len(p)), "undefined": int(len(p) - len(ok))}
    for a in ALPHAS:
        rate = float(np.mean(ok <= a)) if len(ok) else float("nan")
        out[f"reject_at_{a}"] = round(rate, 4)
        out[f"mc_se_at_{a}"] = round(math.sqrt(a * (1 - a) / max(len(ok), 1)), 4)
    return out


def run_null(reps_fast: int, reps_boot: int, b: int) -> list[dict[str, object]]:
    rows = []
    for si, (n_pos, n_neg) in enumerate(SETTINGS):
        for ci, scen in enumerate(SCENARIOS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, si, ci]))
            fast = {"mann_whitney": [], "brunner_munzel": [], "delong": []}
            boot = {"bootstrap_tail_unstratified": [], "bootstrap_tail_stratified": []}
            single_class = 0
            for rep in range(reps_fast):
                s, y = null_sample(scen, n_pos, n_neg, rng)
                fast["mann_whitney"].append(mann_whitney_p(s, y))
                fast["brunner_munzel"].append(brunner_munzel_p(s, y))
                fast["delong"].append(delong_p(s, y))
                if rep < reps_boot:
                    p, undefined = bootstrap_tail_p(s, y, b, rng, stratified=False)
                    boot["bootstrap_tail_unstratified"].append(p)
                    single_class += undefined
                    boot["bootstrap_tail_stratified"].append(bootstrap_tail_p(s, y, b, rng, stratified=True)[0])
            for method, pv in {**fast, **boot}.items():
                row = {"resistant": n_pos, "susceptible": n_neg, "scenario": scen, "method": method}
                row.update(rejection_rates(pv))
                if method == "bootstrap_tail_unstratified":
                    row["single_class_resamples"] = single_class
                rows.append(row)
            print(f"null {n_pos}/{n_neg} {scen}: done", flush=True)
    return rows


def hanley_mcneil_se(a: float, n1: int, n0: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))


def run_power(reps: int) -> list[dict[str, object]]:
    """Binormal, equal variance: susceptible N(0, 1), resistant N(d, 1), d = sqrt(2) * Phi^-1(AUROC)."""
    rows = []
    for si, (n_pos, n_neg) in enumerate(((50, 50), (50, 200))):
        for ai, a in enumerate((0.65, 0.70)):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 100 + si, ai]))
            d = math.sqrt(2) * stats.norm.ppf(a)
            p_bm = []
            for _ in range(reps):
                s = np.r_[rng.normal(d, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)]
                y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
                p_bm.append(brunner_munzel_p(s, y))
            p_bm = np.asarray(p_bm)
            se = hanley_mcneil_se(a, n_pos, n_neg)
            se_deff2 = se * math.sqrt(2)      # an assumed design effect of 2 for repeat isolates per patient
            z = stats.norm.ppf(1 - ALPHAS[0])
            rows.append({
                "resistant": n_pos, "susceptible": n_neg, "true_auroc": a, "alpha_one_sided": ALPHAS[0],
                "power_brunner_munzel_simulated": round(float(np.mean(p_bm <= ALPHAS[0])), 4),
                "power_hanley_mcneil_normal_approx": round(float(stats.norm.sf(z - (a - 0.5) / se)), 4),
                "power_hanley_mcneil_design_effect_2": round(float(stats.norm.sf(z - (a - 0.5) / se_deff2)), 4),
                "half_width_95_hanley_mcneil": round(1.96 * se, 4),
                "reps": reps,
            })
            print(f"power {n_pos}/{n_neg} AUROC {a}: done", flush=True)
    return rows


def main() -> None:
    global BM_ALTERNATIVE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps-fast", type=int, default=4000)
    ap.add_argument("--reps-boot", type=int, default=1000)
    ap.add_argument("--b", type=int, default=1000, help="bootstrap resamples per simulated dataset")
    ap.add_argument("--reps-power", type=int, default=4000)
    args = ap.parse_args()
    started = time.perf_counter()
    BM_ALTERNATIVE = orient_brunner_munzel()
    result = {
        "purpose": "Version 2.0 amendment A: synthetic null rejection rates and power; no research data read",
        "seed": SEED,
        "brunner_munzel_alternative_for_resistant_larger": BM_ALTERNATIVE,
        "settings": [list(x) for x in SETTINGS], "scenarios": list(SCENARIOS),
        "reps_fast": args.reps_fast, "reps_boot": args.reps_boot, "bootstrap_resamples": args.b,
        "null": run_null(args.reps_fast, args.reps_boot, args.b),
        "power": run_power(args.reps_power),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    result["seconds"] = round(time.perf_counter() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} in {result['seconds']} s")


if __name__ == "__main__":
    main()
