# Research demo brief: the frozen ciprofloxacin model

**Date:** 2026-10-04, after the Version 2.0 execution record.

**Status:** an implementation brief. Nothing in it is built yet, and it authorises no experiment, no use of patient
data, no deployment and no change to the served model, API or result page.

**For:** whoever implements the demo, and the owner who commissions it.

## 1. Purpose

The demo shows, for teaching and methods discussion, what this research prototype does:
- a MALDI-TOF spectrum file goes in;
- the frozen model's research score comes out;
- the score is shown beside the published evaluation results and their limits.

It is not a clinical tool and must not look like one. The whole demo is bounded by the rules in sections 5–7.

## 2. Inputs: synthetic spectra and approved aggregate results only

- **Synthetic spectra only.**
  - Generate them deterministically (seed 42), in the accepted format: a `.txt` file of two whitespace-separated
    numeric columns, m/z and intensity, optionally after `#` comment lines and a header line.
  - Each file needs at least 100 points, and at most 200,000 points and 4 MB (the API's limits).
  - Each file's first comment line says `SYNTHETIC: not a measured isolate`.
  - No DRIAMS or MARISMa spectrum, and no spectrum from any laboratory, patient or isolate, may be used. Nothing is
    uploaded from outside.
- **What a synthetic score means: nothing about any organism.** It shows the pipeline's mechanics only (reading,
  preprocessing, scoring, labelling), and the demo says so beside every result.
- **Aggregate results only,** copied from committed files and never recomputed:
  - `results/metrics/v0.4/ecoli_ciprofloxacin/` for the internal test;
  - `results/metrics/v2.0/tables.md` for MARISMa.

## 3. The frozen model (read-only; nothing here may change)

| Item | Value |
|---|---|
| Model | `v0.4.0-tuned_lightgbm-random-seed42`: tuned LightGBM with sigmoid calibration, trained on 2,977 DRIAMS-A *E. coli* spectra for ciprofloxacin |
| Bundle SHA-256 | `d59d6d7deafa1af464c33ebefc0f8f641a80eb70f7b51039e6c2ede0fd841c8b` (the served bundle) |
| Preprocessing | square root, Savitzky–Golay, SNIP baseline, TIC normalisation, 6,000 bins of 3 Da over 2,000–20,000 Da; feature fingerprint `347cbd6d5d956ff9` |
| Research cut-off | 0.14261540693905073, the highest validation threshold with sensitivity ≥ 0.90 |
| Confidence zone (Version 0.6) | score below 0.10255963637363619 = "High-confidence susceptible". There is no confident-resistant zone |
| Serving | the existing CLI (`scripts/predict_spectrum.py`) or the local API, version 1.0.0, bound to 127.0.0.1, with no authentication, used unchanged |
| Response fields | `prediction`, `resistance_probability`, `threshold`, `confidence`, `advice` (uncertain only), `explanation` (optional), `disclaimer`, `model_version`, timings |

## 4. The results the demo shows, with their limits (copied verbatim)

- **Internal test (DRIAMS-A, Version 0.4).** AUROC 0.751 [0.696, 0.807]. At the cut-off, sensitivity 0.827 and
  specificity 0.458.
- **External evaluation (MARISMa, Version 2.0), as registered:** "The frozen ciprofloxacin model achieved AUROC 0.772
  (descriptive 95% interval 0.744–0.798) on 1,145 eligible MARISMa E. coli isolates from 2024, with evidence of
  above-chance ranking under the isolate-independence assumption." Brunner–Munzel inference is approximate and assumes
  independent isolates. Holm adjustment does not repair invalid component p-values. Missing patient linkage leaves
  actual error control uncertain.
- **Shown with equal prominence beside it:**
  - sensitivity 0.897 and specificity 0.393 at the cut-off;
  - under-prediction of resistance (calibration intercept 0.642);
  - the confidence zone's NPV 0.907, with 18 of the 193 isolates in it resistant, which fails its 0.95 research
    target;
  - no demonstrated internal–external gap, which is not equivalence;
  - ceftriaxone not evaluated.
- **Limitations, always on the same screen as the results:**
  - MARISMa has no patient linkage, and its screening status is unknown;
  - its acquisition and preprocessing differ from DRIAMS's;
  - one hospital, 2024 only, one instrument;
  - it is retrospective research data only.

## 5. How each output is worded (research-only interpretation)

| Field | The demo shows | It never shows |
|---|---|---|
| `resistance_probability` | "Model score 0.xx: a research estimate from one model. On MARISMa this model under-predicted resistance." | "probability that this isolate or patient is resistant", "risk" |
| `prediction` (`Resistant` / `Susceptible`) | "Above / below the research cut-off 0.1426" | "Resistant" or "Susceptible" as a result, "positive" or "negative", any colour coding for good or bad |
| `confidence` = High-confidence susceptible | "Inside the research zone (score below 0.1026). On MARISMa this zone failed its target: NPV 0.907, and 18 of the 193 isolates in it were resistant." | "safe", "confirmed", "rule out", "no testing needed", a check mark, green |
| `confidence` = Uncertain | "Outside the research zone", plus the `advice` text verbatim | – |
| `explanation` | "m/z regions that moved this model's score" | protein names, mechanisms, biology |
| `disclaimer` | verbatim, with every output | – |

**The zone is not a safety feature.** It failed its target at MARISMa on its point estimate and its whole interval,
and at DRIAMS-C on its point estimate. The demo may show it only as above, as a research detail with its external
result, and never as a headline.

## 6. Never

- **No antibiotic recommendation.** No treatment, triage or "safe to treat" wording, and no other antibiotic named as
  an alternative.
- **No zone as a validated or safety feature.**
- **No real spectra or patient data,** no storage of inputs, and no prediction history.
- **No deployment beyond localhost** under this brief.
- **No change to the frozen model** or to the served bundle, cut-off, zones, API contract or result page.
- **No new experiment or analysis,** and no number computed for the demo.

## 7. Concrete scope

**Deliverables:**
1. **A generator** that writes three synthetic spectra (seed 42) to a temporary folder, each marked synthetic.
2. **One demo view, local only:** a notebook (preferred) or one static page outside `src/api/`. It scores the synthetic
   files through the existing CLI or local API, unchanged, and shows:
   - the outputs, in the wording of section 5;
   - a results panel copied from section 4, with its limitations.
3. **One test file:**
   - every number shown equals its committed source;
   - the disclaimer appears with every output;
   - a forbidden-phrase check (for example "safe", "treat", "prescribe", "rule out", "confirmed") passes, allowing
     only the `DISCLAIMER` and `ADVICE` constants verbatim;
   - the zone appears only with its external result;
   - no network call goes beyond 127.0.0.1, and no file is written outside a temporary folder.

**Acceptance:** the full test suite and Ruff pass. The served bundle's SHA-256, the production log (115 rows) and the
development log (34 rows) are unchanged.

**Out of scope:**
- hosting, authentication and TLS;
- real or uploaded spectra;
- a ceftriaxone model;
- recalibration or a new cut-off;
- any change to the clinical-use position stated in the README's "Ethics and intended use".
