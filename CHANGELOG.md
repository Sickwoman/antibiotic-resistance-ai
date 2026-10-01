# Changelog

Versions of this research prototype, newest first, each with its pre-registered status: confirmatory, exploratory,
descriptive or engineering. Every number is copied from the
[research report](docs/research_report_v0.1-v1.4.md) (Tables 1, 2 and 5, and sections 5, 10 and 12), which ties each
one to its saved artifact. Nothing here is new. "Not demonstrated" means the 95 % interval includes 0; it never means
"equivalent". **This is not a clinically validated diagnostic tool, and no result supports clinical use.**

Only 1.4.0 is published as a GitHub release; the earlier versions are listed for the record. Their original commits
resolve through the `research-archive/…` tags ([review checklist](docs/review_checklist.md), section 5).

## 1.4.0 — 2026-10-02 (release)

- Closes the research cycle V0.1–V1.4. The [research report](docs/research_report_v0.1-v1.4.md) comes with an
  [evidence map](docs/evidence_map.md), a [reproduction guide](docs/reproduction_guide.md), a
  [research summary](docs/research_summary.md) and a [review checklist](docs/review_checklist.md).
- Contains Versions 1.1–1.4, squash-merged into `main` on 2026-10-01 (PRs #26–#30).
- Model iteration on the ceftriaxone development pool has stopped, by the rule the Version 1.4 plan fixed before its
  run (report, section 10).
- **Conclusions (report, section 12):**
  - Spectra from DRIAMS-A support modest ranking of isolates by resistance for both antibiotics: AUROC about
    0.71–0.75, confirmatory, at a single site.
  - No intervention met its research objective.
  - No generalisation gap was demonstrated, and none could be excluded.
  - The 0.90-sensitivity cut-off did not hold on new data.

## Version 1.4 — 2026-09-30 to 2026-10-01: discrimination audit and screening isolates (descriptive, exploratory)

- Plan: [docs/v1.4_screening_plan.md](docs/v1.4_screening_plan.md) (amendment 11).
- Adding 495 excluded screening isolates to training only: A1 − A0 AUROC −0.007 [−0.056, +0.043], not demonstrated.

## Version 1.3 — 2026-09-30: an uncertainty-aware cut-off over time (exploratory)

- Plan: [docs/v1.3_threshold_plan.md](docs/v1.3_threshold_plan.md) (amendment 10).
- U − E pooled sensitivity +0.115 [+0.019, +0.248]: the primary endpoint was met.
- The research objective was not met: U's specificity was 0.160 (it flags 85 % of isolates), "unhelpful" under the
  pre-set order.

## Version 1.2 — 2026-09-30: calibration and the cut-off, development only (exploratory)

- Plan: [docs/v1.2_calibration_plan.md](docs/v1.2_calibration_plan.md) (amendment 9).
- Brier(B) − Brier(C) +0.0002 [−0.0024, +0.0028], not demonstrated.
- C's cut-off was steadier (SD 0.058 vs 0.151, descriptive).

## Version 1.1 — 2026-09-30: E. coli + ceftriaxone (confirmatory)

- Plan: [docs/v1.1_ceftriaxone_plan.md](docs/v1.1_ceftriaxone_plan.md) (amendment 8). The test parts were scored
  once, taking the production log from 89 to 113 rows.
- Arm T internal AUROC 0.713 (0.604–0.818).
- T − F AUROC +0.009 [−0.021, +0.034], not demonstrated.

## Version 1.0 — 2026-09-28: the result page (engineering)

- Plan: [docs/v1.0_plan.md](docs/v1.0_plan.md) (amendments 6 and 7). A result page served by the API, consolidated
  README sections, and local deployment documentation.

## Version 0.9, with 0.9.1 and 0.9.2 — from 2026-09-25: a read-only HTTP API (engineering)

- Plan: [docs/v0.9_api_plan.md](docs/v0.9_api_plan.md) (amendment 5). A FastAPI service for the frozen Version 0.4
  model. The follow-up releases added error codes, the API contract, verified bundle digests and the `src/api/`
  package.

## Version 0.8 — 2026-09-24: adaptation at DRIAMS-C (confirmatory)

- Plan: [docs/v0.8_adaptive_plan.md](docs/v0.8_adaptive_plan.md) (amendment 4).
- **Recalibration:** Brier(B1) − Brier(A1) −0.0035 [−0.0161, +0.0085], not demonstrated.
- **Refit A + C:** Brier(B1) − Brier(A2) +0.0187 [+0.0086, +0.0290], Holm p 0.001, improved (confirmatory
  secondary). The verdict was "mixed", because the confidence zone reached NPV 0.904 < 0.95.

## Version 0.7 — 2026-09-23 to 2026-09-24: generalisation over time and sites (confirmatory)

- Plan: [docs/v0.7_generalisation_plan.md](docs/v0.7_generalisation_plan.md) (amendment 3).
- **Gaps:** random − 2018 +0.022 [−0.054, +0.094] and random − DRIAMS-D +0.023 [−0.043, +0.085], not demonstrated.
- **A second training site:** A + B vs A at D −0.016 [−0.034, +0.001], not demonstrated.

## Version 0.6 — 2026-09-20: explanations and confidence zones (descriptive, derived)

- Plan: [docs/v0.6_explainability_plan.md](docs/v0.6_explainability_plan.md) (amendment 2).
- Confident-susceptible NPV on stored test predictions: 0.948 [0.910, 0.981].

## Version 0.5 — 2026-09-18: neural networks (confirmatory)

- Plan: [docs/v0.5_deep_learning_plan.md](docs/v0.5_deep_learning_plan.md).
- MLP − V0.4 AUROC −0.039 [−0.085, +0.005], not demonstrated.
- CNN − V0.4 AUROC −0.253 [−0.324, −0.177]: worse.

## Version 0.4 — 2026-09-18: tuning and calibration (confirmatory)

- Plan: [docs/v0.4_search_plan.md](docs/v0.4_search_plan.md). The tuned LightGBM became the served model.
- Internal AUROC 0.751 [0.696, 0.807].
- V0.4 − V0.3 AUROC +0.025 [−0.012, +0.060], not demonstrated.
- Sensitivity at the cut-off was 0.827 [0.760, 0.896], against a target of 0.90.

## Version 0.3 — 2026-09-17: baselines (confirmatory)

- Protocol: [docs/evaluation_protocol.md](docs/evaluation_protocol.md). Logistic regression, random forest and
  LightGBM on the ciprofloxacin `random` and `within_year` test parts (24 log rows).

## Versions 0.1 and 0.2 — 2026-09-17: pair selection, cohort and splits (descriptive)

- E. coli + ciprofloxacin was chosen by rules fixed before any data were examined (`config.yaml → pair_selection`).
- A dataset of 6,410 spectra, with `random`, `within_year`, `temporal` and `external` splits.
