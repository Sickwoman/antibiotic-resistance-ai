# Evidence map, Versions 0.1–1.4

Companion to [the research report](research_report_v0.1-v1.4.md). One entry per reported experiment, linking it to
its question, the information available before it, its data, split, selection and evaluation procedure, the commits
that pre-registered, ran and recorded it, its saved results, its limitations, and whether it is confirmatory or
exploratory. Compiled on 2026-10-01 from the repository at `0806e74` (branch `v1.4-screening`), without running
any experiment. **"Not recorded"** means no committed artifact states it; nothing was filled in from memory.

Commit notes. Versions 0.1–1.0 reached `main` through squash merges, so their original commits are not ancestors of
`main`. Commits of Versions 0.2–0.6 are held by local branches (`v0.2-driams-d`, `v0.3-baselines`,
`v0.4-improved-ml`, `v0.5-deep-learning`, `v0.6-explainability`) that are not on GitHub. The original Version 0.7
and 0.8 commits are on no branch (theirs was deleted). Versions 1.1–1.4 are on GitHub as draft PRs #26–#29, not yet
merged. Every commit this map cites is on `main` or preserved by an archival ref under `refs/archive/`, published on
GitHub as an annotated tag of the same name under `research-archive/` (checked 2026-10-01). The
[review checklist](review_checklist.md), section 5, shows how to fetch the tags and restore a commit.

Abbreviations: R = resistant spectra; CV = cross-validation; OOF = out-of-fold; "spent" = a test part whose
results have been inspected.

---

## Ciprofloxacin (`ecoli_ciprofloxacin`, label 1 = R or I)

### E0.1 — Exploration and pair selection (Version 0.1)
- **Question.** Which species–antibiotic pair meets rules fixed before any data were examined?
- **Prior information.** The pre-registered eligibility rules in `config.yaml → pair_selection` (≥ 500 per class at
  DRIAMS-A, minority share ≥ 0.10, ≥ 3 years with labels, ≥ 50 resistant in the test year, ≥ 2 external sites with
  ≥ 30 per class).
- **Data.** DRIAMS-A, -B, -D metadata (DRIAMS-C not available then): 111,257 / 5,897 / 10,436 rows; *E. coli* rows
  with a binned spectrum 7,320 / 213 / 2,013 (`results/metrics/eda/summary.json`).
- **Outcome.** Ciprofloxacin eligible and selected; ceftriaxone recorded as the benchmark pair, also "eligible" at
  the metadata level (1,086 class-1 rows at A, `pair_candidates.csv`). *Reconciliation:* that count included the
  HospitalHygiene screening isolates; once they were excluded from the cohort (E0.2 rule), ceftriaxone held 428
  resistant spectra in the split cohort, below the 500 minimum (amendment 8, point 5).
- **Commits.** Generated 2026-09-17 02:45 (`summary.json`); producing commit **not recorded** in the artifact.
- **Status.** Descriptive.

### E0.2 — Preprocessing, cohort and splits (Version 0.2)
- **Question.** Can raw spectra be processed reproducibly into model-ready features, and split without patient leakage?
- **Data.** 6,410 spectra, 1,400 R (A 4,259 / 971; B 213 / 59; D 1,938 / 370). Exclusions at A: no ciprofloxacin
  result 2,334; ambiguous result 75; HospitalHygiene workstation 633; malformed spectrum 8; raw file disagrees with
  the published binned file 11 (D: 1) (`results/metrics/v0.2/ecoli_ciprofloxacin/dataset_summary.json`). I counted
  as resistant; an I-excluded variant (6,345 spectra) reuses the same partition.
- **Preprocessing.** Stateless per spectrum (sqrt, Savitzky–Golay, SNIP 20 then 100, TIC, 3 Da bins 2,000–20,000 Da),
  reproducing the published DRIAMS binned files to ≤ 5.9 × 10⁻⁸ relative difference; feature fingerprint
  `347cbd6d5d956ff9`.
- **Splits.** `random` (A, 2,977 / 426 / 856), `within_year` (A, year folder 2017), `temporal` (A, test = 2018),
  `external` (train A; test B and D), later `external_ab` (train A + B, test D). Patient groups (`patient_no`) are
  kept together, **within a year folder only**, because DRIAMS-A re-hashes patient identifiers yearly; B and D have
  no patient identifiers, so each spectrum is its own group.
- **Commits.** Dataset built at `0f16192` (2026-09-17).
- **Limitations.** Cross-year patient overlap is undetectable; B and D intervals are sample-level. The I-excluded
  dataset was built for the protocol's sensitivity analysis 1, but **no model was ever evaluated on it** (no report,
  no log row): that pre-specified analysis was not executed.
- **Status.** Descriptive (data engineering).

### E0.3 — Baseline models (Version 0.3)
- **Question.** How well do simple, fixed-setting models rank new DRIAMS-A patients?
- **Prior information.** Evaluation protocol approved 2026-09-17 (draft `1fbb607`; approved text as recorded at
  `3c0bca7`): AUROC primary, PR-AUC co-primary; cut-off = highest threshold with validation sensitivity ≥ 0.90;
  2,000 patient-group bootstrap resamples; temporal and external parts locked until Version 0.7.
- **Selection and evaluation.** Four fixed models (prevalence, logistic regression, random forest, LightGBM); saved
  model chosen on validation AUROC (random forest, 0.749); `random` and `within_year` test parts plus the
  size-matched run scored once (24 log rows, run commit `25a9bbf`).
- **Saved results.** `results/metrics/v0.3/ecoli_ciprofloxacin/tables.md`. Random test (856 / 197): saved random
  forest AUROC 0.726 [0.666, 0.785]; differences to the other models not demonstrated.
- **Limitations.** Patient-overlap check inconclusive (differences ±0.10–0.13 wide).
- **Status.** Confirmatory for the pre-specified test evaluation of the selected model.

### E0.4 — Tuning and calibration (Version 0.4)
- **Question.** Does a pre-registered search with calibration improve on Version 0.3?
- **Prior information.** Search plan `ba26b34` (2026-09-17), fixed before tuning.
- **Selection and evaluation.** 5-fold `StratifiedGroupKFold` CV on the training part only (logistic regression 100
  settings, random forest 18, LightGBM 30 random draws, RBF SVM 36); sigmoid calibration on OOF predictions; family
  chosen on validation AUROC (LightGBM 0.776). Same `random` test part scored once for the new models (run
  `3c8245b`, 23 rows).
- **Saved results.** `results/metrics/v0.4/ecoli_ciprofloxacin/tables.md`, `best_model_card.json`. The **served
  model**, `v0.4.0-tuned_lightgbm-random-seed42` (threshold 0.14261540693905073): test AUROC 0.751 [0.696, 0.807],
  PR-AUC 0.556, Brier 0.144 [0.120, 0.170], sensitivity 0.827 [0.760, 0.896], specificity 0.458. Against Version 0.3:
  +0.025 [−0.012, +0.060], not demonstrated.
- **Limitations.** The same `random` test part had already been scored in Version 0.3 (for other models); the cut-off
  missed its 0.90 target on test.
- **Status.** Confirmatory for the pre-specified test evaluation; the improvement over Version 0.3 is not demonstrated.

### E0.5 — Neural networks (Version 0.5)
- **Question.** Does an MLP or a 1-D CNN outrank the Version 0.4 model?
- **Prior information.** Plan `37641ec` (2026-09-18).
- **Selection and evaluation.** Same CV and calibration; MLP (16 settings) saved on validation; test scored once (run
  `325ee85`, 11 rows).
- **Saved results.** `results/metrics/v0.5/ecoli_ciprofloxacin/tables.md`. MLP 0.712 [0.664, 0.764], minus Version
  0.4: −0.039 [−0.085, +0.005]; 1-D CNN 0.498 [0.440, 0.559].
- **Limitations.** Small search budget; "networks did not win" means with this budget on these data.
- **Status.** Confirmatory (pre-specified comparison with the saved model).

### E0.6 — Explanations and confidence zones (Version 0.6)
- **Question.** Which m/z regions drive the model, and can it abstain ("uncertain") reliably?
- **Prior information.** Plan `1ecd579` (2026-09-20) and amendment 2.
- **Selection and evaluation.** TreeSHAP and permutation importance on the 426 validation rows; zones fitted on
  validation (95 % target, 5 % minimum coverage); test-side zone figures derived from the stored Version 0.4 test
  probabilities after they reproduced the logged AUROC and Brier (no new scoring). Run `d55e13f`.
- **Saved results.** `results/metrics/v0.6/ecoli_ciprofloxacin/tables.md`, `zone_intervals.json`. Confident-
  susceptible zone (probability < 0.1026): NPV 0.948 [0.910, 0.981] at 24.9 % coverage on the stored test
  predictions. **No confident-resistant zone exists** (best precision 0.84 at 5.9 % validation coverage).
- **Limitations.** Explanation methods agreed weakly (TreeSHAP vs permutation Spearman −0.08); no m/z region has a
  protein identity (amendment 2, point 5).
- **Status.** Explanations descriptive; zones fitted on validation, test figures derived (secondary).

### E0.7 — Generalisation across sites and time (Version 0.7)
- **Question.** How much does the Version 0.4 setting lose at another hospital or in a later year?
- **Prior information.** Plan and amendment 3 at `96d0425` (2026-09-23, before any locked part was scored).
- **Selection and evaluation.** No setting chosen: the Version 0.4 setting refitted unchanged on each split's
  training part (5 seeds; intervals for seed 42); `temporal`, `external`, `external_ab` test parts released and
  scored once (run `8878bfd`, 26 rows; results `64c6c75`). Gaps use an unpaired second-level bootstrap.
- **Saved results.** `results/metrics/v0.7/ecoli_ciprofloxacin/tables.md` (+ `tables_ERRATUM.md`). Gaps (random minus
  site): temporal +0.022 [−0.054, +0.094]; B −0.064 [−0.152, +0.030]; D +0.023 [−0.043, +0.085]; a second training
  site at D −0.016 [−0.034, +0.001]. Zone NPV: later year 0.893 [0.844, 0.937] (below 0.95, but on a
  refitted, recalibrated model — not a clean zone test); saved model at D 0.963; at B 0.922 on 51 covered spectra
  (not rejected under the plan's criterion, which counts an interval covering 0.95 as transfer).
- **Limitations.** Temporal = date-separated with incomplete patient linkage; B/D intervals sample-level; the
  seed-spread table had a grouping error (issue #22), corrected by erratum, table preserved.
- **Status.** Confirmatory (pre-specified gaps, one scoring each); every gap "not demonstrated".

### E0.8 — Adaptation at DRIAMS-C (Version 0.8)
- **Question.** Does local recalibration (A1) or a refit on A + C (A2) improve the saved model at a new hospital?
- **Prior information.** Methodology approved and hashed at `284c0bd` (2026-09-24 14:13, methodology hash
  `e0ceb172…`) before DRIAMS-C was downloaded; amendment 4.
- **Data.** DRIAMS-C ciprofloxacin cohort 889 spectra, 191 R; partition frozen (70 % adaptation 622 / 120 R; 30 %
  protected evaluation 267 / 71 R; seed 42; each spectrum its own group, no patient IDs), `partition.json`.
- **Selection and evaluation.** Arms B0 (prevalence), B1 (saved model), B2 (Version 0.7 external refit), A1, A2;
  primary endpoint Brier, A1 vs B1; A2 vs B1 confirmatory secondary; paired bootstrap (2,000), Holm across the two;
  zone co-endpoint (NPV ≥ 0.95 with coverage floor). Scored once (run `4d82a22`, 5 rows; results `849cad7`).
- **Saved results.** `results/metrics/v0.8/ecoli_ciprofloxacin__site-C/`. Brier(B1) − Brier(A1) −0.0035
  [−0.0161, +0.0085], Holm p 0.576: not demonstrated. Brier(B1) − Brier(A2) +0.0187 [+0.0086, +0.0290], Holm p 0.001,
  but A2's zone NPV 0.904: verdict **mixed**. No arm's zone NPV reached 0.95.
- **Limitations.** 71 resistant spectra; sample-level groups.
- **Status.** Confirmatory. DRIAMS-C's protected part is spent for ciprofloxacin.

### E0.9 / E1.0 — Serving API and result page (Versions 0.9–1.0)
- Engineering, not experiments: plans `8d8fe4e` (squash of PR #12) and `d1718cc`; amendments 5–7. The only
  measurement is request latency on validation spectra (durations only, `results/metrics/v0.9/…/api_timing.json`).
  No model, threshold or result changed. **Status:** not an evaluation.

---

## Ceftriaxone (`ecoli_ceftriaxone`, label 1 = R or I)

### E1.1 — A second antibiotic (Version 1.1)
- **Question.** Does the method carry over to E. coli + ceftriaxone, and does it need its own search?
- **Prior information.** Plan `b6016a3` and amendment 8 (2026-09-30). The test parts' *spectra* had been scored for
  ciprofloxacin; their *ceftriaxone labels* had informed nothing. The pair does not meet the selection rule (428 R
  at A against 500) and proceeds as the designated benchmark.
- **Data.** 6,489 spectra, 681 R (A 4,283 / 439; B 213 / 45; D 1,993 / 197). Cohort = isolates with both results:
  93 ceftriaxone-only isolates (27 R) are in no split. Splits derived from the ciprofloxacin primary.
- **Selection and evaluation.** Arm T = the Version 0.4 search rerun on ceftriaxone (30 draws, CV on the training
  part); arm F = the ciprofloxacin setting refitted. T is the primary family by the plan, whatever validation said.
  Test parts scored once (run `7c26a36`, 24 rows); strict pre-write gate.
- **Saved results.** `results/metrics/v1.1/ecoli_ceftriaxone/tables.md`. Random test (856 / 79): T AUROC 0.713
  (0.604–0.818); T − F +0.009 [−0.021, +0.034]; sensitivity 0.823, specificity 0.422 at a threshold set on 34
  resistant validation spectra. Gaps not demonstrated; DRIAMS-D 0.651 (0.607–0.696).
- **Status.** Confirmatory for its pre-registered endpoints (only "better than chance" met its rule).

### E1.2 — Calibration and cut-off stability (Version 1.2)
- **Question.** Does a cut-off chosen on more (cross-fitted) data improve probability quality (Brier) over the
  Version 1.1 procedure?
- **Prior information.** Plan `db48bff`, pin `af757ae`, amendment 9. No untouched evaluation data exist in A/B/D.
- **Data.** Development pool: `random` training + validation rows never in a spent test part and sharing no patient
  group with one: 2,421 spectra, 247 R, 1,189 patient groups (105 resistant), fingerprint `064a4bbbd05847f2`. Its
  labels had trained and validated V1.1's arm T.
- **Selection and evaluation.** Patient-grouped 5-fold CV, partitions 42, 43, 44; arms R0, B, C, D1; primary
  Brier(B) − Brier(C) on partition 42's pooled OOF predictions, patient-group bootstrap (fits held fixed); forward
  split (fit < 2017, evaluate 2017). Run `ba4144c`, 16 development-log rows; results `025103c`; closure `3fc142b`.
- **Saved results.** `results/metrics/v1.2/ecoli_ceftriaxone/tables_complete.md`. Primary +0.0002 [−0.0024,
  +0.0028]: not demonstrated. C's cut-off steadier (sensitivity SD 0.058 vs 0.151); forward 2017 delivered
  sensitivity 0.556–0.789 against 0.90.
- **Status.** Exploratory (reused development labels).

### E1.3 — Uncertainty-aware cut-off over time (Version 1.3)
- **Question.** Does an order-statistic tolerance rule (U) hold the 0.90 target in later periods better than the
  empirical rule (E), and at what specificity?
- **Prior information.** Plan `2242327`, pin `5c9aa11`, amendment 10; the V1.2 forward result motivated it.
- **Data.** The V1.2 pool; origins 2017-01-01 and 2017-07-01, 7-day label gap; selection on one cross-fitted
  prediction per patient group (39 and 63 resistant patients).
- **Selection and evaluation.** Two rules on one fixed model; two-level patient-group bootstrap (fits held fixed).
  Run `a2d6ee2`, 8 rows; results `4175ce7`; closure `c5cfb49`.
- **Saved results.** `results/metrics/v1.3/ecoli_ceftriaxone/tables.md`. Pooled sensitivity U 0.929, E 0.814; U − E
  +0.115 [+0.019, +0.248]; U specificity 0.160 (0.085–0.393), flag rate 0.849: verdict "unhelpful".
- **Limitations.** U's 95 % statement is a heuristic in this design (cross-fitted scores, cross-year linkage,
  temporal shift); the later periods had been evaluated in aggregate by V1.2.
- **Status.** Exploratory.

### E1.4a — Discrimination audit
- **Question.** What could limit discrimination: labels, missingness, exclusions, repeats, preprocessing, fitted
  transformations, weighting, calibration, shortcuts, population differences?
- **Evidence.** `scripts/discrimination_audit.py` at `4ecb130`; tables `results/metrics/v1.4/audit/`; interpretation
  `docs/discrimination_audit.md` (`775f6bf`, correction note in `94984b5`). Fits and scores nothing.
- **Findings.** No verified defect affecting a historical conclusion; verified constraint of 105 resistant patients.
- **Status.** Descriptive audit (reads development labels and saved predictions).

### E1.4b — Screening isolates as training-only data (Version 1.4)
- **Question.** Do excluded HospitalHygiene isolates, added to training only, improve ranking of clinical isolates?
- **Prior information.** The audit; plan `5600efb`, pin `d3c173a`, amendment 11. The audit had read the screening
  isolates' aggregate label counts; no model had used them.
- **Data.** Evaluation: the V1.2 pool (clinical only). Added training rows: 495 screening spectra, 485 R, from 259
  patient groups (26 pool patients, 233 new), all dated before 2018 (count derivation: report section 8.3).
- **Selection and evaluation.** A0 = V1.2's arm C (required to reproduce it; it did, difference 0); A1 = A0 plus the
  screening rows of patients not held out. Primary: per-fold AUROC difference over 15 folds (3 partitions × 5),
  corrected repeated-CV interval. Implementation `153d3e2`; first run stopped at the count check and was logged;
  addendum and rerun `94984b5`; results `4c278e8`; documentation `0806e74`.
- **Saved results.** `results/metrics/v1.4/ecoli_ceftriaxone/tables.md`. A1 − A0 −0.007 [−0.056, +0.043]: not
  demonstrated. Source separability AUROC 0.937 (0.912–0.959).
- **Status.** Exploratory; triggered the pre-set stop to model iteration on the pool.
