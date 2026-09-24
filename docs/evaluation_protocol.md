# Evaluation protocol (pre-registration)

**Status: APPROVED on 2026-09-17** by the project owner, before any model was trained. The decisions
table (section 10) records the approved choices. This protocol now changes only through a dated
amendment at the end of this file that gives the reason. No change may be motivated by a test-set
result.

This is a research prototype. It predicts a laboratory label from a spectrum; it is not a clinically
validated diagnostic and must never be used to choose a patient's antibiotic.

## 1. Task

Predict from an *E. coli* MALDI-TOF spectrum whether the isolate was reported resistant to ciprofloxacin.
Label 1 = R or I (I counted as resistant), label 0 = S.

## 2. Data

| Dataset | Rows fingerprint | Samples | Resistant | Use |
|---|---|---|---|---|
| `ecoli_ciprofloxacin` | `151415a4d03dbcc5` | 6,410 (A 4,259, B 213, D 1,938) | 1,400 | all main results |
| `ecoli_ciprofloxacin__intermediate-exclude` | `9a23af98751e0770` | 6,345 (A 4,195, B 212, D 1,938) | 1,335 | sensitivity analysis (I removed) |

Both datasets use the same partition: each sample is in the same part of each split in both datasets.
Every run records the dataset name, both fingerprints (rows and `X.npy`), the split name, the git commit,
the full configuration and the random seed.

## 3. Splits and the question each answers

| Split | Question | Train / validation / test | Resistant share in test |
|---|---|---|---|
| `random` | How well does a model work on new patients from the same hospital (A, 2015–2018)? | 2,977 / 426 / 856 | 23.0 % |
| `within_year` | Is `random` optimistic because patient IDs change between years? (A, year folder 2017) | 1,284 / 183 / 366 | 23.2 % |
| `temporal` | Does a model trained on older data work on later data? (A; test = 2018) | 2,505 / 465 / 1,233 | 22.0 % |
| `external` | Does a model trained at hospital A work at other sites? (test = DRIAMS-B and DRIAMS-D) | 3,831 / 428 / 2,151 (B 213, D 1,938) | B 27.7 %, D 19.1 % |

The antibiotic choice was confirmed on DRIAMS-D on 2026-09-17, before any model was trained. DRIAMS-C is
added the same way if it is downloaded. The pre-registered selection rule must be checked on it first.

Pre-specified comparisons:

- **Generalisation gap:** the `random` test result minus the `temporal` test result, and minus the
  `external` test result **for each external site separately**. DRIAMS-B (a hospital) and DRIAMS-D (a
  diagnostic laboratory) differ in size and resistance rate, so a pooled external number is only
  secondary.
- **Patient-overlap check:** `random` versus `within_year`. The `within_year` split has fewer training
  samples, so it is also compared with a `random` model trained on a patient-grouped random subsample of
  its training part of the same size (1,284 samples, seed 42). That way, less training data is not
  mistaken for leakage.

## 4. How each part may be used

- **Train:** fitting the model. Anything learned from data (scaling, PCA, feature selection, class
  weights) is fitted inside a pipeline on training rows only. Hyperparameters are tuned by 5-fold
  `StratifiedGroupKFold` cross-validation within the training part (patient groups kept together).
- **Validation:** choosing between model families and settings, fitting a calibration step if one is
  used, choosing the decision threshold, and early stopping.
- **Test:** evaluated **once** per final model. Every test evaluation is appended to a log (date, commit,
  model, fingerprints, metrics). No setting is changed after looking at a test result. If a model is
  changed anyway, the new result is an additional run, and every run is reported.
- **Final model:** the model trained on the training part is the one tested (no refit on train +
  validation), so the threshold and calibration chosen on validation apply to exactly that model.

## 5. Metrics

- **Primary:** AUROC. It needs no threshold and can be compared between test sets with different
  resistance rates.
- **Co-primary:** PR-AUC (average precision). It is always shown next to the resistant share of that
  test set, which is what a model without any skill would score.
- **Calibration** (needed for the uncertainty work in later versions):
  - Brier score, next to the Brier score of a model that always predicts the training resistance rate.
  - Reliability plot with 10 bins holding equal numbers of samples.
  - Calibration slope and intercept.
- **At the decision threshold chosen on validation:** sensitivity for resistance, specificity, balanced
  accuracy, and the full confusion matrix (TP, FP, TN, FN).
  - Calling a resistant isolate susceptible is the more harmful error, so the threshold rule favours
    sensitivity.
  - The proposed rule is the highest threshold that reaches a validation sensitivity ≥ 0.90.
- **Not primary:** F1 and accuracy, because both depend on how common resistance is. They may be listed
  for completeness.
- **Reference rows in every table:**
  - A model that always predicts the training resistance rate.
  - The best simple baseline from Version 0.3.
  - Published DRIAMS results are not directly comparable (different antibiotic, cohort and splits). Any
    mention of them states these differences.

## 6. Uncertainty of the estimates

- **95 % confidence intervals:** percentile bootstrap with 2,000 resamples, seed 42. Whole patient groups
  are resampled within the test set. In DRIAMS-B and DRIAMS-D each spectrum is its own group.
- **Two models on the same test set:** a paired bootstrap of the difference (the same resamples for
  both). A model is called better only if the 95 % interval of the difference excludes 0.
- **Different test sets** (e.g. `random` versus `temporal`): both intervals are reported, plus the
  difference with an interval from independent bootstraps. The two sets share no rows, so no pairing
  exists: each metric is resampled inside its own test set, then draws are taken independently and with
  replacement from the two bootstrap distributions and subtracted (`unpaired_difference`). This is a
  second-level bootstrap of the convolution, and it is wider than a paired interval, as it should be.
- **Models with random training** (e.g. neural networks, random forests): 5 seeds (42–46); report the
  mean and range, plus the intervals for seed 42.

## 7. Known weak points of the evaluation

- The DRIAMS-B test set has only 59 resistant isolates, so its intervals will be wide and its results are
  indicative only.
- DRIAMS-B and DRIAMS-D have no patient IDs. Repeated isolates of one patient count as independent
  samples, which makes their intervals somewhat too narrow.
- DRIAMS-D spectra start at about 2,000 Da, while A and B spectra usually start near 1,960 Da. The lowest
  bins therefore differ systematically between sites. This matters for models that are trained on more
  than one site.
- Validation parts hold about 40–110 resistant isolates. Thresholds chosen there are noisy, so both
  validation and test sensitivity are reported.
- DRIAMS-A patient IDs change every year, so `random` and `temporal` cannot rule out that one patient
  appears in two parts. `within_year` is the check for this.
- Labels are used as the laboratory reported them. Changes of breakpoints or guidelines between 2015 and
  2018 are not corrected, which can affect the `temporal` split.

## 8. Pre-specified sensitivity analyses

1. **I excluded:** repeat the headline metrics on the `…__intermediate-exclude` dataset, using the same
   partition.
2. **Patient overlap:** `within_year` versus `random` and the size-matched `random` run (section 3).
3. **Later:** DRIAMS-C as a further external site, once it is downloaded and the selection rule is
   confirmed on it.
4. **Optional, later:** hospital-hygiene samples as a separate test set. This needs a small builder
   option first.

## 9. Reporting rules

- Report every run, including failed ones and models worse than the baseline.
- Every table shows n, the number of resistant isolates and the confidence intervals.
- Metrics are never typed by hand; tables are generated from the saved run logs.
- No claim of clinical usefulness and no treatment advice.

## 10. Decisions (approved 2026-09-17)

| # | Decision | Approved choice |
|---|---|---|
| 1 | Primary metric | AUROC, with PR-AUC as co-primary |
| 2 | Threshold rule | highest threshold with validation sensitivity ≥ 0.90 |
| 3 | Final model | trained on the training part only (no refit on train + validation) |
| 4 | Confidence intervals | 2,000 patient-group bootstrap resamples, 95 % percentile interval |
| 5 | Patient-overlap check | add the size-matched `random` training run |
| 6 | Seeds | 5 seeds (42–46) for models with random training |
| 7 | Test parts used before Version 0.7 | `random` and `within_year` (with the size-matched run) only; the `temporal` and `external` test parts stay locked until Version 0.7 so they cannot influence model choices in Versions 0.3–0.6 |

In code, decision 7 is `evaluation.locked_test_splits` in `config.yaml`. Test parts are scored only
when a script is run with `--evaluate-test`, and every scoring is appended to
`results/experiments/test_evaluations.csv`.

## Amendments

### Amendment 1 — 2026-09-18: what the results may be called

Added after an external code review of the code base at Version 0.4. It changes no
decision, no metric and no split; it fixes how results are named and adds one method statement. Nothing
here was prompted by a test result.

1. **The `temporal` split is date-separated, not patient-independent.** DRIAMS-A re-hashes patient IDs
   every year, so a patient who returns in a later year cannot be detected and may appear on both sides
   of the boundary. From now on the temporal result is called a *date-separated evaluation with
   incomplete patient linkage*, never a patient-level generalisation result, and every report of it
   repeats that sentence. The `within_year` split, where patient groups are complete, remains the check
   for how much patient overlap is worth.
2. **External-site intervals are sample-level.** DRIAMS-B and DRIAMS-D carry no patient IDs, so each
   spectrum is its own group and repeated isolates of one patient count as independent. Their intervals
   are therefore reported as *sample-level intervals with unknown within-patient dependence* and are
   expected to be too narrow. Site comparisons must not be read as if these intervals were reliable.
3. **The unpaired difference is a second-level bootstrap** (section 6), stated explicitly so the interval
   is interpretable.
4. **The size-matched run must really be size-matched.** `grouped_subsample` refuses a sample below 99 %
   of the requested size, and the achieved size is recorded as `train_size` in every report row. In the
   Version 0.4 run both parts held exactly 1,284 training samples.
5. **Locked test parts are refused twice**: by the scripts before anything is computed, and by the test
   log itself before anything is written.

### Amendment 2 — 2026-09-20: explanations and confidence zones

Added before any Version 0.6 explanation was computed, together with
[the Version 0.6 plan](v0.6_explainability_plan.md). It changes no decision, no metric, no split and no
model. It states how a new kind of output — an explanation of a prediction, and a three-way confident /
uncertain call — fits the existing rules. Nothing here was prompted by a test result.

1. **Explanations are a validation-part activity.** Contributions, feature importances and example
   explanations are computed on validation rows. Training rows are used only for descriptive comparisons
   (a region's resistant-versus-susceptible intensity difference) and for the shuffled-label control.
   Section 4 already assigns threshold choices to validation; an explanation is a weaker use than that.
2. **The confidence zones are thresholds, so they are fitted on validation**, by the pre-registered rule in
   the Version 0.6 plan (95 % on each side, 5 % minimum coverage, with the fallback fixed in advance). The
   model's decision cut-off from section 5 is unchanged; the zones sit around it and only change what the
   output is *called*.
3. **No test row is scored again in Version 0.6.** The test part was scored once per final model and the
   probabilities were saved. Where a Version 0.6 report gives a test-side number for the confidence zones,
   it is *derived from those stored predictions*, after asserting that they cover exactly that test part's
   rows and reproduce both the logged AUROC and the logged Brier score to 1e-12. The AUROC alone would not
   be enough: it is unchanged by any monotone rescaling of the probabilities, while a confidence zone
   depends on their absolute values. Such a derivation is not a new evaluation: it adds no row to the test
   log, and the run asserts the log is unchanged when it finishes. This is the same treatment the recomputed
   Version 0.4 intervals received under amendment 1, point 3.
4. **Nothing in Version 0.6 may change a model, a setting or a cut-off.** If an explanation suggests a
   change, that change belongs to a later version, is pre-registered there, and is evaluated as a new model.
5. **No m/z region is given a protein or peptide identity.** This project has no MS/MS confirmation and no
   independent panel, so regions are named by their m/z interval only, in every report and every figure.

### Amendment 3 — 2026-09-23: the generalisation experiments

Added before any locked test part was scored, together with
[the Version 0.7 plan](v0.7_generalisation_plan.md). Decision 7 releases the `temporal` and `external` test
parts at Version 0.7, which is what this amendment prepares. It changes no metric, no threshold rule and no
existing split, and it adds no model choice: the hyperparameters are the ones Version 0.4 already selected.
Nothing here was prompted by a test result — none had been seen.

1. **One new split, `external_ab`: train on DRIAMS-A + DRIAMS-B, test on DRIAMS-D.** The specification asks
   for a "train two sites, test a third" experiment, which the four approved splits do not contain: in
   `external`, B is a test site. Validation is 10 % of the combined training sites, patient-grouped, with
   the project seed. It answers whether adding a second, small site to the training data helps at a third
   site. It is listed alongside the section 3 splits from now on, and the same rules apply to it.
2. **The DRIAMS-D test rows are therefore scored under two training regimes** (trained on A, and trained on
   A + B). Both are reported whichever way the difference falls, and because they are the *same* rows the
   difference is compared with a paired bootstrap, pre-specified in the Version 0.7 plan. Neither result may
   be selected afterwards as "the" external result.
3. **The saved project model is scored on the external test part without a refit.**
   `v0.4.0-tuned_lightgbm-random-seed42` was trained on the `random` training part (DRIAMS-A only), whose
   intersection with the `external` test part is 0 rows and 0 patient groups. It keeps the threshold chosen
   on the `random` validation part, which is stated wherever that row appears, because the threshold was not
   chosen on the site it is being applied to. The same model is **not** scored on the `temporal` test part:
   848 of those 1,233 rows are in its training data.
4. **A further external site is added as a separate cohort, never by rebuilding the primary dataset.** A new
   site is built with the same preprocessing settings, so it shares the `feature_fingerprint` and has its own
   `row_fingerprint`. The primary dataset's rows, its saved splits, the saved models and every logged test
   result are left untouched. Rebuilding the primary dataset would change its row fingerprint and invalidate
   the provenance of results that were scored once and may not be scored again. The pre-registered
   pair-selection rule is checked on the new site before it is used, as section 3 already requires.
5. **A Version 0.6 confidence zone may be applied to a new test part, but never refitted there.** Carrying
   the fitted edge across unchanged is a test of the zone; refitting it per site would need that site's
   labels and is a different experiment. The pre-registered reading of the outcome is fixed in the Version
   0.7 plan, including what it means if the zone does not transfer.

### Amendment 4 — 2026-09-24: the Version 0.8 adaptation experiment

Added when the Version 0.8 methodology was approved, **before DRIAMS-C was downloaded, extracted,
partitioned or inspected**, and before any Version 0.8 code existed. It changes no Version 0.7 result and
rewrites no earlier methodology: Versions 0.1–0.7 stand exactly as recorded. It fixes how a new site may be
adapted to and how the outcome is judged. Nothing here was prompted by a result, because none exists.

1. **A locally refitted confidence zone is a different fitted object from the carried-over one, and the two
   may never be compared as though they were the same.** The Version 0.6 edge (probability below 0.1026) was
   fitted once, on the `random` validation part, and *carried* to new data unchanged; its number answers
   "does a zone fitted elsewhere still hold here?". A zone refitted on a new site's adaptation part answers
   a different question — "can a zone be found here, given local labels?" — and will almost always look
   better, because it was fitted where it is measured. Every report must name which of the two a number
   belongs to, and a difference between them is never presented as transfer, improvement or degradation of
   the same object.
2. **The adaptation arms may change only what is listed.** Recalibration-only changes the two Platt
   parameters and the zone edge, and nothing else; the trees, the preprocessing, the feature space and the
   threshold rule stay frozen. The confirmatory A + C refit re-fits the model with the Version 0.4 winning
   setting unchanged. **No hyperparameter search may be run on the new site**, because a search would let
   the adaptation part choose the setting, and a difference could no longer be attributed to adaptation.
3. **The new site's held-out part is protected exactly as a locked test part.** It is listed in
   `evaluation.locked_test_splits` from the moment it is created until the single Version 0.8 scoring, so
   every script refuses it, and it may never influence the adaptation size, the method, the edge, the
   threshold or any hyperparameter. The partition is drawn once, deterministically, by whole patient groups
   with the project seed, before any label distribution in the parts is examined.
4. **The primary endpoint is the Brier score, not AUROC, and the reason is mathematical rather than
   practical.** Recalibration is a monotone map of the probabilities, so it preserves every pairwise
   ordering and leaves AUROC exactly unchanged — verified on the recorded Version 0.7 DRIAMS-D
   probabilities, where five different recalibrations gave bit-identical AUROC. Pairing AUROC with a
   recalibration arm would guarantee a null result by construction. AUROC remains a reported guardrail.
5. **The confidence-zone endpoint is a pair, with a coverage floor.** Negative predictive value on its own
   can be raised arbitrarily by shrinking the zone, so the zone counts as improved only if its NPV reaches
   the pre-registered 0.95 *and* its coverage is within 0.05 of the baseline's. The number of covered
   spectra is reported beside every zone number, because Version 0.7 showed how little a zone number on 51
   spectra establishes.
6. **Eligibility is checked before anything is fitted, on class counts only.** The existing rule
   (`pair_selection.min_per_class_external`) decides whether the site may be used at all; a Version 0.8
   power precondition additionally requires at least that many of each class in the held-out part. A
   failure of either **stops the experiment and is reported with its counts**. Another site is never
   silently substituted, and neither rule may be changed after the counts are seen.
