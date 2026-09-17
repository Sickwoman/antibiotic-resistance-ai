# Evaluation protocol (pre-registration)

**Status: DRAFT – waiting for approval.** Written on 2026-09-17, before any model was trained. After
approval, this protocol changes only through a dated amendment at the end of this file that gives the
reason. No change may be motivated by a test-set result.

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
  difference with an interval from independent bootstraps.
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

## 10. Decisions needed before Version 0.3

| # | Decision | Proposal |
|---|---|---|
| 1 | Primary metric | AUROC, with PR-AUC as co-primary |
| 2 | Threshold rule | highest threshold with validation sensitivity ≥ 0.90 |
| 3 | Final model | trained on the training part only (no refit on train + validation) |
| 4 | Confidence intervals | 2,000 patient-group bootstrap resamples, 95 % percentile interval |
| 5 | Patient-overlap check | add the size-matched `random` training run |
| 6 | Seeds | 5 seeds (42–46) for models with random training |

Approval: not yet approved.

## Amendments

None yet.
