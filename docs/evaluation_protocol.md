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

### Amendment 5 — 2026-09-25: serving the model over HTTP

Added before any Version 0.9 API code existed, together with [the Version 0.9 plan](v0.9_api_plan.md). It
changes no model, no threshold, no split, no metric and no recorded result. It states how a transport layer
fits the existing rules, and what that layer is forbidden from doing.

1. **The API is a read-only consumer of frozen artifacts.** It loads a saved bundle and the Version 0.6
   confidence zones and serves predictions from them. It may not fit, calibrate, tune, score a dataset part,
   write to `results/experiments/test_evaluations.csv`, or add a row to any log. The Version 0.9 test suite
   asserts the append-only log is unchanged.
2. **No protected split is reachable through the API.** The service takes spectra from the request body
   only. It accepts no dataset name, no split name and no filesystem path from a client, so a locked test
   part or the Version 0.8 protected evaluation part cannot be scored through it, deliberately or by
   accident. Serving a prediction for a spectrum that happens to belong to a test part is not an evaluation:
   no label is read, no metric is computed and nothing is recorded.
3. **A latency benchmark is not an evaluation.** Version 0.9 measures request timing on validation-part
   spectra and records **durations only** — no probability, no label, no identifier — so it cannot become a
   covert scoring run. This is the same treatment `inference_timing.json` received in Versions 0.3 and 0.4.
4. **The response contract is an allow-list, and fingerprints are not on it.** Responses are built from
   explicitly declared fields, never by serialising a bundle or a saved report. Feature, row and dataset
   fingerprints, `x_sha256`, git commits, code fingerprints, absolute paths, archive checksums and the
   Version 0.8 Platt coefficients are never served. Neither is any patient, case or order number, any DRIAMS
   spectrum UUID, or any uploaded filename — raw DRIAMS files are named after the spectrum UUID and repeat
   it in their comment lines, so the filename is used for its suffix and then discarded, and comment lines
   are never echoed or logged.
5. **No response may contain a non-finite float.** Several committed reports hold a bare `NaN`, which is not
   valid JSON. Numbers taken from a saved report are converted so that non-finite values become `null`, and
   the tests assert every response body survives strict JSON encoding.
6. **Input limits are function parameters, never feature-definition fields.** A maximum upload size and a
   maximum point count are added as keyword parameters defaulting to today's behaviour. They may not be
   added to `PreprocessingConfig`, whose hash defines the feature fingerprint `347cbd6d5d956ff9` that every
   saved bundle is checked against.
7. **Uploaded bytes are data, never code.** Nothing from a request is deserialised, imported or executed,
   and nothing from a request reaches `joblib.load`.
8. **The clinical framing of section 1 applies unchanged to every response.** Each prediction carries the
   research-prototype disclaimer; an uncertain call carries the recommendation to perform conventional
   antimicrobial susceptibility testing; and no endpoint names, ranks or suggests an antibiotic for a
   patient. `/model-info` reports the external Version 0.7 and Version 0.8 results beside the internal test
   metrics, including the null and mixed outcomes, because serving the internal number alone would overstate
   the model.

### Amendment 6 — 2026-09-28: the result page, and what a summary may claim

Added before any Version 1.0 code existed, together with [the Version 1.0 plan](v1.0_plan.md). It changes no
model, no threshold, no split, no metric and no recorded result. Version 1.0 is engineering and documentation
only; the append-only log gains no row.

1. **The result page is a client of the API, not a second serving path.** It calls the existing endpoints and
   reimplements nothing: not preprocessing, not the threshold, not the confidence zones, not the upload
   limits, not the error mapping. Every number it shows is read from a response at runtime, so it cannot
   disagree with the model. Amendment 5 governs the API it calls, and continues to apply unchanged.
2. **The page may not state a confidence the model does not have.** The fitted zones have no
   high-confidence-resistant side, so that label is unreachable and the page must render a high probability
   as `Uncertain` rather than styling it as a confident call. It names no antibiotic for a patient and uses
   no diagnostic language, which is section 1 and rules 12–13 applied to a user interface.
3. **The page is same-origin and stateless.** No third-party script, style, font or endpoint, so a spectrum
   cannot reach anywhere but this service; nothing is persisted in the browser or on the server; and no
   identifier is displayed, logged or transmitted.
4. **A consolidated summary may not be stronger than its record.** The Version 1.0 README adds Architecture,
   Training, Results, Limitations and Ethics sections across versions. A summary is where a project starts
   sounding better than its evidence, so: no claim may exceed its per-version source; each headline number
   cites the section it came from and is copied from a committed artifact rather than retyped; and the null
   and mixed outcomes appear in Results, not only in Limitations. Specifically, that no generalisation gap
   was demonstrated, that local recalibration was not demonstrated to help, that the A + C refit's verdict is
   *mixed* rather than success, and that no arm reached the 0.95 zone target.
5. **The per-version sections are not rewritten.** They record what was pre-registered and what was found,
   and they stay as they are. The new sections are added in front of them and link down.
6. **Deployment is documented, not performed.** The service has no authentication by design. Version 1.0
   describes running it locally and states what would have to be true before exposure could be considered; it
   ships no container, orchestration or hosting configuration, and does not present a deployment as available.

### Amendment 7 — 2026-09-28: the result page's interface obligations

Added before any Version 1.0 page code existed, because implementation found three things the Version 1.0
plan left unstated. It changes no model, no threshold, no split, no metric and no recorded result, and it
narrows rather than widens what the page may do.

1. **The page must be usable without sight, without a mouse, and without colour vision.** It is this
   project's only user interface, and a prediction a reader cannot perceive is worse than no prediction.
   Concretely: the file input carries a programmatic label; the submit action is reachable and operable from
   the keyboard; the status and result region is an `aria-live` region so a screen reader is told when an
   answer or an error arrives; the document uses semantic landmarks and headings rather than anonymous
   containers; **the confidence label is conveyed as text, never by colour alone**; and focus moves to the
   status region after a render so a keyboard user is not left at the top of the page. Error text says what
   to do next in plain language, consistent with rule 15.

2. **Only module constants may be substituted into the served page, never anything from a request.** The
   page is a static file, and the canonical strings — the disclaimer, the uncertain-call advice, the
   reachable confidence labels — are substituted into it when it is served, so that the page cannot drift
   from `src.predict.DISCLAIMER` and `src.uncertainty.ADVICE` the way a pasted copy would. This is a fixed
   allow-list of project constants and a string replacement, not a template engine, so it adds no
   dependency. **No value derived from a request is ever substituted**: doing so would make the page an
   injection vector, which is the one way a static page could become unsafe.

3. **Verification item 6 is scoped to the result region.** Item 6 of the Version 1.0 plan forbids hard-coded
   result strings in the page, while item 7 requires the page to *explain* that a high-confidence-resistant
   answer is unreachable — which cannot be done without naming that label. The two are reconciled as the
   plan's own wording already implies ("as an achievable outcome"): the **result region is empty in the
   served markup** and is filled only from an API response, while explanatory copy about what the model can
   and cannot answer lives in a separate, clearly marked region. A label appearing in an explanation is not
   a hard-coded result; a label appearing in the result region would be.

4. **The page's JavaScript is not executed by the test suite, and that boundary is recorded rather than
   implied.** Continuous integration installs Python only. So the page is written so that the unsafe code
   *cannot exist* — its script contains no confidence-label literal and no comparison against the returned
   probability or threshold, leaving no branch that could infer a confidence the model did not return — and
   the tests assert that structure on the served HTML. A behavioural test under a JavaScript runtime would
   need its own amendment, because it would add a test-time dependency the Version 1.0 plan forbids.

### Record — 2026-09-29: edits made in place to approved text

This protocol says it "changes only through a dated amendment at the end of this file". A sweep on
2026-09-29, comparing each version of the file against the one approved at commit `3c0bca7`, found that this
was not always followed. The in-place edits are listed here so the history is complete. They are **kept**,
not reverted, because each one corrected or clarified approved text, and reverting would reinstate wording
already found to be inaccurate or under-specified.

| Where | Commit | What changed | Why it is kept |
|---|---|---|---|
| Section 7, "Different test sets" | `d3d4d9e` (PR #8, external code review, 2026-09-18) | "the difference with an interval from independent bootstraps" was expanded in place to say how: each metric resampled inside its own test set, draws taken independently and subtracted (`unpaired_difference`), giving a second-level bootstrap wider than a paired interval | It specifies the method the approved sentence already named; it does not change which method is used |
| Amendment 2, point 3 | `e8d13fb` (PR #10, V0.6 audit fixes, 2026-09-21) | the check before stored test probabilities may be reused was strengthened in place, from "reproduce the logged test AUROC" to "cover exactly that test part's rows and reproduce both the logged AUROC and the logged Brier score", with the reason that AUROC is unchanged by a monotone rescaling | The audit found the original wording described a check too weak to support the claim made; reverting would restore that |

One further change is **not** an exception and is listed only so the sweep is fully accounted for: at `d3d4d9e`
the placeholder "None yet." under **Amendments** was replaced when amendment 1 was written. That placeholder
existed to be replaced.

**The rule from here on is unchanged, and now stated with its reason:** approved text is corrected by a dated
entry at the end of this file, never edited where it stands, so that anyone reading the file can tell what
was decided in advance from what was learned afterwards. The two edits above predate this record; no approved
text has been edited in place since.

### Amendment 8 — 2026-09-30: a second antibiotic (Version 1.1)

Added before any Version 1.1 code exists, together with [the Version 1.1 plan](v1.1_ceftriaxone_plan.md). It
changes no earlier model, threshold, split, metric or recorded result.

1. **A second task, next to the first.** Section 1 still defines the main task. Version 1.1 adds *E. coli* +
   ceftriaxone, with the same label rule (R or I = 1), metrics, threshold rule, bootstrap and reporting rules.
2. **Test parts are used once per task and final model.** The ceftriaxone test parts hold the same spectra as
   the ciprofloxacin ones, which have been scored, but only for ciprofloxacin. Their ceftriaxone labels have
   never informed a model, threshold or choice, so each part may be scored once per final Version 1.1 model.
   Wherever these results appear, they state that the spectra were scored before, for another antibiotic.
3. **No ciprofloxacin result is re-scored.** The comparison between the two antibiotics uses the stored
   Version 0.4 test probabilities, after the check in amendment 2, point 3.
4. **DRIAMS-C is not scored for any label.** Its protected part was spent once in Version 0.8, and Version 1.1
   does not reopen it.
5. **The pair does not meet the pair-selection rule on its cohort** (428 resistant at DRIAMS-A, against a
   minimum of 500). It proceeds as the benchmark antibiotic designated in Version 0.1, not as a pair the rule
   selected, and no result may describe it as meeting the rule. The rule is unchanged.
6. **The cohort is isolates with both results.** The 93 isolates with only a ceftriaxone result, 27 of them
   resistant, are in no split. Results describe the cohort as it is and are not generalised to them.

### Amendment 9 — 2026-09-30: a development-only study of calibration and the cut-off (Version 1.2)

Recorded before any Version 1.2 code or experiment, together with [the Version 1.2
plan](v1.2_calibration_plan.md), on the project owner's instruction. It changes no earlier model, threshold,
split, metric or recorded result.

1. **No untouched evaluation data exist.** Every test part of both antibiotics' A, B and D cohorts has been
   scored and its results inspected, so none of them — and no new random split of their spectra — can give an
   independent confirmation. Version 1.2 therefore runs on development data only, and every conclusion it draws
   is exploratory.
2. **The development pool is derived, not drawn:** the `random` training and validation rows of
   `ecoli_ceftriaxone` that were never in an inspected test part and share no patient group with one (2,421
   spectra, 247 resistant). Model fitting, calibration and the cut-off are chosen inside cross-validation folds
   of that pool only.
3. **Development results never enter the production log.** They go to a separate append-only development log,
   through the same strict pre-write gate.
4. **A cut-off's target must be supported.** The rule is unchanged — the highest cut-off with sensitivity at
   least 0.90 on the selection set — but a selection set with fewer than 50 resistant spectra cannot support
   that target, and a cut-off chosen on one is reported as unsupported.
5. **DRIAMS-C stays closed.** Its ceftriaxone labels are the only unused candidate for a future one-time
   evaluation, and opening them needs a separate dated amendment approved by the owner.

### Amendment 10 — 2026-09-30: an uncertainty-aware cut-off rule, on later development periods (Version 1.3)

Recorded before any Version 1.3 code or experiment, together with [the Version 1.3
plan](v1.3_threshold_plan.md), on the project owner's instruction. It changes no earlier model, threshold,
split, metric or recorded result, and it adds nothing to the production log.

1. **Development only, and exploratory.** Version 1.3 uses the Version 1.2 development pool. Its later periods
   (2017) were already evaluated in aggregate by Version 1.2's forward check, so nothing it finds is independent
   evidence.
2. **Label availability is part of the design.** At each origin only labels dated at least 7 days earlier may be
   used — for fitting, calibration, a cut-off or a recalibration — and never a label from the period evaluated.
3. **A cut-off rule with a stated uncertainty guarantee is compared with the existing rule, and only the rule
   changes.** The order-statistic rule's 95 % statement holds only under assumptions this design does not meet
   (exchangeability over time, a scoring function independent of the selection scores, independent patients),
   so in Version 1.3 it is a heuristic, and no bound is claimed for any later period.
4. **No specificity floor is presented as operationally justified.** A cut-off with specificity below 0.20 is
   labelled as flagging nearly everyone; staying above that line does not make a cut-off useful.
5. **DRIAMS-C stays closed.** Its standing is recorded in `docs/driams_c_status.md`.

### Amendment 11 — 2026-09-30: excluded screening isolates as training data only (Version 1.4)

Recorded before any Version 1.4 code or experiment, together with [the Version 1.4 plan](v1.4_screening_plan.md)
and after [the discrimination audit](discrimination_audit.md), on the project owner's instruction. It changes no
earlier model, threshold, split, metric or recorded result, and it adds nothing to the production log.

1. **Development only, and exploratory.** Version 1.4 evaluates on the Version 1.2 development pool only, whose
   labels have informed earlier decisions; nothing it finds is independent evidence.
2. **Screening isolates may be training data, never evaluation data.** DRIAMS-A HospitalHygiene isolates stay
   excluded from every cohort that is evaluated. Version 1.4 may add those dated before 2018 to training only, each
   joined to its patient's within-year group and left out whenever that patient is held out. The owner's decision
   to exclude them from the primary cohort stands.
3. **One candidate, one baseline, and a stop.** The baseline must reproduce Version 1.2's arm C exactly. After this
   comparison, model iteration on the development pool stops, whatever it finds.
4. **Ranking is not usefulness.** The primary endpoint is AUROC; no specificity or usefulness requirement is
   adopted, and "flags nearly everyone" (specificity below 0.20) remains a research line.
5. **DRIAMS-C stays closed.** Its standing is recorded in `docs/driams_c_status.md`.

### Amendment 12 — 2026-10-02: external validation of the frozen models on MARISMa (Version 2.0)

Recorded before any Version 2.0 code, download or experiment, together with
[the Version 2.0 plan](v2.0_marisma_plan.md). The project owner chose the data source (MARISMa) and the design (both
frozen models, with a multiplicity correction) on 2026-10-02. This amendment changes no earlier model, threshold,
split, metric or recorded result. Its scoring will append rows to the production log, once.

1. **A public external dataset, scored once.** MARISMa version 2.0.0 is the evaluation data: Hospital General
   Universitario Gregorio Marañón, Madrid, 2018–2024. Nothing is fitted on it.
2. **Frozen models only.** They are used exactly as saved: the served Version 0.4 ciprofloxacin model and Version
   1.1's arm T ceftriaxone model, with their thresholds, and the Version 0.6 zone for ciprofloxacin.
3. **Isolate-level units where no patient linkage exists.** MARISMa keeps one identifier per isolate, so intervals
   resample isolates. Every result states that repeat isolates from one patient cannot be detected. This deviates
   from the research report's requirement of patient linkage, by the owner's decision.
4. **Two primary hypotheses, Holm-corrected.** AUROC > 0.5 for each antibiotic, at a family-wise α of 0.05.
5. **Labels sealed until scoring.** Before scoring, only the names of the antibiotic fields and the number of
   non-missing interpretations may be read.
6. **The I-excluded sensitivity analysis is executed** this time, whatever the primary result.
7. **DRIAMS-C stays closed.**

#### Amendment 12, note A — 2026-10-02: pre-data clarifications (recorded before any MARISMa file is downloaded)

Recorded with amendment A of [the Version 2.0 plan](v2.0_marisma_plan.md), which governs where the two differ. The
note is proposed, and nothing in it is approved. Amendment 12 above stays exactly as recorded, and this note is pinned
separately.

1. **Item 3 corrected.** Acceptance of the patient-linkage deviation is pending; it was not given by the owner's choice
   of data source. It is recorded only when the owner gives it, in a later dated entry, and never backdated.
2. **Item 5 made precise.** Before scoring, only the restricted schema reader (plan amendment A2) may open `AMR.csv`.
   - It may match isolates, list antibiotic names and count non-missing interpretations.
   - It may expose no outcome category, prevalence, MIC value, raw row or outcome-bearing log.
3. **Item 4's test.** Each primary hypothesis is tested by a one-sided Brunner–Munzel test, with Holm over the fixed
   two-hypothesis family (plan amendment A3). An antibiotic that is dropped, or falls below the minimums, enters with
   p = 1.

#### Amendment 12, note B — 2026-10-03: final pre-data clarifications (recorded before any MARISMa file is downloaded)

Recorded with amendment B of [the Version 2.0 plan](v2.0_marisma_plan.md), which governs where the two differ. The
note is proposed, and nothing in it is approved. Note A stays exactly as recorded, and this note is pinned separately.

1. **A non-finite test is not estimable.** If the Brunner–Munzel statistic or its p-value is not finite, the primary
   test is reported as "not estimable", and the antibiotic enters the fixed Holm family with p = 1. No fallback test is
   run. Descriptive metrics are kept wherever they are defined.
2. **Error control.** "Brunner–Munzel inference is approximate and assumes independent isolates. Holm adjustment does
   not repair invalid component p-values. Missing patient linkage leaves actual error control uncertain." The
   family-wise α of 0.05 in item 4 is the nominal design level, not a guarantee.
3. **Conclusion wording.** A rejected hypothesis is reported only as "Evidence of above-chance ranking on MARISMa under
   the isolate-independence assumption". This is never read as clinical utility, or as internal performance
   maintained.

#### Amendment 12, note C — 2026-10-03T15:21:12Z: approval of steps 1–3

The owner approved all 17 decisions of the Version 2.0 approval statement (PR #35 at commit
`462a00aa0542991a37f17634469d63ac28bd115f`, including plan amendments A and B and notes A and B). In doing so, the owner
explicitly accepted:
- the missing-patient-linkage limitation;
- the conditional statistical interpretation;
- the non-estimable primary-test rule.

This authorises steps 1–3 of [the Version 2.0 plan](v2.0_marisma_plan.md) only: download, metadata cohort, restricted
schema inspection, reader and frozen features.

It does not authorise model scoring, unrestricted outcome access, fitting, recalibration, deployment, or merging
PR #35. The approval record in the plan quotes the owner's words. Notes A and B stand as recorded.
