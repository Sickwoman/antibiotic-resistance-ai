# Predicting antibiotic resistance from MALDI-TOF spectra (DRIAMS): research report, Versions 0.1–1.4

**Date:** 2026-10-01. **Repository state reviewed:** branch `v1.4-screening` at commit `0806e74`; this report and its
companions were added on branch `v1.4-report`. **Status:** research prototype; not a clinically validated
diagnostic; nothing here recommends a treatment.

Companion documents: [evidence map](evidence_map.md) (per-experiment provenance),
[reproduction guide](reproduction_guide.md), [one-page summary](research_summary.md),
[review checklist](review_checklist.md). Every number below is copied from a committed artifact that is named where it
is used. No model was trained, tuned or scored, no prediction was regenerated, and no log was modified to prepare
this report.

## Abstract

**Aim.** To test, under pre-registered protocols, whether machine-learning models can rank *Escherichia coli*
isolates by resistance from routine MALDI-TOF spectra in the public DRIAMS database, whether performance holds
across hospitals and time, and whether adaptation, cut-off rules or additional training data improve it.

**Methods.** Spectra were preprocessed statelessly (reproducing the published DRIAMS binned files) and split by
patient group. For **ciprofloxacin** (6,410 spectra from DRIAMS-A, -B, -D), baselines, tuned models and neural
networks were compared on a once-scored internal test part; the tuned LightGBM was then carried, unchanged, to a
later year and to two external sites, and adapted at a fourth site (DRIAMS-C). For **ceftriaxone** (6,489 spectra),
the method was repeated with once-scored test parts, followed by three development-only studies on a derived pool
of 2,421 spectra (247 resistant, from 105 resistant patients) whose labels had already been used.

**Results.** Ciprofloxacin: internal test AUROC 0.751 [0.696, 0.807]. No generalisation gap was demonstrated at the
later year or either external site, but the intervals were too wide to exclude gaps of practical size. A
confident-susceptible abstention zone reached NPV 0.948 on the stored test predictions. No confident-resistant zone
existed, and at DRIAMS-C no arm's zone reached its 0.95 target. Local recalibration was not demonstrated to help;
refitting on A + C improved the Brier score but lost the zone target (verdict "mixed"). Ceftriaxone: test AUROC
0.713 (0.604–0.818), above chance but not a usable decision rule (sensitivity 0.823 against a 0.90 research target,
specificity 0.422). Development studies (exploratory) found no demonstrated benefit from a cross-fitted cut-off for
probability quality, an uncertainty-aware cut-off that raised sensitivity only by flagging about 85 % of isolates,
and no demonstrated ranking gain from adding 495 screening isolates to training (AUROC difference −0.007
[−0.056, +0.043]).

**Conclusions.** Confirmatory evidence supports modest ranking ability for both antibiotics at DRIAMS-A. It does not
support useful decision performance, demonstrated transfer or demonstrated benefit from any adaptation or
cut-off intervention tested. Under a stopping rule fixed before the last study, model iteration on the
ceftriaxone development pool has stopped. Further work needs a new, justified hypothesis and evidence no version has
used.

## 1. Research questions and intended use

| # | Question | Antibiotic | Versions |
|---|---|---|---|
| Q1 | How well do models rank new patients' isolates at the development hospital? | ciprofloxacin | 0.3–0.5 |
| Q2 | Does performance hold in a later year and at other sites? | ciprofloxacin | 0.7 |
| Q3 | Can the model abstain reliably (a confident-susceptible / uncertain output)? | ciprofloxacin | 0.6–0.8 |
| Q4 | Does local recalibration or refitting help at a new hospital? | ciprofloxacin | 0.8 |
| Q5 | Does the method carry to a second antibiotic? | ceftriaxone | 1.1 |
| Q6 | Can calibration, the cut-off rule or more training data improve the ceftriaxone model? (development only) | ceftriaxone | 1.2–1.4 |

**Intended research use.** Methodological research into whether MALDI-TOF spectra carry resistance signal, how such
models transfer, and how their evaluation should be designed and reported. **Not intended:** clinical diagnosis,
choosing or ranking antibiotics for a patient, or replacing antimicrobial susceptibility testing (AST). The serving
API and result page (Versions 0.9–1.0) are engineering demonstrations of the frozen ciprofloxacin model, not
validated tools.

## 2. Data

### 2.1 Provenance
DRIAMS ("Database of Resistance Information on Antimicrobials and MALDI-TOF Mass Spectra"; Weis et al., Dryad,
doi:10.5061/dryad.bzkh1899q, CC0) holds spectra and AST results from four Swiss institutions. As recorded in
`config.yaml`:

| Site | Institution | Period | Used here |
|---|---|---|---|
| DRIAMS-A | University Hospital Basel | 11/2015–08/2018 | development, internal tests, temporal test |
| DRIAMS-B | Canton Hospital Basel-Land | 01–06/2018 | external test; a training site in `external_ab` |
| DRIAMS-C | Canton Hospital Aarau | 01–08/2018 | Version 0.8 only (ciprofloxacin) |
| DRIAMS-D | Viollier AG, a diagnostic laboratory | 01–06/2018 | external test |

The A, B and D archives were verified against published checksums (`config.yaml → driams.archives`); C was
downloaded in a browser from Dryad for Version 0.8.

### 2.2 Label policy
Label 1 = R or I (I counted as resistant, following the DRIAMS authors' convention as stated in `config.yaml`), 0 = S.
Bracketed or mixed results (e.g. "R(1), S(1)") are ambiguous and excluded; "-" and empty cells are missing. Labels
are used as reported; breakpoint changes between 2015 and 2018 are not corrected (protocol section 7). At DRIAMS-A,
intermediate results make up 64 of the 971 resistant ciprofloxacin labels and 1 of the 439 resistant ceftriaxone
labels. The pre-specified sensitivity analysis without them was not run (section 8.6).

### 2.3 Cohorts, missingness and exclusions

| Cohort | Spectra (R) | A | B | D | Build |
|---|---|---|---|---|---|
| `ecoli_ciprofloxacin` | 6,410 (1,400) | 4,259 (971) | 213 (59) | 1,938 (370) | `0f16192`, rows `151415a4d03dbcc5` |
| `ecoli_ciprofloxacin__site-C` (V0.8) | 889 (191) | – | – | – | C only, rows `83d504f4d0f83ae8` |
| `ecoli_ceftriaxone` | 6,489 (681) | 4,283 (439) | 213 (45) | 1,993 (197) | `7116e12`, rows `fb65868756a623ee` |
| `ecoli_ceftriaxone_with_screening` (V1.4, training rows only) | 7,138 (1,316) | 4,932 | 213 | 1,993 | `153d3e2`, rows `b55a65935f4073bb` |

Exclusions at DRIAMS-A (first applicable reason; `dataset_summary.json` of each cohort): ciprofloxacin — no result
2,334, ambiguous 75, HospitalHygiene workstation 633, malformed 8, raw file disagrees with the published binned file
11; ceftriaxone — no result 2,332, ambiguous 27, HospitalHygiene 659, malformed 8, disagreeing 11.

**Missingness is informative about sample type, not ignorable.** For ceftriaxone at DRIAMS-A, 32 % of *E. coli*
rows have no result, and the share depends strongly on the workstation: urine 10 %, blood 14 %, deep tissue 26 %,
respiratory 32 %, genital 93 %, stool 99 % (`results/metrics/v1.4/audit/missing_result_by_sample_type.csv`). Every
cohort is therefore "isolates the laboratory tested and reported", dominated by urine, blood and deep tissue.

**Screening isolates.** HospitalHygiene samples (colonisation screening) were excluded from every evaluated cohort
by an owner decision fixed in Version 0.2; the protocol listed them only as an optional *separate test set*
(section 8, item 4). Of 659 such ceftriaxone results at A, 645 are resistant. Version 1.4 used them as training data
only (section 5.6).

**Ceftriaxone cohort selection.** The ceftriaxone cohort is isolates with both a ciprofloxacin and a ceftriaxone
result, because its splits are derived from the ciprofloxacin primary; 93 ceftriaxone-only isolates (27 resistant,
29.0 % against 10.2 %) are in no split (amendment 8, point 6). The pair does not meet the pre-registered selection
rule on this cohort (428 resistant at A against 500). It looked eligible in Version 0.1 (1,086 class-1 rows)
because that metadata-level count included the screening isolates.

### 2.4 Patient linkage limits
- DRIAMS-A re-hashes `patient_no` every year: patients are grouped **within a year folder only**. A patient who
  returns in another year cannot be detected, so random splits and the temporal split may place one person on both
  sides (the `temporal` result is therefore a date-separated evaluation with incomplete patient linkage,
  amendment 1).
- DRIAMS-B, -C and -D carry no patient identifiers: each spectrum is its own group, and their intervals are
  sample-level with unknown within-patient dependence (expected to be too narrow).
- Repeat spectra are common: 76 % of DRIAMS-A *E. coli* ciprofloxacin spectra come from patients with more than one
  spectrum in the same year (`config.yaml` note); in the ceftriaxone development pool, 247 resistant spectra come from
  105 resistant patients, and the 11 patients (10 %) with the most resistant spectra contribute 27 % of them.

## 3. Methods

**Preprocessing and features.** Square-root intensities, Savitzky–Golay smoothing, SNIP baseline (20 then 100
iterations), total-ion-current normalisation, trimming to 2,000–20,000 Da and 3 Da bins (6,000 features),
reproducing the DRIAMS preprocessing (MALDIquant algorithms; Gibb & Strimmer 2012) to at most 5.9 × 10⁻⁸ relative
difference from the published binned files. Preprocessing is stateless; every learned step (scaling, PCA,
feature selection) lives inside model pipelines fitted on training rows only.

**Evaluation rules** (protocol approved 2026-09-17, amended eleven times, each amendment recorded before the work
it governs): AUROC primary, PR-AUC co-primary; cut-off = the highest threshold whose validation sensitivity is at
least 0.90; 95 % percentile intervals from 2,000 bootstrap resamples of whole patient groups (seed 42); paired
bootstrap for two models on the same rows, and an unpaired second-level bootstrap for different test sets; a model is
called better only if the interval of the difference excludes 0; test parts scored once per final model, every
scoring appended to an append-only log (113 rows). **All operating numbers — the 0.90 sensitivity target, the 0.95
zone and tolerance levels, the 0.20 "flags nearly everyone" line, the 7-day label gap and the support minimums —
are research criteria, not clinical standards.**

### Table 1. Experiment chronology

| ID | Version, date | Question | Data and split | Pre-registration → run → results | Status |
|---|---|---|---|---|---|
| E0.1 | 0.1, 2026-09-17 | pair selection | A/B/D metadata | rules in `config.yaml` → EDA (commit not recorded) | descriptive |
| E0.2 | 0.2, 2026-09-17 | cohort, splits | 6,410 cipro spectra | → `0f16192` | descriptive |
| E0.3 | 0.3, 2026-09-17 | baselines | cipro `random`, `within_year` tests | protocol → `25a9bbf` (24 log rows) | confirmatory |
| E0.4 | 0.4, 2026-09-18 | tuning + calibration | cipro `random` test | `ba26b34` → `3c8245b` (23 rows) | confirmatory |
| E0.5 | 0.5, 2026-09-18 | neural networks | cipro `random` test | `37641ec` → `325ee85` (11 rows) | confirmatory |
| E0.6 | 0.6, 2026-09-20 | explanations, zones | cipro validation; stored test predictions | `1ecd579` → `d55e13f` | descriptive / derived |
| E0.7 | 0.7, 2026-09-23/24 | generalisation | cipro `temporal`, `external`, `external_ab` | `96d0425` → `8878bfd` (26 rows) → `64c6c75` | confirmatory |
| E0.8 | 0.8, 2026-09-24 | adaptation at C | cipro C protected 30 % | `284c0bd` → `4d82a22` (5 rows) → `849cad7` | confirmatory |
| – | 0.9–1.0, 2026-09-25/28 | API, result page | none | `8d8fe4e`, `d1718cc` | engineering |
| E1.1 | 1.1, 2026-09-30 | second antibiotic | ceftriaxone test parts | `b6016a3` → `7c26a36` (24 rows) | confirmatory |
| E1.2 | 1.2, 2026-09-30 | calibration, cut-off | ceftriaxone dev pool, 5-fold × 3 | `db48bff` → `ba4144c` → `025103c` | exploratory |
| E1.3 | 1.3, 2026-09-30 | cut-off rules over time | dev pool, 2017 origins | `2242327` → `a2d6ee2` → `4175ce7` | exploratory |
| E1.4a | 1.4, 2026-09-30 | discrimination audit | dev pool, metadata, saved outputs | `4ecb130` → `775f6bf` | descriptive |
| E1.4b | 1.4, 2026-10-01 | screening isolates as training data | dev pool + 495 screening spectra | `5600efb` → `94984b5` → `4c278e8` | exploratory |

## 4. Results — ciprofloxacin

All ciprofloxacin results come from once-scored test parts (confirmatory under the protocol) unless marked.

**Q1 — ranking at the development hospital.** The tuned LightGBM (Version 0.4, the served model) reached AUROC
0.751 [0.696, 0.807] on 856 held-out spectra (197 resistant), PR-AUC 0.556 [0.445, 0.659] against a no-skill 0.230,
Brier 0.144 [0.120, 0.170]. At its validation cut-off it delivered sensitivity 0.827 [0.760, 0.896] against the 0.90
target, and specificity 0.458 [0.411, 0.505]. Its improvement over the Version 0.3 random forest (0.726) was
+0.025 [−0.012, +0.060], not demonstrated; the Version 0.5 MLP scored 0.712 (−0.039 [−0.085, +0.005] against it) and the
1-D CNN 0.498 [0.440, 0.559]. The patient-overlap check (size-matched `random` minus `within_year`) was −0.006
[−0.117, +0.104]: uninformative.

**Q2 — later year and other sites (Version 0.7).** AUROC 0.728 in 2018 (1,233 / 271), 0.815 at DRIAMS-B (213 / 59)
and 0.728 at DRIAMS-D (1,938 / 370), with the setting refitted per split. Gaps (random minus site): +0.022
[−0.054, +0.094], −0.064 [−0.152, +0.030], +0.023 [−0.043, +0.085]. **None was demonstrated, and none was excluded:**
the intervals are 0.13–0.18 wide. Adding DRIAMS-B to training changed AUROC at DRIAMS-D by −0.016 [−0.034, +0.001].

**Q3 — abstention.** On validation, probabilities below 0.1026 formed a confident-susceptible zone (NPV 0.955 at 25.8 %
coverage); on the stored test predictions it reached NPV 0.948 [0.910, 0.981] at 24.9 % coverage (derived, not a new
scoring). **No confident-resistant zone exists:** the best achievable precision was 0.84 at 5.9 % coverage, so every
high-probability spectrum is reported "uncertain". Carried unchanged, the zone met its pre-registered transfer
criterion at DRIAMS-D (saved model, NPV 0.963 [0.939, 0.983]). At DRIAMS-B it could not be judged (0.922 on 51
covered spectra; the criterion counts an interval covering 0.95 as transfer). In 2018 it failed (0.893 [0.844, 0.937]).
At DRIAMS-C no arm reached 0.95: NPV 0.935 (saved model), 0.929, 0.926, 0.904.

**Q4 — adaptation at a new hospital (Version 0.8; 267 spectra, 71 resistant).**

| Arm | Brier | AUROC | Brier(saved model) minus Brier(arm) | Holm p | Verdict |
|---|---|---|---|---|---|
| B1 saved model | 0.1458 | 0.765 | – | – | baseline |
| A1 recalibration only | 0.1493 | 0.765 | −0.0035 [−0.0161, +0.0085] | 0.576 | not demonstrated |
| A2 refit on A + C | 0.1271 | 0.820 | +0.0187 [+0.0086, +0.0290] | 0.001 | **mixed** (zone NPV 0.904 < 0.95) |

## 5. Results — ceftriaxone

### 5.1 Q5 — the method on a second antibiotic (Version 1.1, confirmatory)
On the `random` test part (856 spectra, 79 resistant; spectra previously scored for ciprofloxacin, ceftriaxone labels
unused): arm T AUROC **0.713 (0.604–0.818)**, above chance by the plan's rule; PR-AUC 0.214 against a no-skill 0.092;
Brier 0.083 against 0.084 for predicting the base rate. At a threshold set on 34 resistant validation spectra:
sensitivity 0.823, specificity 0.422, precision 0.127. Retuning versus reusing the ciprofloxacin setting (T − F):
+0.009 [−0.021, +0.034]. Gaps: 2018 −0.002 [−0.131, +0.125]; DRIAMS-B −0.120 [−0.249, +0.008]; DRIAMS-D +0.062
[−0.054, +0.177], the weakest site (0.651, 0.607–0.696). All not demonstrated.

### 5.2 The development pool (Versions 1.2–1.4, exploratory)
After Version 1.1, no untouched evaluation data remained in the A, B and D cohorts (amendment 9). The pool —
`random` training and validation rows never in a spent test part, nor sharing a patient group with one — holds 2,421
spectra (247 resistant; 1,189 patient groups, 105 resistant). Its labels had trained Version 1.1's arm T and were
reused by every later study, so **no result in 5.3–5.6 is independent evidence**, and additional folds, seeds or
resplits of it cannot make it so.

### 5.3 Calibration and cut-off stability (Version 1.2)
Primary, Brier(B) − Brier(C), with B the Version 1.1 procedure and C a cross-fitted cut-off on the whole training fold:
+0.0002 [−0.0024, +0.0028], not demonstrated (partitions 43, 44: −0.0002, +0.0006). C's delivered sensitivity varied
less across the 15 folds (SD 0.058 against 0.151) at lower specificity (0.370 against 0.446); both reached 0.90 in 7 of
15 folds. Fitted before 2017 and applied to 2017, no arm held its target (delivered sensitivity 0.556–0.789).

### 5.4 Cut-off rules over time (Version 1.3)
Rule U (order-statistic tolerance rule; Tong, Feng & Li 2018) against rule E (the protocol rule), at two 2017
origins: pooled sensitivity 0.929 against 0.814 (U − E +0.115 [+0.019, +0.248]), but U's specificity was 0.160
(0.085–0.393) and it flagged 85 % of isolates; verdict under the pre-set order "unhelpful". The 0.56 / 0.44 chances
that E attains 0.90 are theoretical binomial values under exchangeability, not observed rates
(`docs/v1.3_threshold_plan.md`, closure record).

### 5.5 Discrimination audit
No verified defect affected a historical conclusion (labels, preprocessing, fitted transformations, grouping, class
weights, calibration). Pooled AUROC equalled within-half-year and within-sample-type AUROC to ±0.005, and class
weighting did not change cross-validated AUROC in the saved searches (0.771 against 0.773). The verified constraint
was the number of resistant patients (`docs/discrimination_audit.md`).

### 5.6 Screening isolates as training data (Version 1.4)
Adding 495 screening spectra (233 new patients) to training only: AUROC on held-out clinical spectra, A1 − A0,
**−0.007 [−0.056, +0.043], not demonstrated**; every partition's mean was negative. Secondaries (unadjusted): PR-AUC
lower with A1 in all three partitions; at equal sensitivity (0.895) A1's specificity was higher (+0.026
[+0.004, +0.049]); one spectrum per patient −0.026 [−0.071, +0.018]; forward in time +0.001 [−0.046, +0.049]. Resistant
screening and clinical spectra were separable with AUROC 0.937 (0.912–0.959).

### Table 2. Principal results and uncertainty

| Result | Antibiotic | n / resistant | Estimate [95 % interval] | Interval | Status |
|---|---|---|---|---|---|
| Served model, internal AUROC | cipro | 856 / 197 | 0.751 [0.696, 0.807] | group bootstrap | confirmatory |
| V0.4 − V0.3, AUROC | cipro | 856 / 197 | +0.025 [−0.012, +0.060] | paired | not demonstrated |
| Sensitivity at cut-off (target 0.90) | cipro | 197 R | 0.827 [0.760, 0.896] | group bootstrap | confirmatory |
| Gap, random − 2018 | cipro | 1,233 / 271 | +0.022 [−0.054, +0.094] | unpaired 2nd-level | not demonstrated |
| Gap, random − DRIAMS-D | cipro | 1,938 / 370 | +0.023 [−0.043, +0.085] | unpaired, sample-level | not demonstrated |
| Confident-susceptible NPV (stored test) | cipro | 213 covered | 0.948 [0.910, 0.981] | group bootstrap | derived |
| Recalibration at C, Brier(B1) − Brier(A1) | cipro | 267 / 71 | −0.0035 [−0.0161, +0.0085] | paired, Holm | not demonstrated |
| Refit A + C, Brier(B1) − Brier(A2) | cipro | 267 / 71 | +0.0187 [+0.0086, +0.0290] | paired, Holm | mixed |
| Arm T, internal AUROC | ceftriaxone | 856 / 79 | 0.713 (0.604–0.818) | group bootstrap | confirmatory |
| T − F, AUROC | ceftriaxone | 856 / 79 | +0.009 [−0.021, +0.034] | paired | not demonstrated |
| Gap, random − DRIAMS-D | ceftriaxone | 1,936 / 181 | +0.062 [−0.054, +0.177] | unpaired, sample-level | not demonstrated |
| V1.2 Brier(B) − Brier(C) | ceftriaxone | 2,421 / 247 | +0.0002 [−0.0024, +0.0028] | group bootstrap, fits fixed | exploratory |
| V1.3 U − E, pooled sensitivity | ceftriaxone | 1,453 / 156 | +0.115 [+0.019, +0.248] | two-level bootstrap, fits fixed | exploratory |
| V1.4 A1 − A0, AUROC (15 folds) | ceftriaxone | 2,421 / 247 | −0.007 [−0.056, +0.043] | corrected repeated CV (see 8.2) | exploratory |

## 6. Negative findings and unsuccessful interventions

1. Tuning (V0.4) did not demonstrably beat the Version 0.3 baseline; neural networks (V0.5) did not beat tuning; the
   1-D CNN was at chance.
2. The 0.90 sensitivity cut-off, chosen on a validation part, missed its target on test for both antibiotics (0.827,
   0.823) and in every development check of it (V1.2 folds and forward split, V1.3 periods).
3. No confident-resistant zone could be formed; the confident-susceptible zone failed in the later year and at
   DRIAMS-C.
4. Local recalibration at DRIAMS-C was not demonstrated to help; the refit improved probabilities but lost the zone.
5. Adding a second training site did not demonstrably help at a third.
6. Retuning for ceftriaxone did not demonstrably beat reusing the ciprofloxacin setting.
7. A cross-fitted cut-off did not demonstrably improve probability quality (V1.2).
8. An uncertainty-aware cut-off raised sensitivity only by flagging nearly everyone (V1.3).
9. Recent-data intercept recalibration was not demonstrated to help (V1.3).
10. More resistant training data from screening isolates did not demonstrably improve ranking (V1.4).

"Not demonstrated" is never "equivalent" or "no effect" in this report: every such interval also admits effects that
were not excluded.

## 7. Data exposure and remaining evaluation status

### Table 3

| Data | Antibiotic | Used for | Times scored / status |
|---|---|---|---|
| A `random` test part (856) | cipro | V0.3, V0.4, V0.5 final models | once per final model; **spent** |
| A `within_year` test part (366) | cipro | V0.3, V0.4 | **spent** |
| A `temporal` test part (2018) | cipro | V0.7 | once; **spent** |
| B, D `external` test parts | cipro | V0.7 (incl. saved model at B, D) | once; **spent** |
| D `external_ab` test part | cipro | V0.7 | once; **spent** |
| C protected 30 % (267) | cipro | V0.8 | once; **spent** |
| C adaptation 70 % (622) | cipro | V0.8 A1/A2 fitting | used for fitting |
| A `random` test part (856) | ceftriaxone | V1.1 | once; **spent** |
| A 2018, B, D test parts | ceftriaxone | V1.1 generalisation | once; **spent** |
| Development pool (2,421) | ceftriaxone | V1.1 training/validation; V1.2–V1.4 | reused repeatedly; exploratory only |
| A screening isolates < 2018 (495 used) | ceftriaxone | V1.4 training only | aggregate labels read in the audit |
| A screening isolates dated 2018 (145) | ceftriaxone | none | unused |
| DRIAMS-C ceftriaxone labels | ceftriaxone | none by any model | **status uncertain**: aggregate counts (916 with a result, 151 R/I) were viewed on 2026-09-30, the spectra were used for ciprofloxacin, no patient IDs (`docs/driams_c_status.md`); opening them requires an owner-approved amendment |

**No untouched, independent evaluation data remain for either antibiotic in DRIAMS-A, -B or -D.**

## 8. Reporting issues resolved in this review (no experiment rerun)

### 8.1 Corrections and clarifications
| Where | Issue | Resolution |
|---|---|---|
| README "Training" | called the cut-off "the smallest cut-off reaching sensitivity ≥ 0.90"; the code (`threshold_for_sensitivity`) and protocol use the **highest** | corrected in the README |
| README "Training" | "four Swiss hospitals"; DRIAMS-D is a diagnostic laboratory | corrected: four institutions |
| README "Training" | "DRIAMS-B, C and D were never used for training; each was opened once" | corrected: the served model never trained on them, but V0.7's `external_ab` trained on A + B and V0.8's A2 on A + C; B and D test parts were scored for ciprofloxacin (V0.7) and for ceftriaxone labels (V1.1) |
| README "Results" | "a 0.05 AUROC drop … would matter clinically", "effects that would matter clinically" | no clinical threshold was ever justified; reworded as "not excluded" |
| README "Results" | "the zone that held at DRIAMS-B and DRIAMS-D" | B's point estimate was 0.922; the pre-registered criterion counts an interval covering 0.95 as transfer, so B is "not rejected, uninformative", not "held" |
| V0.7 `tables.md` (preserved) | column "Reaches 0.95" means the pre-registered *transfer* criterion (NPV ≥ 0.95 **or** interval covers 0.95), not a point estimate ≥ 0.95 | clarified here; the committed table is not edited |
| README "Results", V0.8 table | "paired Δ vs baseline" did not name its operands | now Brier(B1) − Brier(arm); positive favours the arm |
| V1.1 `tables.md`, `config.yaml` comment, audit document | state that Weis et al. report AUROC 0.74 "for E. coli + ceftriaxone … (LightGBM)" | the verified abstract gives 0.74 for *E. coli* without naming the antibiotic or classifier; the full text was not accessible, so that attribution is **unverified** and not used as a reference point in this report (README and audit document annotated; committed tables kept) |
| V1.4 commit `153d3e2` message | "18 tests" | 17 new tests; already recorded in the V1.4 plan record |

Metric directions were otherwise consistent: every difference in the committed tables names its operands, and the
denominators checked (sensitivity over resistant, specificity over susceptible, precision over flagged, flag rate over
all isolates, NPV over the zone) match the prose.

### 8.2 How the Version 1.4 primary interval treats fitting variability and fold dependence
The 15 per-fold AUROC differences come from 3 repetitions of 5-fold cross-validation **of the same 2,421 spectra**.
Within a repetition the training sets overlap by three quarters; across repetitions every spectrum is reused. They are
**not 15 independent cohorts**, and a naive t-interval (variance s²/15) would be too narrow.

The plan used the "corrected repeated k-fold cv test" of Bouckaert & Frank (2004), which applies Nadeau & Bengio's
correction to repeated cross-validation: variance (1/(k·r) + n₂/n₁)·s² with n₂/n₁ = 1/(k−1) for k-fold, here
(1/15 + 1/4)·s², and a t distribution with k·r − 1 = 14 degrees of freedom. This inflates the variance about 4.75-fold
over the naive one, approximating the extra variability from overlapping training sets. What it includes and what it
does not:
- It reflects, approximately, variability from which patients land in training and test folds. That is the only
  model-fitting variability it captures. The **model seed was fixed** (42) in both arms, so seed variability is
  excluded.
- The correction is a heuristic. Bengio & Grandvalet (2004) proved that no universally unbiased estimator of the
  variance of K-fold cross-validation exists, and the 14 degrees of freedom overstate the independent information.
  Bouckaert & Frank studied the correction on classification accuracy with 10 × 10-fold CV; its use for per-fold AUROC
  with 3 × 5 folds is an extrapolation.
- It describes variability over resamplings of this pool, not over new data from other periods or sites.

**Consequence for the conclusion.** Each of these qualifications could only widen the interval. The verdict "not
demonstrated" (the interval already contains 0) is therefore robust to them. An "improved ranking" verdict would not
have been. The interval should be read as approximate, and not as a guarantee that effects outside it are excluded.

### 8.3 Screening isolates: from 659 excluded to 495 used
| Step | Spectra (resistant) | Rule |
|---|---|---|
| DRIAMS-A HospitalHygiene *E. coli* rows with a single-category ceftriaxone result | 659 (645) | excluded from every evaluated cohort since V0.2 |
| minus those dated 2018 or later | 514 (504) remain; 145 dropped | nothing dated 2018+ is used (spent temporal period) |
| minus spectra of patients with a clinical row outside the pool | 505 remain; 9 dropped | plan section 4 (a within-year patient with any clinical row in a spent part, its patients, or an unsplit ceftriaxone-only isolate) |
| minus spectra failing the builder's checks | **495 (485)** used; 10 dropped | 3 unreadable, 7 disagree with the published binned file |

The 495 come from 259 patient groups: 69 spectra of 26 patients already in the pool and 426 of 233 new patients
(229 resistant). **The dated count amendment:** the audit classified "outside the pool" by patient group and
expected 496 spectra with 8 excluded. The plan's rule is by row, and 15 ceftriaxone-only clinical isolates (in no
split) belong to pool patients. One of those patients also had a resistant screening spectrum. The first run
(`153d3e2`) stopped at the count check before fitting anything and was logged as failed. The rule as written was
kept, the expected counts were corrected in a dated addendum (`94984b5`), and the rerun is the reported run. The
screening-inclusive dataset itself holds 649 of the 659 (the 10 failures excluded), including the 145 dated 2018
that are never used.

### 8.4 Two different test verifications — not the same snapshot
| Result | Code snapshot | Environment | Data and models present | Command |
|---|---|---|---|---|
| 709 passed, 2 skipped | clean detached checkout of `c5cfb49` (V1.3 closure) | this Windows 11 laptop, Python 3.12.10 venv | **no** (git-ignored; the 2 skips are the saved-bundle tests) | `bash -eo pipefail`, `python -m pytest -q` |
| 711 passed | the working tree on which `c5cfb49` was then committed: same code and tests, but documentation and an unrelated config section were being edited while it ran | same | yes | same |
| 729 passed | working tree with the content of `153d3e2` (before commit) | same | yes | same |
| 729 passed (670 s) | working tree with the content of `0806e74` (staged) | same | yes | same |
| see 8.5 | the commit containing this report | same | both, see 8.5 | same |

No GitHub Actions run exists for any commit after `main` (`2f665e1`), because nothing has been pushed; the CI change to
`bash -eo pipefail` (V1.3 closure) has therefore never run on GitHub's runners.

### 8.5 Verification of the commit containing this report
FILLED-IN-BELOW

### 8.6 Pre-specified analyses that were not executed
- **Protocol section 8, item 1 — headline metrics with intermediate (I) results excluded.** The dataset was built
  (`ecoli_ciprofloxacin__intermediate-exclude`, 6,345 spectra, same partition), but no model was ever evaluated on it:
  no report and no log row exist. The effect of counting I as resistant on the ciprofloxacin results is therefore
  **not known** (I is 64 of 971 resistant labels at A). For ceftriaxone the question hardly arises (1 I label).
- **Item 4 (optional) — hospital-hygiene samples as a separate test set.** Not done; Version 1.4 used those samples
  differently, as training data only.
- **Item 3 — DRIAMS-C as a further external site.** Covered in substance by Version 0.8's baseline arm B1 (the saved
  model on C's protected part: AUROC 0.765, Brier 0.1458), under the adaptation protocol rather than as a separate
  generalisation experiment.

### 8.7 What source separability and a null result do and do not show
- Resistant screening spectra were separable from resistant clinical spectra (AUROC 0.937). **This does not
  establish why.** Culture medium, strain lineage mix, patient population, acquisition time and instrument state are
  all candidate causes, and none was tested. Culture medium measurably changes MALDI-TOF spectra in other taxa (Wieme
  et al. 2014; review: Topić Popović et al. 2023), which makes it a plausible candidate here but not an established one.
- Failure to demonstrate an improvement **does not establish equivalence**, and it does not establish a biological
  ceiling on what spectra can reveal about ceftriaxone resistance. The audit's "biological ceiling" is recorded as an
  untested hypothesis.

## 9. Threats to validity

- **Reuse of evaluation data.** The ciprofloxacin `random` test part was scored for three successive versions'
  models. The protocol forbade choosing on test, but repeated looks weaken the independence of later comparisons. The
  ceftriaxone development pool was reused by four versions.
- **Patient linkage.** Cross-year recurrence at A is undetectable; B, C and D have no patient IDs.
- **Small numbers of resistant isolates** in decisive parts: 34 resistant validation spectra set the ceftriaxone
  cut-off; 51 covered spectra judged the zone at B; 71 resistant spectra decided V0.8; 105 resistant patients underlie
  every ceftriaxone development result.
- **Selection of the cohort.** Only isolates with a reported result (strongly sample-type dependent). For ceftriaxone,
  only isolates that also had a ciprofloxacin result. Screening isolates were excluded.
- **Label and time effects.** Breakpoints and laboratory practice over 2015–2018 are uncorrected; resistance prevalence
  and sample-type mix drifted within the development period.
- **Single hospital for development**, three external institutions with small or sample-level evaluations.
- **Estimation choices.** Bootstrap intervals hold fitted models fixed unless stated; the V1.4 interval is approximate
  (8.2). Results for random forests, LightGBM and networks depend on seeds; seed ranges are reported where fitted.
- **Researcher degrees of freedom.** Versions 1.2–1.4 were motivated by earlier results on the same data. Their plans
  were recorded before their code and runs (verified by commit order and plan hashes), but not reviewed by the owner
  before recording.
- **A pre-specified sensitivity analysis was not executed** (I excluded; section 8.6), so the effect of the label
  policy on the ciprofloxacin results is unknown.
- **Engineering claims** (latency, API behaviour) come from one laptop and are not performance evidence.

## Table 4. Claims: supported, unresolved, unsupported

| Claim | Status | Basis |
|---|---|---|
| Spectra carry some ciprofloxacin resistance signal at DRIAMS-A (ranking better than chance) | **supported** | E0.4, AUROC 0.751 [0.696, 0.807], confirmatory |
| Spectra carry some ceftriaxone resistance signal at DRIAMS-A | **supported** | E1.1, 0.713 (0.604–0.818), confirmatory |
| The 0.90-sensitivity cut-off delivers 0.90 on new data | **unsupported** | missed on test for both antibiotics and in development checks |
| The models are usable decision rules | **unsupported** | specificity 0.42–0.46 at the cut-off; no justified operating requirement |
| A confident-susceptible zone is reliable at the development hospital | supported on stored test predictions (derived) | E0.6 |
| The zone transfers to other sites and times | **unresolved / partly unsupported** | met at D, uninformative at B, failed in 2018 and at C |
| Performance generalises to other sites / later time (no gap) | **unresolved** | no gap demonstrated, none excluded |
| Ciprofloxacin results are robust to counting I as resistant | **unresolved** | pre-specified analysis never run (8.6) |
| Local recalibration helps at a new hospital | **unresolved** (not demonstrated) | E0.8 |
| Refitting on the new site improves probability quality | supported for Brier at C, with the zone lost | E0.8, mixed |
| Retuning is needed per antibiotic | **unresolved** | E1.1 T − F not demonstrated |
| A cross-fitted cut-off gives more stable sensitivity | exploratory description only | E1.2 |
| An uncertainty-aware cut-off gives useful performance | **unsupported** | E1.3 |
| Screening isolates improve ranking of clinical isolates | **unresolved** (not demonstrated) | E1.4b |
| The limit is discrimination / a biological ceiling | **unsupported** | never tested; withdrawn wording |
| Screening–clinical separability is caused by culture medium | **unsupported** | cause untested |
| Anything about clinical benefit, treatment or other species | **unsupported** | out of scope |

## 10. Why model iteration stopped

The stop follows the bounded plan, not a judgement that nothing more could be learned. Version 1.4's plan fixed, before
any code, that it would be the last model-iteration study on the ceftriaxone development pool whatever it found
(section 13). Two further reasons support honouring it. First, the pool's labels had informed four versions, so a
further result on them would add little *independent* evidence and would increase the risk of fitting the analysis to
the data. Second, the one data lever the audit found was tested and not demonstrated. Further analysis of these data
could still be informative, for example as descriptive work or as pre-registered hypotheses evaluated on new data. It
cannot provide confirmation.

## 11. Reproducibility

The repository reproduces every committed number from saved outputs; it does not reproduce the historical runs by
default, because most historical commands would retrain models, re-read spent labels, append to logs or overwrite
committed artifacts. The [reproduction guide](reproduction_guide.md) separates read-only artifact verification,
fixture-based tests (which write only to temporary folders; a session guard fails any test run that changes a real
log), and historical commands listed for documentation only. Seeds are 42 throughout (5 seeds, 42–46, where stated);
exact library versions are in `requirements-lock.txt`; the served model is verified by its SHA-256 sidecar
(`d59d6d7d…`).

## 12. Conclusions and requirements for future work

**Defensible conclusions.** For both antibiotics, spectra from DRIAMS-A support modest ranking of isolates by
resistance (AUROC about 0.71–0.75, confirmatory, single site). None of the interventions tested demonstrated a
benefit: tuning, networks, a second training site, local recalibration, cut-off rules or screening isolates as
training data. No generalisation gap was demonstrated, and none could be excluded. The protocol's 0.90-sensitivity
cut-off did not hold on new data. No result supports clinical use.

**A future independent study should require:**
1. **A new, justified hypothesis**, stated before data access, that is not a further tuning of this development
   pool (for example a representation or data source with a mechanistic or empirical rationale).
2. **Evidence no version has used**: isolates from new time periods or institutions, collected or released after the
   hypothesis is recorded, with patient identifiers that allow linkage across time, and the sample type recorded.
3. **A pre-registered analysis** with a primary endpoint, comparator, interval method (stating whether it includes
   fitting variability), a sample size justified by the effect of interest (the development pool could detect only
   AUROC differences of roughly 0.03–0.05), and a stopping rule.
4. **Any operating requirement** (sensitivity, specificity, abstention level) justified externally before results, or
   reported explicitly as a research criterion.
5. **DRIAMS-C's ceftriaxone labels only if** a frozen procedure first earns it on development data and the owner
   approves a dated amendment that settles C's uncertain standing. It should not be opened simply to obtain another
   result.

## References

Bibliographic details were checked against publisher-deposited records (Crossref; DataCite for the dataset) and
content against the original article, its PubMed/PMC record or an author-distributed copy, on 2026-10-01.

1. Weis C, Cuénod A, Rieck B, Dubuis O, Graf S, Lang C, et al. Direct antimicrobial resistance prediction from
   clinical MALDI-TOF mass spectra using machine learning. *Nat Med.* 2022;28(1):164–174.
   doi:10.1038/s41591-021-01619-9. *Verified (abstract, PubMed 35013613):* DRIAMS combines more than 300,000 spectra
   and 750,000 resistance phenotypes from four institutions; AUROC 0.80, 0.74 and 0.74 for *S. aureus*, *E. coli* and
   *K. pneumoniae*. *Unverified:* which antibiotic and classifier produced the *E. coli* 0.74 (full text not
   accessible).
2. Weis C, Cuénod A, Rieck B, Borgwardt K, Egli A. DRIAMS: Database of Resistance Information on Antimicrobials and
   MALDI-TOF Mass Spectra. Dryad; doi:10.5061/dryad.bzkh1899q (current version 12, CC0). *Verified:* DataCite record.
3. Tong X, Feng Y, Li JJ. Neyman-Pearson classification algorithms and NP receiver operating characteristics.
   *Sci Adv.* 2018;4(2):eaao1659. doi:10.1126/sciadv.aao1659. *Verified (PMC5804623):* order-statistic threshold with
   violation bound Σ C(n,j)(1−α)^j α^(n−j), minimum sample size n ≥ log δ / log(1−α), guarantee for a left-out
   sample scored by a separately trained function.
4. Bouckaert RR, Frank E. Evaluating the replicability of significance tests for comparing learning algorithms. In:
   *PAKDD 2004*, LNCS 3056, pp. 3–12. doi:10.1007/978-3-540-24775-3_3. *Verified (full text):* the corrected repeated
   k-fold cv test, (1/(k·r) + n₂/n₁)·σ̂², and the recommendation of 10 × 10-fold CV.
5. Nadeau C, Bengio Y. Inference for the generalization error. *Mach Learn.* 2003;52(3):239–281.
   doi:10.1023/A:1024068626366. *Bibliographic record verified; content not checked in this review* (the text could
   not be retrieved intact); the correction is cited here as described by reference 4.
6. Bengio Y, Grandvalet Y. No unbiased estimator of the variance of K-fold cross-validation. *J Mach Learn Res.*
   2004;5:1089–1105. *Verified (JMLR abstract):* no universal unbiased estimator of that variance exists.
7. Wieme AD, Spitaels F, Aerts M, De Bruyne K, Van Landschoot A, Vandamme P. Effects of growth medium on MALDI-TOF
   mass spectra: a case study of acetic acid bacteria. *Appl Environ Microbiol.* 2014;80(4):1528–1538.
   doi:10.1128/AEM.03708-13. *Verified (PMC3911038 abstract):* medium effects on peaks affected strain-level, not
   species-level, differentiation — in acetic acid bacteria, not *E. coli*.
8. Topić Popović N, Kazazić SP, Bojanić K, Strunjak-Perović I, Čož-Rakovac R. Sample preparation and culture
   condition effects on MALDI-TOF MS identification of bacteria: a review. *Mass Spectrom Rev.*
   2023;42(5):1589–1603. doi:10.1002/mas.21739. *Verified (PubMed 34642960 abstract):* culture media and conditions
   affect MALDI-TOF MS identification performance.
9. Wiesmann N, Enders D, Westendorf A, Koch R, Schaumburg F. Prediction of antimicrobial resistance from MALDI-TOF
   mass spectra using machine learning: a validation study. *J Clin Microbiol.* 2025;63(12):e01186-25.
   doi:10.1128/jcm.01186-25; correction: 2026;64(3):e01884-25 (figures 2–3 only; results and conclusions unchanged).
   *Verified (PMC12710331):* performance was poor when training and test data were unrelated in location and time,
   and declined within 18 months after training.
10. Gibb S, Strimmer K. MALDIquant: a versatile R package for the analysis of mass spectrometry data.
    *Bioinformatics.* 2012;28(17):2270–2271. doi:10.1093/bioinformatics/bts447. *Verified (PubMed 22796955 abstract).*
