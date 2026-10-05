# Research summary: MALDI-TOF resistance prediction on DRIAMS, Versions 0.1–1.4

*One-page technical summary, 2026-10-01. Full report: [research_report_v0.1-v1.4.md](research_report_v0.1-v1.4.md).
Research prototype; not a diagnostic.* *Addendum, 2026-10-04: Version 2.0, the external validation on MARISMa, is
summarised at the end; the text above it is unchanged.*

**Question.** Can routine MALDI-TOF spectra rank *E. coli* isolates by resistance, and does that hold across
hospitals and time? Can adaptation, cut-off rules or more training data make it useful? Every version was
pre-registered (protocol approved 2026-09-17, eleven dated amendments), and test parts were scored once, into an
append-only log.

**Data.** DRIAMS (public, CC0): A = University Hospital Basel (2015–2018, development); B and C = two cantonal
hospitals; D = a diagnostic laboratory (2018). Label 1 = R or I. Patients are linkable only within a year at A, and not
at all at B–D. Screening (HospitalHygiene) isolates were excluded from all evaluated cohorts.

**Ciprofloxacin (6,410 spectra; confirmatory results).**
- Served model (tuned LightGBM): internal test **AUROC 0.751 [0.696, 0.807]** (856 spectra, 197 resistant). Tuning
  did not demonstrably beat baselines; neural networks did not beat tuning.
- A cut-off chosen for 0.90 sensitivity on validation delivered **0.827** on test, at specificity 0.458.
- **No generalisation gap was demonstrated** in 2018 or at B or D, and none was excluded: the intervals are
  0.13–0.18 wide.
- A confident-susceptible zone reached NPV 0.948 on the stored test predictions. **No confident-resistant zone
  exists.** The zone met its transfer criterion at D, was uninformative at B, and fell short in 2018 (on a refitted
  model) and at DRIAMS-C (on its point estimate).
- At DRIAMS-C (267 spectra, 71 resistant), local recalibration was **not demonstrated** to help. Refitting on A + C
  improved the Brier score but lost the zone target: **mixed**.

**Ceftriaxone (6,489 spectra).**
- Confirmatory: test **AUROC 0.713 (0.604–0.818)**, above chance but not a decision rule (sensitivity 0.823,
  specificity 0.422). The pair falls short of the pre-registered selection rule (428 resistant at A against 500).
- Exploratory, on a reused development pool of 2,421 spectra from 105 resistant patients:
  - a cross-fitted cut-off did not demonstrably improve probability quality (V1.2);
  - an uncertainty-aware cut-off raised sensitivity only by flagging 85 % of isolates (V1.3);
  - after an audit found no pipeline defect, adding 495 excluded screening isolates to training did not
    demonstrably improve ranking of clinical isolates: **AUROC −0.007 [−0.056, +0.043]** (V1.4).

**What the evidence supports.**
- Modest ranking signal for both antibiotics at one hospital.
- No intervention met its research objective. Most primary endpoints were not demonstrated. V1.3's primary
  (higher sensitivity) was met at an unhelpful specificity. V0.8's refit improved the Brier score (a confirmatory
  secondary) but lost the confidence zone. A few exploratory secondaries are listed, not claimed (report, Table 5).
- The 0.90-sensitivity cut-off does not hold on new data.

**What it does not support.** Clinical use, transfer to other sites or times, equivalence of any compared methods,
a biological ceiling, or any cause for the separability of screening isolates (source AUROC 0.937).

**Evidence status.**
- Every A, B and D test part is spent for both antibiotics.
- DRIAMS-C's protected part is spent for ciprofloxacin. Its ceftriaxone labels are unused by any model, but their
  standing is uncertain (aggregate counts were viewed), and opening them requires an owner-approved amendment.
- The development pool has informed four versions, so it cannot provide confirmation.

**Decision.** Model iteration on the development pool has **stopped**, as the Version 1.4 plan fixed before its run.
Further analysis could still inform description or new hypotheses, but not confirmation.

**Open issues found in the final review.**
- A pre-specified sensitivity analysis (ciprofloxacin with I excluded) was never executed.
- The attribution of the published AUROC 0.74 to ceftriaxone could not be verified against the paper's full text.
- The Version 1.4 interval is an approximate correction for dependent folds (fixed model seed; extrapolated from its
  validation setting), though its "not demonstrated" verdict is robust to widening.
- The V0.7 and V0.8 plans were revised on their branches after pre-registration: before their runs, plus one
  labelled post-results note. The original texts and the V0.7–V0.9 commits are preserved by archival refs, published
  on GitHub as `research-archive/…` tags, and in a Git bundle whose off-machine copy is pending.

**Requirements for a future independent study.**
1. Exploration of any available data may generate hypotheses. The hypothesis, procedure and analysis must be frozen
   in a dated record before the confirmatory evaluation.
2. Confirmation on data that neither that exploration nor any earlier version has used, with cross-time patient
   linkage and sample type recorded.
3. A pre-registered endpoint, comparator, interval method and a sample-size justification. No adequate power
   analysis exists here; the observed intervals are the only precision evidence.
4. Externally justified operating requirements.
5. DRIAMS-C only through an amendment, and only for a procedure that first earns it on development data.

**Addendum, 2026-10-04: Version 2.0, external validation on MARISMa (confirmatory).** Full record: report section 13
and the plan's execution record. Tables: `results/metrics/v2.0/tables.md`.
- **The design.**
  - The served ciprofloxacin model, frozen, was scored once on MARISMa 2.0.0 (Madrid; public, CC-BY-4.0), under a
    pre-registration and amendments recorded before each step.
  - MARISMa's *E. coli* susceptibility results exist for 2024 only.
  - Ceftriaxone **could not be evaluated** (no interpretations; p = 1 in the Holm family).
- **The result.** "The frozen ciprofloxacin model achieved AUROC 0.772 (descriptive 95% interval 0.744–0.798) on
  1,145 eligible MARISMa E. coli isolates from 2024, with evidence of above-chance ranking under the
  isolate-independence assumption." The Holm-adjusted p was 4.4 × 10⁻⁷¹. Brunner–Munzel inference is approximate and
  assumes independent isolates. Holm adjustment does not repair invalid component p-values. Missing patient linkage
  leaves actual error control uncertain.
- **Equally prominent:**
  - at the frozen cut-off, sensitivity 0.897 (point misses 0.90) and specificity 0.393, with 419 of 690 susceptible
    isolates flagged;
  - under-prediction of resistance (calibration intercept 0.642 [0.581, 0.702]);
  - the confidence zone's NPV 0.907 [0.865, 0.944], with 18 of its 193 isolates R or I, below the 0.95 target;
  - an internal–external gap of −0.021 [−0.086, +0.041], not demonstrated, which is neither equivalence nor
    non-inferiority.
- **Limitations:**
  - no patient linkage, and unknown screening status;
  - acquisition and preprocessing differences the frozen pipeline cannot remove;
  - one hospital, one year and one instrument;
  - reader exclusions of 6.07 % over all years, but 0.81 % in the 2024 folder.
- **A recorded deviation.** An unregistered I-excluded p-value was computed and displayed during verification. It was
  excluded from inference and changed no decision.
- **Status.**
  - Version 2.0 is complete, and its one-time evaluation is spent.
  - Of the requirements for a future study (above), it met independence of institution, country and period, and its
    sample type is recorded. But it has no patient linkage, and its sample types could not identify screening samples.
  - It supports nothing about clinical use.
