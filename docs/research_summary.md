# Research summary: MALDI-TOF resistance prediction on DRIAMS, Versions 0.1–1.4

*One-page technical summary, 2026-10-01. Full report: [research_report_v0.1-v1.4.md](research_report_v0.1-v1.4.md).
Research prototype; not a diagnostic.*

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
  exists.** The zone failed in 2018 and at DRIAMS-C.
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
- No intervention tested demonstrated a benefit.
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
- The original V0.7/V0.8 commits survive only in the reflog and PR refs.

**Requirements for a future independent study.**
1. A new hypothesis, justified and recorded before data access.
2. Data no version has used, with cross-time patient linkage and sample type recorded.
3. A pre-registered endpoint, comparator, interval method and sample size.
4. Externally justified operating requirements.
5. DRIAMS-C only through an amendment, and only for a procedure that first earns it on development data.
