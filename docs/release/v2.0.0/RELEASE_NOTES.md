# v2.0.0: external validation on MARISMa, and a local research demo

*Prepared 2026-10-06; not yet published. This is the code and research release. The model artifact has a separate
release (`model-ecoli-ciprofloxacin-v0.4.0`, [PUBLICATION.md](../model-v0.4.0/PUBLICATION.md)), and this release
contains no model file.*

**Research prototype.** Not a clinically validated diagnostic. It does not replace antimicrobial susceptibility
testing, and nothing here recommends a treatment.

## What is new since 1.4.0

**Version 2.0: a pre-registered external validation.** The served ciprofloxacin model, frozen and unchanged since
Version 0.4 (`v0.4.0-tuned_lightgbm-random-seed42`), was scored once on 1,145 eligible MARISMa *E. coli* isolates from
2024 (Madrid; 455 resistant, 690 susceptible). It was not refitted, recalibrated or re-thresholded.
- **Primary, in the registered wording:** "The frozen ciprofloxacin model achieved AUROC 0.772 (descriptive 95%
  interval 0.744–0.798) on 1,145 eligible MARISMa E. coli isolates from 2024, with evidence of above-chance ranking
  under the isolate-independence assumption." Brunner–Munzel inference is approximate and assumes independent
  isolates. Holm adjustment does not repair invalid component p-values. Missing patient linkage leaves actual error
  control uncertain.
- **With equal weight:**
  - sensitivity 0.897 and specificity 0.393 at the frozen cut-off;
  - calibration intercept 0.642 [0.581, 0.702], meaning resistance was under-predicted;
  - confidence-zone NPV 0.907, with 18 of the 193 isolates in the zone R or I, below its 0.95 research target;
  - internal minus MARISMa AUROC −0.021 [−0.086, +0.041], not demonstrated, which is neither equivalence nor
    non-inferiority.
- **Ceftriaxone could not be evaluated:** MARISMa holds no interpretation for it.
- **Limits:** no patient linkage; unknown screening status; acquisition and preprocessing differences; one hospital,
  one year and one instrument. Clinical usefulness is unproven.
- **The record:**
  - `docs/v2.0_marisma_plan.md`: the plan, its amendments and the execution record;
  - `results/metrics/v2.0/tables.md`;
  - research report section 13;
  - the thesis chapter's section 11.

**A research demo: a software demonstration.**
- `python -m demo` serves a local dashboard. It runs the unchanged prediction command on three synthetic spectra, beside
  the committed evaluation results and their limits.
- The synthetic scores demonstrate the pipeline. They are not results about any organism.
- Guide: `demo/README.md`.

**Model packaging.**
- `scripts/package_model.py` builds a reproducible package of the frozen bundle.
- `scripts/install_model.py` installs one from a local file, after checking it against the SHA-256 pinned in the
  repository.

**Documentation.** The changelog, `CITATION.cff` (2.0.0) and research report section 13, and an addendum to the thesis
chapter with its number trace.

## Verifying it
- **CI:** the unit tests, on Ubuntu and Windows with Python 3.11 and 3.12.
- **On one Windows machine with the bundle:**
  - `docs/demo/verification.md`: the browser check, end to end;
  - `docs/release/model-v0.4.0/HANDOFF_CHECK.md`: a fresh clone and a clean environment.

  These are observations from that machine, not guarantees for others.

## Publishing it (the owner's action; not done)

1. **Merge the branch.** `release/v2.0.0` carries the changelog, `CITATION.cff`, these notes and the thesis addendum.
   Merge it into `main`.
2. **Optionally date it.** In the same or a follow-up commit, add `date-released` to `CITATION.cff`. These records
   leave it unset rather than invent one.
3. **Publish**, at the resulting `main` commit:

   ```powershell
   gh release create v2.0.0 --target <main commit after step 1> `
     --title "v2.0.0: external validation on MARISMa, and a local research demo" `
     --notes-file docs\release\v2.0.0\RELEASE_NOTES.md
   ```

This release has no assets beyond GitHub's source archives, and it is independent of the model release.
