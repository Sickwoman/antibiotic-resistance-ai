# Run 1 (2026-10-03): historical record of an incomplete cohort

**Do not use these files as the Version 2.0 analysis cohort.** They record the first run of steps 2–3, which stopped
at the step 3 gates (amendments A6.3 and A6.4) before any label or outcome count was read. The files are kept
unchanged; only their folder moved, on 2026-10-04.

| | |
|---|---|
| Run | `scripts/v20_prepare_marisma.py` at commit `b691cf7`, 2026-10-03 16:43:44–16:48:19 UTC |
| Antibiotic names | refreshed with `--names` at commit `5391e7f`, 2026-10-04 06:24:11 UTC, after the reader's delimiter fix |
| Reader reference check | `scripts/v20_reader_reference_check.py` at commit `7e0f36a` |

## Why the cohort is incomplete

- **The sample-source rule was not applied.** Its field is only in `AMR.csv`, which amendment A2 did not let any
  code read beyond antibiotic names and counts.
- **The coverage check excluded 60.0 % of isolates.** 10,178 of 16,975 had no replicate passing the 2,000–20,000 Da
  check. The failures are concentrated by instrument and year.
- **The schema check's counts were not run**, because the pause holds.

## Two deviations from the approved plan

Neither was pre-approved. Both are recorded in the plan's deviation record of 2026-10-04.

- **A species-consistency exclusion.** 10 identifiers whose isolate folders sit under more than one genus or species
  were excluded. This rule is not in the approved plan: it was an implementation choice made during step 2.
- **A header inspection beyond amendment A2's outputs.** The restricted reader listed no antibiotic names, because
  `AMR.csv` 2.0.0 is comma-separated with `MIC_` columns, while the reader expected semicolons and `CMI_` columns.
  To diagnose this, two structural probes read the header line.
  - The first printed no column names. It showed the delimiter counts, the number of fields, and how many fields
    start with given prefixes.
  - The second confirmed, by name, that `Identifier` and `Sample` columns exist.

  No field value and no data row was read.

## Files

- **`step3_preparation.json`**: cohort counts, exclusions, the pause check, the batch review, features and the
  spectra-only checks for the 6,797 spectra that passed, and the antibiotic names.
- **`reader_reference_check.json`**: the reader's decoding against nmrglue and maldi-nn, on 400 seeded replicate
  folders. This evidence about the reader stays valid whatever cohort is finally built.
- **`step3_outputs.md`**: the description written with the run.

The per-isolate files of this run are outside the repository, in the data root's
`MARISMa_v2.0.0_work/run1_blocked_2026-10-03` folder, with their SHA-256 values.
