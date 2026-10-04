# Version 2.0, steps 1–3: outputs before any label

These files come from real MARISMa 2.0.0 data, produced after the owner approved steps 1–3 on 2026-10-03
(`docs/v2.0_marisma_plan.md`, approval record). They are spectra and metadata only. They hold aggregate counts, rates
and distributions, and no isolate identifier, archive path, spectrum, feature, label, prediction or model output.

**Status: processing stopped at the step 3 gates (amendments A6.3 and A6.4), before any label or outcome count.**
- 60.0 % of the cohort's isolates have no replicate that passes the reader's checks.
- Every failure is the 2,000–20,000 Da coverage check.
- The failures are concentrated by instrument and year.
- Continuing needs a dated owner decision.

## Files

- **`archive_integrity.json`**, from `scripts/v20_archive_integrity.py`: the archive's SHA-256 against the download's
  provenance record, and every member's CRC-32 and size. Nothing was extracted.
- **`step3_preparation.json`**, from `scripts/v20_prepare_marisma.py`:
  - the cohort counts at each rule;
  - exclusions by reason, and the aggregate pause check (amendment A6.4);
  - every reader attempt's outcome, within each batch (year folder, acquisition month, instrument), for the
    systematic-error review (amendment A6.3);
  - the calibration parameters of the spectra that passed;
  - the frozen features' shape and fingerprint;
  - the spectra-only checks (descriptive, not a gate):
    - each spectrum's correlation with the mean DRIAMS-A *E. coli* training spectrum, with DRIAMS-A's own validation
      spectra as a within-site reference;
    - the alignment of the two mean spectra;
  - the antibiotic names in `AMR.csv`'s header;
  - the protected-state check: frozen bundles, zone file, logs and DRIAMS datasets unchanged.

  The features and the spectra-only checks cover only the spectra that passed. The schema check's counts were not
  run, because the pause holds.
- **`reader_reference_check.json`**, from `scripts/v20_reader_reference_check.py`: the Bruker reader's decoding
  against two independent implementations, nmrglue and maldi-nn, on a seeded sample of the cohort's replicate folders.
  The sample includes spectra the checks refuse.

The per-isolate files stay outside the repository, in the data root's `MARISMa_v2.0.0_work` folder: the cohort's
replicate folders, the selection, the reader attempts, the features and their hashes.

No model has scored any MARISMa spectrum. Scoring needs the separate approval of step 4.
