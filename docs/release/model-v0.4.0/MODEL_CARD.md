# Model card: E. coli ciprofloxacin resistance ranker, `v0.4.0-tuned_lightgbm-random-seed42`

**Research use only.**
- This is not a medical device and not a diagnostic.
- It does not replace antimicrobial susceptibility testing, and it never recommends a treatment.
- It has no regulatory status.

## What it is
- **Input.** One raw MALDI-TOF mass spectrum of an *E. coli* isolate: two text columns, m/z and intensity. The
  repository's frozen pipeline turns it into 6,000 features (feature fingerprint
  `347cbd6d5d956ff9`): square root, Savitzky–Golay smoothing, SNIP baseline, total-ion-current
  normalisation, and 3 Da bins over 2,000–20,000 Da.
- **Model.** LightGBM (300 trees, seed 42). It is wrapped in scikit-learn's
  sigmoid calibration, fitted on out-of-fold predictions of 5 patient-grouped folds of the training part.
- **Output.** A score in [0, 1], where a higher score means resistance (R or I) is more likely by this model's
  ranking. The frozen research threshold, 0.14261540693905073, is the highest validation threshold with
  sensitivity ≥ 0.90.
- **Labels it learned from.** 1 = R or I, and 0 = S, for ciprofloxacin.

## Training data and provenance
- **Data.** DRIAMS-A (University Hospital Basel), 2,977 training spectra from a
  patient-grouped random split. DRIAMS: Weis et al., Dryad, doi:10.5061/dryad.bzkh1899q, CC0 1.0.
- **Saved.** On 2026-09-18 12:54:10, from commit `3c8245b` of https://github.com/Sickwoman/antibiotic-resistance-ai (resolvable through the tag
  `research-archive/v0.4/run`).
- **The bundle.** SHA-256 `d59d6d7deafa1af464c33ebefc0f8f641a80eb70f7b51039e6c2ede0fd841c8b`, 636,912 bytes. It is copied byte for byte from the
  original.

## Evaluation (aggregates, from the repository's committed results)
- **Internal test,** DRIAMS-A, held-out patients (856 spectra, 197 R or I):
  - AUROC 0.751 (0.696–0.807);
  - sensitivity 0.827 and specificity 0.458 at the
    threshold.
- **External test,** MARISMa 2.0.0 (Madrid; 1,145 eligible isolates from 2024, 455 R or I):
  - AUROC 0.772 (descriptive 95% interval 0.744–0.798): "Evidence of above-chance ranking on MARISMa under the isolate-independence assumption."
    Brunner–Munzel inference is approximate and assumes independent isolates. Holm adjustment does not repair invalid component p-values. Missing patient linkage leaves actual error control uncertain.
  - With equal weight:
    - sensitivity 0.897 and specificity 0.393 at the
      threshold;
    - calibration intercept 0.642, meaning resistance was under-predicted;
    - confidence-zone NPV 0.907, with 18 R or I isolates among
      193 zone members, below its 0.95 research target;
    - no internal–external AUROC gap was demonstrated, which is not equivalence.
  - The ceftriaxone evaluation was unavailable.

## Known limitations
- **Narrow training.** One development hospital, one species and one antibiotic. The discrimination is modest.
- **The threshold.** The 0.90-sensitivity threshold did not hold on held-out DRIAMS patients, and missed it on its
  point estimate at MARISMa.
- **The probabilities** were too low on MARISMa: under-predicted.
- **The confidence zone is not a safety feature.**
- **MARISMa's limits.** It has no patient linkage, so its intervals and p-values may be too narrow. Its screening
  status is unknown, and its acquisition differs from DRIAMS's.
- **Not established anywhere else.** Nothing is established for other sites, years, instruments, species or
  antibiotics, or prospectively.

## Using it
- **Environment.** Python 3.12.10, numpy 2.5.3, pandas
  3.0.5, scikit-learn 1.9.1, LightGBM
  4.7.0 and joblib 1.6.0, as pinned in the repository's
  `requirements-lock.txt`.
- **Install.** `python scripts/install_model.py <package.zip>`, from the repository. It checks the bundle against the
  SHA-256 pinned in the repository before anything is written, and never replaces a different model.
- **Load** the bundle only through the repository's loader (`src/predict.py`), and only a copy that passed that check.
  A joblib file is a pickle: loading one from an untrusted source can run arbitrary code. A matching `.sha256` file
  next to it is not proof of origin.

## Citation
Cite the repository (`CITATION.cff`) and DRIAMS (Weis et al., *Nat Med* 2022, doi:10.1038/s41591-021-01619-9; data
doi:10.5061/dryad.bzkh1899q). For the external figures, cite MARISMa (Zenodo doi:10.5281/zenodo.17201597;
descriptor doi:10.1101/2025.05.31.657186).
