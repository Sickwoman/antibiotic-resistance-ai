# Adaptive AI for Rapid Antibiotic Resistance Prediction from MALDI-TOF Mass Spectrometry Data

[![tests](https://github.com/Sickwoman/antibiotic-resistance-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/Sickwoman/antibiotic-resistance-ai/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Research prototype only.** This project is not intended to replace laboratory antimicrobial
> susceptibility testing (AST) or professional medical decision-making, and it never recommends
> treatments. All outputs are AI research predictions on a public, de-identified dataset.

**Status: Version 0.2 – spectrum preprocessing, processed dataset and leakage-safe splits
(no model trained yet).** Version 0.1 (download + exploration) is complete. The evaluation protocol for
the models is drafted in [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) (awaiting approval).
The complete README (architecture, training, results, limitations, ethics) is written at Version 1.0,
once real results exist.

## The idea in simple words

Laboratories normally run AST to find out whether bacteria are resistant to an antibiotic, which takes
time. Many laboratories already record a MALDI-TOF mass spectrum of each isolate to identify the
species. This project trains a model on historical examples where both the spectrum and the true AST
result are known, and then estimates the probability of resistance for a new spectrum within seconds.
The main research question is whether such a model **keeps working across hospitals and time periods**,
and whether adaptation or retraining helps when it does not.

## Dataset: DRIAMS (verified 2026-09-16)

- **DRIAMS** – Database of Resistance Information on Antimicrobials and MALDI-TOF Mass Spectra
  (Weis et al., *Nature Medicine* 2022). License CC0.
- Sources: [Dryad doi:10.5061/dryad.bzkh1899q](https://datadryad.org/dataset/doi:10.5061/dryad.bzkh1899q)
  (current version, Aug 2025) and [Zenodo record 5640517](https://zenodo.org/records/5640517) (v1).

| Site | Institution | Period | Archive | Size |
|---|---|---|---|---|
| DRIAMS-A | University Hospital Basel | 11/2015–08/2018 | `DRIAMS_A.tar.gz` | 86.16 GB |
| DRIAMS-B | Canton Hospital Basel-Land | 01–06/2018 | `DRIAMS_B.tar.gz` | 3.69 GB |
| DRIAMS-C | Canton Hospital Aarau | 01–08/2018 | `DRIAMS_C.tar.gz` | 12.43 GB (Dryad version) |
| DRIAMS-D | Viollier AG (laboratory) | 01–06/2018 | `DRIAMS_D.tar.gz` | 42.56 GB |

Total ≈ 145 GB compressed. Exact byte sizes and checksums are in `config.yaml`.
Dryad replaced `DRIAMS_C.tar.gz` in Aug 2025 (raw-file naming fix), so C must come from Dryad.
Dryad refuses scripted downloads, so C is downloaded in a browser; A, B and D download from Zenodo by script.

Layout inside every archive:

```
DRIAMS-X/
├── raw/<year>/<code>.txt           raw spectrum: '#' comments, header, "m/z intensity" rows (~20,800 rows)
├── preprocessed/<year>/<code>.txt  spectrum after the published preprocessing pipeline
├── binned_6000/<year>/<code>.txt   header "bin_index binned_intensity", 6,000 rows (3 Da bins)
└── id/<year>/<year>_clean.csv      metadata: species, code and one column per antibiotic (R / I / S)
```

What we found in the real files (details in `results/metrics/eda/`):

**DRIAMS-A** (University Hospital Basel)
- 145,341 raw and preprocessed spectra (matches the paper); 111,257 metadata rows, each with a
  binned spectrum (2015: 3,198 · 2016: 34,868 · 2017: 43,122 · 2018: 30,069).
- Each year has `<year>_clean.csv` and `<year>_strat.csv` with identical rows and identical antibiotic
  values; `strat` adds `patient_no`, `case_no`, `order_no`, `acquisition_date`, `acquisition_time` and
  `workstation`, so the loader uses `strat`. `id/2016` also contains a macOS `._` metadata file (ignored).
- `patient_no` / `case_no` are 32-character hashes that never repeat across years, so patients can be
  grouped within a year but not followed across years.
- 232 spectra in the 2018 folder were acquired in 2017.
- Three E. coli "patients" have 274–533 spectra each under a single case and order, spread over
  months and all sample types – most likely placeholder IDs. None of them has a ciprofloxacin result.
- E. coli from hospital-hygiene samples are 78 % ciprofloxacin-resistant versus 20–32 % for the
  clinical sample types – a possible shortcut that must be handled in modelling.

**DRIAMS-B** (Canton Hospital Basel-Land)
- `2018_clean.csv` has 5,897 rows and 47 columns: `species`, `code`, `combined_code` (empty),
  40 antibiotic columns with R/I/S values and 4 screening columns with 0/1 values
  (`ESBL`, `MRSA`, `Cefoxitin_screen`, `Clindamycin_induced`).
- No `patient_no`, `case_no`, `acquisition_date` or `workstation` column.
- `binned_6000` exists only for the 2,386 samples that have resistance results; `raw` and
  `preprocessed` have 6,416 files each.

**DRIAMS-D** (Viollier AG, diagnostic laboratory; checked 2026-09-17)
- `id/2018/2018_clean.csv` has 10,436 rows and 55 columns: `code`, `species`, `genus` and 52 antibiotic
  columns. 16 of these are completely empty, and `Cefuroxime` appears twice in the header.
- No `patient_no`, `case_no`, `acquisition_date` or `workstation` column (like DRIAMS-B).
- Codes are 41 characters: a UUID plus `_3312` or `_3313`. The meaning of this suffix is not documented.
  `raw`, `preprocessed` and `binned_6000` files carry the same name.
- 75,813 raw and preprocessed spectra, of which only 10,436 have a metadata row. Every metadata row has a
  binned spectrum.
- 54 species strings, with no failed identifications and no mixed cultures.
- `Cotrimoxazole` holds only R values (1,280 rows, not a single S), so that column looks selectively
  reported.

**Across sites:**
- Some antibiotic names differ (e.g. `Cotrimoxazole` in A and D, `Cotrimoxazol` in B). Ciprofloxacin and
  ceftriaxone are spelled the same everywhere.
- No spectrum code appears at more than one site.

### Disk space

About 145 GB for all four archives, plus the extracted folders. Only `id/` and `binned_6000/` are
extracted in Version 0.1, which is much smaller: DRIAMS-A 17.6 GB (24 min to extract),
DRIAMS-B 382 MB (under 1 min), DRIAMS-D 1.67 GB (13 min). Raw spectra inside DRIAMS-A alone would be 62.9 GB.
Data lives **outside** the repository in `C:\DRIAMS` (change it in `config.yaml` or with the
`DRIAMS_ROOT` environment variable).

## Setup (Windows, PowerShell)

Requires Python 3.12 (3.11 also works) and Git.

```powershell
cd C:\Projects\antibiotic-resistance-ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Exact package versions of the tested environment are in `requirements-lock.txt`.

## Version 0.1 usage

All commands run from `C:\Projects\antibiotic-resistance-ai` with the virtual environment active.

```powershell
# 1. Tests (synthetic data, no download needed) and code-style check
python -m pytest -q
python -m ruff check .

# 2. Download + verify (resumable: re-run the same command after an interruption)
python scripts/download_driams.py --list
python scripts/download_driams.py --site B          # ~3.7 GB
python scripts/download_driams.py --site A          # ~86 GB, several hours
python scripts/download_driams.py --site D          # ~43 GB (needed from Version 0.7)
# DRIAMS-C: download DRIAMS_C.tar.gz in a browser from the Dryad page, then:
python scripts/download_driams.py --site C --from-file "$env:USERPROFILE\Downloads\DRIAMS_C.tar.gz"

# 3. Extract only the metadata and binned spectra
python scripts/extract_driams.py --site B
python scripts/extract_driams.py --site A
python scripts/extract_driams.py --site D

# 4. Explore (tables + figures, no model)
python scripts/explore_dataset.py
```

The downloader fetches 8 byte ranges in parallel because Zenodo limits each connection to about
0.3 MB/s (DRIAMS-A took about 4 hours at 5–9 MB/s). It waits and retries when Zenodo is temporarily
down (`--max-wait-min`, default 30), and checks the final size and MD5/SHA-256 against the published
values before the archive gets its final name. The extractor refuses unsafe paths (absolute, `..`,
links) and skips macOS `._` files. Both scripts stop Windows from sleeping while they run; closing
the laptop lid still sleeps and pauses the work (re-run the same command to continue).

Outputs of step 4:

- `results/metrics/eda/*.csv` – inventory, column presence, species, label counts, missing values,
  duplicates, pair candidates, per-site/year counts, `summary.json`
- `results/plots/eda/*.png` – samples per site/year, top species, top antibiotics, R/I/S
  distributions, target-species class balance, label coverage heatmap, example spectrum

The notebook `notebooks/01_data_exploration.ipynb` shows the same analysis interactively
(`jupyter notebook`).

## Choosing the species–antibiotic pair

Target species: *Escherichia coli*. Preferred antibiotic: ciprofloxacin. Labels: S → 0, R → 1, and
I → 1 (as in Weis et al. 2022; I counts are reported separately). The eligibility rules were fixed in
`config.yaml` **before** looking at DRIAMS-A: at least 500 resistant and 500 susceptible samples with
spectra in DRIAMS-A, minority class ≥ 10 %, both classes present in ≥ 3 DRIAMS-A years, ≥ 50 resistant
in 2018 (temporal test), and ≥ 30 per class in at least 2 of DRIAMS-B/C/D (external test). If
ciprofloxacin fails, the eligible antibiotic with the largest minority class is used.

**Result: E. coli + ciprofloxacin.** It was selected on 2026-09-16 while one external site was still
missing. It was confirmed on 2026-09-17 once DRIAMS-D was available: it now meets every
pre-registered rule, with both DRIAMS-B and DRIAMS-D having at least 30 samples per class.

| | Resistant (R+I) | Susceptible | Resistant share | Patients R / S |
|---|---|---|---|---|
| DRIAMS-A 2015–2018 | 1,466 (1,371 R + 95 I) | 3,445 | 29.9 % | 667 / 1,797 |
| DRIAMS-A 2018 only | 382 | 993 | 27.8 % | – |
| DRIAMS-B 2018 | 59 (58 R + 1 I) | 154 | 27.7 % | – |
| DRIAMS-D 2018 | 371 (371 R + 0 I) | 1,568 | 19.1 % | – |

Counts are samples with a binned spectrum, before the Version 0.2 exclusions.

- **Resistance rate differs between sites:** it is lower at DRIAMS-D, so external results are reported
  per site.
- **Benchmark pair** E. coli + ceftriaxone (published AUROC 0.74): A 1,086 R+I / 3,875 S, B 45 / 168,
  D 198 / 1,796.

## Version 0.2 – from raw spectrum to model-ready data

### Preprocessing (stateless, one spectrum at a time)

We re-implemented the exact DRIAMS pipeline in NumPy, from the authors' R scripts
(`amr_maldi_ml/DRIAMS_preprocessing/*_preprocessed.r` in BorgwardtLab/maldi_amr) and the MALDIquant
1.22.3 source code:

| Step | Setting | Why |
|---|---|---|
| Validation | ≥ 100 points, finite values, strictly increasing m/z, no negative intensities | corrupt files are excluded, never repaired |
| Intensity transform | square root | stabilises the variance of count data |
| Smoothing | Savitzky–Golay, half-window 10, cubic, MALDIquant edge handling | reduces noise without flattening peaks |
| Baseline | SNIP 20 iterations, then SNIP 100 iterations | the DRIAMS scripts call `removeBaseline(removeBaseline(x, SNIP, 20))`; the outer call uses the default of 100 |
| Normalisation | divide by total ion current (trapezoidal area of the whole spectrum) | removes differences in overall signal strength |
| Trim | 2,000–20,000 Da | the range used by DRIAMS |
| Binning | sum per 3 Da bin → **6,000 features**, `float32` | fixed-length vector for every spectrum |

**Deliberately not done:** peak picking, spectral alignment/warping, scaling/standardisation, PCA,
feature selection, resampling. Peak picking and warping add tuning choices without evidence of
benefit here and would break comparability with DRIAMS. Anything that learns from the data (scaling,
PCA, feature selection) must be fitted on training data only, so it belongs inside the Version 0.3 model
pipelines, not in the dataset.

**Verification:** our features reproduce the published DRIAMS `binned_6000` files. For the 4,472 samples
kept, the largest relative difference is 5.9 × 10⁻⁸ (float32 rounding). A single SNIP pass would differ by
about 5 × 10⁻⁵, which is how the undocumented second pass was confirmed. The builder checks every sample
against its published binned file (`dataset.verify_against_driams_binned`).

### Primary dataset: E. coli + ciprofloxacin (built 2026-09-16)

Rules: I counted as resistant; hospital-hygiene samples excluded; DRIAMS-A metadata from `strat` files.

| | Samples | Resistant (R + I) | Susceptible | Patient groups |
|---|---|---|---|---|
| DRIAMS-A (11/2015–08/2018) | 4,259 | 971 (907 R + 64 I) | 3,288 | 2,137 |
| DRIAMS-B (2018) | 213 | 59 (58 R + 1 I) | 154 | – (no patient IDs) |
| **Total** | **4,472** | **1,030** | **3,442** | |

X = 4,472 × 6,000 `float32` (107.3 MB, memory-mapped); metadata 0.8 MB.

Every E. coli metadata row is either used or excluded with one reason (first reason that applies):

| Reason | DRIAMS-A | DRIAMS-B |
|---|---|---|
| no ciprofloxacin result | 2,334 | 625 |
| ambiguous result (e.g. `R(1), S(1)`) | 75 | 0 |
| hospital-hygiene sample | 633 | not identifiable (no workstation column) |
| malformed raw file (corrupt rows / m/z running backwards) | 8 | 0 |
| raw file differs from the published binned file | 11 | 0 |
| **E. coli rows in metadata** | **7,320** | **838** |

The 11 "differs" cases: DRIAMS ships a raw file and a binned file for the same code that contain two
different measurements (6–38 % relative difference), so the AST label cannot be tied to the raw file with
confidence. Not included either: 61 (A) and 1 (B) mixed-culture rows (`MIX!Escherichia coli`). A further
281 E. coli codes of the 2017 table also have an identical copy of their raw file in the 2018 folder;
these copies are not referenced by any metadata row and are never used.

**Sensitivity dataset** (`--intermediate-as exclude`): 4,407 samples, 965 R / 3,442 S
(A 907 / 3,288, B 58 / 154); 96 I results removed. It uses exactly the same partition as the primary
dataset (see below).

### Splits (row indices only)

| Split | Train | Validation | Test | Resistant in test |
|---|---|---|---|---|
| random (A, stratified, patient-grouped, seed 42) | 2,977 | 426 | 856 | 197 |
| within_year (as random, A year folder 2017 only) | 1,284 | 183 | 366 | 85 |
| temporal (A, by `acquisition_date`) | 2,505 (< 2017-10-01) | 465 (2017-10-01 … 2017-12-31) | 1,233 (≥ 2018-01-01) | 271 |
| external (train A, test B) | 3,831 | 428 | 213 | 59 |

- **Leakage checks:** every split is checked so that no sample and no patient group appears in two
  parts. The temporal split drops 56 training samples whose patient group also appears later.
- **Why `within_year` exists:** DRIAMS-A `patient_no` is re-hashed every year, so a person seen in two
  years cannot be linked, and the pooled `random` split may place them in train and test.
  - Repeat sampling is common: 76 % of the DRIAMS-A spectra (3,247 / 4,259) come from patients with more
    than one spectrum in the same year.
  - Inside one year folder, patient grouping is complete. Comparing `within_year` with `random` shows how
    much this matters.
- **Tied to one dataset build:** each split file stores the fingerprint of its dataset (sample keys,
  labels, patient groups). `load_split(path, meta)` refuses a split made for another build.
- **Sensitivity dataset (I excluded):** instead of drawing new splits, it reuses the primary dataset's
  saved ones (each sample keeps its part), so the two datasets differ only by the removed I samples.
  Sizes: random 2,930 / 421 / 844, within_year 1,268 / 179 / 365, temporal 2,473 / 459 / 1,208,
  external 3,776 / 419 / 212.
  - Before this change, independently drawn splits put 44 % of the shared samples in a different part of
    the random split.
- **Other limitations:** patients cannot be linked across hospitals; DRIAMS-B has no patient IDs, dates
  or sample types.

### Commands

```powershell
# raw E. coli spectra (one pass over each archive; A takes ~20 min)
python scripts/extract_driams.py --site B --folders raw preprocessed --species "Escherichia coli"
python scripts/extract_driams.py --site A --folders raw preprocessed --species "Escherichia coli"

# build dataset + splits + reports (~4-5 min each); the primary dataset first,
# because every other dataset reuses its splits
python scripts/build_dataset.py
python scripts/build_dataset.py --intermediate-as exclude
```

**Outputs:**
- `data/processed/<name>/`: `X.npy`, `metadata.csv`, `exclusions.csv`, `summary.json` and
  `splits/*.json` (git-ignored).
- `results/metrics/v0.2/<name>/`: reports without identifiers.

`summary.json` records a rows fingerprint and the SHA-256 of `X.npy`.

**Loading:** use `src.dataset.load_dataset()`, which checks the fingerprint (`verify_x=True` also
re-hashes `X.npy`), and `src.splits.load_splits(folder, meta)` / `load_split(path, meta)`. The notebook
`notebooks/02_preprocessing.ipynb` shows an example.

**Adding DRIAMS-C/D later:** download, then extract `id`, `binned_6000` and (for E. coli) `raw`. Add the
site to `dataset.sites` and `splits.external.test_sites` in `config.yaml`, and rebuild both datasets.

## Project structure (Version 0.2)

```
antibiotic-resistance-ai/
├── config.yaml                 all paths, archive checksums, label rules, selection thresholds
├── requirements.txt / requirements-lock.txt
├── scripts/
│   ├── download_driams.py      parallel resumable download + checksum verification
│   ├── extract_driams.py       selective streaming extraction + file manifest
│   ├── explore_dataset.py      Version 0.1 exploration report
│   └── build_dataset.py        Version 0.2 dataset, splits and reports
├── src/
│   ├── utils.py                config, paths, seeding, logging, keep-awake
│   ├── data_loader.py          metadata tables, label rules, spectrum readers with validation
│   ├── exploration.py          exploration tables and figures
│   ├── preprocessing.py        MALDIquant-equivalent preprocessing and binning
│   ├── dataset.py              cohort selection, exclusion audit, dataset builder/loader, fingerprints
│   └── splits.py               random / within_year / temporal / external splits, leakage checks,
│                               split reuse for sensitivity datasets
├── docs/evaluation_protocol.md how models will be evaluated (draft, fixed before any training)
├── notebooks/01_data_exploration.ipynb, 02_preprocessing.ipynb
├── tests/                      pytest suite (synthetic data; runs on GitHub Actions for every push)
├── data/ models/ results/      (large files are git-ignored)
├── .github/workflows/tests.yml Ruff + pytest on Ubuntu and Windows
├── ruff.toml                   code-style rules
├── LICENSE, CITATION.cff, CLAUDE.md
```

## License and citation

Code: MIT (see `LICENSE`). The DRIAMS data is not part of this repository; it is CC0 and must be
downloaded from its original source. If you use this work, cite it via `CITATION.cff` and cite
Weis et al. (2022) and the DRIAMS dataset.

## References

1. Weis C, Cuénod A, Rieck B, et al. Direct antimicrobial resistance prediction from clinical MALDI-TOF
   mass spectra using machine learning. *Nature Medicine* 28, 164–174 (2022).
   https://doi.org/10.1038/s41591-021-01619-9
2. Weis C, Cuénod A, Rieck B, Borgwardt K, Egli A. DRIAMS: Database of Resistance Information on
   Antimicrobials and MALDI-TOF Mass Spectra. Dryad (2021, updated 2025).
   https://doi.org/10.5061/dryad.bzkh1899q
3. maldi-learn (BorgwardtLab): https://github.com/BorgwardtLab/maldi-learn
4. maldi_amr, DRIAMS preprocessing scripts (BorgwardtLab):
   https://github.com/BorgwardtLab/maldi_amr/tree/public/amr_maldi_ml/DRIAMS_preprocessing
5. Gibb S, Strimmer K. MALDIquant: a versatile R package for the analysis of mass spectrometry data.
   *Bioinformatics* 28, 2270–2271 (2012). https://doi.org/10.1093/bioinformatics/bts447
