# Adaptive AI for Rapid Antibiotic Resistance Prediction from MALDI-TOF Mass Spectrometry Data

[![tests](https://github.com/Sickwoman/antibiotic-resistance-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/Sickwoman/antibiotic-resistance-ai/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Research prototype only.** This project is not intended to replace laboratory antimicrobial
> susceptibility testing (AST) or professional medical decision-making, and it never recommends
> treatments. All outputs are AI research predictions on a public, de-identified dataset.

**Status: Version 0.1 – dataset acquisition and exploration (no model yet).**
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

**Across sites:** some antibiotic names differ (e.g. `Cotrimoxazole` in A, `Cotrimoxazol` in B).
Ciprofloxacin and ceftriaxone are spelled the same everywhere.

### Disk space

About 145 GB for all four archives, plus the extracted folders. Only `id/` and `binned_6000/` are
extracted in Version 0.1, which is much smaller: DRIAMS-A 17.6 GB (24 min to extract),
DRIAMS-B 382 MB (under 1 min). Raw spectra inside DRIAMS-A alone would be 62.9 GB.
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
# 1. Tests (synthetic data, no download needed)
python -m pytest -q

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

**Result (2026-09-16): E. coli + ciprofloxacin selected**, final confirmation pending one more external
site (C or D).

| | Resistant (R+I) | Susceptible | Patients R / S |
|---|---|---|---|
| DRIAMS-A 2015–2018 | 1,466 (1,371 R + 95 I) | 3,445 | 667 / 1,797 |
| DRIAMS-A 2018 only | 382 | 993 | – |
| DRIAMS-B 2018 | 59 (58 R + 1 I) | 154 | – |

Benchmark pair E. coli + ceftriaxone (published AUROC 0.74): A 1,086 R+I / 3,875 S, B 45 / 168.

## Project structure (Version 0.1)

```
antibiotic-resistance-ai/
├── config.yaml                 all paths, archive checksums, label rules, selection thresholds
├── requirements.txt / requirements-lock.txt
├── scripts/
│   ├── download_driams.py      parallel resumable download + checksum verification
│   ├── extract_driams.py       selective streaming extraction + file manifest
│   └── explore_dataset.py      Version 0.1 exploration report
├── src/
│   ├── utils.py                config, paths, seeding, logging
│   ├── data_loader.py          metadata tables, label encoding, spectrum readers with validation
│   └── exploration.py          exploration tables and figures
├── notebooks/01_data_exploration.ipynb
├── tests/                      pytest suite (synthetic data; runs on GitHub Actions for every push)
├── data/ models/ results/      (large files are git-ignored)
├── .github/workflows/tests.yml
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
