# Reproduction guide, Versions 0.1–1.4

Companion to [the research report](research_report_v0.1-v1.4.md). Written 2026-10-01 against commit `0806e74` and
the report commit on branch `v1.4-report`. **Addendum, 2026-10-04:** section 6 covers Version 2.0 (MARISMa), and
section 1.2 notes the production log's two new rows. Everything else is unchanged.

**Read this first.** The repository is built so that every reported number can be *checked* without being
*recomputed*. Most historical commands would retrain models, re-read labels whose test parts are already spent, append
to the project's append-only logs, or overwrite committed results. So this guide separates three tiers, and only the
first two are meant to be run.

| Tier | What | Safe to run? |
|---|---|---|
| 1 | Artifact and checksum verification | **yes** — read-only |
| 2 | Tests with fixtures and temporary data | **yes** — writes only to temporary folders; a guard fails the run if a real log changes |
| 3 | Historical experiment commands | **no** — documentation only |

## 0. Environment, inputs and access

- **Tested platform:** Windows 11 (build 10.0.26200), PowerShell and Git Bash, Python **3.12.10** in a project venv;
  CPU only (no CUDA), 7.7 GB RAM. CI (GitHub Actions) tests Ubuntu and Windows with Python 3.11 and 3.12; on
  2026-10-01 it ran on the five version branches (PRs #26–#30) and passed on each head.
- **Dependencies:** `requirements.txt` (ranges) for development; **`requirements-lock.txt` (exact pins) to reproduce**:
  `pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu`. Key pins: numpy 2.5.3,
  pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, lightgbm 4.7.0, torch 2.14.0+cpu, joblib 1.6.0, pytest 9.1.1,
  ruff 0.16.8. Each run also records its library versions in its `run_config.json` or model card; historical runs
  used the versions recorded there.
- **Data (not in the repository):** DRIAMS, CC0 (doi:10.5061/dryad.bzkh1899q). Archives A, B and D came from Zenodo
  record 5640517 and C from Dryad via a browser; the expected sizes and checksums are in `config.yaml →
  driams.archives`. The extracted tree lives outside the repository at `C:\DRIAMS` (`config.yaml → paths.driams_root`,
  or the environment variable `DRIAMS_ROOT`). Built datasets live in `data/processed/` and saved models in `models/`,
  both git-ignored.
- **Access restrictions set by the project's protocol (not by the data licence):** every test part of the A, B and D
  cohorts is spent, for both antibiotics; DRIAMS-C's protected part is spent for ciprofloxacin; **DRIAMS-C's
  ceftriaxone labels may be opened only after an owner-approved amendment** (`docs/driams_c_status.md`). Nothing in
  tiers 1–2 reads them.
- **Seeds:** 42 everywhere (patient-grouped folds, partitions, models, bootstraps); stochastic models were also fitted
  with seeds 43–46 where reported; V1.2–V1.4 partitions 42, 43, 44.

## 1. Tier 1 — verify the artifacts (read-only)

Run from the project root with the venv active. Each command only reads.

**1.1 The checkout.**
```bash
git rev-parse --short HEAD        # the commit you are verifying
git status --short                # empty = no local changes
```

**1.2 The append-only logs** (these values must never change):
```bash
sha256sum results/experiments/test_evaluations.csv results/experiments/development_runs.csv
```
| Log | Data rows | SHA-256 |
|---|---|---|
| `test_evaluations.csv` (production) | 113 | `6528eb2abcdfb611d65712354b8a05daed03febdb137a0910ed0eb7764f3ddba` |
| `development_runs.csv` (development) | 34 | `dd5f4a74933a4cfc8382e9a4464ae774aab6944e280b68e8fe340fbbc2d9e894` |

**Update, 2026-10-04 (Version 2.0).** The production log gained its two planned MARISMa rows (results commit
`89e73e8`). It now has **115** data rows with SHA-256 `5a21ba7f809e11fbc783908123539ee384836d5bbd3bf6090df4ecf5121f1557`,
and its first 113 rows are byte-identical to the value above. The development log is unchanged.

**1.3 The saved model bundles** (needs `models/`; `--check` is the default and changes nothing):
```bash
python scripts/write_bundle_checksums.py
```
Expected: all four sidecars match — `v0.3` `e84b4abd…`, **`v0.4` (served) `d59d6d7deafa1af464c33ebefc0f8f641a80eb70f7b51039e6c2ede0fd841c8b`**,
`v0.5` `eabac975…`, `v1.1` `74c62690…`. Never run it with `--write` (that creates sidecars).

**1.4 The datasets** (needs `data/processed/`; hashes every `X.npy`, read-only):
```bash
python -c "from src.dataset import load_dataset; import sys; [print(n, load_dataset('data/processed/'+n, verify_x=True)[2]['row_fingerprint']) for n in sys.argv[1:]]" ecoli_ciprofloxacin ecoli_ciprofloxacin__intermediate-exclude ecoli_ciprofloxacin__site-C ecoli_ceftriaxone ecoli_ceftriaxone_with_screening
```
| Dataset | Spectra | Row fingerprint | `X.npy` SHA-256 |
|---|---|---|---|
| `ecoli_ciprofloxacin` | 6,410 | `151415a4d03dbcc5` | `c27609b6f2663e0d4d044597d8da9f9e7eede0264e2be3a1c5262cfc762e22c6` |
| `ecoli_ciprofloxacin__intermediate-exclude` | 6,345 | `9a23af98751e0770` | `d85b4f983a178a17a1eb773e30918dc0ba05ed0b542a6edc7bedfc31fe423b76` |
| `ecoli_ciprofloxacin__site-C` | 889 | `83d504f4d0f83ae8` | `18329b134b4b52fbb3be2ed2abab53e51072f03dc6a7acfbd7251fbefec1320e` |
| `ecoli_ceftriaxone` | 6,489 | `fb65868756a623ee` | `4f24ddc925c101ad657972c6b9f12969506dc6b6b12d99e4bf4444e4e2df30ab` |
| `ecoli_ceftriaxone_with_screening` | 7,138 | `b55a65935f4073bb` | `9761ef586344a5dbfdeda8c3a9b47c4e45802df52c9ff94c08130bd08e5c5ccc` |

All share the feature fingerprint `347cbd6d5d956ff9` (the preprocessing settings).

**1.5 The pre-registrations and the privacy scan:**
```bash
python -m pytest -q tests/test_preregistration.py tests/test_privacy.py
```
The first checks that each plan's locked text (and the protocol through amendment 11) is unchanged; the second that
no committed file carries a patient, case or order identifier or a spectrum UUID.

**1.6 Numbers in the report.** Every figure in the report names the committed file it came from (mostly
`results/metrics/<version>/…/tables.md`, `tables_complete.md` for V1.2, JSON/CSV beside them). Compare by reading
those files; do not regenerate them (tier 3).

## 2. Tier 2 — tests with fixtures and temporary data

```bash
python -m pytest -q                               # then check the exit code, never pipe into tail/tee
echo $?                                           # bash; in PowerShell: $LASTEXITCODE
python -m ruff check . && python -m ruff check . --target-version py311
```
- **What the suite touches.** Synthetic DRIAMS folders, datasets, models and logs created under pytest's temporary
  folders. End-to-end tests fit small models on synthetic spectra; nothing is fitted on the project's data. A few tests
  *read* committed artifacts (the pre-registration pins, the privacy scan, the V1.2 report generator against its
  saved outputs, the API tests against the saved bundle and zones when present).
- **Isolated logs, enforced.** Every script-running test points its production and development logs into its
  temporary folder; `tests/conftest.py` hashes both real logs when the session starts and **fails the session** if
  either changed (a test of this guard runs it end to end on a throwaway copy).
- **Expected outcome** on the report commit: see the report, section 8.5 — with `data/` and `models/` present, all
  tests pass; on a clean checkout without them, the two saved-bundle tests skip.
- **Runtime:** about 2.5 minutes on the tested laptop; up to about 11 minutes was observed under power throttling.
- **Exit codes.** A pipe such as `pytest | tail -1` returns `tail`'s status. Under plain `bash -e` (GitHub's Linux
  default) a failing test run piped into `tail` exits 0. CI therefore runs every step under `shell: bash`
  (`bash --noprofile --norc -eo pipefail {0}`). That change has not yet run on GitHub.

## 3. Tier 3 — historical experiment commands (documentation only; do not run)

Hazard codes: **R** retrains or refits models; **L** reads labels of spent or restricted parts; **A** appends to an
append-only log; **O** overwrites committed results, reports or saved models; **D** downloads or extracts large data.
"Guard" names protections that exist in code; they reduce, but do not remove, the damage a run would do.

| Version | Command | Ran from | Hazards | Guards and notes |
|---|---|---|---|---|
| 0.1 | `python scripts/download_driams.py --site A` (etc.) | – | D | resumable; verifies checksums; ~145 GB for all sites |
| 0.1 | `python scripts/extract_driams.py --site A --folders raw preprocessed --species "Escherichia coli"` | – | D | writes under `C:\DRIAMS` |
| 0.1 | `python scripts/explore_dataset.py` | not recorded | O | rewrites `results/metrics/eda`, `results/plots/eda` |
| 0.2 | `python scripts/build_dataset.py` (± `--intermediate-as exclude`) | `0f16192` | O | rewrites `data/processed/…` and `results/metrics/v0.2/…`; the row fingerprint is checked on every later load |
| 0.3 | `python scripts/train_baselines.py [--evaluate-test]` | `25a9bbf` | R, O; with `--evaluate-test`: L, A | refuses to re-score logged experiments unless `--allow-rescore` |
| 0.3 | `python scripts/baseline_tables.py` | – | O | rewrites the committed `tables.md` |
| 0.4 | `python scripts/tune_models.py [--evaluate-test]` | `3c8245b` | R (hours), O; L, A with `--evaluate-test` | cached fits; `--evaluate-test` refuses uncached fits and re-scoring |
| 0.5 | `python scripts/tune_models.py --section deep [--evaluate-test]` | `325ee85` | R, O; L, A | as above |
| 0.4–0.5 | `python scripts/tuned_tables.py [--section deep]` | – | O | rewrites committed tables |
| 0.6 | `python scripts/explain_model.py`; `explain_tables.py` | `d55e13f` | R (a shuffled-label control), O | never scores a test row; asserts the log unchanged |
| 0.7 | `python scripts/measure_generalisation.py` | `8878bfd` | R, L, A, O | refuses re-scoring spent parts; **`generalisation_tables.py` would overwrite the preserved V0.7 `tables.md`** (kept with an erratum) |
| 0.8 | `python scripts/build_adaptation_partition.py`; `adapt_model.py` | `284c0bd`, `4d82a22` | L (DRIAMS-C protected labels), R, A, O | refuses unless the locked state matches the approved hash; **the protected part must never be scored again** |
| 0.9 | `python scripts/benchmark_api.py` | – | O | durations only; rewrites `api_timing.json` |
| 1.1 | `python scripts/build_dataset.py --antibiotic Ceftriaxone --report-version v1.1` | `7116e12` | O | shared spectra must equal the primary's byte for byte |
| 1.1 | `python scripts/tune_models.py --section second_antibiotic --dataset ecoli_ceftriaxone --evaluate-test` | `7c26a36` | R, L, A, O | strict pre-write gate; `--allow-rescore` refused |
| 1.1 | `python scripts/measure_generalisation.py --section second_antibiotic_generalisation --dataset ecoli_ceftriaxone` | `7c26a36` | R, L, A, O | strict gate |
| 1.1 | `python scripts/second_antibiotic_report.py` | – | O | reads the log and stored probabilities |
| 1.2 | `python scripts/v12_development.py`; `v12_report.py` | `ba4144c` | R, A (development log), O | clean tree required; strict gate refuses repeated keys |
| 1.3 | `python scripts/v13_threshold.py` | `a2d6ee2` | R, A (development log), O | as above |
| 1.4 | `python scripts/discrimination_audit.py` | `4ecb130` | O; reads DRIAMS-A metadata incl. screening labels in aggregate | fits nothing |
| 1.4 | `python scripts/build_dataset.py --antibiotic Ceftriaxone --keep-workstation HospitalHygiene --name ecoli_ceftriaxone_with_screening --report-version v1.4` | `153d3e2` | O | refuses to reuse the unchanged dataset's name |
| 1.4 | `python scripts/v14_screening.py` | `94984b5` | R, A (development log), O | must reproduce V1.2 arm C exactly or stop; count and ledger checks |
| utility | `python scripts/serve_api.py`; `predict_spectrum.py <file>` | – | none of the above | read-only consumers of the frozen model; bind to 127.0.0.1; no authentication |

**If a historical run must be repeated** (for example, to audit determinism), do it in a **separate clone** with every
output path in `config.yaml` (`evaluation.test_log`, the development log, every `model_dir`, `report_dir`,
`plot_dir`, `dataset.output_dir`) pointed at a scratch folder, check out the run commit listed above, and never copy
anything back. The Version 1.4 run itself shows the expected determinism: its baseline reproduced Version 1.2's saved
held-out probabilities with difference 0 in the current environment.

## 4. Commits

Version commits, run commits and branches are listed in the [evidence map](evidence_map.md) and the
[review checklist](review_checklist.md). The original Version 0.7 and 0.8 commits (`96d0425`, `8878bfd`, `64c6c75`,
`284c0bd`, `4d82a22`, `849cad7`) are on no branch. They are preserved by archival refs under `refs/archive/` and, on
GitHub, by annotated `research-archive/…` tags, which a plain `git fetch` does not bring. Versions 1.1–1.4 were
squash-merged into `main` on 2026-10-01, so their run commits in the table above are not on `main` either. Fetch the
tags before checking any of them out, and restore any archived commit, as the review checklist's section 5 shows.

## 5. Known platform limitations

- **Sleep.** Windows Modern Standby pauses long jobs; the scripts request keep-awake, but a closed lid still stops them.
- **Memory and time.** 7.7 GB: feature matrices are memory-mapped. Version 0.4's LightGBM search took about 40
  minutes; the 40,235 s recorded for its size-matched search included about eleven hours of sleep.
- **Paths.** Git worktrees under deep folders need `git -c core.longpaths=true`; Windows long paths must be enabled.
- **Line endings.** Git normalises text to LF; the pre-registration pins hash LF-normalised bytes, so CRLF checkouts
  still pass.
- **Downloads.** Zenodo throttles single connections (hence the parallel downloader); Dryad blocks scripted downloads.
- **Shells.** PowerShell does not stop on a failing native command unless it is the last one; bash needs `pipefail` to
  see a failure inside a pipe.

## 6. Version 2.0 (MARISMa), added 2026-10-04

**Data, outside the repository.** MARISMa 2.0.0 (Zenodo `doi:10.5281/zenodo.17201597`, CC-BY-4.0):
- `MARISMa.zip` in `C:\DRIAMS\MARISMa_v2.0.0`;
- the sealed `AMR.csv` in `C:\DRIAMS\MARISMa_v2.0.0_sealed`;
- the expected sizes and checksums in `config.yaml → marisma`;
- every file holding identifiers, and the scoring run's state, under `C:\DRIAMS\MARISMa_v2.0.0_work`.

None of these is needed for tiers 1–2.

**Tier 1 (read-only).**
```bash
python scripts/v20_tables.py --check     # the committed tables equal what the committed aggregates generate
git show HEAD:results/metrics/v2.0/marisma_evaluation.json | sha256sum
```
The report's committed (LF) content has SHA-256 `7544d6674f116ecfeeda6de7924b8d10d35f3be00edc1917306c0b803f269204`.
On the Windows machine that ran the evaluation, the working copy holds the same content with CRLF line endings, so
hash the committed blob as shown, or LF-normalised bytes, rather than that file. The pre-registration
test (1.5) also checks the Version 2.0 plan's amendments and its execution record, each pinned as a separate segment.

**Tier 2.** `tests/test_v20_*.py`, `tests/test_bruker.py`, `tests/test_marisma_*.py`, `tests/test_zip_index.py` and
`tests/test_primary_endpoint.py` use synthetic files only. `tests/test_v20_tables.py` also reads the committed
aggregates.

**Tier 3 (documentation only; never run).**

| Command | Ran from | Hazards | Notes |
|---|---|---|---|
| `python scripts/download_marisma.py` | step 1 | D | about 16.8 GB; verifies the published checksums |
| `python scripts/v20_prepare_marisma.py` (its stages) | `47c0e92` (run 2) | O; reads metadata and the restricted counts | writes the work folder and aggregate outputs; run 1 is kept as a historical record |
| `python scripts/v20_score_marisma.py --production` | `74daef3` | L, A, O | **the one-time evaluation: spent.** It refuses to start while its state exists; never delete that state. `--rebuild-report` is only for a report-writing repair |
