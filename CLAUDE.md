# Project rules: Adaptive AI for Antibiotic Resistance Prediction

Real ML research project on MALDI-TOF spectra (DRIAMS). Follow these rules strictly:

1. Never fabricate dataset columns, filenames, statistics or model results.
2. Inspect real data before generating dataset-specific code.
3. Never fabricate accuracy, AUROC, F1, PR-AUC or other metrics.
4. Build incrementally from Version 0.1 to Version 1.0.
5. Do not proceed to the next version until the current one has been executed and verified.
6. Prevent train/test and patient/isolate leakage.
7. Fit learned preprocessing only on training data.
8. Keep raw datasets out of GitHub.
9. Use reproducible random seeds.
10. Run tests before saying something works.
11. Report failed experiments honestly.
12. Research prototype only – not a clinically validated diagnostic system.
13. Never recommend antibiotics to patients.
14. Prefer simple, scientifically justified models over unnecessary complexity.
15. Explain commands and errors in simple language.
16. Execute code and verify outputs instead of merely generating code.

## Working notes

- Windows 11, PowerShell, Python 3.12 venv in `.venv`, 7.7 GB RAM, no CUDA GPU.
- DRIAMS data lives outside the repo at `C:\DRIAMS` (`config.yaml` → `paths.driams_root`). Extracted: A, B and
  D (`id`, `binned_6000`, raw E. coli spectra); DRIAMS-C still needs a browser download from Dryad.
- Run tests: `.\.venv\Scripts\python.exe -m pytest -q`; lint: `.\.venv\Scripts\python.exe -m ruff check .`
  (both run in CI).
- Verified data facts are recorded in `config.yaml` comments and `README.md`.
- Version 0.2 dataset: `.\.venv\Scripts\python.exe scripts\build_dataset.py` writes
  `data/processed/ecoli_ciprofloxacin/` (git-ignored). Never export patient_no / case_no / order_no.
- Build the primary dataset before any variant (e.g. `--intermediate-as exclude`): variants reuse the
  primary dataset's saved splits. Load splits only with `load_split(path, meta)` / `load_splits`, which
  check the dataset fingerprint.
- Anything learned from data (scaling, PCA, feature selection) must be fitted inside model pipelines on
  training rows only; `src/preprocessing.py` stays stateless.
- Models are evaluated as described in `docs/evaluation_protocol.md` (approved 2026-09-17; test parts used
  once per final model; every test evaluation logged). Change it only through a dated amendment, never
  because of test results.
- Version 0.3 baselines: `.\.venv\Scripts\python.exe scripts\train_baselines.py` trains and validates only;
  `--evaluate-test` is for a planned, one-time scoring of the allowed test parts and appends to
  `results/experiments/test_evaluations.csv`. The `temporal` and `external` test parts stay locked until
  Version 0.7 (`evaluation.locked_test_splits`). Never pick settings from test results.
- Research prediction for one file: `.\.venv\Scripts\python.exe scripts\predict_spectrum.py <raw spectrum .txt>`.
  Saved models live in `models/` (git-ignored); only load model files made by this project.
- Version 0.4 tuning: `.\.venv\Scripts\python.exe scripts\tune_models.py` searches, calibrates and validates;
  `--evaluate-test` scores the allowed test parts once and refuses to fit anything that is not cached, so the
  scored models are the ones checked on validation. Searches and fits are cached under `models/v0.4/cache`,
  keyed by data, settings, the text of `src/tuning.py` and `src/train.py`, and library versions: editing either
  file discards hours of computation. Result tables come from `scripts/tuned_tables.py`, never typed by hand.
- Version 0.5 networks: the same script and safeguards with `--section deep` (config section `deep`,
  `models/v0.5`, `results/metrics/v0.5`), compared against the saved Version 0.4 model. `src/deep.py` joins the
  cache key for that section, so editing it discards the cached searches; the section-level `training:` block
  (epochs, batch size, patience) is part of every family's fingerprint too. Tables: `scripts/tuned_tables.py
  --section deep`. PyTorch is CPU-only here; `deep.training.threads` sets the thread count, and predictions
  always run on one thread so a later run reproduces them exactly.
- Version 0.6 explanations: `.\.venv\Scripts\python.exe scripts\explain_model.py` describes the saved model
  (config section `explain`, `results/metrics/v0.6`) and fits the confident / uncertain zones; tables come from
  `scripts\explain_tables.py`. It **never scores a test row**: contributions and importances use the validation
  part only (`explain.rows`, enforced), the test-side zone numbers are derived from the stored
  `test_probabilities.npz` after they cover that test part's rows and reproduce the logged AUROC *and Brier
  score* (AUROC alone survives any monotone rescaling), and the run compares the test log byte for byte
  before and after. Nothing in Version 0.6 may change a model, a setting or a cut-off, and no m/z
  region is ever given a protein or peptide identity (`docs/v0.6_explainability_plan.md`, protocol amendment 2).
  `scripts\predict_spectrum.py` now defaults to the Version 0.4 model and takes `--explain`.
- Version 0.7 generalisation: `.\.venv\Scripts\python.exe scripts\measure_generalisation.py` (config section
  `generalisation`, `results/metrics/v0.7`); tables from `scripts\generalisation_tables.py`. This is the
  version protocol decision 7 released the `temporal` and `external` test parts for, so
  `evaluation.locked_test_splits` is now empty — the guard stays, and putting a split back in that list still
  refuses it everywhere. It chooses **no setting**: the Version 0.4 winning LightGBM setting is read from its
  saved model card and refitted unchanged on each experiment's own training part, so a gap can only come from
  the data. Three checks run before anything is fitted: a locked split is refused, the saved model is refused
  on any part it shares rows or patient groups with (which is what keeps it off the `temporal` part, where 848
  of 1,233 rows are in its own training data), and the refit must reproduce the saved model's logged
  validation AUROC to 1e-9. The `random` result is never re-scored: its probabilities are reused from Version
  0.4 after checking rows, AUROC and Brier. New split `external_ab` (train A+B, test D) is the
  specification's "train two sites, test a third"; add a split with
  `scripts\build_dataset.py --splits-only`, which never rewrites X.npy.
- DRIAMS-C is **reserved for Version 0.8** (`config.yaml` → `reservation`, and the 2026-09-24 addendum in
  `docs/v0.7_generalisation_plan.md`): it is not built, not scored, and not in any split. Version 0.8 needs a
  site whose labels were never spent, and after Version 0.7 B and D no longer qualify. Its partition is fixed
  in advance (70 % adaptation / 30 % held-out, seed 42). A test refuses any config that spends a reserved site.
- Version 0.7 is **recorded and immutable**: run from commit `8878bfd`, results committed as `64c6c75`, and
  the append-only log now holds 84 data rows with SHA-256 `e97480ab…c6c8a048a`. Never rerun it, never edit a
  historical row, never rewrite the log. Findings, stated as the data supports them: **no generalisation gap
  was demonstrated** at any site (every interval includes 0 — which is *not* "no gap exists"; the intervals
  are 0.13–0.18 wide and cannot resolve differences that would matter). DRIAMS-B scored higher (0.815 vs
  0.751) but has 59 resistant isolates and the widest interval, so that is not evidence of better
  performance. Adding B to training showed **no demonstrated benefit** at D (paired −0.016 [−0.034, 0.001]) —
  and no demonstrated harm either. The Version 0.6 confidence zone reached its target at D (0.963, on 298
  covered spectra), was uninformative at B (0.922 on only 51 covered spectra, interval 0.836–0.982), and did
  **not** reach it in the later year (0.893 [0.844, 0.937]). That temporal result is confounded: the saved
  model shares 848 of those 1,233 rows, so a refitted, recalibrated model had to be used, and it is **not a
  clean independent saved-model test**. Do not describe it as one, and do not try to engineer around it.
- Version 0.8 has a **draft** plan at `docs/v0.8_adaptive_plan.md`. It is not approved: three
  **DECISION NEEDED** items (primary target metric, adaptation method, whether several adaptation-set sizes
  are tried) require the owner's approval, and the plan plus a protocol amendment 4 must be committed before
  DRIAMS-C is opened. Note the honest tension recorded there: Version 0.7 found no measurable gap to close,
  so a null adaptation result is the expected outcome and must be reported as prominently as a positive one.
