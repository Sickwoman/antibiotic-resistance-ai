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
- DRIAMS data lives outside the repo at `C:\DRIAMS` (`config.yaml` → `paths.driams_root`).
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
- Models are evaluated as described in `docs/evaluation_protocol.md` (test parts used once per final
  model; every test evaluation logged). Change it only through a dated amendment, never because of test
  results.
