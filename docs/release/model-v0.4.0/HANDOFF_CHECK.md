# Clean-environment handoff check

Run 2026-10-05T11:43:43Z by `scripts/handoff_check.py`, against clone commit `422c0c7`, on Windows-11-10.0.26200-SP0, in 7.6 minutes. **20 of 20 checks passed.**

**Scope.** A fresh clone from GitHub, with a new virtual environment built from this machine's base Python and `requirements-lock.txt`, and the model installed from its package with the documented command. The run had no access to the original checkout's `models/`: every file the clean environment opened was audited. This is a clean-environment test on the same Windows machine; no other operating system was tested.

**Tolerance.** The demo's score (four decimals, as the CLI reports it) must equal the saved evidence. The unrounded probability must lie within 1e-06 of the saved value, which has six decimals.

| Check | Result | Detail |
|---|---|---|
| a fresh clone of the pushed branch, without the model or data | pass | v2.0-research-demo at 422c0c7; models/ and data/ hold only their .gitkeep placeholders |
| a clean virtual environment installed from requirements-lock.txt | pass | Python ['3.12.10'], scikit-learn, LightGBM, numpy, joblib ['1.9.1', '4.7.0', '2.5.3', '1.6.0']; 381 s |
| a missing package is refused | pass | Not installed: No package at C:\Users\Moksh\AppData\Local\Temp\handoff-lfv5n2bz\absent.zip. |
| a damaged package is refused | pass | Not installed: damaged.zip is damaged or not a ZIP package (Bad CRC-32 for file 'best_random.joblib'). |
| a self-consistent forgery is refused by the pin | pass | Not installed: Refused: the bundle's SHA-256 is 18dbb0760c24b1f6da610f9eb4177338cdaf87a2afb9bc008a7f983372ca3ecd (636,912 bytes), not the pinned d59d6d7deafa1af |
| nothing was installed by the refused packages | pass |  |
| a different existing model is never replaced | pass | Not installed: A different file is already at C:\Users\Moksh\AppData\Local\Temp\handoff-lfv5n2bz\other\best_random.joblib; it was left untouched. Move it aside  |
| before installation, the demo reports the missing model and how to obtain it | pass |  |
| the documented command installs the verified bundle | pass | Installed the frozen model at C:\Users\Moksh\AppData\Local\Temp\handoff-lfv5n2bz\clone\models\v0.4\ecoli_ciprofloxacin\best_random.joblib (SHA-256 d59d6d7deafa1 |
| installing again changes nothing | pass |  |
| synthetic-1: the clean environment's demo score equals the saved evidence | pass | 0.1918 against 0.1918 |
| synthetic-2: the clean environment's demo score equals the saved evidence | pass | 0.3941 against 0.3941 |
| synthetic-3: the clean environment's demo score equals the saved evidence | pass | 0.2333 against 0.2333 |
| synthetic-1: the unrounded score matches within 1e-06 | pass | 0.191768627 against 0.191769; difference 3.7e-07 |
| synthetic-2: the unrounded score matches within 1e-06 | pass | 0.394083140 against 0.394083; difference 1.4e-07 |
| synthetic-3: the unrounded score matches within 1e-06 | pass | 0.233275778 against 0.233276; difference 2.2e-07 |
| the clean environment opened nothing inside this checkout (models/ included) | pass | 11289 opens recorded; inside: [] |
| the clone's end-to-end browser check passes in the clean environment | pass | 36 of 36 checks passed. |
| this checkout's models, results, data and both logs are unchanged | pass |  |
| the clone's logs are unchanged (115 and 34 rows) | pass | {'results/experiments/test_evaluations.csv': 115, 'results/experiments/development_runs.csv': 34} |
