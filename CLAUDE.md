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
  D (`id`, `binned_6000`, raw E. coli spectra), and C (downloaded from Dryad in a browser for Version 0.8).
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
- DRIAMS-C **was reserved for Version 0.8 and has now been spent** (`config.yaml` → `reservation`, and the
  2026-09-24 addendum in `docs/v0.7_generalisation_plan.md`). It was reserved because Version 0.8 needed a
  site whose labels were never spent, and after Version 0.7 B and D no longer qualified; its 70 / 30
  partition and seed 42 were fixed before the data was seen. The 30 % protected part has been scored once
  and **must never be scored again**. Note that `adaptation.established.evaluation_scored` still reads
  `false`: it records the state at Phase 3 and is pinned by the published full-section hash
  `fb4c1d49…dcd7bc097`, so it is deliberately not updated. Read it as history, not as current state. A test
  still refuses any config that spends a reserved site.
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
- Version 0.8 is **recorded and immutable**: methodology approved and hashed
  (`e0ceb172…6f91cf60`, `config.yaml` → `adaptation` minus `established`) before DRIAMS-C was downloaded;
  run from commit `4d82a22`, results committed as `849cad7`, log now 89 data rows at `c395fcb3…76e0c7`.
  DRIAMS-C's 30 % protected part has been spent, once. Findings as the data supports them: local
  recalibration was **not demonstrated** to help (paired Brier −0.0035 [−0.0161, +0.0085], p 0.576) and was
  not shown to harm; refitting on A + C did improve Brier (+0.0187 [+0.0086, +0.0290]) but its locally
  refitted zone reached NPV 0.904, below target, so its verdict is **mixed**, not success. **No arm reached
  the 0.95 zone target, including both baselines.** The primary endpoint is the Brier score because AUROC is
  invariant under the monotone map recalibration applies — A1 and B1 give AUROC 0.765019 to twelve decimals.
  Never rerun it, never edit a historical row, never rewrite the log.
- Version 0.9 backend API: `.\.venv\Scripts\python.exe scripts\serve_api.py` (config section `api`,
  binds `127.0.0.1`), latency from `scripts\benchmark_api.py` → `results/metrics/v0.9`. Four endpoints:
  `/health`, `/predict`, `/batch-predict`, `/model-info`. The API is a **read-only consumer of frozen
  artifacts** (`docs/v0.9_api_plan.md`, protocol amendment 5): it may not fit, calibrate, score a dataset
  part or append to any log, and it takes no dataset name, split name or path from a client, so no protected
  split is reachable through it. Four rules to keep:
  - **Input limits are keyword parameters, never `PreprocessingConfig` fields.** That dataclass's hash is
    the feature fingerprint `347cbd6d5d956ff9` every saved bundle is checked against, so a field there
    breaks every bundle and every cache. A test pins the fingerprint.
  - **An uploaded filename never reaches a response or a log.** Raw DRIAMS files are named after their
    spectrum UUID and repeat it in their `#` comments, and the library quotes `path.name` in messages, so
    uploads are saved under a generated name (`upload.txt`).
  - **No response may carry a non-finite float or a fingerprint.** `uncertainty.json` holds a bare `NaN` and
    starlette renders with `allow_nan=False`, so numbers from saved reports go through `finite_or_none`;
    `/model-info` is an explicit allow-list checked against `FORBIDDEN_IN_RESPONSES`.
  - **Uploaded bytes are data, never code.** Nothing from a request reaches `joblib.load`; `.txt` only.
  Honest results to keep stating: there is **no high-confidence-resistant zone** (`upper` is `null`), so a
  spectrum at p = 0.945 is still reported `Uncertain`; and `/batch-predict` is **not faster per sample**
  (51.34 ms vs 40.71 ms) because preprocessing dominates, so it saves round trips, not time. An explanation
  adds ~39 ms (roughly doubling a request), and a cold first request costs ~2.2x a warm one.
- A **hardening review on 2026-09-26** (addendum in `docs/v0.9_api_plan.md`; the pre-registration above it
  is unaltered) found and fixed three defects. Keep these fixed:
  - A degraded `/health` returned an absolute path, because `detail` carried the load error verbatim. The
    privacy list covered `C:/DRIAMS` but not the project directory, and a test asserted `"not found" in
    detail`, so a **passing test was holding the leak in place**. `detail` is now one of two fixed strings
    and a `PATH_LIKE` regex checks every endpoint on success and failure. Lesson: assert on a *safe* public
    string, never on a substring of an exception message.
  - A malformed zones file was logged and ignored, silently reporting `not available` for a model that has
    zones. Unparseable now means **not ready**; absent still serves without a label. `require_model` gates
    on `ready`, not on the bundle alone: `/health` said degraded while `/predict` answered 200 until it did.
  - `/predict` accepted several files and silently scored one. It now requires exactly one (400 otherwise).
  Also: `API_VERSION` is its own constant, never `config.yaml` -> `project.version` (stale at `"0.7.0"`
  through all of Version 0.8); and the dependency is plain `uvicorn`, not `uvicorn[standard]`, so there is
  no `--reload`. Deliberately **not** changed, recorded in the addendum: `/health` still carries liveness
  and readiness together, the error `type` is the exception class name rather than a public code, and the
  module layout stays flat.
- **0.9.2 cleared the last four follow-ups** (#16-#19; addendum in `docs/v0.9_api_plan.md`). Rules to keep:
  - **Zone validation lives in `src/uncertainty.py::Zones.__post_init__`**, not in the API. An edge must be
    None or a finite probability in [0, 1]; the interval is **closed**, because a `(0.0, 1.0)` pair is a
    legitimate fixture. `check_zone_bounds` is gone - once construction validates, a post-load check cannot
    fire, and leaving one would read like a defence that does nothing.
  - **The serving code is the `src/api/` package**: `app.py` (transport only), `inference_service.py`
    (loading, readiness, prediction path), `metadata.py`, `errors.py`, `schemas.py`. `predict_spectrum_file`
    and `TemporaryDirectory` must stay imported in `inference_service.py`, and tests must patch
    **`src.api.inference_service`**, never `src.api`: patching a re-exported alias binds a name nothing
    reads, so the test passes while testing nothing. Same trap as the old `"not found" in detail` assertion.
  - **Bundle digests are verified.** `scripts/write_bundle_checksums.py --write` creates the `.sha256`
    sidecars (they live in gitignored `models/`, so CI never sees them and those tests skip). A *missing*
    sidecar is still accepted - the check is an extra, not a gate - so `/ready` reports `digest_verified`
    rather than pretending the check ran.
  - **`API_VERSION` moves when the contract moves**, and is never sourced from `config.yaml`. It is `0.9.2`
    because 0.9.1 split `/health` and added error `code`, and 0.9.2 added `digest_verified`. The rule the
    test protects is about sourcing, not about the number differing from `project.version`.
  - The suite runs on **httpx2** (starlette 1.7 asks for it); `httpx` stays only for
    `scripts/benchmark_api.py`. The mutation audit now covers **12** defects across three files.
  - The V0.8 plan's "Not executed" header and `adaptation.established.evaluation_scored: false` are both
    **hash-pinned history** - never edit them; a dated addendum records the real state.
- **Version 1.0 is the result page, the consolidated README and deployment documentation**
  (`docs/v1.0_plan.md` pre-registered at `d1718cc`, plus amendment 7). Rules to keep:
  - The page (`src/api/page.html`, served by `src/api/page.py` at `GET /`) **makes no scientific decision**.
    Its script contains no confidence-label text and no comparison against `resistance_probability` or
    `threshold`, so it cannot infer a label. That matters because the zones have `upper: null` - a spectrum at
    p = 0.9453 is correctly `Uncertain`, and a page that styled it as a confident call would be overruling the
    model. Do not add a branch there.
  - The result region is **empty in the markup** and filled only from a response; a failed request clears it
    before the fetch. Explanatory copy may name a label, the result region may not (amendment 7 point 3).
  - `page.py` substitutes **only project constants** (`DISCLAIMER`, `ADVICE`) into the file, never anything
    from a request - that is the one way a static page could become an injection vector.
  - Same-origin and stateless: no external script/style/font/analytics, no `localStorage`/`sessionStorage`/
    `indexedDB`/cookie, no prediction history. Tests assert on **attribute values**, not on the characters
    `//`, which every JS comment contains.
  - `GET /` and `/health` are the **two exemptions** from the degraded-route invariant, asserted positively
    rather than skipped: an unready service should still serve the page that says so.
  - The suite **does not execute the page's JavaScript** (CI installs Python only). The invariants are
    structural and asserted on the served HTML; `node --check` on the extracted script is a local-only extra.
  - `API_VERSION` is `1.0.0` because `GET /` is a contract change. It is still never sourced from config.
  - **Issue #22, fixed**: `scripts/generalisation_tables.py` grouped the seed-variation table by split and
    site only, so the saved-model row (same split, same site, seed 42 like the first refit seed) joined the
    five refit seeds; `nunique()` still read 5 while the mean and range covered six rows. It now groups by
    model too and refuses a group where a seed appears twice. The committed V0.7 `tables.md` is **preserved**
    with the wrong B/D cells (0.816 / 0.715), and `tables_ERRATUM.md` beside it gives the correct ones (0.817 /
    0.717). Never regenerate a committed version's outputs to fix a reporting bug - add an erratum.
- **Version 1.0 is closed** (squash-merged as `69bfd23`, PR #23; release candidate `5afb6e2`). It is the last
  version on the roadmap: anything further is new scope and needs its own pre-registration.
- **Pruning a merged branch does not lose its commits.** GitHub keeps `refs/pull/N/head` for every pull request,
  holding its full history. An earlier record claimed the branches were the only refs holding the audited
  commits; that was wrong and is corrected by addendum in `docs/v0.9_api_plan.md`. The written SHA tables are
  still kept, because a default clone does not fetch `refs/pull/*`.
- **Pre-registrations are append-only, and `tests/test_preregistration.py` enforces it.** Each plan's locked
  text is pinned by length + SHA-256 (not by `git show`, because CI checks out shallowly). A sweep on
  2026-09-29 found three in-place edits: one by me in the 0.9.2 docs pass (V0.9 plan line 26, now restored)
  and two older ones in the protocol (disclosed, not reverted, because they corrected wrong text). When a new
  version's plan is committed, **add it to `LOCKED`** - `test_every_plan_in_docs_is_locked` fails until you do.
- **0.9.1 closed two of those deviations** (issues #14 and #15; addendum in `docs/v0.9_api_plan.md`).
  `/health` is now **liveness only** (always 200 while the process serves, body exactly `{"status": "ok"}`)
  and `/ready` carries `model_loaded`, `zones_loaded`, `model_version` and the fixed public reason, 200 or
  503. The old shape made a missing model look like a dead process, which a liveness probe answers by
  restarting - a restart loop instead of draining traffic. Errors now carry `code` from a closed six-value
  vocabulary (`invalid_request`, `invalid_spectrum`, `payload_too_large`, `unsupported_media_type`,
  `service_not_ready`, `internal_error`), all from one ordered table `ERROR_MAP`; `type` is retained one
  release and deprecated. Two rules to keep: `ERROR_MAP` must stay ordered most specific first, because
  `DataError` is the base of `SpectrumFormatError`, and the emitted code set must equal `ERROR_CODES` - both
  asserted by tests. Consumers of the contract include `scripts/benchmark_api.py`, which read the model
  version from `/health` and had to move to `/ready`: check the scripts when the contract changes.
  Still open: `/model-info` naming and breadth (safe), flat layout (#17), zone validation in the serving
  layer rather than the domain model (#16), no bundle sidecars (#18), starlette deprecation (#19).
- **Version 1.1 is a second antibiotic, E. coli + ceftriaxone** (`docs/v1.1_ceftriaxone_plan.md`, amendment 8).
  Its test parts were **scored once on 2026-09-30 and are spent**: log 113 rows at `6528eb2a…f3ddba`. Never
  re-score them and never regenerate its results. Findings as the data supports them: better than chance
  (AUROC 0.713, 0.604–0.818) but **not a usable decision rule** (sensitivity 0.823 against 0.90, specificity
  0.422, precision 0.127; Brier 0.083 against 0.084 for the base rate); T against F, the two antibiotics and
  every generalisation gap **not demonstrated**, which is never "equivalent"; DRIAMS-D the weakest (0.651).
  The published 0.74 is a reference point, never "matched" or "replicated" - and its attribution to ceftriaxone
  is **unverified** (the Weis et al. abstract gives 0.74 for E. coli without the antibiotic; full text not
  checked). No ceftriaxone model is served.
  - `ecoli_ceftriaxone` comes from `build_dataset.py --antibiotic Ceftriaxone --report-version v1.1`: the
    primary's preprocessing, splits **derived** from the primary (93 ceftriaxone-only isolates are in no
    split), and every shared spectrum must be byte-identical to the primary's row or the build stops.
  - **Every log lookup must include the dataset.** Two antibiotics now share split, model and seed names. The
    V0.7 lookups and the strict pre-write gate were dataset-blind and were fixed; the gate's old key would have
    refused 23 of the 24 V1.1 rows as a second scoring. The key is (dataset, experiment, model, seed).
  - V1.1 runs through config sections of the existing scripts (`second_antibiotic`,
    `second_antibiotic_generalisation`): a section may have no earlier model, a family may name its own
    `seeds`, `primary_family` is saved whatever validation says, and `append_gate: strict` runs the seven
    pre-write checks and writes `pre_write_checks.json` before the log is touched (and refuses
    `--allow-rescore`). V0.4, V0.5 and V0.7 behave exactly as before.
  - `project.version` stamps saved models; the API never reads it. Bump it when a version starts.
  - Provenance kept on record: the first build was stamped `7116e12-dirty` and rebuilt clean,
    byte-identically; the first model was stamped `v1.0.0` and rerun from cache with identical validation.
  - A cut-off chosen on one small validation part has missed its sensitivity target on test twice (0.827
    ciprofloxacin, 0.823 ceftriaxone). Do not describe the 0.90 target as achieved on new patients.
- **Version 1.2 is a development-only study** (`docs/v1.2_calibration_plan.md`, amendment 9; branch
  `v1.2-calibration`). No test part was read, the production log is untouched (113 rows), and **every V1.2
  result is exploratory** - no untouched evaluation data exist. It runs on a derived pool: `ecoli_ceftriaxone`'s
  `random` training + validation rows minus every row, and every patient group, of an inspected test part
  (2,421 spectra, 247 resistant). `python scripts/v12_development.py`; results go to the separate development
  log `results/experiments/development_runs.csv`, which `DevelopmentLog` refuses to point at the production log.
  - Findings, as the data supports them: primary (Brier, B minus C) **not demonstrated**, +0.0002 [-0.0024,
    +0.0028]; C's cut-off far steadier across folds (sensitivity SD 0.058 against 0.151) at lower specificity
    (0.370 against 0.446), reaching 0.90 in 7 of 15 folds; **in the one forward split no arm held its target**
    (0.56-0.79). Resistance was higher in 2017 (11.4 % against 8.3 %) and every arm under-predicted on average
    (intercepts +0.67 to +0.81), but **never write that this explains the loss**: a prevalence change alone
    leaves sensitivity at a fixed cut-off unchanged, and label shift was not established.
  - Brier(B) minus Brier(D1) = +0.0020 is the effect of fitting on 8/8 instead of 7/8 of a fold (no cut-off
    enters Brier), not of the cross-fitted cut-off. Name both operands of every difference.
  - `results/metrics/v1.2/ecoli_ceftriaxone/tables_complete.md` (from `scripts/v12_report.py`, saved outputs
    only) is the complete V1.2 report; the run's own `tables.md` omits three secondaries and is kept as written.
  - DRIAMS-C's standing, by antibiotic, is in `docs/driams_c_status.md`: ciprofloxacin spent; ceftriaxone
    **uncertain** (labels never used by a model, but their aggregate counts were seen on 2026-09-30, the
    spectra were used for ciprofloxacin, and there are no patient IDs). Never open C labels without an
    owner-approved amendment.
  - A cut-off chosen on fewer than 50 resistant spectra is **unsupported**: report it, never describe it as
    targeting 0.90. The Version 1.1 procedure's slice holds about 25.
  - **A `CalibratedClassifierCV` calibrator works on the pipeline's `decision_function` (LightGBM's raw margin),
    not on `predict_proba`.** Map anything through it on that scale; `cross_fitted_probabilities` proves its
    predictions reproduce the calibrator, and `assert_same_scale` that they share the model's scale.
  - Under the plan's section 12, C does not qualify for a held-out evaluation (its primary was not met).
    DRIAMS-C's ceftriaxone labels are the only unused candidate; opening them needs an owner-approved amendment.
- **Version 1.3 is a development-only study of an uncertainty-aware cut-off** (`docs/v1.3_threshold_plan.md`,
  amendment 10; branch `v1.3-threshold`; `python scripts/v13_threshold.py`). Exploratory: its 2017 later periods
  were already evaluated in aggregate by V1.2. Result: rule U reached 0.929 pooled sensitivity against E's 0.814,
  but **only by flagging 85 % of isolates** (specificity 0.160; precision 0.118 against a 0.107 base rate), so
  under the pre-set verdict order it is "unhelpful". E missed 0.90 in both periods. No operational specificity
  floor exists; 0.20 only names "flags nearly everyone". Recent intercept recalibration: not demonstrated.
  - Rule U (`src/threshold_rules.py`): `k* = max{k : P(Binomial(n, 0.10) >= k) >= 0.95}`, the `k*`-th lowest
    resistant score; **infeasible below 29 resistant**. Its 95 % holds only for exchangeable resistant scores from
    a scoring function that did not see them; cross-fitted selection scores make it a heuristic. A confidence
    bound computed after a free search over cut-offs is not a guarantee - never describe one as such.
  - **Count patients, not spectra:** the pool's 247 resistant spectra are 105 resistant patient groups. Select
    cut-offs on one spectrum per patient group; resample whole groups.
  - Time: only labels dated at least 7 days before an origin; later-period spectra of patients already seen
    that year are removed; patients cannot be linked across years.
  - **No frozen procedure so far (V1.2's C, E, U) warrants a DRIAMS-C evaluation.** The rules tested did not
    establish useful performance with this model. That is **not** a universal limit and not proof that
    discrimination is the constraint: never write "the limit is ...". The 0.56 / 0.44 for rule E are
    theoretical (exchangeable scores, a scoring function independent of them), not observed attainment rates; the
    derivation, order-statistic convention and tie handling are in the plan's closure record. Every operating number
    (0.90, 0.95, 0.20, the 7-day gap, 30 / 15 / 50) is a research criterion, never a clinical standard.
- **Version 1.4 tested the one hypothesis the discrimination audit supported, and model iteration then stopped**
  (`docs/discrimination_audit.md`; `docs/v1.4_screening_plan.md`, amendment 11; branch `v1.4-screening`). The audit
  found no defect behind the modest discrimination; the verified constraint was 105 resistant patients in the pool.
  Adding 495 excluded screening (HospitalHygiene) isolates to training only: AUROC on clinical isolates A1 minus A0
  **-0.007 [-0.056, +0.043], not demonstrated**; resistant screening vs clinical spectra are distinguishable (AUROC
  0.937). Rules to keep:
  - **Model iteration on the development pool has stopped** (the plan's section 13). Do not try more settings,
    weights, representations or data on it; the next step is the reproducible research report, and a confirmatory
    claim needs untouched data plus an owner-approved amendment.
  - Screening isolates stay **training-only**: `--keep-workstation HospitalHygiene` builds them into their own
    dataset (`ecoli_ceftriaxone_with_screening`), never under the unchanged name; `src/screening.py` links them to
    within-year patients, drops a held-out patient's, and excludes any patient with a clinical row outside the pool
    (including the 15 unsplit ceftriaxone-only isolates of pool patients - the audit's group-based count missed one).
  - A0 must reproduce Version 1.2's arm C exactly (it did: difference 0). Name both operands: every difference here
    is A1 minus A0.
  - Consensus/semantic-scholar style links carry 32-hex ids that the privacy scan rightly flags; cite by DOI.
- **The research cycle V0.1-V1.4 is closed by `docs/research_report_v0.1-v1.4.md`** (with `evidence_map.md`,
  `reproduction_guide.md`, `research_summary.md`, `review_checklist.md`; branch `v1.4-report`). Its section 8 lists
  the reporting corrections; a pre-specified sensitivity analysis (ciprofloxacin, I excluded) was never run - say so
  wherever the label policy is discussed. Only tier-1 and tier-2 commands of the reproduction guide may be run.
- **Test automation must surface failures** (Version 1.3 closure). CI runs every step under `bash` (GitHub:
  `-eo pipefail`); never pipe pytest into `tail`/`tee` before a commit - read pytest's own exit code.
  `tests/conftest.py` fails the whole session if a test changed `results/experiments/test_evaluations.csv` or
  `development_runs.csv`: every test that runs a script must point both logs into `tmp_path`.
