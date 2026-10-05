# Research demo: the frozen ciprofloxacin model

A local dashboard for showing what this research prototype does. It shows:
- three synthetic MALDI-TOF spectra;
- the frozen model scoring the selected spectrum on this computer;
- beside it, the published Version 2.0 evaluation on MARISMa, with its limits.

It follows [docs/research_demo_brief.md](../docs/research_demo_brief.md). It is not a clinical tool: no real spectra,
no labels, no antibiotic recommendation.

![The dashboard after a run](../docs/demo/screenshots/03_result_synthetic-1.png)

**Quick start**, once the environment and the frozen model are in place (sections 1–3):

```powershell
.\.venv\Scripts\python.exe -m demo        # then open http://127.0.0.1:8050/ ; Ctrl+C stops it
```

## 1. Requirements

- **Operating system.** Windows 10 or 11 with PowerShell is the tested setup: the browser-level verification ran on
  Windows 11. CI also runs the demo's unit tests on Ubuntu, so the server should work on Linux, but it has not been
  checked there in a browser.
- **Python 3.12.** The browser verification used 3.12.10, the version the frozen bundle was saved with. Python 3.11
  passes the unit tests in CI.
- **Git** and a current browser on the same computer.
- **The frozen model bundle,** which a clone does not contain (section 3).

## 2. Clone and install

```powershell
git clone https://github.com/Sickwoman/antibiotic-resistance-ai.git
cd antibiotic-resistance-ai
git switch v2.0-research-demo                     # until the demo is merged into main
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Use the lock file.** The frozen bundle is a pickle. Its model card
(`models/v0.4/ecoli_ciprofloxacin/best_random.json`) records the libraries that saved it: numpy 2.5.3, pandas 3.0.5,
scikit-learn 1.9.1, LightGBM 4.7.0 and joblib 1.6.0. `requirements-lock.txt` pins exactly those, and the browser
verification ran on it.

**The alternative.** `requirements.txt` holds version ranges, and the main README's setup uses it: CPU torch first,
then the ranges. CI installs it fresh on every push and passes the demo's unit tests with it. Loading the frozen
bundle under other library versions has not been verified, and may warn or fail.

The demo adds no dependency of its own: it uses FastAPI and uvicorn, which the project already installs.

## 3. The frozen model

**What is needed, and where:**

| File | Location | In the repository? |
|---|---|---|
| The bundle | `models/v0.4/ecoli_ciprofloxacin/best_random.joblib` (636,912 bytes) | **no**: `models/` is not in Git |
| Its checksum sidecar | `models/v0.4/ecoli_ciprofloxacin/best_random.joblib.sha256` | no |
| The confidence zones the CLI reads | `results/metrics/v0.6/ecoli_ciprofloxacin/uncertainty.json` | yes |

The bundle's SHA-256 is `d59d6d7deafa1af464c33ebefc0f8f641a80eb70f7b51039e6c2ede0fd841c8b`. The same value is published
in [docs/reproduction_guide.md](../docs/reproduction_guide.md), section 1.3, and in the demo brief. It is model
`v0.4.0-tuned_lightgbm-random-seed42`, saved on 2026-09-18 from commit `3c8245b`.

**How an authorised user obtains it.**
- **The package.** The bundle travels as a model package, `ecoli_ciprofloxacin-v0.4.0-tuned_lightgbm-random-seed42.zip`
  (650,235 bytes). It holds the bundle, its checksum, a manifest, a model card and a notice.
- **Not published yet.** No release, artifact store or download script holds it, until the owner decides on the terms
  and publication ([docs/release/model-v0.4.0/PUBLICATION.md](../docs/release/model-v0.4.0/PUBLICATION.md)). Until
  then, ask the project owner for the package through a channel you both trust. The code is MIT-licensed and DRIAMS
  is CC0, but this repository grants nothing about the model itself: its proposed terms are in the package's
  `NOTICE.md`.
- **Installation**, with the environment of section 2:

  ```powershell
  .\.venv\Scripts\python.exe scripts\install_model.py <path to the package .zip>
  ```

  It checks the bundle against the SHA-256 pinned in this repository before writing anything, and never deserialises
  it.
  - It refuses a missing, damaged or different package with exit code 2. A package whose own checksum file and
    manifest were changed to match a different bundle is refused too, by the pin.
  - It never replaces a different model already at the target (exit code 3).
  - Run again on an installed model, it changes nothing.
- **A loose bundle,** without the package, needs the hash checked by hand before first use:

  ```powershell
  (Get-FileHash models\v0.4\ecoli_ciprofloxacin\best_random.joblib -Algorithm SHA256).Hash.ToLower()
  # must print d59d6d7deafa1af464c33ebefc0f8f641a80eb70f7b51039e6c2ede0fd841c8b
  ```

**Why the hash check matters, and what it does not prove.**
- A joblib file is a pickle: loading it runs code stored inside it. Load only a bundle from a trusted source.
- The `.sha256` sidecar sits next to the bundle, and whoever can replace one can replace the other. It catches
  accidental damage, not substitution. A matching sidecar does not establish trust, and the CLI even accepts a
  missing one.
- Trust comes from two things together: who gave you the file, and its SHA-256 matching the value published in this
  repository's history.

**Retraining is not a substitute.** Refitting from DRIAMS would produce a different artifact unless its SHA-256
equals the one above. It is a historical tier-3 command (reproduction guide, section 3) and outside the demo's scope.

**Without the bundle,** the server still starts, and the page and the evaluation panel work.
- The start-up message and the page's status line say the model is missing.
- Every run ends in "The frozen model is not available", with these instructions (see
  [the missing-model screenshot](../docs/demo/screenshots/05_error_missing_model.png)).
- **This is the packaging limitation:** a fresh clone cannot make predictions until the owner supplies the bundle.

## 4. Launch, stop, troubleshoot

```powershell
.\.venv\Scripts\python.exe -m demo                 # serves http://127.0.0.1:8050/
.\.venv\Scripts\python.exe -m demo --open          # the same, and opens the browser
.\.venv\Scripts\python.exe -m demo --port 8051     # if 8050 is taken
```

**Stopping it.** Press Ctrl+C in that window. The server binds to 127.0.0.1 and refuses any other host. The address
works only while this command is running.

**Other options:**
- `--timeout 180` sets the seconds allowed per prediction.
- `--model PATH` passes another bundle to the CLI, for showing the error states.

| What you see | Why | What to do |
|---|---|---|
| The browser cannot reach 127.0.0.1:8050 | the server is not running, or uses another port | start it (above) and check the port it prints |
| "MISSING" at start, "The frozen model is not available" on the page | no bundle at the expected path | section 3 |
| "The frozen model artifacts could not be used … does not match its checksum" | the bundle is damaged, or not the frozen one | obtain it again; never edit the sidecar to match |
| An error that the address is already in use | another process holds the port | `--port 8051`, or find it: `Get-NetTCPConnection -LocalPort 8050` |
| "The demo server did not answer" on the page | the server stopped | start it again and reload the page |
| "The prediction took too long" | a slow first start | `--timeout 300` |
| Unpickling errors or `InconsistentVersionWarning` | library versions differ from the lock | reinstall with `requirements-lock.txt` |

To check the server from a terminal, run `Invoke-RestMethod http://127.0.0.1:8050/api/status`. It reports whether the
bundle is present, without loading it.

## 5. A three-minute walkthrough

1. **Introduce the project** with the "What this project is" card:
   - a LightGBM model trained on 2,977 DRIAMS-A spectra, then frozen;
   - evaluated once on MARISMa under a pre-registered protocol.
2. **Pick "Synthetic spectrum 1".**
   - The plot shows the raw spectrum, with the 2,000–20,000 Da band the preprocessing keeps.
   - Say what it is: generated with seed 42, from no organism, with no ground-truth label.
3. **Press "Run research prediction".**
   - The spinner shows the real work: a fresh process runs `scripts/predict_spectrum.py`, unchanged, which loads and
     checksums the frozen bundle, preprocesses the spectrum and scores it. That took 2.4–2.8 s per run in the
     verification.
   - The card then shows the model score, the frozen threshold (0.1426) and whether the score is above or below it.
   - Read out the two cautions on the card:
     - a synthetic score describes no organism;
     - MARISMa's under-prediction is a finding about 1,145 real isolates, not a correction to this score.
4. **Switch to spectra 2 and 3** to show that each is scored the same way. All three happen to score above the
   threshold. They were fixed before any of them was scored, and that is not evidence about anything.
5. **Walk through the evaluation panel, top to bottom.**
   - **The registered sentence:** AUROC 0.772 (descriptive 95% interval 0.744–0.798), above-chance ranking under the
     isolate-independence assumption.
   - **Then, with equal weight, what fails:**
     - sensitivity 0.897 and specificity 0.393 at the frozen threshold;
     - under-prediction (calibration intercept 0.642);
     - the confidence zone's NPV of 0.907, with 18 R/I isolates among its 193 members, below the research target;
     - ceftriaxone unavailable.
   - **Close with the limits:** no patient linkage, clinical usefulness unproven, screening status unknown,
     different acquisition, and one hospital, one year and one instrument.

## 6. How it works

```
browser ──HTTP on 127.0.0.1──► demo server (demo/app.py, FastAPI)
                                 ├─ GET /api/examples   three synthetic spectra (demo/synthetic.py, seed 42)
                                 ├─ GET /api/evidence   committed aggregates (demo/evidence.py)
                                 └─ POST /api/predict   {"example": id} ─► a temporary file ─► subprocess:
                                                          python scripts/predict_spectrum.py <file>   (unchanged)
                                                          ◄─ its JSON payload, from src.predict.prediction_payload
```

- **Why the CLI, not the API.** It is the simplest path that runs the real model unchanged. There is one server and
  no second port. The CLI prints exactly the payload the API returns, because both build it with
  `src.predict.prediction_payload`.
- **What the page receives.**
  - The server forwards an allow-list: score, threshold, above or below, model version, timings and the disclaimer.
  - The CLI's "Resistant" or "Susceptible" decision, and its confidence-zone label, never reach the page. The demo
    shows "above" or "below the research threshold" instead, from the CLI's own decision.
  - The payload rounds the score and threshold to four decimals. The page also names the exact frozen threshold.
- **Storage, by design.**
  - The selected synthetic spectrum is written to a temporary folder (`amr-demo-*`) for one CLI run, and deleted when
    it returns.
  - Responses are marked `Cache-Control: no-store`. The page uses no browser storage, and the server keeps no
    history.
  - The CLI writes no file and appends to no log; the project logger writes to the console only.
- **The temporary-file limitation.** A server killed in the middle of a run can leave one such folder behind, holding
  only a synthetic spectrum. The next start removes folders older than 15 minutes.
- **Network, by design.**
  - A Content-Security-Policy restricts the page to its own origin.
  - The server binds to 127.0.0.1, answers only loopback host names, and has no upload route.
  - It serves this computer only: there is no authentication, and it must never be exposed to a network.

## 7. Evidence

**Two kinds of check, run in different places:**
- **CI** (GitHub Actions; Ubuntu and Windows; Python 3.11 and 3.12) runs `tests/test_demo_synthetic.py` and
  `tests/test_demo_app.py` on every push.
  - It does so without the model bundle, so the tests that need it skip, and without a browser.
  - It shows that a fresh install from `requirements.txt` imports, serves and refuses what it should.
- **The end-to-end verification,** `python scripts/demo_verify.py`, ran on one Windows 11 machine with the bundle
  present.
  - It used the real server, the unchanged CLI and API, and Microsoft Edge driven over the DevTools protocol.
  - It recorded sockets and file writes in every Python process with an audit hook, and the browser's connections
    with its net log.
  - Report: [docs/demo/verification.md](../docs/demo/verification.md), and the machine-readable
    [verification.json](../docs/demo/verification.json).

**Screenshots** from that run:
- [ready](../docs/demo/screenshots/01_ready_synthetic-1.png)
- [loading](../docs/demo/screenshots/02_loading_synthetic-2.png)
- [result, spectrum 1](../docs/demo/screenshots/03_result_synthetic-1.png)
- [result, spectrum 3](../docs/demo/screenshots/04_result_synthetic-3.png)
- [missing-model error](../docs/demo/screenshots/05_error_missing_model.png)

**What that run observed.** These are observations from one environment, not guarantees for other machines,
browsers or configurations:
- the displayed scores equaled the frozen model's in-process output and the unchanged API's;
- every socket of the server, the CLI runs and the API was on loopback;
- every request of the page went to 127.0.0.1;
- the browser made no TCP connection beyond 127.0.0.1 and sent no UDP data. Edge's own background services were
  blocked from resolving names during that run;
- every file written was a temporary spectrum that no longer existed afterwards;
- models, results, data and both experiment logs were unchanged.

## 8. Limitations

- **Synthetic inputs only.** The scores demonstrate the pipeline and say nothing about any organism. The examples
  come from a simple generator, so they are probably unlike real spectra in ways that matter to the model, and their
  scores should not be read even as typical.
- **The bundle is not distributed** (section 3). A fresh clone cannot make predictions without the owner's copy.
- **Each prediction starts a fresh process,** so a run takes a few seconds. That is the cost of using the unchanged
  CLI.
- **A temporary spectrum can survive a killed server** until the next start (section 6).
- **Localhost only,** with no authentication.
- **The browser-level verification needs Microsoft Edge on Windows,** and has been run on one machine. The unit tests
  run everywhere CI runs.
