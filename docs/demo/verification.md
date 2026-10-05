# Research demo: end-to-end verification

Run 2026-10-05T10:25:36Z by `scripts/demo_verify.py` on commit `7fba910`, on Windows-11-10.0.26200-SP0, with Python 3.12.10 and Edg/154.0.4258.53. **36 of 36 checks passed.**

**Scope.** These are observations from this one run in this environment: one Windows machine, the pinned dependencies of `requirements-lock.txt`, the frozen bundle present, and the browser and commit named above. They are not guarantees for other machines, browsers, versions or configurations. CI does not run this check. CI runs the unit tests on Ubuntu and Windows with Python 3.11 and 3.12, without the model bundle (those tests skip) and without a browser.

The scores below are pipeline demonstrations on synthetic spectra. They are not research results, and no experiment log was touched.

| Synthetic example | Demo score (CLI, 4 decimals) | Frozen model, unrounded | Status | CLI run |
|---|---|---|---|---|
| synthetic-1 | 0.1918 | 0.191769 | above the research threshold | 2.3 s |
| synthetic-2 | 0.3941 | 0.394083 | above the research threshold | 2.2 s |
| synthetic-3 | 0.2333 | 0.233276 | above the research threshold | 2.1 s |

| Check | Result | Detail |
|---|---|---|
| the page is served on 127.0.0.1 with the security headers | pass |  |
| a non-loopback Host header is refused | pass |  |
| synthetic-1: the demo's score is the frozen model's own, through the unchanged CLI | pass | demo 0.1918, library 0.191769, threshold 0.142615 |
| synthetic-2: the demo's score is the frozen model's own, through the unchanged CLI | pass | demo 0.3941, library 0.394083, threshold 0.142615 |
| synthetic-3: the demo's score is the frozen model's own, through the unchanged CLI | pass | demo 0.2333, library 0.233276, threshold 0.142615 |
| synthetic-1: the unchanged API gives the same score | pass | API 0.1918 |
| synthetic-2: the unchanged API gives the same score | pass | API 0.3941 |
| synthetic-3: the unchanged API gives the same score | pass | API 0.2333 |
| synthetic-1: the browser displays the model's score and status | pass | displayed '0.1918', 'Above the research threshold' |
| synthetic-1: no label, treatment or reassurance is displayed | pass |  |
| synthetic-2: the browser displays the model's score and status | pass | displayed '0.3941', 'Above the research threshold' |
| synthetic-2: no label, treatment or reassurance is displayed | pass |  |
| synthetic-3: the browser displays the model's score and status | pass | displayed '0.2333', 'Above the research threshold' |
| synthetic-3: no label, treatment or reassurance is displayed | pass |  |
| the evaluation panel displays the committed figures | pass | missing [] |
| the page keeps nothing in browser storage | pass | [0, 0, 0, 0] |
| a missing bundle is reported as model_unavailable | pass | HTTP 503 |
| the page shows the missing-model error and no score | pass |  |
| a fresh clone has no bundle, and serves the page and the evidence anyway | pass |  |
| a fresh clone's prediction explains how to obtain and verify the bundle | pass | HTTP 503 |
| a damaged bundle is refused by its checksum | pass | model_invalid |
| a spectrum the reader refuses is an inference error | pass | inference_failed |
| a run that takes too long is stopped | pass | timeout |
| every socket of the server, the CLI runs and the API is a loopback socket | pass | 12 socket events in 15 Python processes; outside loopback: [] |
| no Python process looked up a non-loopback host name | pass | looked up [] |
| every request the page made went to 127.0.0.1 | pass | ['127.0.0.1'] |
| every TCP connection of the browser went to 127.0.0.1 | pass | ['127.0.0.1:50846', '127.0.0.1:59193'] |
| the browser sent no UDP data | pass | UDP connects without data (Chromium's IPv6 reachability probe): ['127.0.0.1:443', '[2603:1020:201:10::10f]:443'] |
| every file the demo's processes opened for writing was a temporary file | pass | 31 temporary files; other: [] |
| no temporary spectrum survived its request | pass | left: [] |
| no demo temporary folder is left behind | pass | [] |
| no browser profile stored a prediction (Cache-Control: no-store) | pass | [] |
| the server's console output holds no prediction | pass |  |
| models, results, data and both logs are unchanged | pass | changed: [] |
| the production log has 115 data rows and the development log 34 | pass | {'results/experiments/test_evaluations.csv': 115, 'results/experiments/development_runs.csv': 34} |
| the repository gained no file outside docs/demo/ | pass |  |

**The browser itself.** Edge's own background services tried to reach edge.microsoft.com, mss.office.com, prod.rewardsplatform.microsoft.com, substrate.office.com, www.bing.com. They were not requests of the page. Name resolution was blocked for every host except 127.0.0.1, so none connected: every TCP connection the browser made went to 127.0.0.1. The UDP connects (127.0.0.1:443, [2603:1020:201:10::10f]:443) carried no data: the net log, with socket bytes, records no UDP send. That pattern is Chromium's IPv6 reachability probe.
