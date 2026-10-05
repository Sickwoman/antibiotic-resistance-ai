"""Clean-environment handoff check: a fresh clone and a fresh environment, with the model installed from its package.

    python scripts/handoff_check.py --package dist/ecoli_ciprofloxacin-v0.4.0-tuned_lightgbm-random-seed42.zip

It runs on this machine, and its report says so: it is a clean-environment test, not a test on another system.

1. **Clone.** It clones the repository from GitHub at `--ref` into a new temporary folder. The clone has no model
   and no data.
2. **Environment.** It creates a new virtual environment there with this machine's base Python, and installs
   `requirements-lock.txt` into it.
3. **Package.** It copies the package into the temporary folder, as a download would.
4. **Checks, with the clean environment's Python:**
   - a missing package, a damaged one and a self-consistent forgery (bundle, checksum file and manifest all changed
     together) are refused, and a different existing model is never replaced;
   - before installation, the demo reports the missing model;
   - the documented command installs the package, and a second run changes nothing;
   - the demo then scores the three synthetic examples, and each score must match the saved evidence
     (`docs/demo/verification.json`) within the documented tolerance: the same value at the four decimals the CLI
     reports, and within 1e-6 of the saved unrounded value;
   - the clone's own end-to-end browser check, `scripts/demo_verify.py`, runs.
5. **Audit.** It records every file the clean environment's processes open, through a Python audit hook, and fails if
   any lies inside this checkout. That includes `models/`.
6. **Integrity.** This checkout's models, results, data and both logs must be unchanged, and so must the clone's logs.

It writes `docs/release/model-v0.4.0/HANDOFF_CHECK.md` and `.json`. It publishes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import model_package as mp  # noqa: E402

REPOSITORY = "https://github.com/Sickwoman/antibiotic-resistance-ai.git"
OUT = ROOT / "docs" / "release" / "model-v0.4.0"
LOGS = ("results/experiments/test_evaluations.csv", "results/experiments/development_runs.csv")
TOLERANCE = 1e-6                                   # on the unrounded probability; the 4-decimal score must be equal
AUDIT = r'''
import os, sys
_path = os.environ.get("HANDOFF_AUDIT_LOG")
if _path:
    _out = open(_path, "a", encoding="utf-8", buffering=1)
    def _hook(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            try:
                _out.write(os.path.abspath(os.fsdecode(args[0])) + "\n")
            except Exception:
                pass
    sys.addaudithook(_hook)
'''
UNROUNDED = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from demo import synthetic
from src.predict import load_bundle, predict_features
from src.preprocessing import PreprocessingConfig, preprocess_file
bundle = load_bundle(Path("models/v0.4/ecoli_ciprofloxacin/best_random.joblib"))
out = {}
for example in synthetic.generate():
    path = Path(sys.argv[1]) / f"{example.id}.txt"
    path.write_text(example.to_text(), encoding="utf-8")
    features, _ = preprocess_file(path, PreprocessingConfig())
    out[example.id] = float(predict_features(bundle, features)[0][0])
print(json.dumps(out))
'''


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.facts: dict = {}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
        return bool(ok)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root: Path) -> dict[str, str]:
    out = {p.relative_to(root).as_posix(): sha256(p) for d in ("models", "results")
           for p in sorted((root / d).rglob("*")) if p.is_file()}
    for p in sorted((root / "data").rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = f"{p.stat().st_size}:{p.stat().st_mtime_ns}"
    return out


def run(args: list[str], cwd: Path, env: dict | None = None, timeout: float = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def http(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def demo(python: Path, clone: Path, env: dict, log: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    process = subprocess.Popen([str(python), "-m", "demo", "--port", str(port)], cwd=clone, env=env,
                               stdout=log.open("w", encoding="utf-8"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 120
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the demo exited with code {process.returncode}: {log.read_text()[-500:]}")
        try:
            http("GET", f"{base}/api/status")
            return process, base
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("the demo did not start")


def stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def forged(package: Path, out: Path) -> Path:
    """A self-consistent forgery: one bundle byte changed, and its checksum file and manifest updated to match."""
    with zipfile.ZipFile(package) as archive:
        members = {name: archive.read(name) for name in mp.MEMBERS}
    bundle = bytearray(members[mp.BUNDLE_NAME])
    bundle[len(bundle) // 2] ^= 0x01
    digest = hashlib.sha256(bytes(bundle)).hexdigest()
    manifest_json = json.loads(members["MANIFEST.json"])
    manifest_json["bundle"]["sha256"] = digest
    members.update({mp.BUNDLE_NAME: bytes(bundle), mp.SIDECAR_NAME: f"{digest}  {mp.BUNDLE_NAME}\n".encode(),
                    "MANIFEST.json": json.dumps(manifest_json).encode()})
    out.write_bytes(mp.zip_members(members))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", type=Path, default=ROOT / "dist" / f"{mp.PACKAGE_NAME}.zip")
    parser.add_argument("--ref", default=run(["git", "branch", "--show-current"], ROOT).stdout.strip())
    parser.add_argument("--repository", default=REPOSITORY)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    report = Report()
    base_python = Path(getattr(sys, "_base_executable", sys.executable))
    work = Path(tempfile.mkdtemp(prefix="handoff-"))
    clone, logs = work / "clone", work / "logs"
    logs.mkdir()
    before = manifest(ROOT)
    log_hashes = {log: sha256(ROOT / log) for log in LOGS}
    started = time.time()
    try:
        # 1. Fresh clone, from GitHub, at the pushed commit
        cloned = run(["git", "-c", "core.longpaths=true", "clone", "--quiet", "--branch", args.ref, args.repository,
                      str(clone)], work)
        commit = run(["git", "rev-parse", "HEAD"], clone).stdout.strip()
        local = run(["git", "rev-parse", "HEAD"], ROOT).stdout.strip()
        report.check("a fresh clone of the pushed branch, without the model or data", cloned.returncode == 0 and
                     commit == local and not (clone / mp.TARGET).exists() and not (clone / "data" / "processed")
                     .exists(), f"{args.ref} at {commit[:7]}")

        # 2. A clean virtual environment, from the lock file
        t0 = time.time()
        made = run([str(base_python), "-m", "venv", str(clone / ".venv")], clone)
        python = clone / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt"
                                                                                   else "python")
        pip = run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements-lock.txt",
                   "--extra-index-url", "https://download.pytorch.org/whl/cpu"], clone, timeout=5400)
        (logs / "pip.log").write_text(pip.stdout[-20000:] + pip.stderr[-20000:], encoding="utf-8")
        versions = run([str(python), "-c", "import sys, sklearn, lightgbm, numpy, joblib, fastapi; print(sys.version"
                        ".split()[0], sklearn.__version__, lightgbm.__version__, numpy.__version__, joblib.__version__"
                        ", fastapi.__version__)"], clone).stdout.split()
        report.check("a clean virtual environment installed from requirements-lock.txt", made.returncode == 0 and
                     pip.returncode == 0 and versions[:5] == ["3.12.10", "1.9.1", "4.7.0", "2.5.3", "1.6.0"],
                     f"Python {versions[:1]}, scikit-learn, LightGBM, numpy, joblib {versions[1:5]}; "
                     f"{time.time() - t0:.0f} s")
        report.facts["environment"] = {"os": platform.platform(), "base_python": str(base_python.name),
                                       "versions": versions, "pip_seconds": round(time.time() - t0)}

        # 3. The package, copied in as a download would be; the clean environment audited from here on
        package = work / args.package.name
        shutil.copyfile(args.package, package)
        audit_dir, audit_log = work / "audit", work / "opened.txt"
        audit_dir.mkdir()
        (audit_dir / "sitecustomize.py").write_text(AUDIT, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(audit_dir), "HANDOFF_AUDIT_LOG": str(audit_log),
               "PYTHONDONTWRITEBYTECODE": "1"}
        installer = [str(python), "scripts/install_model.py"]

        # 4. Refusals, before anything is installed
        damaged = work / "damaged.zip"
        data = bytearray(package.read_bytes())
        data[len(data) // 3] ^= 0x01
        damaged.write_bytes(bytes(data))
        for name, path, expected, phrase in (
                ("a missing package is refused", work / "absent.zip", 2, "No package at"),
                ("a damaged package is refused", damaged, 2, "damaged"),
                ("a self-consistent forgery is refused by the pin", forged(package, work / "forged.zip"), 2,
                 "not the pinned")):
            done = run([*installer, str(path)], clone, env)
            report.check(name, done.returncode == expected and phrase in done.stderr, done.stderr.strip()[:160])
        report.check("nothing was installed by the refused packages", not (clone / mp.TARGET).exists())
        other = work / "other" / mp.BUNDLE_NAME
        other.parent.mkdir()
        other.write_bytes(b"a different model")
        done = run([*installer, str(package), "--target", str(other)], clone, env)
        report.check("a different existing model is never replaced", done.returncode == 3 and
                     other.read_bytes() == b"a different model", done.stderr.strip()[:160])

        # 5. The demo before installation reports the missing model
        server, base = demo(python, clone, env, logs / "demo-before.log")
        status = http("GET", f"{base}/api/status")[1]
        code, body = http("POST", f"{base}/api/predict", {"example": "synthetic-1"})
        stop(server)
        report.check("before installation, the demo reports the missing model and how to obtain it",
                     status["model"]["present"] is False and code == 503 and
                     body["error"]["code"] == "model_unavailable" and "demo/README.md" in body["error"]["message"])

        # 6. The documented installation, twice
        first = run([*installer, str(package)], clone, env)
        second = run([*installer, str(package)], clone, env)
        installed = clone / mp.TARGET
        report.check("the documented command installs the verified bundle", first.returncode == 0 and
                     installed.is_file() and sha256(installed) == mp.FROZEN_BUNDLE_SHA256 and
                     installed.with_name(installed.name + ".sha256").read_text() == mp.sidecar_text(),
                     first.stdout.strip()[:160])
        report.check("installing again changes nothing", second.returncode == 0 and "nothing changed" in second.stdout)

        # 7. The demo after installation: the three synthetic examples against the saved evidence
        saved = json.loads((clone / "docs" / "demo" / "verification.json").read_text(encoding="utf-8"))
        saved = saved["synthetic_scores"]
        server, base = demo(python, clone, env, logs / "demo-after.log")
        shown = {}
        for example_id in ("synthetic-1", "synthetic-2", "synthetic-3"):
            code, body = http("POST", f"{base}/api/predict", {"example": example_id})
            shown[example_id] = body
            report.check(f"{example_id}: the clean environment's demo score equals the saved evidence",
                         code == 200 and body["score"] == saved[example_id]["demo_score"]
                         and body["above_threshold"] == saved[example_id]["above_threshold"],
                         f"{body.get('score')} against {saved[example_id]['demo_score']}")
        stop(server)
        scratch = work / "spectra"
        scratch.mkdir()
        unrounded = json.loads(run([str(python), "-c", UNROUNDED, str(scratch)], clone, env).stdout)
        for example_id, value in unrounded.items():
            difference = abs(value - saved[example_id]["library_unrounded"])
            report.check(f"{example_id}: the unrounded score matches within {TOLERANCE:g}", difference <= TOLERANCE,
                         f"{value:.9f} against {saved[example_id]['library_unrounded']}; difference "
                         f"{difference:.1e}")
        report.facts["scores"] = {k: {"demo_score": v["score"], "unrounded": round(unrounded[k], 9),
                                      "saved_demo_score": saved[k]["demo_score"],
                                      "saved_unrounded": saved[k]["library_unrounded"]} for k, v in shown.items()}

        # 8. Nothing the clean environment opened lies inside this checkout
        opened = audit_log.read_text(encoding="utf-8").splitlines() if audit_log.exists() else []
        inside = sorted({p for p in opened if os.path.normcase(p).startswith(os.path.normcase(str(ROOT)) + os.sep)})
        report.check("the clean environment opened nothing inside this checkout (models/ included)", not inside and
                     bool(opened), f"{len(opened)} opens recorded; inside: {inside[:3]}")

        # 9. The clone's own end-to-end browser check, in the clean environment
        browser = run([str(python), "scripts/demo_verify.py"], clone, {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                      timeout=1800)
        (logs / "demo_verify.log").write_text(browser.stdout + browser.stderr, encoding="utf-8")
        summary = [line for line in browser.stdout.splitlines() if "checks passed" in line]
        report.check("the clone's end-to-end browser check passes in the clean environment",
                     browser.returncode == 0, summary[-1] if summary else browser.stderr[-200:])

        # 10. Integrity: this checkout, and the clone's logs
        report.check("this checkout's models, results, data and both logs are unchanged",
                     manifest(ROOT) == before and all(sha256(ROOT / log) == h for log, h in log_hashes.items()))
        clone_logs = run(["git", "diff", "--quiet", "--", *LOGS], clone)
        rows = {log: sum(1 for _ in (clone / log).open(encoding="utf-8")) - 1 for log in LOGS}
        report.check("the clone's logs are unchanged (115 and 34 rows)", clone_logs.returncode == 0 and
                     rows == {LOGS[0]: 115, LOGS[1]: 34}, f"{rows}")
    finally:
        report.facts["minutes"] = round((time.time() - started) / 60, 1)
        report.facts["clone_commit"] = locals().get("commit", "")
        write(report)
        shutil.rmtree(work, ignore_errors=True)
    passed = all(c["ok"] for c in report.checks)
    print(f"\n{sum(c['ok'] for c in report.checks)} of {len(report.checks)} checks passed.")
    return 0 if passed else 1


def write(report: Report) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    env = report.facts.get("environment", {})
    data = {"what": "clean-environment handoff check (scripts/handoff_check.py)", "run_utc": stamp,
            "passed": all(c["ok"] for c in report.checks), **report.facts, "checks": report.checks}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "HANDOFF_CHECK.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = ["# Clean-environment handoff check", "",
             f"Run {stamp} by `scripts/handoff_check.py`, against clone commit "
             f"`{str(report.facts.get('clone_commit', ''))[:7]}`, on {env.get('os', 'unknown')}, in "
             f"{report.facts.get('minutes')} minutes. **{sum(c['ok'] for c in report.checks)} of "
             f"{len(report.checks)} checks passed.**", "",
             "**Scope.** A fresh clone from GitHub, with a new virtual environment built from this machine's base "
             "Python and `requirements-lock.txt`, and the model installed from its package with the documented "
             "command. The run had no access to the original checkout's `models/`: every file the clean environment "
             "opened was audited. This is a clean-environment test on the same Windows machine; no other operating "
             "system was tested.", "",
             f"**Tolerance.** The demo's score (four decimals, as the CLI reports it) must equal the saved evidence. "
             f"The unrounded probability must lie within {TOLERANCE:g} of the saved value, which has six decimals.", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c['check']} | {'pass' if c['ok'] else '**FAIL**'} | {c['detail'].replace('|', '/')} |"
              for c in report.checks]
    (OUT / "HANDOFF_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
