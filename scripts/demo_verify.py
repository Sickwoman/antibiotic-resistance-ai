"""End-to-end verification of the research demo on this machine (needs `models/` and Microsoft Edge).

    python scripts/demo_verify.py

It runs the real demo server, the real CLI, the unchanged API, and a real browser driven over the Chrome DevTools
Protocol on a loopback port. It checks what the demo brief (docs/research_demo_brief.md) and the owner asked for:

- **Scores.** The three synthetic examples are scored through the unchanged CLI. The scores the browser displays
  equal the frozen model's own (an in-process library call) and the unchanged API's.
- **Evidence.** The evaluation panel displays the committed aggregates, and no label, treatment or reassurance.
- **Failures.** A missing bundle, a damaged bundle, a spectrum the reader refuses and a timeout each end in a coded
  error.
- **Network, at runtime.**
  - Every socket of the server, the CLI runs and the API is a loopback socket. A Python audit hook, installed in
    every Python process through `sitecustomize`, records them.
  - Every request the page makes goes to 127.0.0.1 (DevTools network events).
  - Every TCP connection of the browser goes to 127.0.0.1, and no UDP data leaves (Chromium's net log, with socket
    bytes).
- **Persistence, at runtime.**
  - Every file those Python processes open for writing is a temporary file that is gone after the run.
  - The browser keeps no storage, and its profile holds no prediction.
  - No log, model, result or data file changes.

It writes `docs/demo/verification.json`, `docs/demo/verification.md` and the screenshots in
`docs/demo/screenshots/`. It appends to no experiment log and changes no model or result: the check of both logs is
part of its output.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo import synthetic  # noqa: E402
from demo.evidence import load_evidence  # noqa: E402
from demo.inference import DemoInferenceError, predict  # noqa: E402
from src.predict import DISCLAIMER, load_bundle, predict_features, predict_spectrum_file  # noqa: E402
from src.preprocessing import PreprocessingConfig, preprocess_file  # noqa: E402

OUT = ROOT / "docs" / "demo"
SHOTS = OUT / "screenshots"
BUNDLE = ROOT / "models" / "v0.4" / "ecoli_ciprofloxacin" / "best_random.joblib"
LOGS = ("results/experiments/test_evaluations.csv", "results/experiments/development_runs.csv")
EDGE_HOME = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application")
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
FORBIDDEN = [r"\bresistant\b", r"\bsusceptible\b", r"\bsafe\b", r"\bprescri", r"\brule[ -]?out\b", r"\bconfirmed\b",
             r"\btreat(?!ment recommendation)", r"high-confidence"]

# Installed in every Python process of the run (the demo server, each CLI run, the API) through `sitecustomize` on
# PYTHONPATH. It records sockets and every file opened for writing, before any project code runs.
SITECUSTOMIZE = r'''
import json, os, sys, threading
_path = os.environ.get("DEMO_AUDIT_LOG")
if _path:
    _out = open(_path, "a", encoding="utf-8", buffering=1)
    _busy = threading.local()
    _WRITE = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT
    def _emit(record):
        _busy.on = True
        try:
            record["pid"] = os.getpid()
            _out.write(json.dumps(record) + "\n")
        except Exception:
            pass
        finally:
            _busy.on = False
    def _hook(event, args):
        if getattr(_busy, "on", False):
            return
        try:
            if event in ("socket.connect", "socket.bind", "socket.sendto"):
                _emit({"event": event, "address": repr(args[1])})
            elif event == "socket.getaddrinfo":
                _emit({"event": event, "host": repr(args[0])})
            elif event == "open":
                path, mode, flags = args
                writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                    isinstance(flags, int) and flags & _WRITE)
                if writing and not isinstance(path, int):
                    _emit({"event": "open-for-writing", "path": os.fsdecode(path)})
        except Exception:
            pass
    sys.addaudithook(_hook)
    _emit({"event": "process-start", "argv": [str(a) for a in getattr(sys, "orig_argv", [])][:4]})
'''


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.facts: dict = {}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(c["ok"] for c in self.checks)


# --- a minimal WebSocket client and DevTools driver (loopback only) ------------------------------------------------

class WebSocket:
    """Enough of RFC 6455 for the DevTools protocol on 127.0.0.1: text frames, fragmentation, ping, close."""

    def __init__(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.hostname not in LOOPBACK:
            raise ValueError(f"refusing a non-loopback DevTools endpoint: {url}")
        self.sock = socket.create_connection((parts.hostname, parts.port), timeout=180)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {parts.path} HTTP/1.1\r\nHost: {parts.hostname}:{parts.port}\r\nUpgrade: websocket"
                           f"\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
                          .encode())
        head = bytearray()
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("the DevTools endpoint closed during the handshake")
            head += chunk
        status, _, rest = bytes(head).partition(b"\r\n\r\n")
        if b" 101 " not in status.split(b"\r\n", 1)[0]:
            raise ConnectionError(status[:200].decode(errors="replace"))
        self.buffer = bytearray(rest)

    def _exact(self, n: int) -> bytes:
        while len(self.buffer) < n:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise ConnectionError("the DevTools connection closed")
            self.buffer += chunk
        data = bytes(self.buffer[:n])
        del self.buffer[:n]
        return data

    def _frame(self, opcode: int, data: bytes) -> None:
        header = bytearray([0x80 | opcode])
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 1 << 16:
            header += bytes([0x80 | 126]) + n.to_bytes(2, "big")
        else:
            header += bytes([0x80 | 127]) + n.to_bytes(8, "big")
        mask = os.urandom(4)
        self.sock.sendall(bytes(header) + mask + bytes(b ^ mask[i & 3] for i, b in enumerate(data)))

    def send(self, text: str) -> None:
        self._frame(0x1, text.encode("utf-8"))

    def recv(self) -> str:
        message = bytearray()
        while True:
            b0, b1 = self._exact(2)
            n = b1 & 0x7F
            if n == 126:
                n = int.from_bytes(self._exact(2), "big")
            elif n == 127:
                n = int.from_bytes(self._exact(8), "big")
            payload = self._exact(n)
            opcode = b0 & 0x0F
            if opcode == 0x8:
                raise ConnectionError("the browser closed the DevTools connection")
            if opcode == 0x9:
                self._frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            message += payload
            if b0 & 0x80:
                return message.decode("utf-8")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class Browser:
    """Headless Edge with a fresh profile, a net log with socket bytes, and every non-loopback host unresolvable."""

    def __init__(self, work: Path, name: str) -> None:
        self.profile, self.netlog = work / f"edge-profile-{name}", work / f"netlog-{name}.json"
        port = free_port()
        subprocess.Popen([str(EDGE_HOME / "msedge.exe"), "--headless=new", "--disable-gpu", "--no-first-run",
                          "--no-default-browser-check", "--disable-extensions", "--disable-background-networking",
                          "--disable-component-update", "--disable-sync", "--no-pings", "--disable-domain-reliability",
                          "--metrics-recording-only", "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1",
                          f"--user-data-dir={self.profile}", f"--remote-debugging-port={port}",
                          f"--log-net-log={self.netlog}", "--net-log-capture-mode=Everything", "about:blank"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        version = poll_json(f"http://127.0.0.1:{port}/json/version")
        page = next(t for t in poll_json(f"http://127.0.0.1:{port}/json/list") if t["type"] == "page")
        self.browser_url, self.page = version["webSocketDebuggerUrl"], WebSocket(page["webSocketDebuggerUrl"])
        self.version, self.events, self.next_id = version.get("Browser", ""), [], 0
        for method in ("Page.enable", "Runtime.enable", "Network.enable"):
            self.call(method)
        self.viewport(1440, 1000)

    def call(self, method: str, params: dict | None = None, timeout: float = 120) -> dict:
        self.next_id += 1
        msg_id = self.next_id
        self.page.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = json.loads(self.page.recv())
            if message.get("id") == msg_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})
            self.events.append(message)
        raise TimeoutError(method)

    def evaluate(self, expression: str, await_promise: bool = False):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                                "awaitPromise": await_promise})
        return result.get("result", {}).get("value")

    def wait_for(self, expression: str, timeout: float = 180) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.evaluate(expression):
                return
            time.sleep(0.1)
        raise TimeoutError(expression)

    def viewport(self, width: int, height: int) -> None:
        self.call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1,
                                                         "mobile": False})

    def goto(self, url: str) -> None:
        self.call("Page.navigate", {"url": url})
        self.wait_for("document.readyState === 'complete' && document.querySelectorAll('#examples button')"
                      ".length === 3 && !document.querySelector('#evidence-body .facts')")

    def screenshot(self, path: Path, full_page: bool = True, height: int = 1000) -> None:
        if full_page:
            size = self.call("Page.getLayoutMetrics")["cssContentSize"]
            self.viewport(1440, math.ceil(size["height"]))
        else:
            self.viewport(1440, height)
        data = self.call("Page.captureScreenshot", {"format": "png"}, timeout=180)["data"]
        path.write_bytes(base64.b64decode(data))
        self.viewport(1440, 1000)

    def requested_hosts(self) -> set[str]:
        self.evaluate("1")                                     # collect any events still queued
        return {urlsplit(e["params"]["request"]["url"]).hostname or e["params"]["request"]["url"].split(":")[0]
                for e in self.events if e.get("method") == "Network.requestWillBeSent"}

    def close(self) -> None:
        try:
            ws = WebSocket(self.browser_url)
            ws.send(json.dumps({"id": 1, "method": "Browser.close"}))
            ws.close()
        except (OSError, ConnectionError):
            pass
        self.page.close()
        time.sleep(2)
        kill_edge(self.profile)


def kill_edge(profile: Path) -> None:
    try:
        import psutil
    except ImportError:
        return
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            if "msedge" in (proc.info["name"] or "").lower() and any(str(profile) in (c or "")
                                                                     for c in proc.info["cmdline"] or []):
                proc.kill()
        except psutil.Error:
            continue


# --- helpers ---------------------------------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> dict[str, str]:
    """SHA-256 of every model and result file, and size + modification time of every data file."""
    out = {p.relative_to(ROOT).as_posix(): sha256(p) for d in ("models", "results")
           for p in sorted((ROOT / d).rglob("*")) if p.is_file()}
    for p in sorted((ROOT / "data").rglob("*")):
        if p.is_file():
            st = p.stat()
            out[p.relative_to(ROOT).as_posix()] = f"{st.st_size}:{st.st_mtime_ns}"
    return out


def git_status() -> list[str]:
    lines = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True,
                           text=True).stdout.splitlines()
    return sorted(line for line in lines if "docs/demo/" not in line)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method: str, url: str, body: bytes | None = None, headers: dict | None = None) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def poll_json(url: str, seconds: float = 60):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            return json.loads(urllib.request.urlopen(url, timeout=3).read())
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"{url} did not answer")


def post_json(url: str, payload: dict) -> tuple[int, dict]:
    status, _, body = http("POST", url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return status, json.loads(body)


def wait_for_server(url: str, process: subprocess.Popen, seconds: float = 90) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the process for {url} exited early with code {process.returncode}")
        try:
            if http("GET", url)[0] < 500:
                return
        except OSError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"{url} did not answer within {seconds} s")


def start(args: list[str], log: Path) -> subprocess.Popen:
    handle = log.open("w", encoding="utf-8")
    return subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)


def stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def edge_version() -> str:
    versions = [p.name for p in EDGE_HOME.iterdir() if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", p.name)]
    return max(versions, key=lambda v: tuple(int(x) for x in v.split("."))) if versions else "unknown"


def read_netlog(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace").rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:                  # a log cut short when the browser exits: close it
        return json.loads(text.rstrip(",") + ("" if text.endswith("]}") else "]}"))


def netlog_summary(paths: list[Path]) -> dict:
    """What the browser process did on the network: TCP connection attempts, UDP connects and sends, and requests."""
    tcp, udp_connect, udp_sent, background = set(), set(), 0, set()
    for path in paths:
        data = read_netlog(path)
        names = {v: k for k, v in data["constants"]["logEventTypes"].items()}
        for event in data["events"]:
            name, params = names.get(event["type"], ""), event.get("params") or {}
            if name == "TCP_CONNECT_ATTEMPT" and params.get("address"):      # the begin event carries it
                tcp.add(params["address"])
            elif name == "UDP_CONNECT" and params.get("address"):
                udp_connect.add(params["address"])
            elif name in ("UDP_BYTES_SENT", "UDP_SEND_ERROR"):
                udp_sent += 1
            elif name == "URL_REQUEST_START_JOB" and params.get("initiator") == "not an origin":
                background.add(urlsplit(params.get("url", "")).hostname or "")
    host = lambda a: urlsplit("//" + a).hostname or a   # noqa: E731
    return {"tcp_connect_attempts": sorted(tcp),
            "tcp_outside_loopback": sorted(a for a in tcp if host(a) not in LOOPBACK),
            "udp_connects": sorted(udp_connect), "udp_send_events": udp_sent,
            # loopback hosts here are the browser-initiated navigations to the demo page itself
            "browser_background_hosts": sorted(h for h in background if h and h not in LOOPBACK)}


# --- the run ---------------------------------------------------------------------------------------------------------

def run() -> Report:
    report = Report()
    if not BUNDLE.is_file():
        raise SystemExit("models/ is not present: the end-to-end checks need the frozen bundle.")
    if not (EDGE_HOME / "msedge.exe").is_file():
        raise SystemExit("Microsoft Edge was not found; the browser checks need it.")
    SHOTS.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="demo-verify-"))
    audit_dir, audit_log = work / "audit", work / "audit.jsonl"
    audit_dir.mkdir()
    (audit_dir / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    os.environ["PYTHONPATH"] = str(audit_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["DEMO_AUDIT_LOG"] = str(audit_log)
    temp_before = {p.name for p in Path(tempfile.gettempdir()).iterdir()}
    before, status_before = manifest(), git_status()
    log_hashes = {log: sha256(ROOT / log) for log in LOGS}
    report.facts["environment"] = {"python": sys.version.split()[0], "edge": edge_version(),
                                   "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                                            capture_output=True, text=True).stdout.strip()}
    processes: list[subprocess.Popen] = []
    browsers: list[Browser] = []
    page_hosts: set[str] = set()
    demo_scores: dict[str, dict] = {}
    try:
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        server = start(["-m", "demo", "--port", str(port)], work / "server.log")
        processes.append(server)
        wait_for_server(f"{base}/api/status", server)

        # 1. The page, its headers, and the Host guard
        status, headers, _ = http("GET", f"{base}/")
        report.check("the page is served on 127.0.0.1 with the security headers", status == 200 and
                     "default-src 'none'" in headers.get("content-security-policy", "") and
                     headers.get("cache-control") == "no-store")
        report.check("a non-loopback Host header is refused",
                     http("GET", f"{base}/api/status", headers={"Host": "attacker.example"})[0] == 400)

        # 2. The three examples through the demo's API, against the library and the unchanged API
        bundle = load_bundle(BUNDLE)
        library: dict[str, dict] = {}
        for example in synthetic.generate():
            code, body = post_json(f"{base}/api/predict", {"example": example.id})
            demo_scores[example.id] = body
            path = work / f"{example.id}.txt"
            path.write_text(example.to_text(), encoding="utf-8")
            direct = predict_spectrum_file(bundle, path)
            features, _ = preprocess_file(path, PreprocessingConfig())
            unrounded = float(predict_features(bundle, features)[0][0])
            library[example.id] = {"unrounded": unrounded, "above": unrounded >= bundle["threshold"]}
            report.check(f"{example.id}: the demo's score is the frozen model's own, through the unchanged CLI",
                         code == 200 and body.get("runner") == "scripts/predict_spectrum.py (unchanged)"
                         and body["score"] == direct.resistance_probability == round(unrounded, 4)
                         and body["threshold"] == direct.threshold
                         and body["above_threshold"] == (direct.prediction == "Resistant")
                         == library[example.id]["above"],
                         f"demo {body.get('score')}, library {unrounded:.6f}, threshold {bundle['threshold']:.6f}")
        api_port = free_port()
        api = start(["scripts/serve_api.py", "--port", str(api_port)], work / "api.log")
        processes.append(api)
        wait_for_server(f"http://127.0.0.1:{api_port}/ready", api)
        for example in synthetic.generate():
            boundary = "demo-verify-boundary"
            payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"synthetic.txt\""
                       "\r\nContent-Type: text/plain\r\n\r\n").encode() + example.to_text().encode() + \
                f"\r\n--{boundary}--\r\n".encode()
            code, _, raw = http("POST", f"http://127.0.0.1:{api_port}/predict", payload,
                                {"Content-Type": f"multipart/form-data; boundary={boundary}"})
            answer = json.loads(raw)
            report.check(f"{example.id}: the unchanged API gives the same score", code == 200 and
                         answer["resistance_probability"] == demo_scores[example.id]["score"] and
                         answer["threshold"] == demo_scores[example.id]["threshold"],
                         f"API {answer.get('resistance_probability')}")
        stop(api)
        processes.remove(api)

        # 3. The browser: what it displays for each example, the loading state, and screenshots
        evidence = load_evidence()
        browser = Browser(work, "main")
        browsers.append(browser)
        report.facts["environment"]["edge_running"] = browser.version
        browser.goto(f"{base}/?example=synthetic-1")
        browser.screenshot(SHOTS / "01_ready_synthetic-1.png")
        for example in synthetic.generate():
            browser.evaluate(f"document.querySelector('#examples button[data-id=\"{example.id}\"]').click()")
            browser.wait_for("!document.getElementById('run').disabled")
            browser.evaluate("document.getElementById('run').click()")
            browser.wait_for("document.getElementById('state').classList.contains('loading')", timeout=10)
            if example.id == "synthetic-2":
                time.sleep(0.9)
                browser.screenshot(SHOTS / "02_loading_synthetic-2.png", full_page=False, height=1250)
            browser.wait_for("!document.getElementById('result').hidden || "
                             "document.getElementById('state').classList.contains('error')")
            shown = browser.evaluate("document.getElementById('score').textContent")
            position = browser.evaluate("document.getElementById('position').textContent")
            expected = f"{demo_scores[example.id]['score']:.4f}"
            report.check(f"{example.id}: the browser displays the model's score and status",
                         shown == expected and position == ("Above the research threshold"
                                                            if library[example.id]["above"]
                                                            else "Below the research threshold"),
                         f"displayed {shown!r}, {position!r}")
            text = browser.evaluate("document.body.innerText").lower().replace(DISCLAIMER.lower(), "")
            report.check(f"{example.id}: no label, treatment or reassurance is displayed",
                         not any(re.search(p, text) for p in FORBIDDEN))
            if example.id != "synthetic-2":
                browser.screenshot(SHOTS / f"0{3 if example.id == 'synthetic-1' else 4}_result_{example.id}.png")
        panel = browser.evaluate("document.getElementById('evidence-body').innerText")
        wanted = ["1,145 eligible MARISMa E. coli isolates from 2024", "0.772", "0.744\u20130.798", "0.897", "0.393",
                  "0.907", "18 of the 193 zone members were R or I", "unavailable", evidence["error_control"],
                  "Clinical usefulness is unproven", "No patient linkage"]
        report.check("the evaluation panel displays the committed figures",
                     not [w for w in wanted if w not in panel], f"missing {[w for w in wanted if w not in panel]}")
        storage = browser.evaluate("indexedDB.databases().then(d => [localStorage.length, sessionStorage.length, "
                                   "document.cookie.length, d.length])", await_promise=True)
        report.check("the page keeps nothing in browser storage", storage == [0, 0, 0, 0], f"{storage}")
        page_hosts |= browser.requested_hosts()
        browser.close()

        # 4. Failures: a missing bundle (in the browser too), a damaged bundle, a refused spectrum, a timeout
        missing_port = free_port()
        missing = start(["-m", "demo", "--port", str(missing_port), "--model", str(work / "absent" / "x.joblib")],
                        work / "server-missing.log")
        processes.append(missing)
        wait_for_server(f"http://127.0.0.1:{missing_port}/api/status", missing)
        code, body = post_json(f"http://127.0.0.1:{missing_port}/api/predict", {"example": "synthetic-1"})
        report.check("a missing bundle is reported as model_unavailable", code == 503 and
                     body["error"]["code"] == "model_unavailable", f"HTTP {code}")
        browser = Browser(work, "missing")
        browsers.append(browser)
        browser.goto(f"http://127.0.0.1:{missing_port}/?example=synthetic-1")
        browser.evaluate("document.getElementById('run').click()")
        browser.wait_for("document.getElementById('state').classList.contains('error')")
        state_text = browser.evaluate("document.getElementById('state').innerText")
        report.check("the page shows the missing-model error and no score", "The frozen model is not available"
                     in state_text and browser.evaluate("document.getElementById('result').hidden") is True)
        browser.screenshot(SHOTS / "05_error_missing_model.png")
        page_hosts |= browser.requested_hosts()
        browser.close()
        stop(missing)
        processes.remove(missing)

        damaged = work / "damaged" / "best_random.joblib"
        damaged.parent.mkdir()
        data = bytearray(BUNDLE.read_bytes())
        data[len(data) // 2] ^= 0x01
        damaged.write_bytes(bytes(data))
        shutil.copyfile(BUNDLE.with_name(BUNDLE.name + ".sha256"), damaged.with_name(damaged.name + ".sha256"))
        for name, text, kwargs, expected in (
                ("a damaged bundle is refused by its checksum", synthetic.generate()[0].to_text(),
                 {"model": damaged, "timeout": 180}, "model_invalid"),
                ("a spectrum the reader refuses is an inference error", "mass intensity\n2000 1\n1999 2\n" * 60,
                 {"timeout": 180}, "inference_failed"),
                ("a run that takes too long is stopped", synthetic.generate()[0].to_text(), {"timeout": 0.05},
                 "timeout")):
            try:
                predict(text, **kwargs)
                report.check(name, False, "no error was raised")
            except DemoInferenceError as exc:
                report.check(name, exc.code == expected, exc.code)
    finally:
        for browser in browsers:
            kill_edge(browser.profile)
        for process in processes:
            stop(process)
    time.sleep(1.0)

    # 5. Network: every socket of every Python process; every request of the page; every connection of the browser
    events = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    pids = {e["pid"] for e in events if e["event"] == "process-start"}
    sockets = [e for e in events if e["event"] in ("socket.connect", "socket.bind", "socket.sendto")]
    outside = [e for e in sockets if not any(f"'{h}'" in e["address"] for h in LOOPBACK)]
    lookups = sorted({e["host"] for e in events if e["event"] == "socket.getaddrinfo"})
    report.check("every socket of the server, the CLI runs and the API is a loopback socket", not outside,
                 f"{len(sockets)} socket events in {len(pids)} Python processes; outside loopback: {outside[:3]}")
    report.check("no Python process looked up a non-loopback host name",
                 all(h.strip("'b") in LOOPBACK or h == "None" for h in lookups), f"looked up {lookups}")
    report.check("every request the page made went to 127.0.0.1", page_hosts == {"127.0.0.1"}, f"{sorted(page_hosts)}")
    net = netlog_summary(sorted(work.glob("netlog-*.json")))
    report.check("every TCP connection of the browser went to 127.0.0.1", not net["tcp_outside_loopback"],
                 f"{net['tcp_connect_attempts']}")
    report.check("the browser sent no UDP data", net["udp_send_events"] == 0,
                 f"UDP connects without data (Chromium's IPv6 reachability probe): {net['udp_connects']}")
    report.facts["network"] = {"python_processes": len(pids), "python_socket_events": len(sockets),
                               "python_host_lookups": lookups, "page_request_hosts": sorted(page_hosts), **net}

    # 6. Persistence: what was opened for writing, what is left, and what changed
    writes = sorted({e["path"] for e in events if e["event"] == "open-for-writing"})
    temp = os.path.normcase(tempfile.gettempdir())
    spectra = [w for w in writes if os.path.normcase(w).startswith(temp)]
    other = [w for w in writes if w not in spectra and not w.endswith(".pyc") and os.path.basename(w).lower() != "nul"]
    report.check("every file the demo's processes opened for writing was a temporary file", not other,
                 f"{len(spectra)} temporary files; other: {other[:3]}")
    left = [w for w in spectra if Path(w).exists()]
    report.check("no temporary spectrum survived its request", not left, f"left: {left[:3]}")
    leftovers = sorted(n for n in {p.name for p in Path(tempfile.gettempdir()).iterdir()} - temp_before
                       if n.startswith("amr-demo-"))
    report.check("no demo temporary folder is left behind", not leftovers, f"{leftovers}")
    stored = [str(p.relative_to(work)) for p in work.glob("edge-profile-*/**/*") if p.is_file()
              and b"cli_wall_s" in p.read_bytes()]
    report.check("no browser profile stored a prediction (Cache-Control: no-store)", not stored, f"{stored[:3]}")
    logs_text = " ".join(p.read_text(encoding="utf-8") for p in work.glob("server*.log"))
    report.check("the server's console output holds no prediction", "cli_wall_s" not in logs_text and
                 not any(f"{v['score']:.4f}" in logs_text for v in demo_scores.values()))
    after = manifest()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    report.check("models, results, data and both logs are unchanged", not changed and
                 all(sha256(ROOT / log) == h for log, h in log_hashes.items()), f"changed: {changed[:5]}")
    rows = {log: sum(1 for _ in (ROOT / log).open(encoding="utf-8")) - 1 for log in LOGS}
    report.check("the production log has 115 data rows and the development log 34",
                 rows == {LOGS[0]: 115, LOGS[1]: 34}, f"{rows}")
    report.check("the repository gained no file outside docs/demo/", git_status() == status_before)
    report.facts["persistence"] = {"temporary_files_written": len(spectra),
                                   "bytecode_files_written": sum(w.endswith(".pyc") for w in writes),
                                   "log_rows": rows}
    report.facts["synthetic_scores"] = {k: {"demo_score": v["score"], "library_unrounded":
                                            round(library_value, 6), "above_threshold": v["above_threshold"],
                                            "cli_wall_s": v["cli_wall_s"]}
                                        for (k, v), library_value in zip(demo_scores.items(),
                                                                         [x["unrounded"] for x in library.values()],
                                                                         strict=True)}
    shutil.rmtree(work, ignore_errors=True)
    return report


def write(report: Report) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    env = report.facts["environment"]
    data = {"what": "end-to-end verification of the research demo (scripts/demo_verify.py)", "run_utc": stamp,
            "passed": report.passed, **report.facts, "checks": report.checks}
    (OUT / "verification.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    net = report.facts.get("network", {})
    lines = ["# Research demo: end-to-end verification", "",
             f"Run {stamp} by `scripts/demo_verify.py` on commit `{env['commit']}`, with Python {env['python']} and "
             f"{env.get('edge_running') or 'Microsoft Edge ' + env['edge']}. "
             f"**{sum(c['ok'] for c in report.checks)} of {len(report.checks)} checks passed.**", "",
             "The scores below are pipeline demonstrations on synthetic spectra. They are not research results, and "
             "no experiment log was touched.", "",
             "| Synthetic example | Demo score (CLI, 4 decimals) | Frozen model, unrounded | Status | CLI run |",
             "|---|---|---|---|---|"]
    for key, value in report.facts.get("synthetic_scores", {}).items():
        lines.append(f"| {key} | {value['demo_score']:.4f} | {value['library_unrounded']:.6f} | "
                     f"{'above' if value['above_threshold'] else 'below'} the research threshold | "
                     f"{value['cli_wall_s']:.1f} s |")
    lines += ["", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c['check']} | {'pass' if c['ok'] else '**FAIL**'} | {c['detail'].replace('|', '/')} |"
              for c in report.checks]
    lines += ["", "**The browser itself.** Edge's own background services tried to reach "
              f"{', '.join(net.get('browser_background_hosts', [])) or 'no host'}. They were not requests of the "
              "page. Name resolution was blocked for every host except 127.0.0.1, so none connected: every TCP "
              "connection the browser made went to 127.0.0.1. "
              f"The UDP connects ({', '.join(net.get('udp_connects', [])) or 'none'}) carried no data: the net log, "
              "with socket bytes, records no UDP send. That pattern is Chromium's IPv6 reachability probe."]
    (OUT / "verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = run()
    write(result)
    print(f"\n{sum(c['ok'] for c in result.checks)} of {len(result.checks)} checks passed.")
    raise SystemExit(0 if result.passed else 1)
