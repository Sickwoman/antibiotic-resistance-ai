// Research demo page. Everything shown comes from this demo's own server: the synthetic spectra, the frozen
// model's output (through the unchanged prediction CLI), and the committed evaluation results. Nothing is stored in
// the browser, and nothing is fetched from anywhere else.
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(window.location.search);
  const state = { examples: [], selected: null, running: false, evidence: null };
  const MINUS = "−";
  const SVG_NS = "http://www.w3.org/2000/svg";

  const fmt = (x, d = 3) => {
    const s = Number(x).toFixed(d);
    return s.startsWith("-") ? MINUS + s.slice(1) : s;
  };
  const fmtSigned = (x, d = 3) => (x < 0 ? MINUS : "+") + Math.abs(x).toFixed(d);
  const fmtInt = (n) => Number(n).toLocaleString("en-US");
  const range = (m) => `${fmt(m.low)}–${fmt(m.high)}`;
  const sup = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵",
                6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const fmtP = (p) => {
    const [mantissa, exponent] = Number(p).toExponential(1).split("e");
    return `${mantissa} × 10${String(Number(exponent)).split("").map((c) => sup[c]).join("")}`;
  };

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (key === "class") node.className = value; else node.setAttribute(key, value);
    }
    for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    return node;
  }

  function svg(tag, attrs = {}, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  async function getJSON(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (err) {
      const e = new Error("The local demo server could not be reached. Is `python -m demo` still running?");
      e.code = "network";
      throw e;
    }
    let body = null;
    try { body = await response.json(); } catch (err) { body = null; }
    if (!response.ok) {
      const e = new Error((body && body.error && body.error.message) || `The server answered HTTP ${response.status}.`);
      e.code = (body && body.error && body.error.code) || "http_error";
      e.status = response.status;
      throw e;
    }
    return body;
  }

  // --- the spectrum plot --------------------------------------------------------------------------------------
  function drawPlot(example) {
    const plot = $("plot");
    plot.replaceChildren();
    const W = 640, H = 300, L = 62, R = 14, T = 14, B = 44;
    const xs = example.plot.mz, ys = example.plot.intensity;
    const xMin = 1800, xMax = 20300, yMax = Math.max(...ys) * 1.06;
    const X = (m) => L + ((m - xMin) / (xMax - xMin)) * (W - L - R);
    const Y = (v) => T + (1 - v / yMax) * (H - T - B);
    plot.append(svg("rect", { x: X(2000), y: T, width: X(20000) - X(2000), height: H - T - B, fill: "#edf1f7" }));
    for (let m = 2000; m <= 20000; m += 2000) {
      plot.append(svg("line", { x1: X(m), y1: H - B, x2: X(m), y2: H - B + 5, stroke: "#8a94a3" }));
      plot.append(svg("text", { x: X(m), y: H - B + 18, "text-anchor": "middle", "font-size": 10, fill: "#5b6573" },
                      fmtInt(m)));
    }
    const step = Math.pow(10, Math.floor(Math.log10(yMax / 4)));
    const tick = Math.ceil(yMax / 4 / step) * step;
    for (let v = 0; v <= yMax; v += tick) {
      plot.append(svg("line", { x1: L - 5, y1: Y(v), x2: W - R, y2: Y(v), stroke: v ? "#e3e7ec" : "#8a94a3" }));
      plot.append(svg("text", { x: L - 8, y: Y(v) + 3, "text-anchor": "end", "font-size": 10, fill: "#5b6573" },
                      fmtInt(v)));
    }
    plot.append(svg("line", { x1: L, y1: T, x2: L, y2: H - B, stroke: "#8a94a3" }));
    const d = xs.map((m, i) => `${i ? "L" : "M"}${X(m).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join("");
    plot.append(svg("path", { d, fill: "none", stroke: "#1f3c66", "stroke-width": 1, "stroke-linejoin": "round" }));
    plot.append(svg("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", "font-size": 11, fill: "#1d2733" },
                    "m/z (Da)"));
    plot.append(svg("text", { x: 14, y: (T + H - B) / 2, "text-anchor": "middle", "font-size": 11, fill: "#1d2733",
                              transform: `rotate(-90 14 ${(T + H - B) / 2})` }, "intensity (counts)"));
    plot.append(svg("text", { x: W - R - 4, y: T + 14, "text-anchor": "end", "font-size": 11, fill: "#8a4b08",
                              "font-weight": 600 }, "SYNTHETIC"));
  }

  // --- examples and the run -----------------------------------------------------------------------------------
  function renderExamples() {
    const box = $("examples");
    box.replaceChildren();
    for (const example of state.examples) {
      const button = el("button", { type: "button", "aria-pressed": "false", "data-id": example.id }, example.title);
      button.addEventListener("click", () => select(example.id));
      box.append(button);
    }
  }

  function select(id) {
    const example = state.examples.find((e) => e.id === id) || state.examples[0];
    state.selected = example;
    for (const button of $("examples").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.id === example.id));
    }
    $("example-facts").textContent = `${example.title}: ${fmtInt(example.n_points)} points, m/z ` +
      `${fmtInt(Math.round(example.mz_min))}–${fmtInt(Math.round(example.mz_max))} Da, generated with seed ` +
      `${example.seed}. Synthetic: no organism, no ground-truth label.`;
    drawPlot(example);
    $("result").hidden = true;
    $("state").hidden = true;
    updateButton();
  }

  function updateButton() {
    $("run").disabled = state.running || !state.selected;
    $("run").textContent = state.running ? "Running…" : "Run research prediction";
  }

  function showState(kind, ...nodes) {
    const box = $("state");
    box.hidden = false;
    box.className = `state ${kind}`;
    box.replaceChildren(...nodes);
  }

  const ERROR_TITLES = {
    model_unavailable: "The frozen model is not available",
    model_invalid: "The frozen model artifacts could not be used",
    inference_failed: "The prediction failed",
    timeout: "The prediction took too long",
    unknown_example: "Unknown example",
    network: "The demo server did not answer",
  };

  async function run() {
    if (state.running || !state.selected) return;
    const example = state.selected;
    state.running = true;
    updateButton();
    $("result").hidden = true;
    const started = performance.now();
    const elapsed = el("span", { class: "elapsed" }, "0.0 s");
    showState("loading", el("span", { class: "spinner", "aria-hidden": "true" }),
              `Running the frozen model locally on ${example.title}… `, elapsed, el("br"),
              el("small", {}, "Each run starts the project's prediction CLI in a fresh process and loads the frozen " +
                              "bundle, so it takes a few seconds."));
    const timer = setInterval(() => { elapsed.textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`; },
                              100);
    try {
      const result = await getJSON("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ example: example.id }),
      });
      if (state.selected !== example) return;
      renderResult(result);
      showState("success", `Scored by the frozen model through the unchanged scripts/predict_spectrum.py: ` +
                           `${example.title} (synthetic).`);
    } catch (err) {
      if (state.selected !== example) return;
      showState("error", el("strong", {}, ERROR_TITLES[err.code] || "Something went wrong"), el("br"), err.message);
    } finally {
      clearInterval(timer);
      state.running = false;
      updateButton();
    }
  }

  function renderResult(result) {
    $("score").textContent = fmt(result.score, 4);
    $("threshold").textContent = fmt(result.threshold, 4);
    $("threshold-note").textContent = state.evidence
      ? `frozen at ${state.evidence.threshold}; the CLI reports both numbers to four decimals, and the status is ` +
        "its own decision on the unrounded score"
      : "the CLI reports both numbers to four decimals; the status is its own decision on the unrounded score";
    $("position").textContent = result.above_threshold ? "Above the research threshold"
                                                       : "Below the research threshold";
    $("model-version").textContent = result.model_version;
    $("runner").textContent = result.runner;
    $("timing").textContent = `${Number(result.cli_wall_s).toFixed(1)} s for the CLI run (starting Python and ` +
      `loading the bundle included); inside it, preprocessing ${fmt(result.preprocessing_ms, 1)} ms and the model ` +
      `${fmt(result.inference_ms, 1)} ms`;
    $("disclaimer").textContent = result.disclaimer;
    $("intercept-inline").textContent = state.evidence ? fmt(state.evidence.calibration_intercept.estimate)
                                                       : "see the evaluation panel";
    $("result").hidden = false;
  }

  // --- the evaluation panel -----------------------------------------------------------------------------------
  function tile(title, value, sub, caution = false) {
    return el("div", { class: caution ? "tile caution" : "tile" }, el("p", { class: "label" }, title),
              el("p", { class: "value" }, value), el("p", { class: "sub" }, sub));
  }

  function renderEvidence(e) {
    const p = e.population;
    const body = $("evidence-body");
    body.replaceChildren(
      el("p", {}, el("strong", {}, "Who was evaluated. "),
         `${fmtInt(p.n)} eligible MARISMa E. coli isolates from 2024, all acquired on one instrument (MBT-WIN10): ` +
         `${fmtInt(p.n_ri)} with an R or I ciprofloxacin interpretation and ${fmtInt(p.n_s)} with S. They came ` +
         `from ${fmtInt(p.candidates)} prepared 2024 isolates, of which ${fmtInt(p.matched)} had a susceptibility ` +
         "record. All sample sources were kept, because screening status could not be determined. The model was " +
         "scored once, with nothing refitted."),
      el("p", { class: "statement" },
         `The frozen ciprofloxacin model achieved AUROC ${fmt(e.auroc.estimate)} (descriptive 95% interval ` +
         `${range(e.auroc)}) on ${fmtInt(p.n)} eligible MARISMa E. coli isolates from 2024, with evidence of ` +
         "above-chance ranking under the isolate-independence assumption."),
      el("div", { class: "tiles" },
         tile("AUROC (ranking)", fmt(e.auroc.estimate),
              `interval ${range(e.auroc)}; one-sided Holm-adjusted p ${fmtP(e.holm_adjusted_p)}`),
         tile("Sensitivity at the frozen threshold", fmt(e.sensitivity.estimate),
              `share of R/I isolates scored above the threshold; interval ${range(e.sensitivity)}; the point ` +
              "estimate misses the 0.90 research target", true),
         tile("Specificity at the frozen threshold", fmt(e.specificity.estimate),
              `share of S isolates scored below the threshold; interval ${range(e.specificity)}: most S isolates ` +
              "were flagged", true),
         tile("Confidence zone NPV", fmt(e.zone.npv.estimate),
              `${fmtInt(e.zone.ri)} of the ${fmtInt(e.zone.n)} zone members were R or I; interval ` +
              `${range(e.zone.npv)}, below the 0.95 research target. Not a safety feature.`, true),
         tile("Calibration intercept", fmt(e.calibration_intercept.estimate),
              `interval ${range(e.calibration_intercept)}: observed in aggregate, the model under-predicted ` +
              "resistance on MARISMa", true),
         tile("Internal minus MARISMa AUROC", fmtSigned(e.gap.estimate),
              `interval ${fmtSigned(e.gap.low)} to ${fmtSigned(e.gap.high)}: a gap was not demonstrated, which is ` +
              "not equivalence or non-inferiority"),
         tile("Ceftriaxone", e.ceftriaxone, "MARISMa holds no ceftriaxone interpretation for these isolates, so " +
              "the second model could not be evaluated", true)),
      el("p", { class: "quote" }, `Registered conclusion: “${e.conclusion}”`),
      el("h3", {}, "Limits that come with every number above"),
      el("ul", { class: "limits" },
         el("li", {}, el("strong", {}, "No patient linkage. "),
            "MARISMa removed patient information, so several isolates from one patient may count as independent, " +
            "and the intervals and p-values may be too narrow. In the protocol's words: ",
            el("span", { class: "quote" }, `“${e.error_control}”`)),
         el("li", {}, el("strong", {}, "Clinical usefulness is unproven. "),
            "Above-chance ranking is not a decision rule. At the frozen threshold most S isolates were flagged, " +
            "the probabilities were too low on aggregate, and the confidence zone missed its target."),
         el("li", {}, el("strong", {}, "Screening status unknown. "),
            "No sample-source category could be identified as screening, so all sources were kept."),
         el("li", {}, el("strong", {}, "Different acquisition. "),
            "MARISMa spectra start near 2,000 Da rather than 1,960 Da, and the features scale by about 1 %; the " +
            "frozen preprocessing cannot remove either."),
         el("li", {}, el("strong", {}, "One hospital, one year, one instrument. "),
            "Nothing here covers other years, instruments or sites.")),
      el("p", { class: "internal" }, el("strong", {}, "For comparison, the internal test "),
         `(DRIAMS-A, held-out patients): AUROC ${fmt(e.internal.auroc.estimate)} (${range(e.internal.auroc)}), ` +
         `sensitivity ${fmt(e.internal.sensitivity.estimate)}, specificity ${fmt(e.internal.specificity.estimate)}.`),
      el("p", { class: "source" },
         `Read when the page loaded from ${e.source.report} (SHA-256 of its committed content ` +
         `${e.source.report_sha256.slice(0, 16)}…, written ${e.source.written_utc} by the one-time run, ` +
         `authorised code ${e.source.authorised_commit}) and ${e.source.internal}.`),
    );
  }

  function showModelStatus(model) {
    const box = $("model-status");
    if (model.present) {
      box.className = "model-status";
      box.textContent = `Frozen model found: ${model.path}` +
        (model.sidecar ? " (its SHA-256 sidecar is checked on every run)." : " (no checksum sidecar).");
    } else {
      box.className = "model-status missing";
      box.textContent = `Frozen model not found at ${model.path}: predictions will fail until it is restored.`;
    }
  }

  async function init() {
    $("run").addEventListener("click", run);
    try {
      const [status, examples] = await Promise.all([getJSON("/api/status"), getJSON("/api/examples")]);
      showModelStatus(status.model);
      state.examples = examples;
      renderExamples();
      select(params.get("example") || examples[0].id);
    } catch (err) {
      showState("error", el("strong", {}, "The demo server did not answer."), el("br"), err.message);
      return;
    }
    try {
      state.evidence = await getJSON("/api/evidence");
      renderEvidence(state.evidence);
    } catch (err) {
      $("evidence-body").replaceChildren(el("p", { class: "state error" },
        `The committed evaluation results could not be loaded: ${err.message}`));
    }
    if (params.get("run") === "1") run();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
