"""Build the frozen model's distribution package from the trusted original (run on the machine that holds it).

    python scripts/package_model.py                 # dist/<package>.zip, dist/SHA256SUMS.txt, docs/release/...

It copies the bundle byte for byte (no retraining, conversion or re-serialisation) after checking it against the
pinned SHA-256 in `src/model_package.py`. The manifest, model card and notice are rendered from committed records:
- the bundle's saved model card (`models/.../best_random.json`) for its versions and parameters;
- `results/metrics/v0.4/.../test_intervals.json` and `results/metrics/v2.0/marisma_evaluation.json` for the
  aggregate results.

No figure is typed. The ZIP is reproducible: the same inputs give the same bytes. It writes copies of the text members,
the asset list and the release notes to `docs/release/model-v0.4.0/`, for review. Nothing is published.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_package import (  # noqa: E402
    BUNDLE_NAME,
    FROZEN_BUNDLE_SHA256,
    FROZEN_BUNDLE_SIZE,
    MODEL_VERSION,
    PACKAGE_NAME,
    SIDECAR_NAME,
    TARGET,
    build,
    sha256_bytes,
)
from src.utils import project_path  # noqa: E402

REPOSITORY = "https://github.com/Sickwoman/antibiotic-resistance-ai"
RELEASE_TAG = "model-ecoli-ciprofloxacin-v0.4.0"
DOCS = Path("docs/release/model-v0.4.0")
CARD = TARGET.with_suffix(".json")
INTERNAL = Path("results/metrics/v0.4/ecoli_ciprofloxacin/test_intervals.json")
EXTERNAL = Path("results/metrics/v2.0/marisma_evaluation.json")
COPYRIGHT = "Copyright (c) 2026 Moksh"


def f3(x: float) -> str:
    return f"{x:.3f}".replace("-", "−")


def interval(m: dict) -> str:
    return f"{f3(m['low'])}–{f3(m['high'])}"


def evidence(root: Path) -> dict:
    internal = json.loads((root / INTERNAL).read_bytes())["random"]["intervals"]["tuned_lightgbm"]
    external = json.loads((root / EXTERNAL).read_bytes().replace(b"\r\n", b"\n"))
    primary, family = external["primary"], external["holm_family"]
    m, zone = primary["metrics"], primary["zone"]
    in_zone_s = round(m["zone_npv"]["estimate"] * zone["n_in_zone"])
    return {"internal": {"n": external["internal_side"]["n"], "n_resistant": external["internal_side"]["n_resistant"],
                         "auroc": internal["roc_auc"], "sensitivity": internal["sensitivity"],
                         "specificity": internal["specificity"]},
            "external": {"n": primary["n"], "n_resistant": primary["n_resistant"], "auroc": m["roc_auc"],
                         "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                         "calibration_intercept": m["calibration_intercept"], "zone_npv": m["zone_npv"],
                         "zone_n": zone["n_in_zone"], "zone_ri": zone["n_in_zone"] - in_zone_s,
                         "conclusion": family["conclusions"]["ciprofloxacin"],
                         "error_control": family["error_control"], "ceftriaxone": family["conclusions"]["ceftriaxone"],
                         "gap": external["gap"]}}


def manifest(card: dict, ev: dict) -> dict:
    return {
        "package": PACKAGE_NAME, "package_format": "amr-model-package/1",
        "model": {key: card[key] for key in ("model_version", "model", "model_kind", "species", "antibiotic",
                                              "label_map", "threshold", "threshold_rule", "n_features",
                                              "feature_fingerprint", "calibration", "seed", "params", "created")},
        "bundle": {"file": BUNDLE_NAME, "bytes": FROZEN_BUNDLE_SIZE, "sha256": FROZEN_BUNDLE_SHA256,
                   "format": "amr-model-bundle/1",
                   "serialisation": "joblib (a zlib-compressed pickle): loading it runs code, so load only a copy "
                                    "whose SHA-256 matches the value pinned in the repository",
                   "install_to": TARGET.as_posix(), "checksum_file": SIDECAR_NAME},
        "provenance": {"repository": REPOSITORY, "source_commit": card["git_commit"],
                       "source_commit_resolvable_via": "tag research-archive/v0.4/run",
                       "sha256_pinned_in": ["src/model_package.py", "docs/reproduction_guide.md (section 1.3)"],
                       "training_data": {"dataset": "DRIAMS-A, E. coli isolates with a ciprofloxacin result",
                                         "doi": "10.5061/dryad.bzkh1899q", "licence": "CC0 1.0",
                                         "split": card["split"]["name"], "train_size": card["split"]["train_size"],
                                         "row_fingerprint": card["dataset"]["row_fingerprint"]},
                       "bundle_contents": "the fitted pipeline and aggregate metadata; the calibration folds are "
                                          "stored as integer row positions into the training part. No identifier, "
                                          "spectrum, label or per-isolate prediction."},
        "environment": {"python": card["versions"]["python"], **{k: v for k, v in card["versions"].items()
                                                                if k != "python"},
                        "install": "pip install -r requirements-lock.txt --extra-index-url "
                                   "https://download.pytorch.org/whl/cpu",
                        "loader": "src/predict.load_bundle (unchanged), via scripts/predict_spectrum.py or the demo"},
        "evaluation": {"internal_test_driams_a": {"n": ev["internal"]["n"], "n_r_or_i": ev["internal"]["n_resistant"],
                                                  "auroc": ev["internal"]["auroc"],
                                                  "sensitivity": ev["internal"]["sensitivity"],
                                                  "specificity": ev["internal"]["specificity"],
                                                  "source": INTERNAL.as_posix()},
                       "external_marisma_2024": {"n": ev["external"]["n"], "n_r_or_i": ev["external"]["n_resistant"],
                                                 "auroc": ev["external"]["auroc"],
                                                 "sensitivity": ev["external"]["sensitivity"],
                                                 "specificity": ev["external"]["specificity"],
                                                 "calibration_intercept": ev["external"]["calibration_intercept"],
                                                 "zone_npv": ev["external"]["zone_npv"],
                                                 "zone_members": ev["external"]["zone_n"],
                                                 "zone_r_or_i": ev["external"]["zone_ri"],
                                                 "conclusion": ev["external"]["conclusion"],
                                                 "error_control": ev["external"]["error_control"],
                                                 "source": EXTERNAL.as_posix()}},
        "terms": {"status": "proposed: in force only once the copyright holder publishes this package",
                  "proposed_licence": "MIT", "notice": "NOTICE.md"},
        "intended_use": "methodological research only; not a medical device or diagnostic; never a treatment "
                        "recommendation",
        "installation": "python scripts/install_model.py <this package's .zip>",
    }


def model_card(card: dict, ev: dict) -> str:
    i, e = ev["internal"], ev["external"]
    return f"""# Model card: E. coli ciprofloxacin resistance ranker, `{MODEL_VERSION}`

**Research use only.**
- This is not a medical device and not a diagnostic.
- It does not replace antimicrobial susceptibility testing, and it never recommends a treatment.
- It has no regulatory status.

## What it is
- **Input.** One raw MALDI-TOF mass spectrum of an *E. coli* isolate: two text columns, m/z and intensity. The
  repository's frozen pipeline turns it into {card["n_features"]:,} features (feature fingerprint
  `{card["feature_fingerprint"]}`): square root, Savitzky–Golay smoothing, SNIP baseline, total-ion-current
  normalisation, and 3 Da bins over 2,000–20,000 Da.
- **Model.** LightGBM ({card["params"]["n_estimators"]} trees, seed {card["seed"]}). It is wrapped in scikit-learn's
  sigmoid calibration, fitted on out-of-fold predictions of 5 patient-grouped folds of the training part.
- **Output.** A score in [0, 1], where a higher score means resistance (R or I) is more likely by this model's
  ranking. The frozen research threshold, {card["threshold"]}, is the highest validation threshold with
  sensitivity ≥ 0.90.
- **Labels it learned from.** 1 = R or I, and 0 = S, for ciprofloxacin.

## Training data and provenance
- **Data.** DRIAMS-A (University Hospital Basel), {card["split"]["train_size"]:,} training spectra from a
  patient-grouped random split. DRIAMS: Weis et al., Dryad, doi:10.5061/dryad.bzkh1899q, CC0 1.0.
- **Saved.** On {card["created"]}, from commit `{card["git_commit"]}` of {REPOSITORY} (resolvable through the tag
  `research-archive/v0.4/run`).
- **The bundle.** SHA-256 `{FROZEN_BUNDLE_SHA256}`, {FROZEN_BUNDLE_SIZE:,} bytes. It is copied byte for byte from the
  original.

## Evaluation (aggregates, from the repository's committed results)
- **Internal test,** DRIAMS-A, held-out patients ({i["n"]} spectra, {i["n_resistant"]} R or I):
  - AUROC {f3(i["auroc"]["estimate"])} ({interval(i["auroc"])});
  - sensitivity {f3(i["sensitivity"]["estimate"])} and specificity {f3(i["specificity"]["estimate"])} at the
    threshold.
- **External test,** MARISMa 2.0.0 (Madrid; {e["n"]:,} eligible isolates from 2024, {e["n_resistant"]} R or I):
  - AUROC {f3(e["auroc"]["estimate"])} (descriptive 95% interval {interval(e["auroc"])}): \"{e["conclusion"]}\"
    {e["error_control"]}
  - With equal weight:
    - sensitivity {f3(e["sensitivity"]["estimate"])} and specificity {f3(e["specificity"]["estimate"])} at the
      threshold;
    - calibration intercept {f3(e["calibration_intercept"]["estimate"])}, meaning resistance was under-predicted;
    - confidence-zone NPV {f3(e["zone_npv"]["estimate"])}, with {e["zone_ri"]} R or I isolates among
      {e["zone_n"]} zone members, below its 0.95 research target;
    - no internal–external AUROC gap was demonstrated, which is not equivalence.
  - The ceftriaxone evaluation was {e["ceftriaxone"]}.

## Known limitations
- **Narrow training.** One development hospital, one species and one antibiotic. The discrimination is modest.
- **The threshold.** The 0.90-sensitivity threshold did not hold on held-out DRIAMS patients, and missed it on its
  point estimate at MARISMa.
- **The probabilities** were too low on MARISMa: under-predicted.
- **The confidence zone is not a safety feature.**
- **MARISMa's limits.** It has no patient linkage, so its intervals and p-values may be too narrow. Its screening
  status is unknown, and its acquisition differs from DRIAMS's.
- **Not established anywhere else.** Nothing is established for other sites, years, instruments, species or
  antibiotics, or prospectively.

## Using it
- **Environment.** Python {card["versions"]["python"]}, numpy {card["versions"]["numpy"]}, pandas
  {card["versions"]["pandas"]}, scikit-learn {card["versions"]["scikit-learn"]}, LightGBM
  {card["versions"]["lightgbm"]} and joblib {card["versions"]["joblib"]}, as pinned in the repository's
  `requirements-lock.txt`.
- **Install.** `python scripts/install_model.py <package.zip>`, from the repository. It checks the bundle against the
  SHA-256 pinned in the repository before anything is written, and never replaces a different model.
- **Load** the bundle only through the repository's loader (`src/predict.py`), and only a copy that passed that check.
  A joblib file is a pickle: loading one from an untrusted source can run arbitrary code. A matching `.sha256` file
  next to it is not proof of origin.

## Citation
Cite the repository (`CITATION.cff`) and DRIAMS (Weis et al., *Nat Med* 2022, doi:10.1038/s41591-021-01619-9; data
doi:10.5061/dryad.bzkh1899q). For the external figures, cite MARISMa (Zenodo doi:10.5281/zenodo.17201597;
descriptor doi:10.1101/2025.05.31.657186).
"""


def notice() -> str:
    licence = (project_path("LICENSE")).read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    return f"""# Notice: `{PACKAGE_NAME}`

## Redistribution terms (proposed)
These are the terms the project proposes to its copyright holder. **They take effect only when the copyright holder
publishes this package.** Until then it is not licensed for redistribution.

Proposed: this package (the model bundle and these documents) under the MIT License, the same terms as the
repository's code:

```
{licence}
```

The repository's code licence does not by itself cover the model; the publication would make this grant.

## Data the model was derived from
- **Training data:** DRIAMS (Weis C, Cuénod A, Rieck B, Borgwardt K, Egli A; Dryad, doi:10.5061/dryad.bzkh1899q),
  released under CC0 1.0. It places no condition on derived models; its authors ask to be cited.
- **External evaluation only:** MARISMa 2.0.0 (Zenodo, doi:10.5281/zenodo.17201597; CC BY 4.0). No MARISMa data was
  used to train the model or is in this package. The model card quotes aggregate evaluation figures with attribution.

## Third-party software
The bundle holds serialised objects of scikit-learn (BSD-3-Clause), LightGBM (MIT) and NumPy (BSD-3-Clause). It
contains none of their code: install those libraries yourself, under their own licences.

## What the bundle contains
The fitted pipeline and aggregate metadata: parameters, versions, fingerprints and summary metrics. The calibration
folds are stored as integer row positions into the training part.

It contains no isolate, patient or spectrum identifier, no spectrum, no label and no per-isolate prediction.

## Security
A joblib file is a pickle, and loading it runs code. Load this bundle only if its SHA-256 is
`{FROZEN_BUNDLE_SHA256}`, the value pinned in the repository. The `.sha256` file in this package can be replaced
together with the bundle, so it proves nothing on its own. The repository's installer checks the pin.

## No warranty; research use
Provided as is, without warranty: see the licence text above. It is a research artifact, and not a medical device.
"""


def release_notes(assets: list[dict], ev: dict) -> str:
    e = ev["external"]
    rows = "\n".join(f"| `{a['name']}` | {a['bytes']:,} | `{a['sha256']}` |" for a in assets)
    return f"""# Frozen model: E. coli ciprofloxacin `{MODEL_VERSION}` (research use only)

The exact frozen model behind this repository's Version 0.4–2.0 results and its research demo, packaged for local
installation.
- **Byte-identical original.** It was not retrained, converted or re-serialised.
- **Not a diagnostic.** It is not for clinical use and never recommends a treatment.

| Asset | Bytes | SHA-256 |
|---|---|---|
{rows}

The bundle inside is `{BUNDLE_NAME}`, {FROZEN_BUNDLE_SIZE:,} bytes, SHA-256 `{FROZEN_BUNDLE_SHA256}`. The same value
is pinned in the repository's `src/model_package.py` and `docs/reproduction_guide.md`.

## Install
From a clone of the repository, with the environment of `requirements-lock.txt`:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\install_model.py <path to {PACKAGE_NAME}.zip>
.\\.venv\\Scripts\\python.exe -m demo        # http://127.0.0.1:8050/
```

The installer checks the bundle against the pinned hash before writing anything, refuses a mismatch, and never
replaces a different model. Load the bundle only after that check: a joblib file is a pickle.

## What it can and cannot claim
On MARISMa it showed above-chance ranking under the isolate-independence assumption: AUROC
{f3(e["auroc"]["estimate"])}, descriptive 95% interval {interval(e["auroc"])}. With equal weight:
- sensitivity {f3(e["sensitivity"]["estimate"])} and specificity {f3(e["specificity"]["estimate"])} at the frozen
  threshold;
- resistance was under-predicted (calibration intercept {f3(e["calibration_intercept"]["estimate"])});
- its confidence zone missed its target (NPV {f3(e["zone_npv"]["estimate"])}; {e["zone_ri"]} of {e["zone_n"]} R or I);
- there was no patient linkage.

Clinical usefulness is unproven. See `MODEL_CARD.md` and `NOTICE.md` in the package.

## Terms
The MIT License, granted by the copyright holder by publishing this release (see `NOTICE.md` in the package).
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=project_path(TARGET), help="the trusted original bundle")
    parser.add_argument("--out", type=Path, default=project_path("dist"), help="where the assets are written")
    args = parser.parse_args(argv)
    root = project_path(".")
    bundle = args.source.read_bytes()
    if sha256_bytes(bundle) != FROZEN_BUNDLE_SHA256:
        print(f"Refused: {args.source} is not the frozen bundle (SHA-256 mismatch).", file=sys.stderr)
        return 2
    card = json.loads(args.source.with_suffix(".json").read_text(encoding="utf-8"))
    ev = evidence(root)
    files = {"MANIFEST.json": json.dumps(manifest(card, ev), indent=2, ensure_ascii=False) + "\n",
             "MODEL_CARD.md": model_card(card, ev), "NOTICE.md": notice()}
    package = build(bundle, json.loads(files["MANIFEST.json"]), files["MODEL_CARD.md"], files["NOTICE.md"])
    args.out.mkdir(parents=True, exist_ok=True)
    zip_path = args.out / f"{PACKAGE_NAME}.zip"
    zip_path.write_bytes(package)
    sums = f"{sha256_bytes(package)}  {zip_path.name}\n"
    (args.out / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")
    assets = [{"name": zip_path.name, "bytes": len(package), "sha256": sha256_bytes(package)},
              {"name": "SHA256SUMS.txt", "bytes": len(sums.encode()), "sha256": sha256_bytes(sums.encode())}]
    docs = root / DOCS
    docs.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (docs / name).write_text(text, encoding="utf-8", newline="\n")
    (docs / "ASSETS.json").write_text(json.dumps({"release_tag": RELEASE_TAG, "assets": assets,
                                                  "bundle": {"name": BUNDLE_NAME, "bytes": FROZEN_BUNDLE_SIZE,
                                                             "sha256": FROZEN_BUNDLE_SHA256}}, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    (docs / "RELEASE_NOTES.md").write_text(release_notes(assets, ev), encoding="utf-8", newline="\n")
    for asset in assets:
        print(f"{asset['name']}: {asset['bytes']:,} bytes, SHA-256 {asset['sha256']}")
    print(f"written to {args.out}; review copies in {DOCS.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
