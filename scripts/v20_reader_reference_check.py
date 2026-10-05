"""Check the Bruker reader (src/bruker.py) against two independent implementations, on real MARISMa spectra.

Version 2.0, step 3 (docs/v2.0_marisma_plan.md). Spectra only: no label, no outcome file, no model.

The references are third-party code, loaded from a folder given on the command line. They are not dependencies of this
project and are never imported by it:
- nmrglue 0.12 (BSD): `fileio.bruker.read_jcamp` parses the acqu file and `read_binary` decodes the fid's 32-bit
  integers in the stated byte order. This checks the parameter parsing, the byte order and the integer decoding.
- maldi-nn 0.2.5 (MIT): `SpectrumObject.from_bruker` and `tof2mass`, a separate Python port of readBrukerFlexData's
  conversion without HPC. Only these two methods are taken from its source, verbatim, because importing the module
  would need torch. This checks the time-of-flight axis and the calibration equation.

A third check needs no code from anyone: every converted mass must satisfy the calibration's inverse,
tof = ML2 + ML3*m + sqrt(1e12/ML1)*sqrt(m), the equation readBrukerFlexData documents.

The sample is seeded and drawn from all of the cohort's replicate folders, so it includes spectra the registered
checks refuse: the decoding (`decode`) is compared for every spectrum, and whether `convert` accepts it is counted.
Folders are extracted one at a time to a temporary folder outside the repository and deleted afterwards. The output
holds aggregate counts and deviations only: no identifier and no path.

    python scripts/v20_reader_reference_check.py --references <folder>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import tempfile
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bruker import BrukerReadError, _double, _text, convert, decode, tof2mass  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.zip_index import iter_members, read_member  # noqa: E402

PARAMETERS = ("TD", "DELAY", "DW", "ML1", "ML2", "ML3", "BYTORDA")
CHECKS = ("zip_equals_disk", "parameters_equal", "fid_equal", "maldi_nn_mz_within_1e-9_da",
          "maldi_nn_intensity_equal", "mz_equals_tof2mass_outside_hpc")


def maldi_nn_reference(spectrum_py: Path):
    """maldi-nn's `tof2mass` and `from_bruker`, verbatim, in a minimal class (its module needs torch to import)."""
    source = spectrum_py.read_text(encoding="utf-8")
    lines = source.splitlines()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "SpectrumObject")
    methods = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"tof2mass", "from_bruker"}:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
            methods.append(textwrap.dedent("\n".join(lines[start:node.end_lineno])))
    if len(methods) != 2:
        raise SystemExit("maldi-nn's spectrum.py does not define tof2mass and from_bruker as expected")
    body = "def __init__(self, mz, intensity):\n    self.mz, self.intensity = mz, intensity\n\n" + "\n\n".join(methods)
    code = "class Reference:\n" + textwrap.indent(body, "    ")
    namespace = {"np": np}
    exec(compile(code, f"{spectrum_py} (tof2mass, from_bruker)", "exec"), namespace)   # noqa: S102
    return namespace["Reference"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(folder: Path, streamed_fid: bytes, ngb, Reference) -> tuple[dict[str, bool], dict[str, float], str, str]:
    """One spectrum: which checks agree, the deviations, the instrument, and whether `convert` accepts it."""
    acqu = folder / "acqu" if (folder / "acqu").is_file() else folder / "acqus"
    lines = acqu.read_text(encoding="latin-1").splitlines()
    fid = (folder / "fid").read_bytes()
    ours = decode(lines, fid)
    try:
        convert(lines, fid)
        verdict = "accepted"
    except BrukerReadError as exc:
        verdict = exc.reason
    from_zip = decode(lines, streamed_fid)
    ok = {"zip_equals_disk": np.array_equal(ours.mz, from_zip.mz) and np.array_equal(ours.intensity,
                                                                                      from_zip.intensity)}
    worst = {}
    td, delay, dw, ml1, ml2, ml3, bytorda = (_double(lines, p) for p in PARAMETERS)

    dic = ngb.read_jcamp(str(acqu))
    theirs = [float(dic[p]) for p in PARAMETERS]
    _, raw = ngb.read_binary(str(folder / "fid"), shape=(-1,), cplex=False, big=bytorda == 1, isfloat=False)
    ok["parameters_equal"] = theirs == [td, delay, dw, ml1, ml2, ml3, bytorda]
    raw = np.asarray(raw).ravel()[: int(td)].astype(np.float64)
    raw[raw < 0] = 0
    ok["fid_equal"] = np.array_equal(raw, ours.intensity)

    tof = delay + np.arange(ours.mz.size) * dw
    base = tof2mass(tof, ml1, ml2, ml3)
    implied = ml2 + ml3 * base + math.sqrt(1e12 / ml1) * np.sqrt(base)
    worst["inverse_calibration_max_abs_residual"] = float(np.max(np.abs(implied - tof)))
    if ours.calibration == "tof2mass+hpc":
        lo, hi = _double(lines, "HPClBLo"), _double(lines, "HPClBHi")
        outside = (base < lo) | (base > hi)
        ok["mz_equals_tof2mass_outside_hpc"] = np.array_equal(ours.mz[outside], base[outside])
    else:
        ok["mz_equals_tof2mass_outside_hpc"] = np.array_equal(ours.mz, base)

    ref = Reference.from_bruker(str(acqu), str(folder / "fid"))
    ref_mz, ref_intensity = np.asarray(ref.mz, dtype=np.float64), np.asarray(ref.intensity, dtype=np.float64)
    same_shape = ref_mz.shape == base.shape and ref_intensity.shape == ours.intensity.shape
    worst["maldi_nn_max_abs_mz_difference_da"] = float(np.max(np.abs(ref_mz - base))) if same_shape else math.inf
    ok["maldi_nn_mz_within_1e-9_da"] = same_shape and worst["maldi_nn_max_abs_mz_difference_da"] <= 1e-9
    ok["maldi_nn_intensity_equal"] = same_shape and np.array_equal(ref_intensity, ours.intensity)
    return ok, worst, _text(lines, "INSTRUM"), verdict


def main(argv: list[str] | None = None) -> int:
    data_root = Path(load_config()["paths"]["driams_root"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=data_root / "MARISMa_v2.0.0_work")
    parser.add_argument("--references", type=Path, required=True,
                        help="folder holding the unpacked nmrglue 0.12 and maldi-nn 0.2.5 wheels")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "results/metrics/v2.0/reader_reference_check.json")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.references))
    import nmrglue.fileio.bruker as ngb  # noqa: PLC0415 - third-party reference, from --references only

    Reference = maldi_nn_reference(args.references / "maldi_nn/spectrum.py")
    zip_path = json.loads((args.work / "inputs.json").read_text(encoding="utf-8"))["zip"]
    frame = pd.read_csv(args.work / "replicate_folders.csv", dtype=str, keep_default_na=False)
    frame = frame.sort_values("folder").reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    rows = frame.iloc[np.sort(rng.choice(len(frame), size=min(args.n, len(frame)), replace=False))]

    wanted = {f"{f}/{m}" for f in rows["folder"] for m in ("fid", "acqu", "acqus")}
    members = {name: member for name, member in iter_members(zip_path) if name in wanted}
    per_group: dict[str, Counter] = defaultdict(Counter)
    worst: dict[str, float] = defaultdict(float)
    errors: Counter = Counter()
    with open(zip_path, "rb") as fh, tempfile.TemporaryDirectory() as tmp:
        for k, folder_name in enumerate(rows["folder"]):
            folder = Path(tmp) / str(k)
            folder.mkdir()
            for part in ("fid", "acqu", "acqus"):
                if f"{folder_name}/{part}" in members:
                    (folder / part).write_bytes(read_member(fh, members[f"{folder_name}/{part}"]))
            try:
                ok, deviations, instrument, verdict = compare(
                    folder, read_member(fh, members[f"{folder_name}/fid"]), ngb, Reference)
            except Exception as exc:                  # noqa: BLE001 - any failure is counted, never hidden
                errors[type(exc).__name__] += 1
                continue
            finally:
                for f in folder.iterdir():
                    f.unlink()
                folder.rmdir()
            group = f"{instrument} / {'accepted' if verdict == 'accepted' else 'refused: ' + verdict}"
            per_group[group]["spectra"] += 1
            for check, agreed in ok.items():
                per_group[group][check] += int(agreed)
            for key, value in deviations.items():
                worst[key] = max(worst[key], value)

    checked = sum(c["spectra"] for c in per_group.values())
    all_agree = not errors and all(c[check] == c["spectra"] for c in per_group.values() for check in CHECKS)
    result = {
        "what": "Version 2.0 step 3: the Bruker reader's decoding against independent implementations, on real "
                "MARISMa spectra (spectra only; no label, no outcome file, no model).",
        "sample": {"from": "all of the cohort's replicate folders (accepted and refused by the registered checks)",
                   "n": int(len(rows)), "of": int(len(frame)), "seed": args.seed, "compared": checked},
        "references": {
            "nmrglue": {"version": "0.12", "licence": "BSD", "used": "fileio.bruker.read_jcamp, read_binary",
                        "bruker_py_sha256": sha256(args.references / "nmrglue/fileio/bruker.py")},
            "maldi_nn": {"version": "0.2.5", "licence": "MIT", "used": "SpectrumObject.tof2mass, from_bruker",
                         "spectrum_py_sha256": sha256(args.references / "maldi_nn/spectrum.py")},
            "documented_equation": "tof = ML2 + ML3*m + sqrt(1e12/ML1)*sqrt(m) (readBrukerFlexData .tof2mass)",
        },
        "checks": list(CHECKS),
        "by_instrument_and_verdict": {g: dict(c) for g, c in sorted(per_group.items())},
        "worst": dict(worst),
        "errors": dict(errors),
        "all_agree": all_agree,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if all_agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
