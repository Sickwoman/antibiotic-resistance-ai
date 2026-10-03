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

The spectra checked are a seeded sample of the spectra selected for the cohort. Their folders are extracted one at a
time to a temporary folder outside the repository and deleted afterwards. The output holds aggregate counts and
deviations only: no identifier and no path.

    python scripts/v20_reader_reference_check.py --work C:/DRIAMS/MARISMa_v2.0.0_work --references <folder>
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
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bruker import BrukerReadError, _double, convert, read_spectrum, tof2mass  # noqa: E402
from src.zip_index import iter_members, read_member  # noqa: E402

PARAMETERS = ("TD", "DELAY", "DW", "ML1", "ML2", "ML3", "BYTORDA")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="the step-3 work folder (outside the repository)")
    parser.add_argument("--references", type=Path, required=True,
                        help="folder holding the unpacked nmrglue 0.12 and maldi-nn 0.2.5 wheels")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "results/metrics/v2.0/reader_reference_check.json")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.references))
    import nmrglue.fileio.bruker as ngb  # noqa: PLC0415 - third-party reference, from --references only

    Reference = maldi_nn_reference(args.references / "maldi_nn/spectrum.py")
    provenance = json.loads((args.work / "inputs.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(args.work / "selection.csv", dtype=str, keep_default_na=False)
    selected = selected[selected["status"] == "selected"].sort_values("isolate").reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    rows = selected.iloc[np.sort(rng.choice(len(selected), size=min(args.n, len(selected)), replace=False))]

    tally: dict[str, int] = {"checked": 0, "reader_refused": 0, "zip_equals_disk": 0, "nmrglue_error": 0,
                             "parameters_equal": 0, "fid_equal": 0, "maldi_nn_error": 0, "maldi_nn_compared": 0,
                             "maldi_nn_mz_within_1e-9_da": 0, "maldi_nn_intensity_equal": 0, "hpc_applied": 0,
                             "mz_equals_tof2mass_outside_hpc": 0}
    worst = {"maldi_nn_max_abs_mz_difference_da": 0.0, "inverse_calibration_max_abs_residual": 0.0}
    wanted = {f"{f}/{m}" for f in rows["spectrum_folder"] for m in ("fid", "acqu", "acqus")}
    members = {name: member for name, member in iter_members(provenance["zip"]) if name in wanted}
    with open(provenance["zip"], "rb") as fh, tempfile.TemporaryDirectory() as tmp:
        for k, row in enumerate(rows.itertuples(index=False)):
            folder = Path(tmp) / str(k)
            folder.mkdir()
            for part in ("fid", "acqu", "acqus"):
                name = f"{row.spectrum_folder}/{part}"
                if name in members:
                    (folder / part).write_bytes(read_member(fh, members[name]))
            tally["checked"] += 1
            try:
                ours = read_spectrum(folder)       # from the extracted files, as readBrukerFlexData reads a folder
                acqu_name = "acqu" if (folder / "acqu").is_file() else "acqus"
                from_zip = convert((folder / acqu_name).read_text(encoding="latin-1").splitlines(),
                                   read_member(fh, members[f"{row.spectrum_folder}/fid"]))
            except BrukerReadError:
                tally["reader_refused"] += 1          # a selected spectrum should never be refused here
                continue
            tally["zip_equals_disk"] += int(np.array_equal(ours.mz, from_zip.mz)
                                             and np.array_equal(ours.intensity, from_zip.intensity))
            acqu = folder / "acqu" if (folder / "acqu").is_file() else folder / "acqus"
            lines = acqu.read_text(encoding="latin-1").splitlines()
            td, delay, dw, ml1, ml2, ml3, bytorda = (_double(lines, p) for p in PARAMETERS)

            # nmrglue: parameter parsing, byte order and integer decoding
            try:
                dic = ngb.read_jcamp(str(acqu))
                theirs = [float(dic[p]) for p in PARAMETERS]
                _, raw = ngb.read_binary(str(folder / "fid"), shape=(-1,), cplex=False, big=bytorda == 1,
                                         isfloat=False)
            except Exception:                         # noqa: BLE001 - any failure of the reference is counted
                tally["nmrglue_error"] += 1
            else:
                tally["parameters_equal"] += int(theirs == [td, delay, dw, ml1, ml2, ml3, bytorda])
                raw = np.asarray(raw).ravel()[: int(td)].astype(np.float64)
                raw[raw < 0] = 0
                tally["fid_equal"] += int(np.array_equal(raw, ours.intensity))

            # the documented equation, and maldi-nn's port of readBrukerFlexData (which has no HPC)
            tof = delay + np.arange(ours.mz.size) * dw
            base = tof2mass(tof, ml1, ml2, ml3)
            implied = ml2 + ml3 * base + math.sqrt(1e12 / ml1) * np.sqrt(base)
            worst["inverse_calibration_max_abs_residual"] = max(worst["inverse_calibration_max_abs_residual"],
                                                                float(np.max(np.abs(implied - tof))))
            if ours.calibration == "tof2mass+hpc":
                tally["hpc_applied"] += 1
                lo, hi = _double(lines, "HPClBLo"), _double(lines, "HPClBHi")
                outside = (base < lo) | (base > hi)
                tally["mz_equals_tof2mass_outside_hpc"] += int(np.array_equal(ours.mz[outside], base[outside]))
            else:
                tally["mz_equals_tof2mass_outside_hpc"] += int(np.array_equal(ours.mz, base))
            try:
                ref = Reference.from_bruker(str(acqu), str(folder / "fid"))
                ref_mz = np.asarray(ref.mz, dtype=np.float64)
                ref_intensity = np.asarray(ref.intensity, dtype=np.float64)
            except Exception:                         # noqa: BLE001 - any failure of the reference is counted
                tally["maldi_nn_error"] += 1
            else:
                tally["maldi_nn_compared"] += 1
                if ref_mz.shape == base.shape and ref_intensity.shape == ours.intensity.shape:
                    diff = float(np.max(np.abs(ref_mz - base)))
                    worst["maldi_nn_max_abs_mz_difference_da"] = max(worst["maldi_nn_max_abs_mz_difference_da"],
                                                                     diff)
                    tally["maldi_nn_mz_within_1e-9_da"] += int(diff <= 1e-9)
                    tally["maldi_nn_intensity_equal"] += int(np.array_equal(ref_intensity, ours.intensity))
                else:
                    worst["maldi_nn_max_abs_mz_difference_da"] = float("inf")
            for f in folder.iterdir():
                f.unlink()
            folder.rmdir()

    result = {
        "what": "Version 2.0 step 3: the Bruker reader against independent implementations, on real MARISMa "
                "spectra (spectra only; no label, no outcome file, no model).",
        "sample": {"from": "spectra selected for the cohort", "n": int(tally["checked"]), "seed": args.seed,
                   "of": int(len(selected))},
        "references": {
            "nmrglue": {"version": "0.12", "licence": "BSD", "used": "fileio.bruker.read_jcamp, read_binary",
                        "bruker_py_sha256": sha256(args.references / "nmrglue/fileio/bruker.py")},
            "maldi_nn": {"version": "0.2.5", "licence": "MIT", "used": "SpectrumObject.tof2mass, from_bruker",
                         "spectrum_py_sha256": sha256(args.references / "maldi_nn/spectrum.py")},
            "documented_equation": "tof = ML2 + ML3*m + sqrt(1e12/ML1)*sqrt(m) (readBrukerFlexData .tof2mass)",
        },
        "counts": tally,
        "worst": worst,
    }
    n = tally["checked"]
    ok = (tally["reader_refused"] == 0 and tally["nmrglue_error"] == 0 and tally["maldi_nn_error"] == 0
          and n == tally["zip_equals_disk"] == tally["parameters_equal"] == tally["fid_equal"]
          == tally["maldi_nn_mz_within_1e-9_da"] == tally["maldi_nn_intensity_equal"]
          == tally["mz_equals_tof2mass_outside_hpc"])
    result["all_agree"] = ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
