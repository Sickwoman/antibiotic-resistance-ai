# Frozen model: E. coli ciprofloxacin `v0.4.0-tuned_lightgbm-random-seed42` (research use only)

The exact frozen model behind this repository's Version 0.4–2.0 results and its research demo, packaged for local
installation.
- **Byte-identical original.** It was not retrained, converted or re-serialised.
- **Not a diagnostic.** It is not for clinical use and never recommends a treatment.

| Asset | Bytes | SHA-256 |
|---|---|---|
| `ecoli_ciprofloxacin-v0.4.0-tuned_lightgbm-random-seed42.zip` | 650,235 | `9980930f35a4f592ab4269aed9811f3e869844ea6a13679a7e60f8fead62618d` |
| `SHA256SUMS.txt` | 126 | `2b127337f6e0ef14481bd83fb35f9195abd82d3dba5f186a2cf0c3672fad667c` |

The bundle inside is `best_random.joblib`, 636,912 bytes, SHA-256 `d59d6d7deafa1af464c33ebefc0f8f641a80eb70f7b51039e6c2ede0fd841c8b`. The same value
is pinned in the repository's `src/model_package.py` and `docs/reproduction_guide.md`.

## Install
From a clone of the repository, with the environment of `requirements-lock.txt`:

```powershell
.\.venv\Scripts\python.exe scripts\install_model.py <path to ecoli_ciprofloxacin-v0.4.0-tuned_lightgbm-random-seed42.zip>
.\.venv\Scripts\python.exe -m demo        # http://127.0.0.1:8050/
```

The installer checks the bundle against the pinned hash before writing anything, refuses a mismatch, and never
replaces a different model. Load the bundle only after that check: a joblib file is a pickle.

## What it can and cannot claim
On MARISMa it showed above-chance ranking under the isolate-independence assumption: AUROC
0.772, descriptive 95% interval 0.744–0.798. With equal weight:
- sensitivity 0.897 and specificity 0.393 at the frozen
  threshold;
- resistance was under-predicted (calibration intercept 0.642);
- its confidence zone missed its target (NPV 0.907; 18 of 193 R or I);
- there was no patient linkage.

Clinical usefulness is unproven. See `MODEL_CARD.md` and `NOTICE.md` in the package.

## Terms
The MIT License, granted by the copyright holder by publishing this release (see `NOTICE.md` in the package).
