# Publishing the frozen model: the owner's decision

**Status, 2026-10-06: prepared, not published.**
- The package is built and verified, and a clean environment installed it from the package.
- On 2026-10-06 it was re-verified against the checksums recorded here, and rebuilt byte-identically from `main`'s
  code (`b4c8d31`).
- Nothing has been uploaded or released, and no release tag exists.
- Publishing needs the owner's decisions below, then the one action at the end.

**Two different releases.**
- **This is the model artifact release** (tag `model-ecoli-ciprofloxacin-v0.4.0`). It carries the unchanged Version 0.4
  model, `v0.4.0-tuned_lightgbm-random-seed42`.
- **The code and research release `v2.0.0` is separate** (`docs/release/v2.0.0/RELEASE_NOTES.md`). It contains no
  model file. Either can be published without the other.

## What is ready

| Asset | Bytes | SHA-256 |
|---|---|---|
| `ecoli_ciprofloxacin-v0.4.0-tuned_lightgbm-random-seed42.zip` | 650,235 | `9980930f35a4f592ab4269aed9811f3e869844ea6a13679a7e60f8fead62618d` |
| `SHA256SUMS.txt` | 126 | `2b127337f6e0ef14481bd83fb35f9195abd82d3dba5f186a2cf0c3672fad667c` |

- **Where.** In `dist/` of the owner's checkout. Git ignores it, and it is not committed.
- **The package** holds five stored members:
  - `best_random.joblib`: 636,912 bytes, SHA-256 `d59d6d7deafa1af464c33ebefc0f8f641a80eb70f7b51039e6c2ede0fd841c8b`,
    byte-identical to the original;
  - its `.sha256` file;
  - `MANIFEST.json`, `MODEL_CARD.md` and `NOTICE.md`. Their review copies are in this folder.
- **It rebuilds byte for byte** with `python scripts/package_model.py`, which refuses any source but the pinned
  original.
- **Release text:** [RELEASE_NOTES.md](RELEASE_NOTES.md). **Asset list:** [ASSETS.json](ASSETS.json).
- **Clean-environment test:** [HANDOFF_CHECK.md](HANDOFF_CHECK.md).

## Rights review

**Provenance.**
- The project trained the model from DRIAMS-A, at commit `3c8245b` (tag `research-archive/v0.4/run`, on GitHub).
  DRIAMS (Weis et al., Dryad, doi:10.5061/dryad.bzkh1899q) is released under CC0 1.0, which places no condition on
  derived models.
- MARISMa (CC BY 4.0) was used only to evaluate the model. None of its data is in the model or the package; the model
  card cites its aggregate results with attribution.

**What the bundle holds.** It was inspected on 2026-10-05: the decompressed object was examined after its hash
matched.
- **It holds:** the fitted pipeline (LightGBM inside scikit-learn's sigmoid calibration) and aggregate metadata
  (parameters, versions, fingerprints, summary metrics). The calibration folds are stored as integer row positions
  into the training part.
- **It does not hold:** any identifier, local path, spectrum, label or per-isolate prediction.

**Third-party software.** The bundle holds objects of scikit-learn (BSD-3-Clause), LightGBM (MIT) and NumPy
(BSD-3-Clause), and none of their code. Users install those libraries themselves.

**The repository's MIT licence covers its code, not automatically the model.** The model is the copyright holder's to
license.

**Nothing here clears the right to distribute the model.** That right cannot be inferred from:
- who owns the GitHub repository;
- the code's licence;
- the public availability of the training data.

Only the copyright holder's confirmation (decision 2) can clear it.

## Decisions only the owner can make (unresolved)

1. **The model's licence.** MIT is **proposed, pending your acceptance**.
   - The package's `NOTICE.md` proposes the MIT License, the same terms as the code. Those terms take effect only
     when the copyright holder publishes the package.
   - `RELEASE_NOTES.md` states MIT as granted on publication, so it is correct only if you accept MIT.
   - Choosing other terms, such as CC BY 4.0, means editing `notice()` in `scripts/package_model.py`, rebuilding, and
     re-running the handoff check. The package's hash changes.
2. **The authority to license it.**
   - Confirm that you alone hold the rights.
   - The thesis chapter leaves "Affiliation" and "Supervisor" as "[to be completed]". If this work falls under a
     university's or employer's IP policy, or under a thesis agreement, confirm that it allows public release under
     the chosen terms.
   - This repository cannot settle that.
3. **Whether and when to publish.**

## Recommended destination

**A GitHub Release of this public repository** (https://github.com/Sickwoman/antibiotic-resistance-ai), with tag
`model-ecoli-ciprofloxacin-v0.4.0`.
- **Target commit:** `b4c8d31138f1bd9cec3e98ef394fe6aa40a511f8`, PR #36's merge on `main`. Its tree contains the
  installer, the pinned hash and the first version of this record.
- **Not latest.** Mark it not-latest, so that the code release (`v1.4.0`, later `v2.0.0`) stays the latest.
- **Assets and text.** Attach the two assets above, and use `RELEASE_NOTES.md` as the text.

Why there:
- The code and the pinned hash live in the same place, and GitHub serves release assets unchanged.
- No new account or service is needed.
- The installer stays local-only: a user downloads the asset, then runs
  `python scripts/install_model.py <zip>`, which checks the pin before writing anything.

## The exact action that needs your approval

Once decisions 1 and 2 are confirmed, from the owner's checkout:

```powershell
git -C . cat-file -e b4c8d31138f1bd9cec3e98ef394fe6aa40a511f8   # the target exists locally (fetch origin first if not)
.\.venv\Scripts\python.exe scripts\package_model.py         # must print SHA-256 9980930f35a4f592…62618d again
gh release create model-ecoli-ciprofloxacin-v0.4.0 `
  dist\ecoli_ciprofloxacin-v0.4.0-tuned_lightgbm-random-seed42.zip dist\SHA256SUMS.txt `
  --target b4c8d31138f1bd9cec3e98ef394fe6aa40a511f8 --latest=false `
  --title "Frozen model: E. coli ciprofloxacin v0.4.0 (research use only)" `
  --notes-file docs\release\model-v0.4.0\RELEASE_NOTES.md
```

**If the rebuild prints another hash, stop:** the package changed, and the evidence above no longer applies to it.

**Once it is published,** a follow-up documentation change can name the release URL in `demo/README.md`. The install
command does not change.
