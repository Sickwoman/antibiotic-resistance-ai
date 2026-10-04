# Version 2.0, steps 2–3, run 2: the prepared cohort under amendments D and E

**Status: ready for the owner's step 3 review.** It is not accepted. Nothing has been scored, no interpretation
category has been inspected, and no prevalence has been computed.

This run applies amendments D and E of `docs/v2.0_marisma_plan.md`:
- **Amendment D** holds the owner's decisions. It was recorded at `6a5ea90` and pinned at `8bea756`.
- **Amendment E** holds the source mapping. It was recorded at `83ea59e` and pinned at `26e1517`.

Both were recorded before this run, which used commit `47c0e92`. The numbers are in `step3_run2.json`. Run 1's
historical record is in `run1_blocked_2026-10-03/`.

**Cohort label:** "identifiable screening sources excluded (none was identifiable)".

## Sequential accounting

| Step | Rule | Excluded | Remaining |
|---|---|---|---|
| Archive | identifiers of all isolate folders | | 202,596 |
| Species | MARISMa's species field, Escherichia/Coli | | 16,985 |
| Identity (D5) | identifier filed under more than one genus or species (none has an `AMR.csv` record) | 10 | 16,975 |
| Year | year folders 2018–2024 | 0 | 16,975 |
| Source (E) | identifiable screening or colonisation sources | 0 | 16,975 |
| Replicate folder | at least one replicate folder | 0 | 16,975 |
| Reader (D2) | a replicate passing the reader's checks, in the A6.2 order | 1,031 | 15,944 |
| `AMR.csv` record | matched by identifier: a record, not yet a usable interpretation | 14,779 | 1,165 |

**Usable interpretation** (amendment D6: the restricted schema counts only). These count isolates with a non-missing
interpretation, out of the 1,165 matched.

| Antibiotic | Isolates | Availability gate (≥ 100) |
|---|---|---|
| Ciprofloxacin | 1,146 | met |
| Ceftriaxone | 0 | **not met: dropped** |

- **Ceftriaxone is dropped under amendment A6.1.** It enters the fixed two-hypothesis Holm family with p = 1.
- **Ciprofloxacin proceeds alone.** Nothing is substituted for ceftriaxone.
- **The study stops only if both antibiotics are dropped.**

## The evaluation population (D6)

- **Records exist for 2024 only.** All 1,165 matched isolates are from the 2024 year folder and the MBT-WIN10
  instrument. MARISMa's descriptor says why: "For other species, AMR data is currently limited to the year 2024,
  following a change in the hospital's information system that enabled broader extraction of AST results."
- **What results may claim.** Any result applies only to the eligible, tested 2024 subset. It does not apply across
  2018–2024, across instruments, or to drift over time within MARISMa.
- **Unavailable analyses:**
  - endpoint 6's period analyses (2018–2019, 2020–2021, 2022–2024);
  - every ceftriaxone endpoint.
- **The all-sources sensitivity analysis is identical** to the primary analysis, because no isolate was excluded by
  source.

## Sources (E)

Of the 16,975 cohort isolates, by source:
- **Clinical specimen type:** 1,030 isolates, in 29 named categories.
- **Ambiguous:** 142 isolates.
  - 77 are in 3 named categories:
    - `EXUDADO RECTAL`: 31;
    - `ASPIRADO TRAQUEAL`: 36;
    - `JUGO GASTRICO`: 10.
  - 65 are in the 39 suppressed categories.
- **Unknown, no `AMR.csv` record:** 15,803.

No category was identifiable as screening. The rectal-swab risk (E3) stays open, and changing that category's class
needs a dated owner decision before scoring.

## The aggregate pause (A6.4, D3)

- **The figure.** 1,031 of 16,975 isolates (6.07 %) have no passing replicate. That is the figure D3 recorded.
- **Every exclusion reason is an investigated one.** Over failed attempts of excluded isolates, the reasons were:
  - a feature bin without data (1,166);
  - a sampling interval other than 2 ns (1);
  - a start below 1,960 Da (1).
- **The owner's disposition applies.** Without it, processing would have stopped.

## Reader and feature checks

**The reader.** The 15,944 selected spectra were checked as follows.
- **Sampling interval.** All have a 2 ns interval, read from `##$DW` in their `acqu` files.
- **Start of the window.** Their first point lies between 1,992.6 and 2,002.9 Da, so it is at or above 1,960 Da and
  below 2,003 Da. Every feature bin holds an acquired point.
- **Calibration.** It is `tof2mass` throughout, with no HPC, CTOF2 or LIFT spectrum.

**Decoding.** The reader's decoding was checked earlier against nmrglue and maldi-nn on 400 spectra:
`run1_blocked_2026-10-03/reader_reference_check.json`.

**The features.**
- **Shape.** 15,944 spectra × 6,000 features, float32, all finite and non-negative.
- **Fingerprint.** It is `347cbd6d5d956ff9`, from both bundles' stored settings.
- **Non-zero bins.** The median is 5,994.
- **Where they are kept.** The features and their hashes are outside the repository. No model was invoked.

**Spectra-only checks.** These are descriptive, not a gate.
- **Correlation with the mean DRIAMS-A *E. coli* training spectrum.** The median is 0.715 (5–95 %: 0.393–0.886).
  DRIAMS-A's own validation spectra give 0.838 (0.584–0.923). By year the medians are 0.62–0.66 for 2018–2019 and
  0.73–0.76 for 2020–2024; 2024 is 0.726.
- **Alignment of the mean spectra.** No bin lag, and a median peak offset of 0.17 Da.

## Limitations accepted by the owner (D2), and still unresolved

- **Residual edge effects.** The first 10–16 feature bins differ for essentially every MARISMa spectrum, and the last
  21–41 differ for spectra ending near 20,000 Da (FLEX-PC, 2018–2019).
- **Sparse edge bins.** A covered bin may hold a single point.
- **Window-dependent normalisation.** It is about 1 % for MBT-WIN10, which acquires to about 21,000 Da.
- **No equivalence.** The features are not shown to be equivalent to DRIAMS features. Feature importance is not used
  as evidence that these differences are harmless.
