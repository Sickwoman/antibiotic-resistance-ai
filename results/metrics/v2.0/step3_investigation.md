# Version 2.0, step 3 blockers: label-blind investigation (2026-10-04)

This note covers the work done under plan amendment C: the sample source, species identity and the coverage check.
Everything here comes from spectra, archive metadata and count-only outputs of the restricted `AMR.csv` reader. No
label, outcome count, prediction or model output was produced, and the coverage gate stays as approved. Each finding
links to the file that holds its numbers.

## Matching `AMR.csv` to the archive (`matching_diagnostic.json`)

- **`AMR.csv` has 36,141 rows and 29,679 distinct identifiers.** Every identifier matches an archive isolate folder.
- **Labelled E. coli isolates exist for 2024 only.** E. coli isolates match only in 2024: 1,172 of 1,607. From 2018
  to 2023, none of 15,378 matches, although other species do match in those years.
- **This is not a key-format problem.** Upper- or lower-casing the archive identifiers adds no match, and every
  `AMR.csv` identifier is already matched.

## Sample source (`source_categories.json`; amendment C4)

- **The categories.** Among the 1,172 matched isolates, `Sample` has 71 categories, with none missing and none
  conflicting.
  - 32 categories, holding 1,107 isolates, meet the size and content rules.
  - 39 categories hold fewer than 5 isolates each, 65 isolates in all (5.55 %).
  - No category fails a content rule.
- **The free-text rule fixed in C4 fired.** Suppressed categories hold more than 5 % of matched isolates. So no
  category is named and no mapping is applied.
- **What tripped it.** A long tail of rare categories, not free text or identifying content.

## Species identity (`species_identity.json`; amendment C6)

- **The 10 identifiers.** Each is filed under E. coli and under another genus or species:
  - **Five** pair E. coli with *Campylobacter coli*, in the same year, acquired 0–4 days apart.
  - **Five** pair a late-December E. coli folder with a different organism in the next year folder, in January or
    February.
- **All 10 hold two different organisms.** The highest correlation between the two folders' spectra is 0.04–0.22.
  Two replicates of one E. coli isolate have a median of 0.90, with 95 % above 0.45. None of the 10 looks like one
  organism identified twice.
- **None of them has an `AMR.csv` record.** Because `AMR.csv` links by identifier alone, such an identifier could link
  an isolate to the other organism's record.

## Coverage (`coverage_investigation.json`, `coverage_matching.json`, `../../plots/v2.0/coverage_edge_effects.png`; amendment C7)

### What the frozen pipeline does at an edge

These facts come from the code and are confirmed by the experiments below.
- **Smoothing.** Savitzky–Golay estimates the first and last 10 points from a one-sided fit.
- **Baseline.** SNIP never clips points within its window of an edge.
- **Normalisation.** The total ion current is the area under the whole acquired spectrum, computed before trimming
  to 2,000–20,000 Da.
- **Binning.** A bin is the *sum* of the points in it. An empty bin is 0, a partly covered bin is low, and denser
  sampling makes every bin larger.

### Sampling

- **Spacing.** Points are 0.417–0.419 Da apart at 2,000 Da. At 20,000 Da they are 1.32–1.37 Da apart.
- **Points per bin.** A covered first bin holds about 7 points, and the last bin 2–3.

### The 3 Da statement, checked on every spectrum's own grid

All 21,199 cohort replicate folders were checked.
- **The equivalence holds exactly.** A spectrum has no empty feature bin if and only if its first point is below
  2,003 Da and its last point at or above 19,997 Da. 20,008 spectra meet both conditions, and none of them has an
  empty bin.
- **"Data" can mean a single point.** A covered bin may hold one point instead of seven.

### Controlled truncation

The test spectra were synthetic, 150 DRIAMS-A raw and 150 MARISMa MBT-WIN10. The fully covering spectra were cut, and
their frozen features compared before and after.
- **Endpoint rounding does not matter.** Starting at 1,999.9 Da (passes the approved check) or at 2,000.05 Da (fails)
  gives the same features. So does ending at 20,000.5 Da (passes) or at 19,999.9 Da (fails).
- **Missing margin matters, wherever the boundary falls.** Without margin below 2,000 Da, the first 10–16 bins change
  by more than 1 % of the mean bin. Without margin above 20,000 Da, the last 21–41 bins change. DRIAMS-A's own end, at
  about 20,130 Da, changes nothing.
- **Interior bins are not reached.** Bins 200–5,800 never change beyond float32 rounding (about 1e-5 of the mean
  bin).
- **The global scale shifts.** The total ion current follows the whole acquired range. A MARISMa MBT-WIN10 spectrum
  (acquired to about 21,000 Da) is scaled about 1 % differently from the same spectrum in a DRIAMS-A-like window.

### Non-standard acquisitions

There are 3 in the cohort, all FLEX-PC from 2018–2019, and none has an `AMR.csv` record.
- **Two use a 1 ns sampling interval.** Their features are 1.75× and 2.17× the standard median.
- **One starts at about 0 Da.** Its features are 0.57× the median. It passes the approved check and was the spectrum
  selected for its isolate in run 1.

### Isolates excluded

| Rule | Overall | 2018 FLEX-PC | 2019 FLEX-PC | 2020–2023 MBT-WIN10 | 2024 MBT-WIN10 | 2024 isolates matched to `AMR.csv` and kept |
|---|---|---|---|---|---|---|
| Approved: first ≤ 2,000 Da and last ≥ 20,000 Da | 59.96 % | 94.9 % | 96.1 % | 30.2–47.7 % | 18.4 % | 961 of 1,172 |
| Every feature bin has acquired data | 6.06 % | 12.9 % | 13.1 % | 0.1–1.2 % | 0.8 % | 1,165 of 1,172 |
| The same, with the training spectra's acquisition settings (2 ns; start ≥ 1,960 Da) | 6.07 % | 13.0 % | 13.1 % | 0.1–1.2 % | 0.8 % | 1,165 of 1,172 |

## Not resolved by any coverage rule

Removing any of these would need a change to the frozen pipeline, which is not allowed.
- **The low edge.** The first 10–16 feature bins differ for essentially every MARISMa spectrum, because all of them
  start at about 2,000 Da. DRIAMS-A spectra start at about 1,960 Da.
- **The high edge.** The last 21–41 bins differ for FLEX-PC spectra, which end at about 20,000 Da. None of them has an
  `AMR.csv` record.
- **Partial support** in edge bins.
- **About 1 % global scaling** of MBT-WIN10 features, from the longer acquisition window.

Their effect on the frozen models' outputs cannot be measured without invoking the models, which is not authorised.
For the ciprofloxacin model, Version 0.6's committed SHAP outputs give 0.13 % of total importance to the first 2 and
last 6 bins, and 0.54 % to bins 0–14 and 5,950–5,999. No comparable evidence exists for the ceftriaxone model.
