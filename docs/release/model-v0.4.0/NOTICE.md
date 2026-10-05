# Notice: `ecoli_ciprofloxacin-v0.4.0-tuned_lightgbm-random-seed42`

## Redistribution terms (proposed)
These are the terms the project proposes to its copyright holder. **They take effect only when the copyright holder
publishes this package.** Until then it is not licensed for redistribution.

Proposed: this package (the model bundle and these documents) under the MIT License, the same terms as the
repository's code:

```
MIT License

Copyright (c) 2026 Moksh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
`d59d6d7deafa1af464c33ebefc0f8f641a80eb70f7b51039e6c2ede0fd841c8b`, the value pinned in the repository. The `.sha256` file in this package can be replaced
together with the bundle, so it proves nothing on its own. The repository's installer checks the pin.

## No warranty; research use
Provided as is, without warranty: see the licence text above. It is a research artifact, and not a medical device.
