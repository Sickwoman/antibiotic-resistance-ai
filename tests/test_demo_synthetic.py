"""The research demo's synthetic spectra (demo/synthetic.py): reproducible, labelled as synthetic, never labelled
as resistant or susceptible, and accepted by the project's real reader and frozen preprocessing.

No model is loaded here: these checks hold before anything is scored.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from demo import synthetic
from src.data_loader import read_raw_spectrum
from src.preprocessing import PreprocessingConfig, preprocess_file


def test_the_examples_are_the_first_three_draws_of_seed_42_and_reproducible():
    first, second = synthetic.generate(), synthetic.generate()
    assert [e.id for e in first] == ["synthetic-1", "synthetic-2", "synthetic-3"]
    assert synthetic.SEED == 42 and synthetic.N_EXAMPLES == 3
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.intensity, b.intensity) and np.array_equal(a.mz, b.mz)
        assert a.to_text() == b.to_text()
    assert not np.array_equal(first[0].intensity, first[1].intensity)
    # the n-th example does not depend on how many are drawn after it
    assert np.array_equal(synthetic.generate(n=1)[0].intensity, first[0].intensity)


def test_an_example_carries_no_label():
    names = {f.name for f in dataclasses.fields(synthetic.Example)}
    assert names == {"id", "title", "mz", "intensity"}
    text = synthetic.generate()[0].to_text().lower()
    assert "resistant" not in text and "susceptible\n" not in text
    assert text.startswith(synthetic.SYNTHETIC_MARK.lower())


def test_the_real_reader_and_the_frozen_preprocessing_accept_every_example(tmp_path):
    cfg = PreprocessingConfig()
    assert cfg.fingerprint() == "347cbd6d5d956ff9"
    for example in synthetic.generate():
        path = tmp_path / f"{example.id}.txt"
        path.write_text(example.to_text(), encoding="utf-8")
        assert path.stat().st_size < 4_000_000                       # the served API's upload limit
        values = read_raw_spectrum(path, max_points=200_000)
        assert values.shape == (synthetic.N_POINTS, 2)
        features, _ = preprocess_file(path, cfg)
        assert features.shape == (6000,) and np.isfinite(features).all() and features.any()


def test_the_plot_series_keeps_the_peaks():
    example = synthetic.generate()[0]
    series = synthetic.plot_points(example)
    assert len(series["mz"]) == len(series["intensity"]) <= 1_200
    assert max(series["intensity"]) == float(example.intensity.max())
    assert series["mz"] == sorted(series["mz"])
