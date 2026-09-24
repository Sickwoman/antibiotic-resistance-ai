"""Version 0.8, Phase 3: create and freeze the pre-registered DRIAMS-C partition. Nothing is scored.

    python scripts/build_adaptation_partition.py

This script partitions and validates. It does not fit, calibrate, score or evaluate anything, and it never
loads the feature matrix -- only the metadata is needed to draw the partition, so the protected rows'
spectra are not even read here.

The partition is fixed by docs/v0.8_adaptive_plan.md and protocol amendment 4: 70 % adaptation / 30 %
protected evaluation, seed 42, whole patient groups (DRIAMS-C carries no patient identifier, so each
spectrum is its own group). Everything it writes is a record of that one draw, including two independent
fingerprints, so a later run can prove it reproduced the same partition rather than a new one.

Writes (committed; aggregate counts and fingerprints only, no identifiers):
    results/metrics/v0.8/<cohort>/partition.json
Writes (git-ignored; the row indices themselves):
    models/v0.8/<cohort>/partition_rows.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adaptation import (  # noqa: E402
    AdaptationError,
    assert_disjoint,
    partition_fingerprint,
    split_adaptation_and_holdout,
)
from src.data_loader import DataError  # noqa: E402
from src.dataset import load_dataset  # noqa: E402
from src.utils import ConfigError, get_logger, git_commit, load_config, project_path, show_path  # noqa: E402

log = get_logger("partition")

STAGE = "v0.8-partition"


def class_counts(y: np.ndarray, rows: np.ndarray) -> dict[str, int]:
    part = y[np.asarray(rows, dtype=np.int64)]
    return {"n": int(part.size), "resistant": int((part == 1).sum()), "susceptible": int((part == 0).sum())}


def draw(meta: Any, rows: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return split_adaptation_and_holdout(meta, rows, fraction, seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Version 0.8 Phase 3: freeze the DRIAMS-C partition.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        ad = config["adaptation"]
        cohort = str(ad["cohort"]["name"])
        data_dir = project_path(config["dataset"]["output_dir"]) / cohort

        # The metadata alone decides the partition. verify_x=False keeps this from touching X.npy at all.
        _, meta, summary = load_dataset(data_dir, verify_x=False)
        y = meta["label"].to_numpy().astype(np.int64)
        rows = np.arange(len(meta), dtype=np.int64)

        wanted = str(ad["cohort"]["require_feature_fingerprint"])
        if summary["feature_fingerprint"] != wanted:
            raise ConfigError(
                f"{cohort} has feature fingerprint {summary['feature_fingerprint']}, but the protocol "
                f"requires {wanted}: it was built with other preprocessing and is not comparable with the "
                "model being adapted. Stop condition 9.")

        seed = int(ad["partition"]["seed"])
        fraction = float(ad["partition"]["adaptation_fraction"])
        log.info("%s: %d rows, %d resistant, %d susceptible; drawing %.0f%% / %.0f%% at seed %d",
                 cohort, rows.size, int((y == 1).sum()), int((y == 0).sum()),
                 100 * fraction, 100 * (1 - fraction), seed)

        adaptation, holdout = draw(meta, rows, fraction, seed)

        # --- integrity, all of it fatal ---------------------------------------------------------------
        if adaptation.size + holdout.size != rows.size:
            raise AdaptationError(f"{adaptation.size} + {holdout.size} != {rows.size}: rows were lost.")
        if not np.array_equal(np.union1d(adaptation, holdout), rows):
            raise AdaptationError("The two parts do not cover every row exactly once.")
        for name, part in (("adaptation", adaptation), ("evaluation", holdout)):
            if np.unique(part).size != part.size:
                raise AdaptationError(f"The {name} part contains duplicate rows.")
        assert_disjoint(meta, adaptation, holdout)

        counts = {"cohort": class_counts(y, rows), "adaptation": class_counts(y, adaptation),
                  "evaluation": class_counts(y, holdout)}
        floor = int(ad["eligibility"]["e2_min_per_class_holdout"])
        for cls in ("resistant", "susceptible"):
            got = counts["evaluation"][cls]
            if got < floor:
                raise AdaptationError(f"The evaluation part holds {got} {cls} rows, below the "
                                      f"pre-registered minimum of {floor} (stop condition 3).")

        fp_adapt = partition_fingerprint(meta, adaptation)
        fp_hold = partition_fingerprint(meta, holdout)

        # --- the reproducibility check the plan requires: draw it again, independently ----------------
        again_adapt, again_hold = draw(meta, rows, fraction, seed)
        repeat = {"adaptation": partition_fingerprint(meta, again_adapt),
                  "evaluation": partition_fingerprint(meta, again_hold)}
        identical = bool(repeat["adaptation"] == fp_adapt and repeat["evaluation"] == fp_hold
                         and np.array_equal(again_adapt, adaptation) and np.array_equal(again_hold, holdout))
        if not identical:
            raise AdaptationError("A second draw with the same cohort, method, seed, grouping rule and "
                                  "ratio produced a different partition. The partition is not "
                                  "reproducible; refusing to freeze it.")
        log.info("reproducibility check: a second independent draw gives the same partition "
                 "(adaptation %s, evaluation %s)", fp_adapt, fp_hold)

        record: dict[str, Any] = {
            "stage": STAGE, "status": "frozen", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": time.strftime("%Z%z"), "git_commit": git_commit(),
            "cohort": cohort, "rows_fingerprint": summary["row_fingerprint"],
            "x_sha256": summary["x_sha256"], "feature_fingerprint": summary["feature_fingerprint"],
            "archive_checksum": str(ad["cohort"]["archive_checksum"]),
            "seed": seed, "adaptation_fraction": fraction,
            "holdout_fraction": float(ad["partition"]["holdout_fraction"]),
            "method": str(ad["partition"]["method"]), "unit": str(ad["partition"]["unit"]),
            "counts": counts,
            "partition_fingerprint": {"adaptation": fp_adapt, "evaluation": fp_hold},
            "reproducibility_check": {"repeated": True, "identical": identical, "fingerprints": repeat},
            "group_limitation": (
                "DRIAMS-C provides no patient identifier in the available metadata. Each spectrum is "
                "therefore treated as an independent group for partitioning. Consequently, uncertainty "
                "intervals for C are sample-level and may be narrower than appropriate if multiple spectra "
                "originate from the same unobserved patient."),
            "evaluation_scored": False,
            "evaluation_used_for": {"training": False, "calibration": False, "tuning": False,
                                    "model_selection": False, "adaptation_size_selection": False},
        }

        report_dir = project_path(ad["report_dir"]) / cohort
        model_dir = project_path(ad["model_dir"]) / cohort
        report_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "partition.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        np.savez_compressed(model_dir / "partition_rows.npz", adaptation=adaptation, evaluation=holdout)

        log.info("adaptation %d rows (%d resistant, %d susceptible); evaluation %d rows (%d resistant, "
                 "%d susceptible)", counts["adaptation"]["n"], counts["adaptation"]["resistant"],
                 counts["adaptation"]["susceptible"], counts["evaluation"]["n"],
                 counts["evaluation"]["resistant"], counts["evaluation"]["susceptible"])
        log.info("partition frozen in %s; nothing was scored", show_path(report_dir / "partition.json"))
        return 0
    except (DataError, ConfigError, AdaptationError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
