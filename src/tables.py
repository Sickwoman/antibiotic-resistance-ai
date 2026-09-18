"""Small helpers for building Markdown result tables from saved reports.

The evaluation protocol requires that result tables are generated from the saved files, never typed by
hand; these helpers are shared by the Version 0.3 and Version 0.4 table scripts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

MODEL_NAMES = {"prevalence": "Prevalence only", "v0.3_prevalence": "Prevalence only (Version 0.3 run)",
               "logistic_regression": "Logistic regression", "random_forest": "Random forest",
               "lightgbm": "LightGBM", "svm_rbf": "SVM (RBF)",
               "tuned_logistic_regression": "Logistic regression (tuned)",
               "tuned_random_forest": "Random forest (tuned)", "tuned_lightgbm": "LightGBM (tuned)",
               "tuned_svm_rbf": "SVM (RBF, tuned)", "v0.3_random_forest": "Random forest (Version 0.3)"}


def model_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def md_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table from a list of dicts (the first row's keys are the columns)."""
    if not rows:
        return ""
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def with_ci(entry: dict[str, float], digits: int = 3) -> str:
    return f"{entry['estimate']:.{digits}f} [{entry['low']:.{digits}f}, {entry['high']:.{digits}f}]"


def signed_ci(entry: dict[str, float], digits: int = 3) -> str:
    return f"{entry['estimate']:+.{digits}f} [{entry['low']:+.{digits}f}, {entry['high']:+.{digits}f}]"


def number(value: Any, digits: int = 3) -> str:
    """Format a number for a table; missing values become a dash."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else f"{value:,}"
    return str(value)


def seconds(value: Any) -> str:
    if value is None or pd.isna(value):
        return "–"
    value = float(value)
    return f"{value:.1f} s" if value < 90 else f"{value / 60:.0f} min"
