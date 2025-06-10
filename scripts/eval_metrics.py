"""
eval_metrics.py
===============

Utility helpers for summarizing TextGrad run logs.  Designed to be
lightweight so they can run inside CI without extra dependencies
beyond pandas + scikit-learn.

Current functionality
---------------------
* `report()` – prints classification report & writes it to disk.

Planned extensions
------------------
* SQLite logger for experiment tracking.
* Custom cost-weighted metrics (e.g., FN penalties).
"""

from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.metrics import classification_report


def _load_column(df: pd.DataFrame) -> pd.Series:
    """Return the final column (assumes target label is last)."""
    return df.iloc[:, -1]


def report(run_log: Union[str, Path], truth_csv: Union[str, Path], out_txt: Path):
    """
    Generate precision/recall/F1 report for a TextGrad job.

    Parameters
    ----------
    run_log
        CSV produced by TextGrad; must contain a `prediction` column.
    truth_csv
        Ground-truth CSV used during evaluate(); label column must be last.
    out_txt
        File path to store the textual report.

    Side effects
    ------------
    Writes *classification_report* text to `out_txt`
    and prints it to stdout for CI visibility.
    """
    run_df = pd.read_csv(run_log)
    truth_df = pd.read_csv(truth_csv).iloc[: len(run_df)]  # align rows safe-ish

    preds = run_df["prediction"]
    y_true = _load_column(truth_df)

    txt = classification_report(y_true, preds, digits=3)
    Path(out_txt).write_text(txt)

    print(txt)
