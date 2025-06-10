#!/usr/bin/env python
"""
run_textgrad.py
===============

Command-line entry point for running a TextGrad experiment end-to-end.

Current behavior
----------------
1. Load a YAML config (`configs/*.yaml`).
2. Train + evaluate via `textgrad.Trainer`.
3. Write the raw TextGrad run log to `outputs/{timestamp}/`.
4. Call `eval_metrics.report()` for a quick P/R/F1 readout.

Future extensions (optional)
----------------------------
* SQLite or DuckDB writer – store run metadata for long-term tracking.
* Hyper-param sweeps – loop over multiple configs in one invocation.
* Accuracy threshold – exit non-zero if F1 below a CLI flag.
"""

import datetime as _dt
import os
from pathlib import Path

import typer
import yaml
from textgrad import Trainer

from eval_metrics import report

# ---------------------------------------------------------------------------
#  Typer app definition
# ---------------------------------------------------------------------------

app = typer.Typer(help="Run TextGrad fit/eval cycle for a single dataset.")


@app.command()
def run(
    cfg_path: Path = typer.Argument(..., exists=True, help="Path to YAML config."),
    csv_path: Path = typer.Argument(..., exists=True, help="CSV with training data."),
    out_root: Path = typer.Option("outputs", help="Root dir for run artifacts."),
):
    """Execute a single TextGrad job and print eval metrics."""
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = out_root / ts
    save_dir.mkdir(parents=True, exist_ok=True)

    # -- Trainer --
    trainer = Trainer(str(cfg_path))
    trainer.fit(str(csv_path))

    # TextGrad evaluate() returns the run-log path; store under save_dir
    run_log = trainer.evaluate(str(csv_path), save_path=save_dir)

    # -- Persist best prompt for transparency
    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.setdefault("prompt", {})["optimized"] = trainer.best_prompt

    with open(save_dir / "optimized_prompt.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # -- Quick metrics report
    report(run_log, csv_path, save_dir / "metrics.txt")

    typer.secho(f"✓ Completed: artifacts in {save_dir}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
