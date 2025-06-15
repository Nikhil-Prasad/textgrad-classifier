#!/usr/bin/env python
"""
run_textgrad.py
===============

Command-line entry point for running a TextGrad experiment end-to-end.

Current behavior
----------------
1. Load a YAML config (`configs/*.yaml`).
2. Train + evaluate via custom TextGradRunner (low-level TextGrad API).
3. Write results and optimized prompts to `outputs/{timestamp}/`.
4. Exit with success/failure based on metric threshold.

Future extensions (optional)
----------------------------
* SQLite or DuckDB writer – store run metadata for long-term tracking.
* Hyper-param sweeps – loop over multiple configs in one invocation.
"""

import datetime as _dt
import sys
from pathlib import Path

import typer
import yaml

from core.textgrad_runner import TextGradRunner

# ---------------------------------------------------------------------------
#  Typer app definition
# ---------------------------------------------------------------------------

app = typer.Typer(help="Run TextGrad fit/eval cycle for a single dataset.")


@app.command()
def run(
    cfg_path: Path = typer.Argument(..., help="Path to YAML config."),
    csv_path: Path = typer.Option(None, help="CSV path (overrides config)."),
    out_root: Path = typer.Option("outputs", help="Root dir for run artifacts."),
):
    """Execute a single TextGrad job using low-level optimization."""
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = out_root / ts
    save_dir.mkdir(parents=True, exist_ok=True)

    # Validate config file exists
    if not cfg_path.exists():
        typer.secho(f"Error: Config file not found at {cfg_path}", fg=typer.colors.RED)
        raise typer.Exit(1)
    
    # Load config to get CSV path if not provided
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    
    # Use provided CSV path or get from config
    if csv_path is None:
        csv_path = Path(config['data']['csv_path'])
    
    # Validate CSV exists
    if not csv_path.exists():
        typer.secho(f"Error: CSV file not found at {csv_path}", fg=typer.colors.RED)
        typer.secho("Run: python scripts/prepare_data.py contact", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    
    # Initialize and run TextGrad
    runner = TextGradRunner(str(cfg_path))
    
    try:
        success = runner.fit(str(csv_path), str(save_dir))
        
        # Save final optimized config
        config['prompt']['optimized'] = runner.system_prompt.value
        with open(save_dir / "optimized_config.yaml", "w") as f:
            yaml.safe_dump(config, f)
        
        typer.secho(f"✓ Completed: artifacts in {save_dir}", fg=typer.colors.GREEN)
        
        # Exit with appropriate code
        if not success:
            typer.secho(f"✗ Metric below threshold ({runner.success_threshold})", fg=typer.colors.RED)
            sys.exit(1)
            
    except Exception as e:
        typer.secho(f"✗ Error during training: {e}", fg=typer.colors.RED)
        sys.exit(1)


if __name__ == "__main__":
    app()
