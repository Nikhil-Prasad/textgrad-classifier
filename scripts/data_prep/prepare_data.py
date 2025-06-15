"""
prepare_data.py
===============

Utility script for generating or fetching CSV datasets used in the
TextGrad-classifier repo.

Two sub-commands exposed via Typer:

* `contact` – create a synthetic UBS contact-note dataset
* `issues`  – download & normalize the NLBSE’24 GitHub issue dataset

Usage
-----
# generate both datasets (default paths in ./data)
python scripts/prepare_data.py contact
python scripts/prepare_data.py issues

Pass ``--help`` on either command for CLI options.
"""

import random
from pathlib import Path
from typing import Final

import pandas as pd
import requests
import typer
from faker import Faker

app = typer.Typer(help="Dataset generator for TextGrad demo repo")

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
DATA_DIR: Final[Path] = Path("data")
DATA_DIR.mkdir(exist_ok=True)

NLBSE_URL: Final[str] = (
    "https://raw.githubusercontent.com/"
    "nlbse2024/issue-report-classification/main/data/issues_train.csv"
)
DEFAULT_CONTACT_ROWS: Final[int] = 40
DEFAULT_ISSUE_LIMIT: Final[int] = 3000


# --------------------------------------------------------------------------- #
#  Commands
# --------------------------------------------------------------------------- #
@app.command()
def contact(rows: int = DEFAULT_CONTACT_ROWS, force: bool = typer.Option(False)):
    """
    Generate a **synthetic UBS contact-note** dataset.

    Parameters
    ----------
    rows : int, default ``40``
        Number of rows to create.
    force : bool, default ``False``
        Overwrite the existing CSV if it already exists.

    Output
    ------
    * ``data/contact_notes.csv`` with columns::

          note_content,satisfactory
          "The client was pleased with...",satisfactory
          "Follow-up required on the missing docs",needs_follow_up
    """
    out_path = DATA_DIR / "contact_notes.csv"
    if out_path.exists() and not force:
        typer.secho(f"{out_path} already exists – skipping. Use --force to overwrite.",
                    fg=typer.colors.YELLOW)
        raise typer.Exit()

    fake = Faker()
    df = pd.DataFrame(
        {
            "note_content": [fake.paragraph(nb_sentences=5) for _ in range(rows)],
            "satisfactory": [
                random.choice(["satisfactory", "needs_follow_up"]) for _ in range(rows)
            ],
        }
    )
    df.to_csv(out_path, index=False)
    typer.secho(f"Wrote {out_path}", fg=typer.colors.GREEN)


@app.command()
def issues(
    limit: int = DEFAULT_ISSUE_LIMIT,
    force: bool = typer.Option(False),
):
    """
    Download & preprocess a **Linear/Jira-style issue** dataset.

    * Fetches the NLBSE’24 GitHub issue corpus (public MIT license).
    * Keeps only the first *limit* rows for quick demo runs.
    * Maps ``enhancement`` → ``feature`` so labels become
      ``bug`` / ``feature`` / ``question``.

    Parameters
    ----------
    limit : int, default ``3000``
        Truncate the dataset for lightweight CI runs.
    force : bool, default ``False``
        Overwrite the existing CSV if it already exists.

    Output
    ------
    * ``data/issues.csv`` with columns::

          title,body,label
          "Crash when clicking X","Steps to reproduce ...",bug
    """
    out_path = DATA_DIR / "issues.csv"
    if out_path.exists() and not force:
        typer.secho(f"{out_path} already exists – skipping. Use --force to overwrite.",
                    fg=typer.colors.YELLOW)
        raise typer.Exit()

    typer.echo("Downloading NLBSE’24 issue dataset …")
    resp = requests.get(NLBSE_URL, timeout=30)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text)).head(limit)
    df["label"] = df["label"].replace({"enhancement": "feature"})
    df.to_csv(out_path, index=False)
    typer.secho(f"Wrote {out_path}", fg=typer.colors.GREEN)


# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app()
