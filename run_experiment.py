"""Paleidžia visą PM2.5 notebook viena komanda (dabartinio Python branduolio)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "PM25_prognozavimas.ipynb"


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"Nerastas notebook: {NOTEBOOK}")
        return 1
    print("Vykdoma:", NOTEBOOK)
    print("Python:", sys.executable)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=7200,
        kernel_name="pm25_egz_env",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(nb, NOTEBOOK)
    print("Done. Notebook saved with outputs.")
    print("Figures:", ROOT / "figures")
    print("Tables:", ROOT / "results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
