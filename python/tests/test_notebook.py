from pathlib import Path

import pytest

nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")


def test_the_walkthrough_runs_from_start_to_finish():
    path = Path(__file__).parents[2] / "notebooks" / "walkthrough.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(notebook, timeout=240, kernel_name="python3").execute()
