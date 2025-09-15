# tests/test_source2silver_plot_unit.py
from pathlib import Path
import re
import pytest
import responses
from .utils_loader import load_functions

FILE = Path("src/source2silver_plot.py")
def _load(names): 
    # Dynamically load functions from the specified file
    return load_functions(FILE, names)

@responses.activate
def test_fetch_plot_ok():
    """
    Test that fetch_plot returns the expected plot string or dictionary when the API responds with a valid plot.
    """
    fn = _load(["fetch_plot"])["fetch_plot"]
    if fn is None:
        pytest.skip("fetch_plot not found; skipping")

    # Mock the GET request to the OMDB API endpoint
    responses.add(
        responses.GET,
        re.compile(r".*omdbapi\.com.*", re.IGNORECASE),
        json={"Plot": "A moving tale of friendship.", "Response": "True", "Title": "X", "Year": "1999"},
        status=200,
    )
    out = fn("tt0000001")  # The function expects a single argument
    if isinstance(out, dict):
        assert out.get("plot") == "A moving tale of friendship."
    else:
        assert out == "A moving tale of friendship."

@responses.activate
def test_fetch_plot_na_returns_none():
    """
    Test that fetch_plot returns None or "N/A" when the API responds with "N/A" as the plot.
    """
    fn = _load(["fetch_plot"])["fetch_plot"]
    if fn is None:
        pytest.skip("fetch_plot not found; skipping")

    # Mock the GET request to the OMDB API endpoint with "N/A" plot
    responses.add(
        responses.GET,
        re.compile(r".*omdbapi\.com.*", re.IGNORECASE),
        json={"Plot": "N/A", "Response": "True"},
        status=200,
    )
    out = fn("tt0000002")
    if isinstance(out, dict):
        assert out.get("plot") in (None, "N/A")
    else:
        assert out is None or out == "N/A"
