# tests/test_gold_training_structural.py
from pathlib import Path
import re
import pytest

FILE = Path("src/gold_prediction_training_v4.py")

def test_has_movieapi_and_constants_text():
    """
    Test that the MovieAPI class and required constants are present in the source file.
    """
    text = FILE.read_text(encoding="utf-8")

    # Check for the MovieAPI class definition
    assert re.search(r"^\s*class\s+MovieAPI\b", text, re.MULTILINE)

    # Check for the presence of required constants in the file
    needed = [
        "MODEL_SKLEARN_NAME",
        "API_MODEL_NAME",
        "GOLD_CATALOG_DB_TABLE",
        "NUMERIC_COLS",
        "TEXT_COL",
        "GENRES_COL",
        "LABEL_COL",
    ]
    missing = [c for c in needed if c not in text]
    assert not missing, f"Missing constants: {missing}"
