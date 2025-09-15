# tests/test_source2landing_unit.py
from pathlib import Path
import pytest
from .utils_loader import load_functions

FILE = Path("src/source2landing_imdb.py")

def test_has_datasets_keys():
    """
    Test that the source file contains references to the expected dataset keys.
    """
    text = Path(FILE).read_text(encoding="utf-8")
    assert "title.basics.tsv.gz" in text
    assert "title.ratings.tsv.gz" in text
    assert "name.basics.tsv.gz" in text
