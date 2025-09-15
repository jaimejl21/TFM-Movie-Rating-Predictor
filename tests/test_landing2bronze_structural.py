# tests/test_landing2bronze_structural.py
from pathlib import Path

def test_uses_dbutils_and_blob():
    """
    Test that the source file uses dbutils.widgets, BlobServiceClient, and expects TSV format.
    """
    text = Path("src/landing2bronze_imdb.py").read_text(encoding="utf-8")
    # Check for dbutils.widgets usage
    assert "dbutils.widgets" in text
    # Check for BlobServiceClient usage
    assert "BlobServiceClient" in text
    # Check for TSV format usage
    assert "sep='\\t'" in text or "\\t" in text
