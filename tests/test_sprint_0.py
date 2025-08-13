import pytest
from baselines.run import run_baseline_evaluation
import os
import pandas as pd

def setup_module(module):
    """Create dummy files for testing."""
    os.makedirs("sample_data/images", exist_ok=True)
    pd.DataFrame({
        "filename": ["test1.png", "test2.png"],
        "label": [0, 1],
        "class_name": ["Normal", "Rare"]
    }).to_csv("sample_data/metadata.csv", index=False)
    # Create dummy image files
    from PIL import Image
    Image.new('RGB', (100, 100)).save("sample_data/images/test1.png")
    Image.new('RGB', (100, 100)).save("sample_data/images/test2.png")


def test_baseline_script_runs(capsys):
    """
    Tests if the baseline script runs without throwing an error.
    It captures stdout to check if metrics are printed.
    """
    run_baseline_evaluation()
    captured = capsys.readouterr()
    assert "--- Baseline Metrics ---" in captured.out
    assert "Acceptance: Script finished" in captured.out