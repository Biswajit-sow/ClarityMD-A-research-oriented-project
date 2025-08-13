import pytest
import os
import pandas as pd
import shutil
from PIL import Image

# --- Add Python Path Fix to the test file as well ---
import sys
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(script_dir)
sys.path.insert(0, PROJECT_ROOT)

from data_prep.dataset import MedicalImageDataset
from data_prep.augmentations import get_train_transforms
from data_prep.preview import run_preview

# This fixture is now updated to create a directory structure, not a CSV
@pytest.fixture(scope="module")
def setup_data():
    """Create a dummy directory structure mimicking the Kaggle dataset."""
    base_dir = "tests/temp_data_sprint1" # Use a new name to avoid conflicts
    # Define paths for NORMAL and PNEUMONIA classes
    normal_dir = os.path.join(base_dir, "NORMAL")
    pneumonia_dir = os.path.join(base_dir, "PNEUMONIA")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(pneumonia_dir, exist_ok=True)

    # Create dummy image files in their respective class folders
    Image.new('RGB', (128, 128)).save(os.path.join(normal_dir, "normal1.jpeg"))
    Image.new('RGB', (128, 128)).save(os.path.join(pneumonia_dir, "pneu1.jpeg"))

    # The fixture now returns the path to the base directory of the test data
    yield {"img_dir": base_dir}

    # Teardown: remove the temporary directory after tests are done
    shutil.rmtree(base_dir)


def test_dataset_loading(setup_data):
    """
    Tests if MedicalImageDataset loads data correctly from a directory structure.
    """
    # --- CORRECTED ---
    # The constructor no longer takes 'metadata_path'. It only needs the image directory.
    dataset = MedicalImageDataset(
        img_dir=setup_data["img_dir"],
        transform=get_train_transforms(img_size=64)
    )
    assert len(dataset) == 2
    # We can check that both labels (0 and 1) are present
    assert 0 in dataset.labels
    assert 1 in dataset.labels

    sample = dataset[0]
    assert "image" in sample
    assert "label" in sample
    assert sample['image'].shape == (3, 64, 64) # C, H, W

def test_data_preview_script_runs(setup_data, monkeypatch):
    """
    Tests if the preview script runs without errors using the new data structure.
    """
    # Patch the constants in the preview script to point to our test data
    monkeypatch.setattr('data_prep.preview.DATA_DIR', setup_data["img_dir"])
    monkeypatch.setattr('data_prep.preview.OUTPUT_DIR', os.path.join(setup_data["img_dir"], "output"))

    # --- CORRECTED ---
    # The 'METADATA_PATH' attribute no longer exists, so we don't patch it.
    run_preview()

    # Check if output files were created in the temporary output directory
    output_dir = os.path.join(setup_data["img_dir"], "output")
    assert os.path.exists(output_dir)
    output_files = os.listdir(output_dir)
    assert any(f.endswith(".png") for f in output_files)