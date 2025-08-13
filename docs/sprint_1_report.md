# Sprint 1 Report: Data Pipeline & Preprocessing

## Status: Completed

### 1. What was done:
-   **PyTorch Dataset Class**: Implemented `data_prep/dataset.py` containing `MedicalImageDataset`. This class is responsible for:
    -   Loading image paths and labels from a CSV file.
    -   Reading image files using OpenCV.
    -   Applying transformations.
    -   Includes a placeholder `anonymize_filename` function to demonstrate a basic privacy-preserving step.
-   **Augmentation Pipeline**: Created `data_prep/augmentations.py` using the `albumentations` library. It provides two separate pipelines:
    -   `get_train_transforms`: For training, includes geometric and color augmentations (flips, rotation, brightness/contrast) to improve model generalization.
    -   `get_val_transforms`: For validation/testing, includes only necessary preprocessing (resizing, normalization).
-   **Data Preview Script**: Developed `data_prep/preview.py`. This utility script:
    -   Loads a few sample images using `MedicalImageDataset`.
    -   Generates and displays the original image, a preprocessed version, and several randomly augmented versions.
    -   Saves the comparison as a PNG file in the `output/` directory for easy visual verification.
-   **Unit Tests**: Added `tests/test_sprint_1.py` to validate the dataset loader and the preview script.

### 2. Results & Acceptance Criteria:
-   **Acceptance Met**: Running `python data_prep/preview.py` successfully loads images from the `sample_data` folder and generates output images in `output/data_preview/`. The saved images clearly show the original, the resized/normalized, and various augmented versions, confirming the entire pipeline works as expected.

### 3. Challenges & Blockers:
-   None. The use of `albumentations` and a standard PyTorch `Dataset` structure made this sprint straightforward.

### 4. Next Steps (Sprint 2):
-   Focus on strategies to handle class imbalance, which is central to this project.
-   Implement and experiment with weighted loss functions (e.g., Focal Loss, `nn.CrossEntropyLoss` with `weight` parameter).
-   Implement a weighted random sampler to oversample the minority class during training.
-   Begin implementation of a simple Variational Autoencoder (VAE) in the `synth/` directory to generate synthetic images of the rare disease class.