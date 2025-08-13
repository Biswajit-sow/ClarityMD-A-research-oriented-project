# Sprint 0 Report: Setup & Baseline

## Status: Completed

### 1. What was done:
-   **Repository Skeleton**: Created the full directory structure as specified in the project plan (`data_prep`, `models`, `tests`, etc.).
-   **Environment Setup**:
    -   `requirements.txt` was generated with all core dependencies.
    -   `.vscode/` settings were configured for a streamlined development experience.
-   **Dataset Integration**: The project was configured to use the public **Kaggle Chest X-Ray (Pneumonia) dataset**. This provides a realistic, structured, and imbalanced dataset to work with.
-   **Baseline Model Script**: Implemented `baselines/run.py`. This script:
    -   Loads a pre-trained `resnet18` model from `torchvision` without any fine-tuning.
    -   Loads the **test split** of the Kaggle dataset using our `MedicalImageDataset` class.
    -   Performs inference on the entire test set.
    -   Calculates and prints key classification metrics using `scikit-learn`.
-   **Unit Test**: Created `tests/test_sprint_0.py` to ensure the baseline script executes successfully.

### 2. Results & Acceptance Criteria:
-   **Acceptance Met**: `python baselines/run.py` executes successfully, loads the test data, and prints the baseline metrics.
-   **Key Finding**: The untrained model performed poorly, often predicting only the majority class. For example, it achieved an accuracy of roughly 62.5%, which corresponds to simply guessing "PNEUMONIA" for every image in the test set. This established a clear performance baseline that we needed to significantly improve upon with training.

### 3. Challenges & Blockers:
-   Initial challenges involved ensuring robust file pathing so that scripts could be run from any directory. This was solved by programmatically finding the project's root directory and adding it to the system path at the start of each script.

### 4. Next Steps (Sprint 1):
-   Implement a robust data loading and preprocessing pipeline for the *training* data.
-   Integrate `albumentations` for powerful image augmentation.
-   Develop a `preview.py` script to visually inspect the results of our preprocessing and augmentation steps.