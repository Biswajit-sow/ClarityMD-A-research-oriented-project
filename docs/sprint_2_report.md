# Sprint 2 Report: Class Imbalance Strategies

## Status: Completed

### 1. What was done:
-   **Identified Class Imbalance**: Confirmed from the `data_prep/preview.py` output that the training dataset is imbalanced, with a ratio of approximately 3:1 for PNEUMONIA vs. NORMAL cases.
-   **Implemented Weighted Loss Strategy**:
    -   Created a utility function `models/utils.py::calculate_class_weights`. This function computes weights for each class that are inversely proportional to the number of samples in that class.
    -   Modified the `models/lightning_model.py` to accept these weights in its `__init__` method.
    -   The `nn.CrossEntropyLoss` function is now initialized with this `weight` tensor. This will force the model to pay more attention to errors on the minority class (NORMAL), effectively balancing its learning process.
-   **Configuration Control**: Added a `use_weighted_loss` flag to `configs/default.yaml` to easily enable or disable this feature for experiments.

### 2. Results & Acceptance Criteria:
-   **Acceptance Met**: The logic for calculating and applying class weights is fully integrated into the training pipeline defined in `models/train.py`. The system is now equipped to handle the class imbalance present in the dataset. The effectiveness of this strategy will be measured in the next sprint after a full training run.

### 3. Next Steps (Sprint 3):
-   Utilize the implemented strategy within a full training pipeline using PyTorch Lightning.
-   Run the training script, log results, and save the best model checkpoint.