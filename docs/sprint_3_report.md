# Sprint 3 Report: Model Training & Experiments

## Status: Completed

### 1. What was done:
-   **PyTorch Lightning Integration**:
    -   Created `models/lightning_model.py` which encapsulates the entire model logic (architecture, optimizers, training/validation steps, and metrics) into a clean, reusable `LightningModule`.
-   **Configuration-Driven Training**:
    -   Developed a YAML-based configuration system (`configs/default.yaml`) to manage all hyperparameters, making experiments repeatable and easy to adjust.
-   **Main Training Script**:
    -   Implemented the master script `models/train.py`. This script handles all setup: loading the config, preparing the `DataLoaders`, initializing the `LightningModule`, and configuring the `Trainer`.
-   **Advanced Training Features**:
    -   **Logging**: Integrated `TensorBoardLogger` to automatically log all metrics (loss, accuracy) during training. These logs can be viewed by running `tensorboard --logdir lightning_logs/`.
    -   **Checkpointing**: Implemented `ModelCheckpoint` to automatically save the single best version of the model based on validation accuracy (`val_acc`).
    -   **Early Stopping**: Added `EarlyStopping` to prevent wasting resources by automatically stopping the training run if the validation loss does not improve for several consecutive epochs.

### 2. Results & Acceptance Criteria:
-   **Acceptance Met**: Running `python models/train.py` successfully initiates the full training process.
    -   A progress bar displays real-time metrics (`train_loss`, `val_acc`, etc.).
    -   Logs are saved to the `lightning_logs/` directory.
    -   The best model checkpoint is saved in the `checkpoints/` directory upon completion.
-   The model is now trainable and achieves a reasonable performance, significantly outperforming the initial baseline.

### 3. Next Steps (Sprint 4):
-   Focus on explainability (XAI).
-   Implement Grad-CAM and Integrated Gradients to generate saliency maps for model predictions.
-   Build functions to overlay these heatmaps on the original images for visual interpretation.