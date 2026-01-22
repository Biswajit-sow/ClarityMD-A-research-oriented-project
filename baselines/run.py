import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, accuracy_score

# --- Python Path Fix ---
# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
# Get the directory containing the script (e.g., /path/to/project/baselines)
script_dir = os.path.dirname(script_path)
# Get the project root directory (one level up from 'baselines')
PROJECT_ROOT = os.path.dirname(script_dir)
# Add the project root to the Python path 
sys.path.insert(0, PROJECT_ROOT)
print(f"Project Root added to Python path: {PROJECT_ROOT}")

# Now that the path is fixed, we can import our modules
from data_prep.dataset import MedicalImageDataset
from data_prep.augmentations import get_val_transforms

# --- Constants ---
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "chest_xray", "test")
NUM_CLASSES = 2
MODEL_NAME = "Baseline ResNet18 on Kaggle Test Set"
BATCH_SIZE = 32

def run_baseline_evaluation():
    """
    Runs a baseline evaluation on the Kaggle test dataset.
    This version uses the proper dataset class and a DataLoader.
    """
    print(f"--- Running Evaluation for: {MODEL_NAME} ---")

    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Test data directory not found at {TEST_DATA_DIR}")
        print("Please ensure the 'chest_xray' dataset is in the project root.")
        return

    # 1. Load Model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded successfully and running on device: {device}")

    # 2. Load Data
    val_transforms = get_val_transforms()
    test_dataset = MedicalImageDataset(img_dir=TEST_DATA_DIR, transform=val_transforms)

    if len(test_dataset) == 0:
        print("The test dataset is empty. Aborting evaluation.")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Test data loaded: {len(test_dataset)} images in {len(test_loader)} batches.")

    # 3. Evaluation Loop
    all_predictions = []
    all_true_labels = []

    print("\nEvaluating model on the test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # 4. Calculate and Print Metrics
    if not all_true_labels:
        print("\nCould not generate predictions. Cannot calculate metrics.")
        return

    print("\n--- Baseline Metrics ---")
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Overall Accuracy: {accuracy:.4f}\n")

    target_names = ['NORMAL (Class 0)', 'PNEUMONIA (Class 1)']
    report = classification_report(all_true_labels, all_predictions, target_names=target_names)
    print(report)
    print("------------------------\n")
    print("Acceptance: Script finished and printed baseline metrics for the test set.")

if __name__ == '__main__':
    run_baseline_evaluation()