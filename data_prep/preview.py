import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# --- Python Path Fix ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(script_dir)
sys.path.insert(0, PROJECT_ROOT)
print(f"Project Root added to Python path: {PROJECT_ROOT}")

from data_prep.dataset import MedicalImageDataset
from data_prep.augmentations import get_train_transforms

# --- Constants ---
DATA_DIR = os.path.join(PROJECT_ROOT, "chest_xray", "train")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "data_preview")
NUM_SAMPLES_PER_CLASS = 2
NUM_AUGMENTATIONS = 4

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes a tensor image for visualization."""
    mean = np.array(mean)
    std = np.array(std)
    tensor = tensor.clone().permute(1, 2, 0).cpu().numpy()
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor

def run_preview():
    """
    Loads a few images from the dataset, applies transformations,
    and saves them for visual inspection.
    """
    print("--- Running Data Preparation Preview ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory not found at {DATA_DIR}")
        print("Please ensure you have the 'chest_xray' dataset in the project root.")
        return

    # 1. Initialize datasets
    train_transforms = get_train_transforms()
    preview_dataset = MedicalImageDataset(img_dir=DATA_DIR, transform=train_transforms)

    if len(preview_dataset) == 0:
        print("Dataset is empty. Cannot generate previews.")
        return

    # 2. Get sample indices from each class
    indices = []
    normal_indices = [i for i, label in enumerate(preview_dataset.labels) if label == 0]
    pneumonia_indices = [i for i, label in enumerate(preview_dataset.labels) if label == 1]

    if normal_indices:
        indices.extend(np.random.choice(normal_indices, min(NUM_SAMPLES_PER_CLASS, len(normal_indices)), replace=False))
    if pneumonia_indices:
        indices.extend(np.random.choice(pneumonia_indices, min(NUM_SAMPLES_PER_CLASS, len(pneumonia_indices)), replace=False))

    print(f"Generating previews for {len(indices)} images...")

    # 3. Generate and save previews for the selected samples
    for i, sample_idx in enumerate(indices):
        original_sample = MedicalImageDataset(img_dir=DATA_DIR, transform=None)[sample_idx]
        original_image = original_sample['image']
        label_name = "PNEUMONIA" if original_sample['label'] == 1 else "NORMAL"

        augmented_images = []
        for _ in range(NUM_AUGMENTATIONS):
            aug_sample = preview_dataset[sample_idx]
            augmented_images.append(aug_sample['image'])

        # 4. Create plot
        fig, axes = plt.subplots(1, 1 + NUM_AUGMENTATIONS, figsize=(16, 4))
        fig.suptitle(f"Sample from '{label_name}' Class", fontsize=16)

        axes[0].imshow(original_image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        for j, aug_img in enumerate(augmented_images):
            axes[1 + j].imshow(denormalize(aug_img))
            axes[1 + j].set_title(f"Augmented Sample {j+1}")
            axes[1 + j].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(OUTPUT_DIR, f"preview_class_{label_name}_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  - Saved preview for sample index {sample_idx} to {save_path}")

    print("\nAcceptance: Preview script completed and saved output images.")

if __name__ == "__main__":
    run_preview()