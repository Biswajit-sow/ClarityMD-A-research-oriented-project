import os
import sys
import torch
import yaml
from PIL import Image
import cv2
import glob
import numpy as np

# --- Python Path Fix ---
# This is still necessary so we can find the 'models' and 'data_prep' packages
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(script_dir)
sys.path.insert(0, PROJECT_ROOT)

from models.lightning_model import LitClassificationModel
from data_prep.augmentations import get_val_transforms
# --- CORRECTED LINES ---
# Use a relative import (.) to find modules in the same 'xai' package
from .captum_utils import generate_gradcam_attributions
from .visualizer import overlay_heatmap_on_image

# --- Configuration ---
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'default.yaml')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, 'chest_xray', 'test', 'PNEUMONIA')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'xai_explanations')
NUM_IMAGES_TO_EXPLAIN = 5

def find_best_checkpoint(checkpoint_dir):
    """Finds the best-performing checkpoint file in the directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}. Please run training first.")
    return checkpoint_files[0]

def explain_predictions():
    """Main function to generate and save XAI explanations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- Generating XAI Explanations ---")
    
    # 1. Load Config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Model from Checkpoint
    checkpoint_path = find_best_checkpoint(CHECKPOINT_DIR)
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = LitClassificationModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    # 3. Get image paths to explain
    image_paths = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpeg"))[:NUM_IMAGES_TO_EXPLAIN]
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {TEST_IMAGE_DIR}")

    # 4. Define transforms
    preprocess = get_val_transforms()

    # 5. Loop through images and generate explanations
    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        original_image = Image.open(img_path).convert("RGB")
        transformed_image = preprocess(image=np.array(original_image))['image']
        
        input_tensor = transformed_image.unsqueeze(0)
        input_tensor.requires_grad = True

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities.max().item()
        
        class_names = {0: "NORMAL", 1: "PNEUMONIA"}
        print(f"  -> Prediction: {class_names[predicted_class]} (Confidence: {confidence:.2f})")
        
        target_layer = model.model.layer4
        heatmap = generate_gradcam_attributions(model, input_tensor.squeeze(0), predicted_class, target_layer)
        
        overlayed_image = overlay_heatmap_on_image(original_image, heatmap)
        
        text = f"Pred: {class_names[predicted_class]} ({confidence:.2f})"
        cv2.putText(overlayed_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        output_path = os.path.join(OUTPUT_DIR, f"explanation_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, overlayed_image)
        print(f"  -> Saved explanation to {output_path}")

    print("\nAcceptance: XAI script completed and saved output images.")

if __name__ == "__main__":
    explain_predictions()