import gradio as gr
import os
import sys
import torch
import yaml
from PIL import Image
import numpy as np
import cv2
import glob

# --- Python Path Fix & Imports ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from models.lightning_model import LitClassificationModel
from data_prep.augmentations import get_val_transforms
from xai.captum_utils import generate_gradcam_attributions
from xai.visualizer import overlay_heatmap_on_image
from xai.text_generator import generate_llm_explanation

# --- Global Model Loading ---
# Load the model and config once when the script starts
def load_model_and_config():
    print("Loading model, please wait...")
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError("No model checkpoint found. Please run `python models/train.py` first.")
    best_checkpoint = checkpoint_files[0]
    
    model = LitClassificationModel.load_from_checkpoint(best_checkpoint)
    model.eval()
    print("Model loaded successfully!")
    return model

model = load_model_and_config()

# --- The Main Analysis Function for Gradio ---
def analyze_xray(input_image):
    """
    This function takes an uploaded image and runs the entire pipeline.
    Args:
        input_image (np.ndarray): The image uploaded by the user via Gradio.
    Returns:
        tuple: A tuple containing all the outputs for the Gradio interface.
    """
    if input_image is None:
        return None, None, "No Prediction", "N/A", "Please upload an image to analyze."

    # 1. Prepare Image
    original_image_pil = Image.fromarray(input_image).convert("RGB")
    preprocess = get_val_transforms()
    transformed_image = preprocess(image=input_image)['image']
    input_tensor = transformed_image.unsqueeze(0)
    input_tensor.requires_grad = True

    # 2. Get Model Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities.max().item()
    
    class_names = {0: "NORMAL", 1: "PNEUMONIA"}
    predicted_class_name = class_names[predicted_class_idx]

    # 3. Generate Saliency Map
    target_layer = model.model.layer4
    heatmap = generate_gradcam_attributions(model, input_tensor.squeeze(0), predicted_class_idx, target_layer)
    overlayed_image_bgr = overlay_heatmap_on_image(original_image_pil, heatmap)
    # Convert from BGR (for cv2) to RGB for Gradio display
    overlayed_image_rgb = cv2.cvtColor(overlayed_image_bgr, cv2.COLOR_BGR2RGB)

    # 4. Generate LLM Summary
    explanation_text = generate_llm_explanation(predicted_class_name, confidence)
    
    # 5. Return all outputs
    return (
        original_image_pil,
        overlayed_image_rgb,
        predicted_class_name,
        f"{confidence * 100:.2f}%",
        explanation_text
    )

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), title="Explainable Medical Diagnosis Assistant") as demo:
    gr.Markdown("# 🩺 Explainable Medical Diagnosis Assistant")
    gr.Markdown(
        "<h3 style='color: #d9534f;'>Disclaimer: This is a research prototype and NOT a certified medical device. "
        "The predictions are for demonstration purposes only and should not be used for actual clinical diagnosis.</h3>"
    )
    
    with gr.Row():
        # --- INPUT COLUMN ---
        with gr.Column(scale=1, min_width=300):
            image_input = gr.Image(type="numpy", label="Upload X-ray Scan")
            submit_btn = gr.Button("Analyze Image", variant="primary")
            
            gr.Examples(
                examples=[
                    os.path.join(PROJECT_ROOT, "chest_xray/test/NORMAL/IM-0001-0001.jpeg"),
                    os.path.join(PROJECT_ROOT, "chest_xray/test/PNEUMONIA/person1_virus_6.jpeg")
                ],
                inputs=image_input
            )

        # --- OUTPUT COLUMN ---
        with gr.Column(scale=2):
            gr.Markdown("## Analysis Results")
            with gr.Row():
                original_output = gr.Image(label="Original Image")
                saliency_output = gr.Image(label="Explanation (Saliency Map)")
            
            with gr.Row():
                diagnosis_output = gr.Label(label="Predicted Diagnosis")
                confidence_output = gr.Label(label="Model Confidence")
            
            explanation_output = gr.Markdown(label="AI-Generated Clinical Summary")

    # Connect the button click to the analysis function
    submit_btn.click(
        fn=analyze_xray,
        inputs=image_input,
        outputs=[original_output, saliency_output, diagnosis_output, confidence_output, explanation_output]
    )

if __name__ == "__main__":
    demo.launch()