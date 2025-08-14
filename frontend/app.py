import streamlit as st
import os
import sys
import torch
import yaml
from PIL import Image
import numpy as np
import cv2
import glob
import base64

# --- Python Path Fix & Imports ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from models.lightning_model import LitClassificationModel
from data_prep.augmentations import get_val_transforms
from xai.captum_utils import generate_gradcam_attributions
from xai.visualizer import overlay_heatmap_on_image
from xai.text_generator import generate_llm_explanation

# --- Helper function to encode images for CSS display ---
def get_image_as_base64(image_array, format='jpeg'):
    """Converts a numpy array image to a base64 string."""
    _, buffer = cv2.imencode(f'.{format}', image_array)
    return base64.b64encode(buffer).decode()

# --- Custom CSS for Medical Theme & Neon Shadows ---
def app_styling():
    st.markdown("""
    <style>
        /* Main page background */
        .main {
            background-color: #1a1a2e; /* Dark navy background */
            color: #e0e0e0; /* Light text color */
        }

        /* Title Logo Container */
        .title-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0.5rem 0;
        }

        .title-icon {
            font-size: 2.8em;
            color: #00bfff; /* DeepSkyBlue */
            text-shadow: 0 0 10px #00bfff, 0 0 20px #00bfff;
            padding-right: 15px;
        }
        
        .title-text {
            font-size: 2.5em;
            font-weight: bold;
            color: #f0f8ff; /* AliceBlue */
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.8);
        }

        /* Subtle Tagline */
        .tagline {
            text-align: center;
            font-weight: bold;
                
            color: #87CEFA!important;
            font-size: 40px
            
        }
        
        /* The container for our image with the neon shadow */
        .image-container {
            padding: 10px;
            border-radius: 15px;
            background-color: transparent;
            box-shadow:
                0 0 7px rgba(0, 191, 255, 0.7),
                0 0 14px rgba(0, 191, 255, 0.5),
                0 0 28px rgba(0, 191, 255, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1rem;
        }

        .image-container img {
            max-width: 100%;
            border-radius: 10px;
        }

        /* Style for st.metric */
        .st-emotion-cache-1f1fig6 {
            background-color: #2a2a4e;
            border: 1px solid #4a4a7e;
            padding: 1rem;
            border-radius: 10px;
        }
        
        /* Style for st.info box */
        .st-emotion-cache-1wivap2 {
            background-color: #16213e;
            border-left: 5px solid #0f3460;
            color: #e0e0e0;
        }
        
        /* Style for st.success box */
        .st-emotion-cache-zt5ig0 {
            background-color: #102a24;
            border-left: 5px solid #198754;
            color: #a3cfbb;
        }

    </style>
    """, unsafe_allow_html=True)


# --- Configuration & Model Loading ---
@st.cache_resource
def load_model_and_config():
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        return None
    best_checkpoint = checkpoint_files[0]
    model = LitClassificationModel.load_from_checkpoint(best_checkpoint)
    model.eval()
    return model

# --- Main Application ---
st.set_page_config(page_title="ClarityMD", layout="wide")
app_styling()

st.markdown("""
<div class='title-container'>
    <span class='title-icon'>🩺</span>
    <span class='title-text'>ClarityMD</span>
</div>
""", unsafe_allow_html=True)
st.markdown("<h3 class='tagline'>Illuminating the AI 'Black Box' with Medical Diagnostic Saliency Maps.</h3>", unsafe_allow_html=True)


st.warning(
    "**Disclaimer:** This is a research prototype and **NOT** a certified medical device. "
    "The predictions are for demonstration purposes only and should not be used for actual clinical diagnosis."
)

model = load_model_and_config()

if model is None:
    st.error("Model checkpoint not found. Please run the training script `python models/train.py` first.")
else:
    st.sidebar.title("Upload Scan")
    uploaded_file = st.sidebar.file_uploader("Choose an X-ray image...", type=["jpeg", "jpg", "png"])
    
    if uploaded_file is not None:
        # --- ANALYSIS VIEW ---
        st.header("Analysis Results")
        
        original_image = Image.open(uploaded_file).convert("RGB")
        original_image_np = np.array(original_image)
        
        preprocess = get_val_transforms()
        transformed_image = preprocess(image=original_image_np)['image']
        input_tensor = transformed_image.unsqueeze(0)
        input_tensor.requires_grad = True

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities.max().item()
        
        class_names = {0: "NORMAL", 1: "PNEUMONIA"}
        predicted_class_name = class_names[predicted_class_idx]

        target_layer = model.model.layer4
        heatmap = generate_gradcam_attributions(model, input_tensor.squeeze(0), predicted_class_idx, target_layer)
        overlayed_image = overlay_heatmap_on_image(original_image, heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            original_b64 = get_image_as_base64(cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR))
            st.markdown(f'<div class="image-container"><img src="data:image/jpeg;base64,{original_b64}" /></div>', unsafe_allow_html=True)
        with col2:
            st.subheader("Explanation (Saliency Map)")
            overlay_b64 = get_image_as_base64(overlayed_image)
            st.markdown(f'<div class="image-container"><img src="data:image/jpeg;base64,{overlay_b64}" /></div>', unsafe_allow_html=True)
            
        st.subheader("Prediction Details")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Predicted Diagnosis", value=predicted_class_name)
        with metric_col2:
            st.metric(label="Model Confidence", value=f"{confidence*100:.2f}%")
        
        st.subheader("AI-Generated Clinical Summary")
        with st.spinner('Generating summary with Groq LLM...'):
            explanation_text = generate_llm_explanation(predicted_class_name, confidence)
            st.info(explanation_text)

    else:
        # --- WELCOME VIEW ---
        st.markdown(
            "<h3 style='color:#B0E0E6; font-size: 32px'>About This Project: From Black Box to Glass Box AI</h3>",
            unsafe_allow_html=True
        )

        
        st.markdown(
            "Traditional AI models often operate as a **'black box'**: they provide an answer but offer no insight into their reasoning. "
            "In high-stakes fields like medicine, this is a major barrier to trust and adoption. A doctor can't rely on a decision if they don't understand the 'why' behind it."
        )

        st.markdown(
            "This project transforms that paradigm by creating a **'glass box'** assistant. Our goal is not just to provide a prediction, but to open up the model's 'thought process' for human review. It achieves this by combining two key technologies:"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3 style='color:#B0E0E6; font-size: 28px'> 1. Visual Explanation (Saliency Maps)",
                         unsafe_allow_html=True)
            st.write(
                "The AI generates a heatmap, or 'saliency map', overlaid on the X-ray. This map acts like a highlighter, showing the exact pixels and regions the model found most suspicious or important for its decision. This provides direct, visual evidence for a clinician to scrutinize."
            )

        with col2:
            st.markdown("<h3 style='color:#B0E0E6; font-size: 28px'> 2. Conversational Explanation (LLM)",
                         unsafe_allow_html=True)
            st.write(
                "Using a powerful Large Language Model, the tool translates the complex findings into a clear, clinical summary. It even explains how to interpret the saliency map's colors, bridging the gap between the model's data and human understanding."
            )
        
        st.success("The result is a trustworthy 'second opinion' that builds confidence and fosters collaboration between human expertise and artificial intelligence.")

        st.markdown("<h3 style='color:#B0E0E6; font-size: 32px'>How to Use This Tool",
                  unsafe_allow_html=True)
        st.markdown("""
            1.  **Upload a Scan:** Use the "Browse files" button in the sidebar on the left to select a chest X-ray image.
            2.  **View AI Analysis:** The tool will automatically process the image and display the prediction, confidence score, and the visual Saliency Map.
            3.  **Read the Summary:** An AI-generated summary will explain the findings in plain language.
        """)

        st.divider()

        with st.expander("Explore the Technology Stack"):
            st.markdown("""
                -   **Machine Learning Framework:** PyTorch & PyTorch Lightning
                -   **Explainable AI (XAI):** Captum (for Grad-CAM)
                -   **Language Model (LLM) & Prompting:** Llama 3 (via Groq API), LangChain, Prompt Engineering
                -   **Web Frameworks:** Streamlit 
                -   **Data Science & Imaging:** Scikit-learn, Pandas, NumPy, OpenCV, Albumentations
            """)
st.markdown("<div class='footer'>© 2025 Medical Diagnostic Saliency Maps using XAI | Engineered by SweetPoison</div>", unsafe_allow_html=True)