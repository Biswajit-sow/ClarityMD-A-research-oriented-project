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
from datetime import datetime

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

# --- VANTA.JS ANIMATED NETWORK EFFECT + RESPONSIVE DESIGN ---
def app_styling():
    st.markdown("""
    <style>
        /* --- RESPONSIVE TYPOGRAPHY & LAYOUT --- */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background: #0a0f1a;
        }

        /* --- VANTA.JS BACKGROUND --- */
        .main {
            position: relative;
            background-color: #0a0f1a;
            overflow-x: hidden;
            min-height: 100vh;
            width: 100%;
        }

        .main {
            background: transparent !important;
        }

        /* Push all content above the canvas */
        [data-testid="stAppViewContainer"] {
            position: relative;
            z-index: 10;
            width: 100%;
        }

        /* --- ENHANCED GLASSMORPHISM COMPONENTS --- */
        div[data-testid="stMetric"],
        div[data-testid="stInfo"],
        div[data-testid="stSuccess"],
        div[data-testid="stExpander"],
        div[data-testid="stWarning"] {
            background-color: rgba(10, 15, 26, 0.92);
            backdrop-filter: blur(28px);
            -webkit-backdrop-filter: blur(28px);
            border-radius: 1.25rem;
            border: 1px solid rgba(56, 189, 248, 0.5);
            padding: 1.75rem;
            box-shadow:
                0 25px 50px rgba(0, 0, 0, 0.98),
                0 0 100px rgba(56, 189, 248, 0.4),
                inset 0 1px 0 rgba(255,255,255,0.08);
            margin-bottom: 2rem;
            position: relative;
            z-index: 20;
        }

        /* --- RESPONSIVE TITLE SYSTEM --- */
        .title-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1.5rem 1rem;
            position: relative;
            z-index: 20;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .title-icon {
            font-size: clamp(2.5rem, 8vw, 4rem);
            color: #38bdf8;
            text-shadow: 0 0 16px #38bdf8, 0 0 24px rgba(56, 189, 248, 0.4);
            animation: pulse-icon-subtle 2.5s infinite;
            filter: drop-shadow(0 0 8px #38bdf8);
            flex-shrink: 0;
        }

        .title-text {
            font-size: clamp(2rem, 7vw, 3.25rem);
            font-weight: 900;
            color: #f0f8ff;
            text-shadow: 0 0 24px rgba(56, 189, 248, 0.4), 0 0 48px rgba(56, 189, 248, 0.2);
            letter-spacing: -0.02em;
        }

        /* --- UPDATED TAGLINE STYLE --- */
        .tagline {
            font-size: clamp(1.20rem, 4.85vw, 1.93rem) !important;
            font-weight: 800;
            text-align: center;
            font-family: 'Times New Roman', Times, serif !important;
            max-width: 1200px;
            margin: 2rem auto 4rem auto;
            line-height: 1.2;
            background: linear-gradient(to bottom, #93c5fd, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 10px 30px rgba(56, 189, 248, 0.2);
            letter-spacing: -0.02em;
            position: relative;
            z-index: 20;
        }

        /* --- RESPONSIVE IMAGE CONTAINERS --- */
        .image-container {
            padding: clamp(1rem, 4vw, 2rem);
            border-radius: 1.5rem;
            background: rgba(10, 15, 26, 0.88);
            backdrop-filter: blur(20px);
            border: 2px solid rgba(56, 189, 248, 0.7);
            box-shadow:
                0 0 40px rgba(56, 189, 248, 0.5),
                0 0 80px rgba(56, 189,248, 0.3),
                0 0 120px rgba(56, 189, 248, 0.15),
                inset 0 0 40px rgba(255,255,255,0.08);
            transition: all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            z-index: 20;
            overflow: hidden;
            width: 100%;
        }

        .image-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
            transition: left 0.7s;
            z-index: 1;
        }

        .image-container:hover::before {
            left: 100%;
        }

        .image-container:hover {
            transform: translateY(-16px) scale(1.04);
            border-color: rgba(56, 189, 248, 1);
            box-shadow:
                0 45px 90px rgba(56, 189, 248, 0.6),
                0 0 140px rgba(56, 189, 248, 0.8),
                inset 0 0 40px rgba(255,255,255,0.1);
        }

        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 1rem;
            transition: transform 0.6s ease;
            position: relative;
            z-index: 2;
            display: block;
        }

        .image-container:hover img {
            transform: scale(1.06);
        }

        /* --- RESPONSIVE SIDEBAR --- */
        section[data-testid="stSidebar"] {
            background: rgba(10, 15, 26, 0.99) !important;
            backdrop-filter: blur(28px);
            border-right: 1px solid rgba(56, 189, 248, 0.5) !important;
            z-index: 20;
        }

        /* Metric containers */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(56,189,248,0.25), rgba(59,130,246,0.25));
            border-radius: 1rem;
            border: 1px solid rgba(56,189,248,0.4);
        }

        /* --- RESPONSIVE TYPOGRAPHY --- */
        h1 {
            font-size: clamp(1.75rem, 6vw, 2.75rem);
            color: #f0f8ff;
            text-shadow: 0 0 24px rgba(56, 189, 248, 0.3);
            margin: 1rem 0.5rem;
            padding: 0 1rem;
            line-height: 1.3;
        }

        h2 {
            font-size: clamp(1.5rem, 5vw, 2rem) !important; 
            font-weight: 500 !important;
            color: #FFFFF0 !important;
            margin-top: 2rem !important;
            margin-bottom: 0.5rem !important;
            letter-spacing: -0.04em;
            padding: 0 1rem;
            line-height: 1.1;
            text-shadow: 0 0 30px rgba(56, 189, 248, 0.4);
        }

        h3 {
            font-size: clamp(1.3rem, 4.75vw, 1.75rem) !important;
            font-weight: 400 !important;
            color: #38bdf8 !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
            letter-spacing: -0.03em;
        }

        /* --- LARGE LABELS FOR ANALYSIS RESULTS --- */
        .analysis-label {
            font-size: clamp(1.3rem, 4.7vw, 1.5rem) !important;
            font-weight: 400 !important;
            color: #f1f5f9 !important;
            margin: 1rem 0 1rem 0 !important;
            display: flex;
            align-items: center;
            gap: 1rem;
            padding-left: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        p, li, span {
            font-size: clamp(0.875rem, 2.5vw, 1rem);
            line-height: 1.6;
            color: #e5e7eb;
        }

        /* --- COLUMN LAYOUT FIX FOR MOBILE --- */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        /* Expander styling */
        [data-testid="stExpander"] button {
            font-size: clamp(0.875rem, 2.5vw, 1rem);
        }

        /* Mobile optimization */
        @media (max-width: 768px) {
            .title-container { padding: 1rem 0.5rem; gap: 0.25rem; }
            .title-icon { margin-right: 0.5rem; }
            .tagline { margin: 0.75rem 0; padding: 0 0.5rem; }
            .image-container { padding: 1rem; border-radius: 1rem; margin-bottom: 1.5rem; }
            div[data-testid="stMetric"], div[data-testid="stInfo"], div[data-testid="stSuccess"], div[data-testid="stExpander"], div[data-testid="stWarning"] {
                padding: 1.25rem; margin-bottom: 1.5rem; border-radius: 1rem; font-size: clamp(0.875rem, 2vw, 0.95rem);
            }
            section[data-testid="stSidebar"] { width: 100% !important; }
            h1, h2, h3 { padding: 0 0.5rem; }
        }

        /* Tablet optimization */
        @media (min-width: 769px) and (max-width: 1024px) {
            .title-container { padding: 1.25rem 0.75rem; }
            .image-container { padding: 1.5rem; }
        }

        /* ANIMATIONS */
        @keyframes pulse-icon-subtle {
            0%, 100% { opacity: 1; transform: scale(1); filter: drop-shadow(0 0 8px #38bdf8); }
            50% { opacity: 0.88; transform: scale(1.08); filter: drop-shadow(0 0 12px #38bdf8); }
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 1rem;
            color: rgba(148, 163, 184, 0.95);
            font-size: clamp(0.75rem, 2vw, 1rem);
            border-top: 1px solid rgba(56, 189, 248, 0.3);
            margin-top: 4rem;
            position: relative;
            z-index: 20;
            backdrop-filter: blur(12px);
            background: rgba(10, 15, 26, 0.8);
            line-height: 1.5;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: rgba(10, 15, 26, 0.7); }
        ::-webkit-scrollbar-thumb { background: linear-gradient(#38bdf8, #0ea5e9); border-radius: 5px; border: 2px solid rgba(10, 15, 26, 0.7); }
        ::-webkit-scrollbar-thumb:hover { background: linear-gradient(#60a5fa, #38bdf8); }

        [data-testid="stAppViewContainer"] > div { max-width: 100%; overflow-x: hidden; }
        button { font-size: clamp(0.85rem, 2vw, 1rem) !important; }
        input, textarea, select { font-size: clamp(0.85rem, 2vw, 1rem) !important; }
    </style>

    <!-- THREE.JS & VANTA.JS Libraries -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"></script>

    <script>
        // Kill existing instances
        if (window.vantaEffect) {
            window.vantaEffect.destroy();
            window.vantaEffect = null;
        }

        // Initialize Vanta.NET
        function initVanta() {
            if (typeof VANTA !== 'undefined' && VANTA.NET) {
                window.vantaEffect = VANTA.NET({
                    el: 'body',
                    mouseControls: true, touchControls: true, gyroControls: true,
                    minHeight: 200.00, minWidth: 200.00, scale: 1.00, scaleMobile: 1.00,
                    color: 0x38bdf8, backgroundColor: 0x0a0f1a,
                    points: 22, maxDistance: 32.0, spacing: 20.0
                });
            }
        }

        // Robust initialization
        function robustInit() {
            if (!window.vantaEffect) {
                initVanta();
            }
        }
        
        window.addEventListener('load', () => setTimeout(robustInit, 500));
        
        if (window.MutationObserver) {
            new MutationObserver(mutations => robustInit()).observe(document.body, { childList: true, subtree: true });
        }
        setInterval(robustInit, 5000);
    </script>
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
st.set_page_config(
    page_title="ClarityMD", layout="wide", page_icon="🩺", initial_sidebar_state="expanded",
    menu_items={'About': "### ClarityMD - Explainable AI Medical Diagnostics"}
)
app_styling()

# --- NEW: Initialize Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- HEADER ---
st.markdown("""
<div class='title-container'>
    <span class='title-icon'>🩺</span>
    <span class='title-text'>ClarityMD</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<p class='tagline'>Illuminating the AI 'Black Box' with Medical Diagnostic Saliency Maps</p>", unsafe_allow_html=True)

st.warning(
    "**Disclaimer:** This is a research prototype and **NOT** a certified medical device. "
    "The predictions are for demonstration purposes only and should not be used for actual clinical diagnosis.",
    icon="⚠️"
)

model = load_model_and_config()

if model is None:
    st.error("❌ Model checkpoint not found. Please run the training script `python models/train.py` first.")
else:
    # --- SIDEBAR ---
    st.sidebar.title("📁 Upload Scan")
    st.sidebar.markdown("**Supported formats:** JPEG, PNG")
    uploaded_file = st.sidebar.file_uploader("Choose an X-ray image...", type=["jpeg", "jpg", "png"])

    # --- NEW: History Section in Sidebar ---
    st.sidebar.divider()
    st.sidebar.title("📜 Analysis History")
    if not st.session_state.history:
        st.sidebar.write("No analyses yet.")
    else:
        # Display history items in reverse chronological order
        for item in reversed(st.session_state.history):
            with st.sidebar.expander(f"{item['timestamp']} - {item['filename']}"):
                st.image(item['thumbnail'], use_container_width=True)
                st.write(f"**Prediction:** {item['prediction']}")
                st.write(f"**Confidence:** {item['confidence']:.1f}%")

    # --- MAIN CONTENT ---
    if uploaded_file is not None:
        # --- ANALYSIS VIEW ---
        st.header("🔬 Analysis Results")
        
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

        # --- NEW: Save the result to history ---
        if not st.session_state.history or st.session_state.history[-1]['filename'] != uploaded_file.name:
            thumbnail = original_image.copy()
            thumbnail.thumbnail((100, 100))
            now = datetime.now().strftime("%H:%M:%S")
            st.session_state.history.append({
                "filename": uploaded_file.name,
                "prediction": predicted_class_name,
                "confidence": confidence * 100,
                "thumbnail": thumbnail,
                "timestamp": now
            })

        # --- Display Results ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="analysis-label">📷 Original Image</p>', unsafe_allow_html=True)
            original_b64 = get_image_as_base64(cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR))
            st.markdown(f'<div class="image-container"><img src="data:image/jpeg;base64,{original_b64}" /></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<p class="analysis-label">🎯 Explanation - Saliency Map</p>', unsafe_allow_html=True)
            overlay_b64 = get_image_as_base64(overlayed_image)
            st.markdown(f'<div class="image-container"><img src="data:image/jpeg;base64,{overlay_b64}" /></div>', unsafe_allow_html=True)
            
        st.markdown('<p class="analysis-label">📊 Prediction Details</p>', unsafe_allow_html=True)
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Predicted Diagnosis", value=predicted_class_name)
        with metric_col2:
            st.metric(label="Model Confidence", value=f"{confidence*100:.1f}%")
        
        st.markdown('<p class="analysis-label">💬 AI-Generated Clinical Summary</p>', unsafe_allow_html=True)
        with st.spinner('🤖 Generating summary with Groq LLM...'):
            explanation_text = generate_llm_explanation(predicted_class_name, confidence)
            st.info(explanation_text)

    else:
        # --- WELCOME VIEW ---
        st.markdown("<h2>About This Project: From Black Box to Glass Box in Medical AI</h2>", unsafe_allow_html=True)
        st.markdown("Traditional AI models often operate as a **'black box'**: they provide an answer but offer no insight into their reasoning...")
        st.markdown("This project transforms that paradigm by creating a **'glass box'** assistant...")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>🔍 1. Visual Explanation (Saliency Maps)</h3>", unsafe_allow_html=True)
            st.write("The AI generates a heatmap, or 'saliency map', overlaid on the X-ray...")
        with col2:
            st.markdown("<h3>🤖 2. Conversational Explanation (LLM)</h3>", unsafe_allow_html=True)
            st.write("Using a powerful Large Language Model, the tool translates the complex findings...")
        
        st.success("✅ The result is a trustworthy 'second opinion'...")

        st.markdown("<h2>🚀 How to Use This Tool</h2>", unsafe_allow_html=True)
        st.markdown("""
            **1. Upload a Scan:** ...
            **2. View AI Analysis:** ...
            **3. Read Summary:** ...
        """)
        st.divider()
        with st.expander("🔧 Explore the Technology Stack", expanded=False):
            st.markdown("""
            **Core ML Stack:** ...
            **LLM Integration:** ...
            **Frontend:** ...
            **Data Processing:** ...
            """)

st.markdown("""
<div class='footer'>
    © 2026 ClarityMD - Medical Diagnostic Saliency Maps using XAI | 
    Engineered by SweetPoison | Powered by PyTorch Lightning & Groq
</div>
""", unsafe_allow_html=True)