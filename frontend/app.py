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
import json



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


# --- HISTORY MANAGEMENT FUNCTIONS ---
def save_prediction_to_history(patient_name, patient_id, original_image_np, overlayed_image, predicted_class_name, confidence, explanation_text):
    """Save prediction to persistent storage"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert images to base64 for storage
    original_b64 = get_image_as_base64(cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR))
    overlay_b64 = get_image_as_base64(overlayed_image)
    
    prediction_data = {
        'id': prediction_id,
        'timestamp': timestamp,
        'patient_name': patient_name,
        'patient_id': patient_id,
        'predicted_class': predicted_class_name,
        'confidence': float(confidence),
        'original_image': original_b64,
        'overlay_image': overlay_b64,
        'explanation': explanation_text
    }
    
    # Get existing history
    history = get_prediction_history()
    history.append(prediction_data)
    
    # Save to storage (limited to last 50 predictions)
    if len(history) > 50:
        history = history[-50:]
    
    try:
        st.session_state.prediction_history = history
    except Exception as e:
        st.error(f"Error saving prediction: {e}")


def get_prediction_history():
    """Retrieve prediction history from session state"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    return st.session_state.prediction_history


def display_prediction_detail(prediction):
    """Display a single prediction in detail view"""
    st.markdown('<p class="analysis-label">🔬 Analysis Results</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="analysis-label">Original Image</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="image-container"><img src="data:image/jpeg;base64,{prediction["original_image"]}" /></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="analysis-label">Explanation - Saliency Map</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="image-container"><img src="data:image/jpeg;base64,{prediction["overlay_image"]}" /></div>', unsafe_allow_html=True)
    
    st.markdown('<p class="analysis-label">📊 Prediction Details</p>', unsafe_allow_html=True)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric(label="Patient Name", value=prediction['patient_name'])
    with metric_col2:
        st.metric(label="Patient ID", value=prediction['patient_id'])
    with metric_col3:
        st.metric(label="Predicted Diagnosis", value=prediction['predicted_class'])
    with metric_col4:
        st.metric(label="Model Confidence", value=f"{prediction['confidence']*100:.1f}%")
    
    st.markdown('<p class="analysis-label">💬 AI-Generated Clinical Summary</p>', unsafe_allow_html=True)
    st.info(prediction['explanation'])
    
    st.caption(f"📅 Analysis Date: {prediction['timestamp']}")


# --- VANTA.JS ANIMATED NETWORK EFFECT + RESPONSIVE DESIGN ---
def app_styling():
    st.markdown("""
    <style>
        /* ANIMATED TECH GRID BACKGROUND - GARURA STYLE */
        .stApp {
            background-color: #000000 !important;
            background-image: 
                linear-gradient(rgba(14, 165, 233, 0.15) 1px, transparent 1px),
                linear-gradient(90deg, rgba(14, 165, 233, 0.15) 1px, transparent 1px),
                linear-gradient(rgba(14, 165, 233, 0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(14, 165, 233, 0.06) 1px, transparent 1px) !important;
            background-size: 100px 100px, 100px 100px, 20px 20px, 20px 20px !important;
            background-position: 0 0, 0 0, 0 0, 0 0 !important;
            animation: gridSlide 30s linear infinite !important;
        }

        @keyframes gridSlide {
            0% {
                background-position: 0 0, 0 0, 0 0, 0 0;
            }
            100% {
                background-position: 100px 100px, 100px 100px, 20px 20px, 20px 20px;
            }
        }

        /* Animated radial gradient spotlight */
        .stApp::before {
            content: '' !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
            background: 
                radial-gradient(ellipse at 50% 0%, rgba(14, 165, 233, 0.2) 0%, transparent 50%),
                radial-gradient(ellipse at 0% 50%, rgba(14, 165, 233, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 100% 50%, rgba(14, 165, 233, 0.1) 0%, transparent 50%) !important;
            animation: spotlightPulse 8s ease-in-out infinite !important;
            pointer-events: none !important;
            z-index: 0 !important;
        }

        @keyframes spotlightPulse {
            0%, 100% {
                opacity: 0.6;
                transform: scale(1);
            }
            50% {
                opacity: 1;
                transform: scale(1.05);
            }
        }

        /* Animated glowing particles with movement */
        .stApp::after {
            content: '' !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
            background-image: 
                radial-gradient(circle at 20% 30%, rgba(14, 165, 233, 0.4) 3px, transparent 3px),
                radial-gradient(circle at 80% 20%, rgba(56, 189, 248, 0.35) 2.5px, transparent 2.5px),
                radial-gradient(circle at 40% 70%, rgba(14, 165, 233, 0.4) 2px, transparent 2px),
                radial-gradient(circle at 70% 60%, rgba(56, 189, 248, 0.35) 3px, transparent 3px),
                radial-gradient(circle at 15% 80%, rgba(14, 165, 233, 0.4) 2px, transparent 2px),
                radial-gradient(circle at 90% 40%, rgba(56, 189, 248, 0.35) 2.5px, transparent 2.5px),
                radial-gradient(circle at 30% 50%, rgba(14, 165, 233, 0.4) 2px, transparent 2px),
                radial-gradient(circle at 85% 75%, rgba(56, 189, 248, 0.35) 3px, transparent 3px),
                radial-gradient(circle at 50% 15%, rgba(14, 165, 233, 0.4) 2.5px, transparent 2.5px),
                radial-gradient(circle at 60% 85%, rgba(56, 189, 248, 0.35) 2px, transparent 2px),
                radial-gradient(circle at 25% 25%, rgba(14, 165, 233, 0.4) 2px, transparent 2px),
                radial-gradient(circle at 75% 35%, rgba(56, 189, 248, 0.35) 2.5px, transparent 2.5px) !important;
            background-size: 100% 100% !important;
            animation: particleFloat 20s ease-in-out infinite !important;
            pointer-events: none !important;
            z-index: 0 !important;
        }

        @keyframes particleFloat {
            0%, 100% {
                transform: translate(0, 0);
                opacity: 0.7;
            }
            25% {
                transform: translate(15px, -10px);
                opacity: 0.9;
            }
            50% {
                transform: translate(-10px, 15px);
                opacity: 1;
            }
            75% {
                transform: translate(10px, 8px);
                opacity: 0.85;
            }
        }

        /* Ensure content is above animation */
        [data-testid="stAppViewContainer"] > div {
            position: relative !important;
            z-index: 1 !important;
        }
        /* --- RESPONSIVE TYPOGRAPHY & LAYOUT --- */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* Main container with solid background fallback */
        .main {
            position: relative;
            overflow-x: hidden;
            min-height: 100vh;
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

        .title-text .md-highlight {
            color: #38bdf8;
            font-size: inherit;
            font-weight: inherit;
        }

        
        /* --- UPDATED TAGLINE STYLE --- */
        .tagline {
            font-size: clamp(1.5rem, 5vw, 2.5rem) !important;
            font-weight: 700 !important;
            text-align: center;
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
            max-width: 1400px;
            margin: 3rem auto 4rem auto;
            padding: 0 2rem;
            line-height: 1.4;
            background: linear-gradient(135deg, #60a5fa 0%, #38bdf8 50%, #0ea5e9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: none;
            letter-spacing: -0.01em;
            position: relative;
            z-index: 20;
            animation: taglineGlow 3s ease-in-out infinite;
        }

        @keyframes taglineGlow {
            0%, 100% {
                filter: drop-shadow(0 0 20px rgba(56, 189, 248, 0.4));
            }
            50% {
                filter: drop-shadow(0 0 30px rgba(56, 189, 248, 0.7));
            }
        }

        /* Decorative line under tagline */
        .tagline::after {
            content: '';
            display: block;
            width: 200px;
            height: 3px;
            background: linear-gradient(90deg, transparent, #38bdf8, transparent);
            margin: 1.5rem auto 0;
            border-radius: 2px;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.8);
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

        /* --- RESPONSIVE SIDEBAR WITH NEON EFFECT --- */
        section[data-testid="stSidebar"] {
            background: rgba(10, 15, 26, 0.95) !important;
            backdrop-filter: blur(28px);
            border-right: 2px solid rgba(56, 189, 248, 0.8) !important;
            box-shadow: 
                0 0 20px rgba(56, 189, 248, 0.6),
                0 0 40px rgba(56, 189, 248, 0.4),
                0 0 60px rgba(56, 189, 248, 0.2),
                inset 0 0 20px rgba(56, 189, 248, 0.1) !important;
            z-index: 20;
            position: relative;
        }

        /* Neon glow animation for sidebar */
        section[data-testid="stSidebar"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                180deg,
                rgba(56, 189, 248, 0.1) 0%,
                transparent 50%,
                rgba(56, 189, 248, 0.1) 100%
            );
            animation: sidebarNeonPulse 4s ease-in-out infinite;
            pointer-events: none;
            z-index: -1;
        }

        @keyframes sidebarNeonPulse {
            0%, 100% {
                opacity: 0.5;
            }
            50% {
                opacity: 1;
            }
        }

        /* Neon effect on sidebar buttons */
        section[data-testid="stSidebar"] button {
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            box-shadow: 
                0 0 10px rgba(56, 189, 248, 0.3),
                inset 0 0 10px rgba(56, 189, 248, 0.1) !important;
            transition: all 0.3s ease !important;
        }

        section[data-testid="stSidebar"] button:hover {
            border: 1px solid rgba(56, 189, 248, 1) !important;
            box-shadow: 
                0 0 20px rgba(56, 189, 248, 0.8),
                0 0 40px rgba(56, 189, 248, 0.5),
                inset 0 0 15px rgba(56, 189, 248, 0.3) !important;
            transform: translateX(5px);
        }

        /* FORCE SIDEBAR SCROLLBAR TO SHOW */
        section[data-testid="stSidebar"] > div:first-child {
            overflow-y: scroll !important;
        }

        section[data-testid="stSidebar"]::-webkit-scrollbar {
            width: 12px !important;
            display: block !important;
        }

        section[data-testid="stSidebar"]::-webkit-scrollbar-track {
            background: rgba(10, 15, 26, 0.95) !important;
            border-radius: 8px !important;
            border: 2px solid rgba(56, 189, 248, 0.3) !important;
        }

        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #60a5fa, #38bdf8, #0ea5e9, #38bdf8, #60a5fa) !important;
            border-radius: 8px !important;
            border: 2px solid rgba(10, 15, 26, 0.7) !important;
            box-shadow: 
                0 0 15px rgba(56, 189, 248, 1),
                0 0 30px rgba(56, 189, 248, 0.8),
                inset 0 0 15px rgba(56, 189, 248, 0.4) !important;
            animation: sidebar-neon-pulse 3s ease-in-out infinite !important;
        }

        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #93c5fd, #60a5fa, #38bdf8, #60a5fa, #93c5fd) !important;
            box-shadow: 
                0 0 20px rgba(56, 189, 248, 1),
                0 0 40px rgba(56, 189, 248, 1),
                0 0 60px rgba(56, 189, 248, 0.8),
                inset 0 0 20px rgba(56, 189, 248, 0.6) !important;
        }

        @keyframes sidebar-neon-pulse {
            0%, 100% {
                box-shadow: 
                    0 0 15px rgba(56, 189, 248, 1),
                    0 0 30px rgba(56, 189, 248, 0.8),
                    inset 0 0 15px rgba(56, 189, 248, 0.4);
            }
            50% {
                box-shadow: 
                    0 0 25px rgba(56, 189, 248, 1),
                    0 0 50px rgba(56, 189, 248, 1),
                    inset 0 0 25px rgba(56, 189, 248, 0.6);
            }
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
            color: #38bdf8 !important;
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
            .title-container {
                padding: 1rem 0.5rem;
                gap: 0.25rem;
            }

            .title-icon {
                margin-right: 0.5rem;
            }

            .tagline {
                margin: 0.75rem 0;
                padding: 0 0.5rem;
            }

            .image-container {
                padding: 1rem;
                border-radius: 1rem;
                margin-bottom: 1.5rem;
            }

            div[data-testid="stMetric"],
            div[data-testid="stInfo"],
            div[data-testid="stSuccess"],
            div[data-testid="stExpander"],
            div[data-testid="stWarning"] {
                padding: 1.25rem;
                margin-bottom: 1.5rem;
                border-radius: 1rem;
                font-size: clamp(0.875rem, 2vw, 0.95rem);
            }
            
            section[data-testid="stSidebar"] {
                width: 100% !important;
            }

            h1, h2, h3 {
                padding: 0 0.5rem;
            }
        }

        /* Tablet optimization */
        @media (min-width: 769px) and (max-width: 1024px) {
            .title-container {
                padding: 1.25rem 0.75rem;
            }

            .image-container {
                padding: 1.5rem;
            }
        }

        /* ANIMATIONS */
        @keyframes pulse-icon-subtle {
            0%, 100% { 
                opacity: 1; 
                transform: scale(1); 
                filter: drop-shadow(0 0 8px #38bdf8);
            }
            50% { 
                opacity: 0.88; 
                transform: scale(1.08); 
                filter: drop-shadow(0 0 12px #38bdf8);
            }
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
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(10, 15, 26, 0.7);
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#38bdf8, #0ea5e9);
            border-radius: 5px;
            border: 2px solid rgba(10, 15, 26, 0.7);
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(#60a5fa, #38bdf8);
        }

        /* Ensure images and content don't overflow */
        [data-testid="stAppViewContainer"] > div {
            max-width: 100%;
            overflow-x: hidden;
        }

        /* Button responsiveness */
        button {
            font-size: clamp(0.85rem, 2vw, 1rem) !important;
        }

        /* Input field responsiveness */
        input, textarea, select {
            font-size: clamp(0.85rem, 2vw, 1rem) !important;
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
st.set_page_config(
    page_title="ClarityMD", 
    layout="wide", 
    page_icon="🩺", 
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### ClarityMD - Explainable AI Medical Diagnostics\nPowered by PyTorch Lightning & Groq"
    }
)
app_styling()


# Initialize page state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'selected_prediction' not in st.session_state:
    st.session_state.selected_prediction = None


# --- RESPONSIVE TITLE WITH SUBTLE GLOW ---
# --- RESPONSIVE TITLE WITH CUSTOM LOGO ---
logo_path = os.path.join(APP_ROOT, 'logo.png')
if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        logo_b64 = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
    <div class='title-container'>
        <img src='data:image/png;base64,{logo_b64}' class='title-icon' style='width: clamp(80px, 15vw, 130px); height: auto;' alt='ClarityMD Logo'/>
        <span class='title-text'>Clarity<span class='md-highlight'>MD</span></span>
    </div>
    """, unsafe_allow_html=True)
else:
    # Fallback to emoji if logo not found
    st.markdown("""
    <div class='title-container'>
        <span class='title-icon'>🩺</span>
        <span class='title-text'>Clarity<span class='md-highlight'>MD</span></span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<p class='tagline'>Illuminating the AI 'Black Box' with Medical Diagnostic Saliency Maps</h3>", unsafe_allow_html=True)

st.warning(
    "**Disclaimer:** This is a research prototype and **NOT** a certified medical device. "
    "The predictions are for demonstration purposes only and should not be used for actual clinical diagnosis.",
    icon="⚠️"
)



model = load_model_and_config()



if model is None:
    st.error("❌ Model checkpoint not found. Please run the training script `python models/train.py` first.")
else:
    # --- SIDEBAR NAVIGATION ---
    st.sidebar.title("🧭 Navigation")
    
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = 'home'
        st.session_state.selected_prediction = None
        st.rerun()
    
    if st.sidebar.button("📜 Prediction History", use_container_width=True):
        st.session_state.current_page = 'history'
        st.session_state.selected_prediction = None
        st.rerun()
    
    st.sidebar.divider()
    
    # --- PAGE ROUTING ---
    if st.session_state.current_page == 'history':
        # HISTORY PAGE
        st.markdown('<p class="analysis-label">📜 Prediction History</p>', unsafe_allow_html=True)
        
        history = get_prediction_history()
        
        if not history:
            st.info("No predictions yet. Upload an X-ray to get started!")
        else:
            # Show selected prediction detail or list
            if st.session_state.selected_prediction is not None:
                selected = st.session_state.selected_prediction
                
                if st.button("⬅️ Back to History List"):
                    st.session_state.selected_prediction = None
                    st.rerun()
                
                st.divider()
                display_prediction_detail(selected)
            else:
                # Display history list
                st.write(f"**Total Predictions:** {len(history)}")
                
                for idx, pred in enumerate(reversed(history)):
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 2, 2, 1])
                        
                        with col1:
                            st.write(f"**{pred['patient_name']}**")
                        with col2:
                            st.write(f"🆔 {pred['patient_id']}")
                        with col3:
                            st.write(f"📅 {pred['timestamp']}")
                        with col4:
                            color = "🟢" if pred['predicted_class'] == "NORMAL" else "🔴"
                            st.write(f"{color} {pred['predicted_class']} ({pred['confidence']*100:.1f}%)")
                        with col5:
                            if st.button("View", key=f"view_{idx}"):
                                st.session_state.selected_prediction = pred
                                st.rerun()
                        
                        st.divider()
    
    else:
        # HOME PAGE - UPLOAD & ANALYSIS
        st.sidebar.title("📁 Upload Scan")
        st.sidebar.markdown("**Supported formats:** JPEG, PNG")
        uploaded_file = st.sidebar.file_uploader("Choose an X-ray image...", type=["jpeg", "jpg", "png"])
        
        # Patient information inputs (mandatory)
        st.sidebar.markdown("### 👤 Patient Information")
        patient_name = st.sidebar.text_input("Patient Name *", value="", placeholder="Enter patient name")
        patient_id = st.sidebar.text_input("Patient ID *", value="", placeholder="Enter patient ID")
        
        st.sidebar.markdown("<small style='color: #f87171;'>* Required fields</small>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Check if patient information is provided
            if not patient_name.strip() or not patient_id.strip():
                st.error("⚠️ **Patient Information Required**")
                st.warning("Please enter both **Patient Name** and **Patient ID** in the sidebar before proceeding with the analysis.")
                st.stop()
            
            # --- ANALYSIS VIEW ---
            st.markdown('<p class="analysis-label">🔬 Analysis Results</p>', unsafe_allow_html=True)
            
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
                st.markdown('<p class="analysis-label">Original Image</p>', unsafe_allow_html=True)
                original_b64 = get_image_as_base64(cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR))
                st.markdown(f'<div class="image-container"><img src="data:image/jpeg;base64,{original_b64}" /></div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<p class="analysis-label">Explanation - Saliency Map</p>', unsafe_allow_html=True)
                overlay_b64 = get_image_as_base64(overlayed_image)
                st.markdown(f'<div class="image-container"><img src="data:image/jpeg;base64,{overlay_b64}" /></div>', unsafe_allow_html=True)
                
            st.markdown('<p class="analysis-label">📊 Prediction Details</p>', unsafe_allow_html=True)
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric(label="Patient Name", value=patient_name)
            with metric_col2:
                st.metric(label="Patient ID", value=patient_id)
            with metric_col3:
                st.metric(label="Predicted Diagnosis", value=predicted_class_name)
            with metric_col4:
                st.metric(label="Model Confidence", value=f"{confidence*100:.1f}%")
            
            st.markdown('<p class="analysis-label">💬 AI-Generated Clinical Summary</p>', unsafe_allow_html=True)
            with st.spinner('🤖 Generating summary with Groq LLM...'):
                explanation_text = generate_llm_explanation(predicted_class_name, confidence)
                st.info(explanation_text)
            
            # Automatically save to history
            prediction_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            if f'saved_{prediction_id}' not in st.session_state:
                save_prediction_to_history(
                    patient_name, 
                    patient_id,
                    original_image_np, 
                    overlayed_image, 
                    predicted_class_name, 
                    confidence, 
                    explanation_text
                )
                st.session_state[f'saved_{prediction_id}'] = True
                st.success(f"✅ Prediction automatically saved to history for {patient_name} (ID: {patient_id})!")


        else:
            # --- WELCOME VIEW ---
            st.markdown(
                "<h2 style='color: #38bdf8 !important;'>About This Project: From Black Box to Glass Box in Medical AI</h2>",
                unsafe_allow_html=True
            )
            
            st.markdown("""
            Traditional AI models often operate as a **'black box'**: they provide an answer but offer no insight into their reasoning. 
            In high-stakes fields like medicine, this is a major barrier to trust and adoption. 
            A doctor can't rely on a decision if they don't understand the 'why' behind it.
            """)


            st.markdown("""
            This project transforms that paradigm by creating a **'glass box'** assistant. 
            Our goal is not just to provide a prediction, but to open up the model's 'thought process' for human review. 
            It achieves this by combining two key technologies:
            """)


            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h3>🔍 1. Visual Explanation (Saliency Maps)</h3>", unsafe_allow_html=True)
                st.write("""
                    The AI generates a heatmap, or 'saliency map', overlaid on the X-ray. 
                    This map acts like a highlighter, showing the exact pixels and regions 
                    the model found most suspicious or important for its decision. 
                    This provides direct, visual evidence for a clinician to scrutinize.
                """)


            with col2:
                st.markdown("<h3>🤖 2. Conversational Explanation (LLM)</h3>", unsafe_allow_html=True)
                st.write("""
                    Using a powerful Large Language Model, the tool translates the complex 
                    findings into a clear, clinical summary. It even explains how to interpret 
                    the saliency map's colors, bridging the gap between the model's data 
                    and human understanding.
                """)


            st.markdown("<h2>🚀 How to Use This Tool</h2>", unsafe_allow_html=True)
            st.markdown("""
                **1. Upload a Scan:** Use the "Browse files" button in the sidebar to select a chest X-ray image.
                
                **2. View AI Analysis:** The tool automatically processes the image and displays:
                - Prediction & confidence score
                - Visual Saliency Map (heatmap explanation)
                
                **3. Read Summary:** AI-generated clinical explanation in plain language.
                
                **4. Save to History:** Click "Save to History" to store the prediction for later review.
                
                **5. View Past Predictions:** Use the "Prediction History" button in the sidebar.
            """)


            st.divider()


            with st.expander("🔧 Explore the Technology Stack", expanded=False):
                st.markdown("""
                **Core ML Stack:**
                - PyTorch & PyTorch Lightning
                - Captum (Grad-CAM for XAI)
                
                **LLM Integration:**
                - Llama 3 via Groq API
                - LangChain Prompt Engineering
                
                **Frontend:**
                - Streamlit + Vanta.js NET Effect (Animated Network Background)
                
                
                **Data Processing:**
                - Scikit-learn, Pandas, NumPy
                - OpenCV, Albumentations
               
                """)


st.markdown("""
<div class='footer'>
    © 2026 ClarityMD - Medical Diagnostic Saliency Maps using XAI | 
    Engineered by SweetPoison | Powered by PyTorch Lightning & Groq
</div>
""", unsafe_allow_html=True)