# 🩺 ClarityMD - Explainable Medical Diagnosis Assistant

<div align="center">
  
  **Illuminating the AI 'Black Box' with Medical Diagnostic Saliency Maps**
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  
</div>

---

## 🎯 Project Overview

**ClarityMD** is a proof-of-concept AI-powered clinical support system designed to assist doctors in diagnosing diseases from medical images, with a primary focus on **pneumonia detection from chest X-rays**.

The core mission is **not to replace medical professionals**, but to **augment clinical decision-making** by providing a fast, transparent, and explainable **second opinion**. The system directly addresses the **"black box" problem** in medical AI by combining visual explanations, confidence scores, clinical summaries, and AI-driven patient guidance.

> 🎓 **Goal:** Transform medical AI from a **black box** into a **glass box** by making predictions interpretable, trustworthy, and clinically useful.

---

## 🌟 Key Features

### 1️⃣ **Visual Explainability - Saliency Maps**
- **Grad-CAM** heatmaps highlight critical regions in X-rays
- Color-coded importance (Red/Yellow = High, Blue/Green = Low)
- Enables clinicians to visually verify AI reasoning

### 2️⃣ **AI-Generated Clinical Summaries**
- **LLM-powered** (Llama 3 via Groq API) professional explanations
- Interprets prediction confidence in medical terminology
- Bridges the gap between AI output and clinical understanding

### 3️⃣ **Patient Information Management**
- Mandatory patient name and ID before diagnosis
- Complete prediction history tracking
- Timestamp and confidence logging

### 4️⃣ **Interactive Web Interface**
- Modern, responsive Streamlit UI
- Animated grid background with neon effects
- Side-by-side comparison of original and saliency maps

### 5️⃣ **Prediction History**
- Browse all past diagnoses
- Filter by patient ID and name
- View detailed analysis for any prediction

---

## 📸 Screenshots

### 🏠 Home Page - Clean Interface
<img src="ClarityMD-A-research-oriented-project/docs/home.png" alt="ClarityMD Home" width="800"/>

*Professional dashboard with animated grid background and clear navigation*

### 🔬 Analysis Results - Side-by-Side Comparison
<img src="ClarityMD-A-research-oriented-project/docs/analysis.png" alt="Analysis Results" width="800"/>

*Original X-ray alongside AI-generated saliency map with detailed metrics*

### 💬 AI Clinical Summary
<img src="ClarityMD-A-research-oriented-project/docs/summary.png" alt="Clinical Summary" width="800"/>

*LLM-generated professional explanation with saliency map interpretation*

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/claritymd.git
cd claritymd
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

4. **Download the dataset**
```bash
# Kaggle Pneumonia Dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

5. **Train the model** (optional - pre-trained checkpoint provided)
```bash
python models/train.py
```

6. **Run the application**
```bash
cd frontend
streamlit run app.py
```

Visit `http://localhost:8501` in your browser

---

## 🏗️ Project Structure

```
claritymd/
├── data_prep/
│   ├── augmentations.py          # Image preprocessing & augmentation
│   └── dataset.py                 # PyTorch Dataset classes
├── models/
│   ├── lightning_model.py         # PyTorch Lightning training module
│   └── train.py                   # Training script
├── xai/
│   ├── captum_utils.py           # Grad-CAM implementation
│   ├── visualizer.py             # Saliency map overlay
│   └── text_generator.py         # LLM clinical summary generation
├── frontend/
│   ├── app.py                    # Streamlit web interface
│   └── logo.png                  # ClarityMD logo
├── checkpoints/                   # Saved model weights
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Stack

### Deep Learning
- **Framework:** PyTorch Lightning
- **Architecture:** ResNet-18 (pre-trained on ImageNet)
- **Loss Function:** Weighted Cross-Entropy (handles class imbalance)
- **Optimizer:** AdamW with learning rate scheduling

### Explainable AI
- **Library:** Captum (PyTorch XAI toolkit)
- **Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Visualization:** OpenCV + NumPy heatmap overlay

### LLM Integration
- **Model:** Llama 3 (via Groq API)
- **Framework:** LangChain for prompt engineering
- **Purpose:** Generate clinician-friendly summaries

### Frontend
- **Framework:** Streamlit
- **Styling:** Custom CSS with animated grid background
- **UI/UX:** Glassmorphism design with neon accents

### Data Pipeline
- **Augmentation:** Albumentations
- **Processing:** NumPy, Pillow, OpenCV
- **Storage:** Session-based prediction history

---

## 🎓 How It Works

### 1. **Image Upload**
Doctor uploads chest X-ray via web interface

### 2. **Preprocessing**
- Resize to 224x224
- Normalize using ImageNet statistics
- Apply validation transforms

### 3. **Model Prediction**
- ResNet-18 outputs diagnosis (NORMAL/PNEUMONIA)
- Confidence score calculated via softmax

### 4. **Explainability (Grad-CAM)**
- Generates saliency heatmap
- Highlights influential regions
- Overlays on original image

### 5. **Clinical Summary**
- LLM interprets results
- Generates professional explanation
- Provides saliency map reading guide

### 6. **Display Results**
- Side-by-side image comparison
- Metrics dashboard (Patient info, diagnosis, confidence)
- AI-generated summary with recommendations

### 7. **History Tracking**
- Auto-saves to session storage
- Searchable by patient ID/name
- View past predictions anytime

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.2% |
| **Precision (Pneumonia)** | 92.8% |
| **Recall (Pneumonia)** | 96.1% |
| **F1-Score** | 94.4% |
| **AUC-ROC** | 0.97 |

*Trained on 5,856 chest X-ray images from Kaggle Pneumonia Dataset*

---

## 🎯 Innovation Highlights

### ✅ **End-to-End MLOps Pipeline**
Complete workflow from data preprocessing to deployment

### ✅ **Multimodal Explainability**
Visual (Grad-CAM) + Textual (LLM) explanations

### ✅ **LLM-Powered Clinical Narratives**
Professional medical summaries using Llama 3

### ✅ **Real-World Data Challenges**
Handles class imbalance with weighted loss

### ✅ **Production-Ready Interface**
Modern web UI with patient management

---

## ⚠️ Disclaimer

**This is a research prototype and NOT a certified medical device.**

- Predictions are for **demonstration purposes only**
- Should **NOT** be used for actual clinical diagnosis
- Always consult qualified medical professionals
- Not FDA-approved or clinically validated

---

## 🔮 Future Roadmap

- [ ] Multi-disease detection (TB, COVID-19, lung cancer)
- [ ] EHR system integration
- [ ] FDA-compliant validation pipeline
- [ ] Multimodal data fusion (reports + images)
- [ ] Real-time hospital deployment
- [ ] Mobile application (iOS/Android)
- [ ] Support for CT scans and MRI

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Framework:** PyTorch Lightning, Streamlit
- **XAI Library:** Captum
- **LLM Provider:** Groq (Llama 3)
- **Inspiration:** Making medical AI transparent and trustworthy

---

## 📧 Contact

**Developed by:** SweetPoison  
**Email:** your.email@example.com  
**GitHub:** [@yourusername](https://github.com/yourusername)

---

<div align="center">
  
  **⭐ If you find this project useful, please give it a star! ⭐**
  
  Made with ❤️ and 🤖 for better healthcare
  
</div>


