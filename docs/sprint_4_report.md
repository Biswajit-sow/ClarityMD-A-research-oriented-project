# Sprint 4 Report: Explainability & Saliency

## Status: Completed

### 1. What was done:
-   **Explainability Framework**: The `captum` library has been integrated as the core XAI framework.
-   **Grad-CAM Implementation**:
    -   Created `xai/captum_utils.py` containing a function to compute Grad-CAM attributions. This function takes a model, an input image, and a target layer (`model.model.layer4` for ResNet18) to produce a raw heatmap.
-   **Visualization Utility**:
    -   Developed `xai/visualizer.py` with a helper function `overlay_heatmap_on_image`. This function uses OpenCV to apply a colormap to the raw heatmap and blend it transparently over the original image, making the explanation intuitive.
-   **Main Explanation Script**:
    -   Built the runnable script `xai/explain.py`. This script automates the entire process:
        1.  Finds and loads the best saved model checkpoint from the `checkpoints/` directory.
        2.  Loads a few sample images from the test set.
        3.  Makes a prediction for each image.
        4.  Generates the Grad-CAM heatmap based on the model's prediction.
        5.  Overlays the heatmap and saves the final visual explanation to the `output/xai_explanations/` directory.

### 2. Results & Acceptance Criteria:
-   **Acceptance Met**: Running `python xai/explain.py` successfully generates and saves several PNG/JPEG files in the `output/xai_explanations/` folder.
-   Each output image clearly shows the original X-ray with a colored heatmap highlighting the areas the model focused on (e.g., regions of lung opacity for a "PNEUMONIA" prediction). The image also includes the model's prediction and confidence score as text.
-   This provides a trustworthy visual explanation of the model's decision-making process.

### 3. Next Steps (Sprint 5 & 6):
-   **Plain-Language Explanations**: Integrate a simple text generator (template-based or LLM-based) that converts the XAI output into a human-readable sentence.
-   **Frontend Demo**: Build the interactive Streamlit application where a doctor can upload a scan and see the image, prediction, confidence score, and the saliency map overlay all in one dashboard.