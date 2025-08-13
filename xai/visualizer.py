import numpy as np
import cv2
import torch
from PIL import Image

def overlay_heatmap_on_image(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays a heatmap on an image for visualization.

    Args:
        image (PIL.Image or np.ndarray): The original, full-size image.
        heatmap (torch.Tensor or np.ndarray): The heatmap to overlay (H, W).
        alpha (float): The transparency of the heatmap.
        colormap (int): The OpenCV colormap to use for the heatmap.

    Returns:
        np.ndarray: The image with the heatmap overlay, in BGR format for OpenCV.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image.convert("RGB"))
    
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()

    # --- THE FIX IS HERE ---
    # Resize the heatmap to be the same size as the original image.
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize the heatmap to the 0-255 range
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8) # Add epsilon for stability
    heatmap = np.uint8(255 * heatmap)
    
    # Apply the colormap to the heatmap
    colored_heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Blend the heatmap with the original image
    # We need to convert the original image from RGB to BGR for cv2
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlayed_image = cv2.addWeighted(image_bgr, alpha, colored_heatmap, 1 - alpha, 0)

    return overlayed_image