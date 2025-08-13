import torch
from captum.attr import LayerGradCam, LayerAttribution

def generate_gradcam_attributions(model, input_tensor, target_class, target_layer):
    """
    Generates Grad-CAM attributions for a given model and input.

    Args:
        model (nn.Module): The model to explain.
        input_tensor (torch.Tensor): The preprocessed input image tensor (C, H, W).
                                     It must require gradients.
        target_class (int): The class index for which to generate the explanation.
        target_layer (torch.nn.Module): The convolutional layer to use for Grad-CAM.

    Returns:
        torch.Tensor: The raw attribution heatmap.
    """
    # Initialize LayerGradCam with the model and the target layer
    lgc = LayerGradCam(model, target_layer)

    # Upsample the attribution map to the input image size
    # attribute() expects a batch, so we add a batch dimension and remove it later
    attributions = lgc.attribute(
        input_tensor.unsqueeze(0),
        target=target_class,
        relu_attributions=True  # Use ReLU on gradients for cleaner maps
    )
    
    # Upsample the heatmap to the same size as the input image
    upsampled_attributions = LayerAttribution.interpolate(attributions, input_tensor.shape[1:], "bilinear")
    
    # Return the heatmap, removing the batch and channel dimensions
    return upsampled_attributions.squeeze().cpu().detach()