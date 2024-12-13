import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_heatmap(image_tensor, model, target_layer):
    """
    Generate a heatmap using Grad-CAM to identify critical regions.

    Args:
        image_tensor (torch.Tensor): Preprocessed input tensor for the model.
        model: Pre-trained model.
        target_layer: Layer to focus Grad-CAM.

    Returns:
        np.ndarray: Heatmap highlighting critical regions.
    """
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    targets = [ClassifierOutputTarget(0)]  # Example: Target the first class
    heatmap = cam(input_tensor=image_tensor, targets=targets)
    return heatmap

def apply_region_specific_attack(image_tensor, model, target_layer):
    """
    Apply region-specific attacks on an image tensor by distorting critical regions.

    Args:
        image_tensor (torch.Tensor): Input image tensor.
        model: Pre-trained model.
        target_layer: Target layer for Grad-CAM.

    Returns:
        np.ndarray: Perturbed image with region-specific distortions.
    """
    # Preprocess image tensor for Grad-CAM
    preprocessed_image = preprocess_image(image_tensor.cpu().numpy(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Generate Grad-CAM heatmap
    heatmap = generate_heatmap(preprocessed_image, model, target_layer)
    heatmap_resized = cv2.resize(heatmap, (image_tensor.shape[2], image_tensor.shape[3]))

    # Convert tensor to image
    image = image_tensor.permute(1, 2, 0).cpu().numpy() * 255
    image = image.astype(np.uint8)

    # Apply Gaussian blur to critical regions
    blurred = cv2.GaussianBlur(image, (51, 51), 0)
    region_attack = np.where(heatmap_resized > 0.5, blurred, image)

    # Convert back to tensor
    region_attack_tensor = torch.from_numpy(region_attack).permute(2, 0, 1).float() / 255
    return region_attack_tensor
