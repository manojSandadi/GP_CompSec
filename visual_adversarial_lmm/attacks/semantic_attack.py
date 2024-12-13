import cv2
import numpy as np

def apply_semantic_attack(image: np.ndarray, hue_shift: int = 60) -> np.ndarray:
    """
    Apply a semantic attack by shifting the hue of the image.

    Args:
        image (np.ndarray): Input image in BGR format with shape (H, W, C).
        hue_shift (int): Amount to shift the hue (default is 60 degrees).

    Returns:
        np.ndarray: Adversarial image with shifted colors, in BGR format.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    # Check if the image has three channels (BGR)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (BGR format).")

    # Convert image to uint8 if necessary
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply the hue shift
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180

    # Convert back to BGR format
    adversarial_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return adversarial_image
