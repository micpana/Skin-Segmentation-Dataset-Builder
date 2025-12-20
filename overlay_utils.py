"""
Utilities for overlaying masks during inference.
"""

import cv2
import numpy as np


def overlay_mask_on_image(
    original_image,
    mask,
    color=(0, 255, 0),
    alpha=0.4
):
    """
    Overlays a segmentation mask on the original image.

    Args:
        original_image (BGR)
        mask (binary 0/255)
        color (BGR)
        alpha (float)

    Returns:
        overlayed_image
    """
    overlay = original_image.copy()
    color_mask = np.zeros_like(original_image)
    color_mask[mask > 0] = color

    cv2.addWeighted(
        color_mask,
        alpha,
        overlay,
        1 - alpha,
        0,
        overlay
    )

    return overlay
