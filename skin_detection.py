"""
Skin detection and extraction logic.
"""

import cv2
import numpy as np
import mediapipe as mp


mp_face = mp.solutions.face_detection


def detect_face(image):
    """
    Detects the primary face in an image.
    """
    with mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as detector:
        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return None

        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = image.shape

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = x1 + int(bbox.width * w)
        y2 = y1 + int(bbox.height * h)

        return image[y1:y2, x1:x2]


def skin_color_mask(face_img):
    """
    Generates a binary skin mask using HSV + YCrCb thresholds.
    """
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)

    hsv_mask = cv2.inRange(
        hsv,
        (0, 40, 60),
        (25, 255, 255)
    )

    ycrcb_mask = cv2.inRange(
        ycrcb,
        (0, 135, 85),
        (255, 180, 135)
    )

    mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def extract_skin(image_path):
    """
    Extracts skin-only image and mask.
    Works for:
    - Face images (selfies)
    - Partial skin images
    - Skin-only datasets
    """
    USE_FACE_DETECTION = True
    FACE_REQUIRED = False

    image = cv2.imread(image_path)
    if image is None:
        return None, None

    # Try face detection first
    face = detect_face(image)

    if (face is not None) and USE_FACE_DETECTION:
        region = face
    else:
        if FACE_REQUIRED:
            return None, None
        # Fallback to full image
        region = image

    mask = skin_color_mask(region)

    # Reject images with insufficient skin pixels
    skin_pixel_ratio = np.sum(mask > 0) / mask.size
    if skin_pixel_ratio < 0.05:
        return None, None

    skin_only = cv2.bitwise_and(region, region, mask=mask)

    return skin_only, mask
