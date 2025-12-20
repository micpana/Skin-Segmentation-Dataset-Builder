"""
I/O utilities for dataset building.
"""

import os
import cv2
import numpy as np


def get_class_names(dataset_original):
    train_dir = os.path.join(dataset_original, "train")
    return sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])


def get_image_paths(dataset_original, split, class_names):
    paths = []

    for cls in class_names:
        cls_dir = os.path.join(dataset_original, split, cls)
        if not os.path.exists(cls_dir):
            continue

        for file in os.listdir(cls_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                paths.append((
                    os.path.join(cls_dir, file),
                    cls
                ))

    return paths


def prepare_output_dirs(dataset_output):
    for sub in ["images", "masks"]:
        for split in ["train", "valid", "test"]:
            os.makedirs(
                os.path.join(dataset_output, sub, split),
                exist_ok=True
            )


def save_image(image, dataset_output, split, filename):
    path = os.path.join(dataset_output, "images", split, filename)
    cv2.imwrite(path, image)


def save_mask(mask, dataset_output, split, filename):
    path = os.path.join(dataset_output, "masks", split, filename)
    cv2.imwrite(path, mask)


def save_classes_txt(dataset_output, class_names):
    with open(os.path.join(dataset_output, "classes.txt"), "w") as f:
        for cls in class_names:
            f.write(f"{cls}\n")


# ------------------------
# Colorize mask for preview
# ------------------------
def colorize_mask(mask, class_names):
    """
    Converts integer mask into RGB color image for human inspection.
    mask: H x W, values 0..N
    class_names: list of class names
    """
    h, w = mask.shape
    mask_color = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign a unique color per class
    np.random.seed(42)  # for consistent colors
    colors = {0: (0, 0, 0)}  # background = black
    for idx, cls in enumerate(class_names, start=1):
        colors[idx] = tuple(np.random.randint(0, 256, 3).tolist())

    for cls, color in colors.items():
        mask_color[mask == cls] = color

    return mask_color


def save_mask_preview(mask, dataset_output, split, filename, class_names):
    """
    Saves colorized mask preview to masks_preview folder.
    """
    preview_dir = os.path.join(dataset_output, "masks_preview", split)
    os.makedirs(preview_dir, exist_ok=True)
    mask_color = colorize_mask(mask, class_names)
    preview_path = os.path.join(preview_dir, filename)
    cv2.imwrite(preview_path, mask_color)
