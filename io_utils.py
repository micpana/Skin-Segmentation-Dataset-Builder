"""
I/O utilities for dataset building.
"""

import os
import cv2


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
