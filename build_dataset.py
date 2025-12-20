"""
Skin Segmentation Dataset Builder

Converts a classification dataset into a segmentation dataset
by automatically extracting skin-only regions and masks.

Author: Michael Panashe Mudimbu
License: MIT
"""

import os
from tqdm import tqdm
from io_utils import (
    get_class_names,
    get_image_paths,
    prepare_output_dirs,
    save_image,
    save_mask,
    save_classes_txt
)
from skin_detection import extract_skin
from overlay_utils import overlay_mask_on_image
import numpy as np


DATASET_ORIGINAL = "dataset_original"
DATASET_OUTPUT = "dataset"


def process_split(split: str, class_names: list):
    """
    Processes one dataset split (train / valid / test)
    """
    image_paths = get_image_paths(DATASET_ORIGINAL, split, class_names)
    class_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}

    for img_path, class_name in tqdm(image_paths, desc=f"Processing {split}"):
        skin_image, binary_mask = extract_skin(img_path)

        if skin_image is None or binary_mask is None:
            continue

        # -----------------------------
        # Convert binary mask to class-labeled mask
        # -----------------------------
        mask = np.zeros_like(binary_mask, dtype=np.uint8)
        mask[binary_mask > 0] = class_to_id[class_name]
        
        # sanity check: ensure mask class IDs are within valid range
        assert mask.max() <= len(class_names), f"Mask ID exceeds number of classes: {mask.max()}"
        # sanity check: ensure mask has no negative values
        assert mask.min() >= 0, f"Mask has negative values: {mask.min()}"

        filename = os.path.basename(img_path)

        save_image(
            skin_image,
            DATASET_OUTPUT,
            split,
            filename
        )

        save_mask(
            mask,
            DATASET_OUTPUT,
            split,
            filename
        )


def main():
    class_names = get_class_names(DATASET_ORIGINAL)

    prepare_output_dirs(DATASET_OUTPUT)
    save_classes_txt(DATASET_OUTPUT, class_names)

    for split in ["train", "valid", "test"]:
        process_split(split, class_names)

    print("Dataset build complete.")


if __name__ == "__main__":
    main()
