"""
Preprocessing pipeline for RecycleVision.

Does:
1. Finds all images under the dataset's class folders.
2. Splits them into train / val / test sets (stratified by class).
3. Resizes + copies them into data/processed/{train,val,test}/{class}/...
   (Augmentation itself is applied on-the-fly during training via
   ImageDataGenerator / tf.keras preprocessing layers -- see train.py --
   this script only prepares clean, resized, split image folders.)

Usage:
    python preprocessing.py
"""

import os
import shutil
import random
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

from config import (
    get_dataset_config,
    get_images_dir,
    IMG_SIZE,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    PROCESSED_DIR,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_class_folders(images_dir):
    """Returns {class_name: [list of image file paths]}"""
    class_to_files = defaultdict(list)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    for class_name in sorted(os.listdir(images_dir)):
        class_path = os.path.join(images_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                class_to_files[class_name].append(os.path.join(class_path, fname))

    return class_to_files


def split_files(files, seed=SEED):
    """Shuffles and splits a list of file paths into train/val/test."""
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test


def process_and_save(file_list, dest_dir, img_size=IMG_SIZE):
    os.makedirs(dest_dir, exist_ok=True)
    for src_path in file_list:
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")            # handles grayscale/RGBA edge cases
                img = img.resize(img_size)
                fname = os.path.basename(src_path)
                dest_path = os.path.join(dest_dir, fname)
                img.save(dest_path, quality=95)
        except Exception as e:
            print(f"  [WARN] Skipping corrupt/unreadable image: {src_path} ({e})")


def preprocess(variant: str = None):
    cfg = get_dataset_config(variant)
    images_dir = get_images_dir(variant)

    print(f"Dataset: {cfg['name']}")
    print(f"Reading images from: {images_dir}")

    class_to_files = find_class_folders(images_dir)
    if not class_to_files:
        raise RuntimeError(
            f"No class folders/images found under {images_dir}. "
            "Check config.py's 'images_subpath' matches the actual folder structure."
        )

    print(f"\nFound {len(class_to_files)} classes:")
    total = 0
    for cls, files in class_to_files.items():
        print(f"  {cls}: {len(files)} images")
        total += len(files)
    print(f"Total images: {total}\n")

    # Clean previous processed data for this run
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    for cls, files in class_to_files.items():
        train, val, test = split_files(files)

        print(f"[{cls}] train={len(train)} val={len(val)} test={len(test)}")

        process_and_save(train, os.path.join(PROCESSED_DIR, "train", cls))
        process_and_save(val, os.path.join(PROCESSED_DIR, "val", cls))
        process_and_save(test, os.path.join(PROCESSED_DIR, "test", cls))

    print(f"\nDone. Processed data saved under: {PROCESSED_DIR}")
    print(f"Structure: {PROCESSED_DIR}/(train|val|test)/<class_name>/*.jpg, resized to {IMG_SIZE}")


if __name__ == "__main__":
    preprocess()