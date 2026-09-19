"""
Central config for RecycleVision.
Switch DATASET_VARIANT to work with any of the 3 supported Kaggle datasets.
"""

import os

# ---------------------------------------------------------------------------
# 1. CHOOSE DATASET VARIANT HERE: "6class" | "12class" | "10class"
# ---------------------------------------------------------------------------
DATASET_VARIANT = "6class"

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
DATASETS = {
    "6class": {
        "name": "Garbage Classification (6 Classes)",
        "kaggle_slug": "asdasdasasdas/garbage-classification",
        "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
        "approx_images": 2467,
        "raw_dir": "data/raw/6class",
        "images_subpath": "Garbage classification/Garbage classification",
    },
    "12class": {
        "name": "Garbage Classification (12 Classes)",
        "kaggle_slug": "mostafaabla/garbage-classification",
        "classes": [
            "paper", "cardboard", "biological", "metal", "plastic",
            "green-glass", "brown-glass", "white-glass",
            "battery", "clothes", "shoes", "trash",
        ],
        "approx_images": 15150,
        "raw_dir": "data/raw/12class",
        "images_subpath": "",
    },
    "10class": {
        "name": "Garbage Classification V2 (10 Classes)",
        "kaggle_slug": "sumn2u/garbage-classification-v2",
        "classes": None,  # populated dynamically from folder names after download
        "approx_images": None,
        "raw_dir": "data/raw/10class",
        "images_subpath": "",
    },
}

def get_dataset_config(variant: str = None):
    variant = variant or DATASET_VARIANT
    if variant not in DATASETS:
        raise ValueError(f"Unknown dataset variant '{variant}'. Choose from {list(DATASETS)}")
    return DATASETS[variant]

def get_images_dir(variant: str = None):
    """Returns the actual folder that directly contains the class subfolders."""
    cfg = get_dataset_config(variant)
    subpath = cfg.get("images_subpath", "")
    return os.path.join(cfg["raw_dir"], subpath) if subpath else cfg["raw_dir"]

# ---------------------------------------------------------------------------
# Image / training params
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

EPOCHS = 20
LEARNING_RATE = 1e-4

# Backbones to try for transfer learning
BACKBONES = ["MobileNetV2", "EfficientNetB0", "ResNet50"]

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)