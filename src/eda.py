"""
Exploratory Data Analysis for RecycleVision.

Generates and saves to reports/:
1. class_distribution.png -- bar chart of image counts per class (train set)
2. sample_images.png       -- grid of example images from each class
3. pixel_intensity.png     -- average pixel intensity distribution per class

Usage:
    python eda.py
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import PROCESSED_DIR, REPORTS_DIR, SEED

random.seed(SEED)


def get_class_counts(split="train"):
    split_dir = os.path.join(PROCESSED_DIR, split)
    counts = {}
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if os.path.isdir(cls_dir):
            counts[cls] = len(os.listdir(cls_dir))
    return counts


def plot_class_distribution(counts, save_path):
    classes = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(9, 5))
    bars = plt.bar(classes, values, color="#4C9A2A")
    plt.title("Class Distribution (Training Set)")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=30)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 2, str(val),
                  ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_images(split="train", n_per_class=3, save_path=None):
    split_dir = os.path.join(PROCESSED_DIR, split)
    classes = sorted(os.listdir(split_dir))

    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 2.5, len(classes) * 2.5))

    for row, cls in enumerate(classes):
        cls_dir = os.path.join(split_dir, cls)
        files = os.listdir(cls_dir)
        sample_files = random.sample(files, min(n_per_class, len(files)))

        for col in range(n_per_class):
            ax = axes[row, col] if len(classes) > 1 else axes[col]
            if col < len(sample_files):
                img = Image.open(os.path.join(cls_dir, sample_files[col]))
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=10)
                ax.text(-0.1, 0.5, cls, fontsize=11, rotation=90,
                         va="center", ha="center", transform=ax.transAxes)

    plt.suptitle("Sample Images per Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_pixel_intensity(split="train", n_samples_per_class=50, save_path=None):
    split_dir = os.path.join(PROCESSED_DIR, split)
    classes = sorted(os.listdir(split_dir))

    plt.figure(figsize=(9, 5))
    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        files = os.listdir(cls_dir)
        sample_files = random.sample(files, min(n_samples_per_class, len(files)))

        intensities = []
        for f in sample_files:
            img = Image.open(os.path.join(cls_dir, f)).convert("L")  # grayscale
            arr = np.array(img).flatten()
            intensities.append(arr.mean())

        plt.hist(intensities, bins=20, alpha=0.5, label=cls)

    plt.title("Average Pixel Intensity Distribution per Class")
    plt.xlabel("Mean Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def run_eda():
    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(
            f"{PROCESSED_DIR} not found. Run preprocessing.py first."
        )

    print("Running EDA...\n")

    counts = get_class_counts("train")
    print("Class counts (train):")
    for cls, n in counts.items():
        print(f"  {cls}: {n}")

    plot_class_distribution(counts, os.path.join(REPORTS_DIR, "class_distribution.png"))
    plot_sample_images(save_path=os.path.join(REPORTS_DIR, "sample_images.png"))
    plot_pixel_intensity(save_path=os.path.join(REPORTS_DIR, "pixel_intensity.png"))

    print(f"\nAll EDA plots saved in: {REPORTS_DIR}/")


if __name__ == "__main__":
    run_eda()