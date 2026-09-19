"""
Evaluation script for RecycleVision.

Loads the best trained model, evaluates it on the held-out test set, and
generates:
1. Overall + per-class metrics (Accuracy, Precision, Recall, F1-Score)
2. A confusion matrix plot
3. A text evaluation report

Usage:
    python evaluate.py                        # evaluates MobileNetV2 model
    python evaluate.py --backbone EfficientNetB0
"""

import argparse
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

from config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, IMG_SIZE, BATCH_SIZE
from model import _get_backbone


def load_class_indices(backbone):
    path = os.path.join(MODELS_DIR, f"class_indices_{backbone}.json")
    with open(path, "r") as f:
        class_indices = json.load(f)  # {class_name: index}
    # invert to {index: class_name}
    idx_to_class = {v: k for k, v in class_indices.items()}
    return idx_to_class


def get_test_generator(backbone):
    _, preprocess_fn = _get_backbone(backbone, IMG_SIZE + (3,))
    test_dir = os.path.join(PROCESSED_DIR, "test")

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,  # important: keep order so predictions align with labels
    )
    return test_gen


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def evaluate(backbone: str = "MobileNetV2", suffix: str = ""):
    model_path = os.path.join(MODELS_DIR, f"best_model_{backbone}{suffix}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py (and fine_tune.py) first.")

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    idx_to_class = load_class_indices(backbone)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    test_gen = get_test_generator(backbone)
    y_true = test_gen.classes

    print("Running predictions on test set...")
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # ---- Overall metrics ----
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"\n=== Overall Test Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # ---- Per-class report ----
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\n=== Per-Class Report ===")
    print(report)

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(REPORTS_DIR, f"confusion_matrix_{backbone}{suffix}.png"))

    # ---- Save full text report ----
    report_path = os.path.join(REPORTS_DIR, f"evaluation_report_{backbone}{suffix}.txt")
    with open(report_path, "w") as f:
        f.write(f"RecycleVision Evaluation Report - Backbone: {backbone}{suffix}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy:  {acc:.4f}\n")
        f.write(f"Overall Precision: {precision:.4f}\n")
        f.write(f"Overall Recall:    {recall:.4f}\n")
        f.write(f"Overall F1-Score:  {f1:.4f}\n\n")
        f.write("Per-Class Report:\n")
        f.write(report)
    print(f"\nSaved evaluation report to: {report_path}")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", default="MobileNetV2",
        choices=["MobileNetV2", "EfficientNetB0", "ResNet50"],
    )
    parser.add_argument("--suffix", default="",
                         help="Model suffix to evaluate: '' (base), '_finetuned', or '_finetuned_v2'")
    args = parser.parse_args()

    evaluate(backbone=args.backbone, suffix=args.suffix)