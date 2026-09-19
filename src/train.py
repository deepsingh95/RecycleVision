"""
Training script for RecycleVision.

Loads the processed train/val data, applies augmentation, computes class
weights to handle imbalance, trains the transfer-learning model, and saves
the best checkpoint to models/.

Usage:
    python train.py                       # trains with default backbone (MobileNetV2)
    python train.py --backbone EfficientNetB0
    python train.py --backbone ResNet50
"""

import argparse
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from config import (
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    SEED,
)
from model import build_model


def get_data_generators(preprocess_fn):
    train_dir = os.path.join(PROCESSED_DIR, "train")
    val_dir = os.path.join(PROCESSED_DIR, "val")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
        shuffle=False,
    )

    return train_gen, val_gen


def get_class_weights(train_gen):
    """Computes balanced class weights from the training generator's labels."""
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights (to handle imbalance):", class_weight_dict)
    return class_weight_dict


def train(backbone: str = "MobileNetV2", epochs: int = EPOCHS):
    print(f"\n=== Training with backbone: {backbone} ===\n")

    # We need a preliminary generator just to know num_classes before building model
    tmp_datagen = ImageDataGenerator()
    tmp_gen = tmp_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "train"), target_size=IMG_SIZE, batch_size=1
    )
    num_classes = tmp_gen.num_classes
    class_indices = tmp_gen.class_indices  # {class_name: index}

    model, preprocess_fn = build_model(num_classes=num_classes, backbone=backbone)
    model.summary()

    train_gen, val_gen = get_data_generators(preprocess_fn)
    class_weights = get_class_weights(train_gen)

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, f"best_model_{backbone}.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Save class index mapping (needed later for evaluate.py / the Streamlit app)
    class_indices_path = os.path.join(MODELS_DIR, f"class_indices_{backbone}.json")
    with open(class_indices_path, "w") as f:
        json.dump(class_indices, f, indent=2)
    print(f"Saved class index mapping to: {class_indices_path}")

    # Save training history for later plotting
    history_path = os.path.join(REPORTS_DIR, f"history_{backbone}.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"Saved training history to: {history_path}")

    print(f"\nBest model saved to: {checkpoint_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", default="MobileNetV2",
        choices=["MobileNetV2", "EfficientNetB0", "ResNet50"],
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    train(backbone=args.backbone, epochs=args.epochs)