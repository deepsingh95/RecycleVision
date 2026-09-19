"""
Fine-tuning script for RecycleVision.

Loads a trained model, unfreezes the last N layers of the backbone, and
continues training with a low learning rate. Supports resuming from an
already fine-tuned checkpoint (--resume) to do a second round of deeper
fine-tuning (more unfrozen layers) without losing the previous best model.

Usage:
    python fine_tune.py                              # round 1: from base model, unfreeze 30 layers
    python fine_tune.py --resume --unfreeze 60 --epochs 25 --lr 1e-5
                                                       # round 2: from round-1 model, unfreeze 60 layers
"""

import argparse
import os
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, IMG_SIZE, BATCH_SIZE, SEED
from model import _get_backbone
from train import get_class_weights


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
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", seed=SEED, shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", seed=SEED, shuffle=False,
    )
    return train_gen, val_gen


def find_backbone_layer(model):
    """Finds the nested pretrained backbone (e.g. 'mobilenetv2_1.00_224') inside the model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("Could not find nested backbone layer inside the model.")


def fine_tune(backbone: str = "MobileNetV2", unfreeze_last_n: int = 30,
              epochs: int = 15, resume: bool = False, lr: float = 1e-5):

    if resume:
        # Round 2+: start from the previous fine-tuned checkpoint
        source_path = os.path.join(MODELS_DIR, f"best_model_{backbone}_finetuned.keras")
        checkpoint_path = os.path.join(MODELS_DIR, f"best_model_{backbone}_finetuned_v2.keras")
        history_path = os.path.join(REPORTS_DIR, f"history_{backbone}_finetuned_v2.json")
    else:
        # Round 1: start from the base frozen-backbone model
        source_path = os.path.join(MODELS_DIR, f"best_model_{backbone}.keras")
        checkpoint_path = os.path.join(MODELS_DIR, f"best_model_{backbone}_finetuned.keras")
        history_path = os.path.join(REPORTS_DIR, f"history_{backbone}_finetuned.json")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Model not found: {source_path}.")

    print(f"Loading model to fine-tune from: {source_path}")
    model = tf.keras.models.load_model(source_path)

    backbone_layer = find_backbone_layer(model)
    backbone_layer.trainable = True

    for layer in backbone_layer.layers[:-unfreeze_last_n]:
        layer.trainable = False
    for layer in backbone_layer.layers[-unfreeze_last_n:]:
        layer.trainable = True

    print(f"Unfroze last {unfreeze_last_n} layers of {backbone_layer.name}")
    print(f"Learning rate: {lr}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    _, preprocess_fn = _get_backbone(backbone, IMG_SIZE + (3,))
    train_gen, val_gen = get_data_generators(preprocess_fn)
    class_weights = get_class_weights(train_gen)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)

    print(f"\nFine-tuned model saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", default="MobileNetV2",
        choices=["MobileNetV2", "EfficientNetB0", "ResNet50"],
    )
    parser.add_argument("--unfreeze", type=int, default=30,
                         help="Number of layers to unfreeze from the end of the backbone")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the already fine-tuned checkpoint instead of the base model")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning")
    args = parser.parse_args()

    fine_tune(backbone=args.backbone, unfreeze_last_n=args.unfreeze,
              epochs=args.epochs, resume=args.resume, lr=args.lr)