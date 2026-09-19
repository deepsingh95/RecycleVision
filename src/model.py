"""
Model architecture definitions for RecycleVision.

Uses transfer learning: a pretrained backbone (frozen) + custom
classification head on top. Supports MobileNetV2, EfficientNetB0, ResNet50.

Usage (as a module):
    from model import build_model
    model = build_model(num_classes=6, backbone="MobileNetV2")
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMG_SIZE


def _get_backbone(name: str, input_shape):
    """Returns (backbone_model, preprocess_function) for the given backbone name."""
    name = name.lower()

    if name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    elif name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    elif name == "resnet50":
        base = tf.keras.applications.ResNet50(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input

    else:
        raise ValueError(f"Unknown backbone '{name}'. Choose MobileNetV2, EfficientNetB0, or ResNet50.")

    return base, preprocess


def build_model(num_classes: int, backbone: str = "MobileNetV2", fine_tune_last_n: int = 0):
    """
    Builds a transfer-learning classification model.

    Args:
        num_classes: number of output classes.
        backbone: one of "MobileNetV2", "EfficientNetB0", "ResNet50".
        fine_tune_last_n: if > 0, unfreezes the last N layers of the backbone
                           for fine-tuning. 0 means backbone is fully frozen.

    Returns:
        (model, preprocess_function)
    """
    input_shape = IMG_SIZE + (3,)
    base, preprocess = _get_backbone(backbone, input_shape)

    # Freeze base layers by default (feature extraction mode)
    base.trainable = False
    if fine_tune_last_n > 0:
        base.trainable = True
        for layer in base.layers[:-fine_tune_last_n]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"RecycleVision_{backbone}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, preprocess


if __name__ == "__main__":
    # Quick sanity check: build and summarize a model
    model, _ = build_model(num_classes=6, backbone="MobileNetV2")
    model.summary()