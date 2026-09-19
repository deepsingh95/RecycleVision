"""
RecycleVision - Streamlit App

A simple web UI: upload a garbage/waste image, get the predicted category
with a confidence score and top-3 predictions.

Usage (from the project root):
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Make src/ importable regardless of where streamlit is launched from
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.append(SRC_DIR)

from model import _get_backbone  # noqa: E402
from config import IMG_SIZE  # noqa: E402


# ---------------------------------------------------------------------------
# Config: which trained model to serve
# ---------------------------------------------------------------------------
BACKBONE = "EfficientNetB0"
MODEL_SUFFIX = "_finetuned"   # matches the best model we trained

# Build absolute paths based on this file's location + src/models,
# so this works no matter which directory `streamlit run` is launched from.
MODELS_DIR_ABS = os.path.join(SRC_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR_ABS, f"best_model_{BACKBONE}{MODEL_SUFFIX}.keras")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR_ABS, f"class_indices_{BACKBONE}.json")


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)  # {class_name: index}
    idx_to_class = {v: k for k, v in class_indices.items()}
    _, preprocess_fn = _get_backbone(BACKBONE, IMG_SIZE + (3,))
    return model, idx_to_class, preprocess_fn


def predict(image: Image.Image, model, idx_to_class, preprocess_fn):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_fn(arr)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:3]

    results = [(idx_to_class[i], float(probs[i])) for i in top_indices]
    return results


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RecycleVision", page_icon="♻️", layout="centered")

st.title("♻️ RecycleVision")
st.caption("Garbage Image Classification using Deep Learning")

st.write(
    "Upload a photo of a waste item and the model will predict which "
    "recycling category it belongs to: **cardboard, glass, metal, paper, "
    "plastic, or trash**."
)

model, idx_to_class, preprocess_fn = load_model_and_classes()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        results = predict(image, model, idx_to_class, preprocess_fn)

    top_class, top_conf = results[0]

    with col2:
        st.subheader("Prediction")
        st.markdown(f"### 🏷️ {top_class.capitalize()}")
        st.metric("Confidence", f"{top_conf * 100:.1f}%")

        st.write("**Top-3 predictions:**")
        for cls, conf in results:
            st.progress(conf, text=f"{cls.capitalize()}: {conf * 100:.1f}%")

    st.divider()
    st.caption(
        f"Model: EfficientNetB0 (fine-tuned) · Trained on RecycleVision dataset"
    )
else:
    st.info("👆 Upload an image to get a prediction.")

with st.sidebar:
    st.header("About")
    st.write(
        "RecycleVision classifies waste images into 6 categories to help "
        "automate recycling and reduce manual sorting effort."
    )
    st.write("**Classes:** cardboard, glass, metal, paper, plastic, trash")
    st.write("**Model:** EfficientNetB0 (Transfer Learning, fine-tuned)")
    st.write("**Test Accuracy:** ~85.4%")