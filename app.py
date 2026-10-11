import json
from pathlib import Path
import requests

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
LABELS_PATH = BASE_DIR / "class_names.json"
IMAGE_SIZE = (128, 128)

# Hugging Face direct download link
MODEL_URL = "https://huggingface.co/sikandarWithCode/fruit-veg-model/resolve/main/model.h5"

st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {background: linear-gradient(180deg, #f7fbf6 0%, #ffffff 55%);}
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px;}
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 22px;
        background: linear-gradient(120deg, #123c2b, #287a4b);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin-bottom: .35rem;}
    .hero p {color: #e4f5e8; font-size: 1.05rem; margin-bottom: 0;}
    .result {
        border: 1px solid #d8eadb; background: #f3faf3;
        border-radius: 18px; padding: 1.2rem 1.4rem;
    }
    .small-note {color: #5b6b60; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        with st.spinner("Downloading model from Hugging Face... Please wait."):
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Model download nahi ho saka.")
                return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    if not LABELS_PATH.exists():
        return None
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

with st.sidebar:
    st.header("🍏 Classifier settings")
    st.write("Upload a clear picture of one fruit or vegetable.")
    st.caption("The prediction is based on the classes used to train your model.")
    st.divider()
    st.write("**Model input:** 128 × 128 pixels")
    st.write("**Model type:** CNN")

st.markdown("""
<div class="hero">
    <h1>🍎 Fruit & Vegetable Classifier</h1>
    <p>Upload an image and let your trained CNN predict what it sees.</p>
</div>
""", unsafe_allow_html=True)

model = load_model()
class_names = load_labels()

if model is None or class_names is None:
    st.error("Model files are missing. Check `class_names.json` or model download link.")
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("📤 Upload an image")
    uploaded_file = st.file_uploader(
        "Choose a fruit or vegetable photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP",
    )
    st.caption("Tip: use a well-lit image with the item clearly visible.")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("This image could not be opened. Please try another image.")
        st.stop()

    with left:
        st.image(image, caption="Your uploaded image", use_container_width=True)

    with right:
        st.subheader("🔎 Prediction")
        with st.spinner("Analyzing image..."):
            resized = image.resize(IMAGE_SIZE)
            image_array = np.asarray(resized, dtype=np.float32)
            batch = np.expand_dims(image_array, axis=0)
            predictions = model.predict(batch, verbose=0)[0]

        if len(predictions) != len(class_names):
            st.error(
                f"Model returned {len(predictions)} scores, but class_names.json has "
                f"{len(class_names)} labels. Re-save the class names from the same training dataset."
            )
            st.stop()

        top_indices = np.argsort(predictions)[::-1][:3]
        best_idx = int(top_indices[0])
        best_label = class_names[best_idx]
        confidence = float(predictions[best_idx]) * 100

        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.markdown("**Predicted class**")
        st.title(f"🍃 {best_label.replace('_', ' ').title()}")
        st.progress(min(max(confidence / 100, 0.0), 1.0))
        st.metric("Model confidence", f"{confidence:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("**Top 3 predictions**")
        for idx in top_indices:
            idx = int(idx)
            score = float(predictions[idx]) * 100
            st.write(f"{class_names[idx].replace('_', ' ').title()} — **{score:.2f}%**")
            st.progress(min(max(score / 100, 0.0), 1.0))

        st.caption("Confidence is the model's score, not a guarantee that the prediction is correct.")
else:
    with right:
        st.subheader("✨ Ready when you are")
        st.write("Your result will appear here after you upload a picture.")
        st.markdown("""
        - Choose a picture from your computer.
        - Click **Browse files** above.
        - The app will resize the image and run your CNN.
        - View the predicted class and top 3 scores.
        """)

st.divider()
st.markdown(
    '<p class="small-note">Built with Streamlit and TensorFlow • '
    'For best results, upload images similar to those used during training.</p>',
    unsafe_allow_html=True,
)
