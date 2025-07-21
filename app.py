import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("🧠 Kidney Stone Predictor")
st.write("Upload an ultrasound scan to check for kidney stones.")

# Model file settings
MODEL_URL = "https://drive.google.com/uc?id=1xZrR5K1kbiBUP6pBZqobpo0zwpLLEanJ"
MODEL_PATH = "kidney_stone_model.keras"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Upload image
uploaded_file = st.file_uploader("📤 Upload Ultrasound Scan", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Scan", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]
    st.write(f"Prediction Score: `{pred:.4f}`")

    # Adjusted threshold to reduce false positives
    if pred > 0.7:
        st.error("🛑 Stone Predicted")
    else:
        st.success("✅ No Stone Predicted")
