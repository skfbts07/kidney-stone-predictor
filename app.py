import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# Title
st.title("🧠 Kidney Stone Detection from Ultrasound Scan")

# Step 1: Download model from Google Drive (if not already present)
model_path = "kidney_stone_model.keras"
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download("https://drive.google.com/uc?id=1xZrR5K1kbiBUP6pBZqobpo0zwpLLEanJ", model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Image upload
uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "🟠 Stone Detected" if prediction > 0.5 else "✅ No Stone Detected"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
