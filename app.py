import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("kidney_stone_model.h5")

model = load_model()

st.title("🩺 Kidney Stone Detection")
st.markdown("Upload an ultrasound scan to detect kidney stones.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Ultrasound Scan", use_column_width=True)
    img_array = np.array(img.resize((224, 224))) / 255.0
    prediction = model.predict(np.expand_dims(img_array, 0))[0][0]
    if prediction > 0.5:
        st.error("⚠️ Kidney Stone Detected")
    else:
        st.success("✅ No Kidney Stone Detected")
