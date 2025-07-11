import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Battery CT Scanner", layout="centered", initial_sidebar_state="collapsed")
st.title("🔋 Lithium Battery CT Scanner")

MODEL_PATH = "battery_model.h5"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded = st.file_uploader("📸 Upload IR Battery Image", type=["jpg", "png"])
if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1, 64, 64, 1) / 255.0
    try:
        preds = model.predict(gray)
        label_idx = np.argmax(preds)
        labels = ["Healthy ✅", "Bulging ⚠️", "Cracked ❌"]
        st.subheader(f"🧠 AI Diagnosis: **{labels[label_idx]}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
