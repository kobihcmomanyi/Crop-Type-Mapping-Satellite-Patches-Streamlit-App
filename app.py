import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Crop Type Classifier", layout="centered")

IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("crop_classifier.keras")
    with open("class_indices.json") as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    return model, idx_to_class

model, idx_to_class = load_model()

st.title("Crop Type Mapping from Satellite Patches")
st.write("Upload a EuroSAT-style satellite image patch to classify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized).astype("float32") / 255.0  # same rescale=1./255 as training
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Classifying..."):
        probs = model.predict(arr)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])

    st.subheader(f"Prediction: **{pred_label}**")
    st.write(f"Confidence: {confidence:.2%}")

    st.bar_chart({idx_to_class[i]: float(p) for i, p in enumerate(probs)})