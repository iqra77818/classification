import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# UI configs
st.set_page_config(page_title="Cats vs Dogs Classifier", layout="centered")

st.markdown("""
<center>

# 🐶 Cats vs Dogs Classifier 🐱  
Upload an image & the AI model will tell whether it's a **Cat** or a **Dog**.

</center>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_dogs_model.keras", compile=False)

model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

def preprocess(image):
    image = image.resize((150,150))
    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0).astype("float32")
    return img

if uploaded_file:
    st.markdown("### Preview")
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, width=350)

    with st.spinner("Predicting..."):
        img = preprocess(image)
        result = model.predict(img)[0][0]

    st.markdown("---")

    if result > 0.5:
        st.success(f"Prediction: **Dog** (Confidence: {result:.2f})")
        st.markdown("<center><h2>🐶</h2></center>", unsafe_allow_html=True)
    else:
        conf = 1-result
        st.success(f"Prediction: **Cat** (Confidence: {conf:.2f})")
        st.markdown("<center><h2>🐱</h2></center>", unsafe_allow_html=True)

else:
    st.info("Please upload an image to analyze.")

