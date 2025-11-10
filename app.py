import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cats vs Dogs Classifier", layout="centered")

# custom CSS ui premium glassmorphism
page_bg = """
<style>
.main {
    background: linear-gradient(135deg, #ebf4ff 0%, #fdf2f8 100%);
}
.upload-box {
    background: rgba(255,255,255,0.55);
    border-radius: 18px;
    padding: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
}
.predict-btn {
    width:100%;
    font-size:18px;
    border-radius:10px;
    height:55px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; font-size:40px; color:#1a1a1a;'>🐶 Cats vs Dogs AI Classifier 🐱</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#4B5563;'>Upload an image & let AI detect which one it is</h4>", unsafe_allow_html=True)
st.write("")

model = tf.keras.models.load_model("cats_dogs_model.keras")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

def preprocess(img):
    img = img.resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype("float32")
    return img

st.markdown('<div class="upload-box">', unsafe_allow_html=True)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image Preview", width=350)

    if st.button("🔍 Predict", type="primary"):
        p = preprocess(img)
        pred = model.predict(p)[0][0]

        if pred > 0.5:
            st.markdown("<h2 style='text-align:center; color:#059669;'>🐶 Dog Detected!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align:center; color:#DC2626;'>🐱 Cat Detected!</h2>", unsafe_allow_html=True)
else:
    st.info("Upload an image to start prediction 🖼️")

st.markdown('</div>', unsafe_allow_html=True)

