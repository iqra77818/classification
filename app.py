import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.markdown("""
<div style="text-align:center;">
    <h1>🐶🐱 Cat vs Dog Image Classifier</h1>
    <p style="font-size:18px; font-weight:500;">
        Upload an image and our AI model will predict if it is a Dog or a Cat!
    </p>
</div>
""", unsafe_allow_html=True)

# Load Model
model = tf.keras.models.load_model("cats_dogs_model.keras")

# File Upload Box
uploaded_file = st.file_uploader("Upload Image Here", type=["png", "jpg", "jpeg"])

def preprocess(img):
    img = img.resize((200,200))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.markdown("### 👇 Uploaded Image")
    st.image(image, width=350)

    img = preprocess(image)
    prediction = model.predict(img)[0][0]

    result = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    prob_dog = float(prediction * 100)
    prob_cat = float((1 - prediction) * 100)

    st.markdown("---")
    st.markdown(f"<h2 style='text-align:center;'>Prediction: {result}</h2>", unsafe_allow_html=True)

    st.markdown("### Probability Score")
    st.progress(prob_dog/100)  # simple bar

    st.write(f"🐶 Dog probability: **{prob_dog:.2f}%**")
    st.write(f"🐱 Cat probability: **{prob_cat:.2f}%**")
