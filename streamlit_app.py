import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('cat_dog_model.h5')

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.markdown("<h1 style='text-align: center;'>🐱 Cat vs Dog Classifier 🐶</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to identify whether it's a cat or a dog.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    resized = image.resize((128, 128))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = np.expand_dims(np.array(resized) / 255.0, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

    st.markdown(f"<h3 style='text-align: center;'>Prediction: {label}</h3>", unsafe_allow_html=True)
