import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image

# Load model once at startup
@st.cache_resource
def load_model():
    model = keras.models.load_model("b_CNN.h5")
    return model

model = load_model()
st.title("🐾 Dog vs Cat Classifier")

st.write("Upload an image of a dog or a cat, and the model will predict which it is.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Convert image to array
    img = np.array(image)

    # Ensure it's in RGB
    if img.shape[-1] == 4:  # Handle PNG with alpha channel
        img = img[:, :, :3]

    # Resize and preprocess
    img_resized = cv2.resize(img, (256, 256))
    test_input = img_resized.reshape((1, 256, 256, 3))

    # Predict
    output = model.predict(test_input)

    prob = float(output[0][0])

    prediction = "🐶 Dog" if prob > 0.5 else "🐱 Cat"

    st.write(f"### Prediction: {prediction}")
    st.write(f"Raw Model Output: {prob:.4f}")

    
