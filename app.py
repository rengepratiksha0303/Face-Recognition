import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Face Emotion Recognition",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ TITLE ------------------
st.title("😊 Face Emotion Recognition App")

# ------------------ EMOTION LABELS ------------------
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.h5")  # change if needed

model = load_emotion_model()

# ------------------ IMAGE PREPROCESSING ------------------
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]

    st.success(f"### 😃 Predicted Emotion: **{emotion}**")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Built using FER2013 Dataset • Streamlit App")
