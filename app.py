import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from keras.models import load_model

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

model = load_model("emotion_cnn.keras", compile=False)

st.title("Facial Emotion Recognition")
st.write("Take a photo or upload one. Keep your face close to the camera.")

photo = st.camera_input("Webcam")
upload = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

file = photo if photo is not None else upload

if file is not None:
    img = Image.open(file).convert("L").resize((48, 48))   # gray + 48x48 like the training data
    x = np.array(img) / 255
    x = x.reshape(1, 48, 48, 1)

    probs = model.predict(x)[0]
    pred = class_names[probs.argmax()]

    st.image(file, width=250)
    st.subheader(f"Prediction: {pred} ({probs.max() * 100:.1f}%)")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=class_names))
