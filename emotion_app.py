import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

model_path = r"C:\Users\Nebras-PC\Downloads\Project_5\emotion_recognition_model.keras"

@st.cache_resource
def load_my_model():
    if os.path.exists(model_path):
        return load_model(model_path, compile=False)
    else:
        fallback_path = r"C:\Users\Nebras-PC\Downloads\Project_5\emotion_recognition_model.h5"
        if os.path.exists(fallback_path):
            return load_model(fallback_path, compile=False)
    raise FileNotFoundError("Model file not found!")

model = load_my_model()

# The 7 emotion classes in FER-2013 order
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.title("😊 Facial Emotion Recognition App")
st.write("Detect expressions using your deep learning model by uploading an image or using your camera!")

# اختيار طريقة الإدخال (رفع صورة أو استخدام الكاميرا)
input_option = st.radio("Choose input method:", ("Upload an Image", "Take a Picture with Camera"))

image = None

if input_option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(camera_file)

if image is not None:
    st.image(image, caption='Selected Image', use_container_width=True)

    st.write("Analyzing the image...")

    image = image.convert('RGB') 
    image = image.resize((48, 48))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 48, 48, 3)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_class]
    confidence = np.max(prediction) * 100

    st.success(f"### Predicted Emotion: {predicted_emotion} (Confidence: {confidence:.2f}%)")

    st.write("#### Probability Breakdown for All Emotions:")
    for i, emotion in enumerate(emotion_labels):
        st.write(f"- {emotion}: {prediction[0][i]*100:.2f}%")
