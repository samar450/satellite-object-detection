import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import glob
import shutil
import numpy as np

# Load YOLOv8 model only once with caching
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Path to your trained model
    model = YOLO(model_path)
    return model

model = load_model()

st.title("Satellite Object Detection with YOLOv8")

# Upload image
uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Clean previous predictions
    if os.path.exists("runs/detect/predict"):
        shutil.rmtree("runs/detect/predict")

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image temporarily
    temp_image_path = "temp_image.jpg"
    image.convert("RGB").save(temp_image_path)

    # Predict with YOLOv8
    results = model.predict(temp_image_path, save=True, conf=0.25)

    # Locate saved result image
    result_dir = "runs/detect/predict"
    result_files = glob.glob(f"{result_dir}/*.jpg")

    if result_files:
        result_file = result_files[0]
        result_image = Image.open(result_file)
        st.image(result_image, caption="Detection Result", use_container_width=True)
    else:
        st.warning("No prediction result found.")

    # Clean up temporary file
    os.remove(temp_image_path)

    # Show labels/classes detected
    st.subheader("Detected Classes")
    if results[0].boxes is not None and results[0].boxes.cls.numel() > 0:
        detected_classes = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names
        unique_classes = set(detected_classes)
        for cls_id in unique_classes:
            st.write(f"- {class_names[int(cls_id)]}")
    else:
        st.write("No objects detected.")
