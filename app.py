import sys
import subprocess

# Safely force the headless version of OpenCV to bypass Streamlit's linux environment bugs
if sys.platform.startswith("linux"):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    except Exception:
        pass

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import glob

# Try to finding best.pt locally (for cloud deployment) or find the latest trained model
def get_latest_model():
    local_model = "best.pt"
    if os.path.exists(local_model):
        return local_model

    base_dir = r"C:\Users\sathw\runs\detect\PCB_Defect_Project"
    if not os.path.exists(base_dir):
        return None
    
    # Get all unified_model_v* folders
    folders = glob.glob(os.path.join(base_dir, "unified_model_v*"))
    if not folders:
        return None
    
    # Sort folders by modification time to get the newest
    latest_folder = max(folders, key=os.path.getmtime)
    model_path = os.path.join(latest_folder, "weights", "best.pt")
    
    if os.path.exists(model_path):
        return model_path
    return None

st.set_page_config(page_title="PCB Defect Detector", layout="wide")

st.title("🧩 PCB Defect Detection App")
st.write("Upload an image of a PCB to automatically detect scratches and solder anomalies.")

# Find latest model automatically or let user specify
latest_model_path = get_latest_model()
model_path = st.text_input("Path to your trained model (.pt):", value=latest_model_path if latest_model_path else "yolov8n.pt")

@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

if model_path:
    model = load_model(model_path)
    if model:
        uploaded_file = st.file_uploader("Drag and drop your PCB image here...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            # Predict
            with st.spinner("Detecting defects..."):
                # Convert PIL to numpy for YOLO
                img_array = np.array(image.convert("RGB"))
                results = model.predict(source=img_array, conf=0.25)
                
                # Get annotated image
                res = results[0]
                annotated_img = res.plot()  # BGR format natively from ultralytics
                
                # Convert BGR string to RGB for PIL/Streamlit
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
            # Layout Side by Side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("Detected Defects")
                st.image(annotated_img_rgb, use_container_width=True)
                
            # Display summary metrics if defects are found
            if len(res.boxes) > 0:
                st.warning(f"⚠️ Found {len(res.boxes)} defects in this PCB!")
                
                classes = res.names
                detected_classes = [classes[int(c)] for c in res.boxes.cls]
                
                # Create a simple breakdown table
                import pandas as pd
                breakdown = pd.Series(detected_classes).value_counts().reset_index()
                breakdown.columns = ["Defect Type", "Count"]
                st.dataframe(breakdown, hide_index=True)
            else:
                st.success("✅ No defects detected. The board looks perfectly clean!")
