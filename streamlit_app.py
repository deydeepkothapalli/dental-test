import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# Optional: Download model from Hugging Face on first run
@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id="nsitnov/8024-yolov8-model", filename="8024.pt")
    return YOLO(model_path)

model = load_model()

st.title("🦷 Dental X-ray Analysis with YOLOv8")
st.write("Upload a dental X-ray image to detect dental conditions (e.g., caries, implants, crowns).")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        results = model(temp_file.name)

    # Show detection image
    st.subheader("Detected Results")
    results[0].save(filename="prediction.jpg")
    st.image("prediction.jpg", caption="YOLOv8 Detections", use_column_width=True)

    # Show classes + confidences
    detections = results[0].boxes
    if detections:
        st.write("### Detected Classes")
        for box in detections:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls_id]
            st.write(f"🦷 **{label}** – confidence: `{conf:.2f}`")
    else:
        st.write("No dental anomalies detected.")

