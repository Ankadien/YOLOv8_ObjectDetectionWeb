import streamlit as st
from PIL import Image
import os
import uuid
from ultralytics import YOLO

# Set page config
st.set_page_config(page_title="YOLOv8 Detection", layout="centered")

# ====== Title ======
st.markdown("## 🧠 Object Detection Web App using YOLOv8")
st.markdown("👤 **Author:** Nguyễn Hoài An")
st.markdown("---")

# Apply custom UI
st.markdown("""
    <style>
    .stButton button {
        background-color: #3b3b3b;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .stFileUploader {
        background-color: #1e1e1e;
        border: 1px solid #3b3b3b;
        border-radius: 10px;
        padding: 1em;
    }
    .stTextInput, .stSelectbox, .stTextArea {
        background-color: #1e1e1e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown("### 🖼️ Image Config")
st.write("Choose an image...")

# File uploader
uploaded_file = st.file_uploader(
    "Drag and drop file here", 
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    label_visibility="collapsed"
)

# Check if file uploaded
if uploaded_file:
    # Show file info
    st.write(f"📂 **{uploaded_file.name}** — {round(uploaded_file.size / 1024, 1)} KB")

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Preview", use_container_width=True)

    # Create temp folder
    os.makedirs("uploads", exist_ok=True)
    temp_path = f"uploads/{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Button for detection
    if st.button("🔍 Detect Objects"):
        # Load YOLO model (yolov8n for speed)
        model = YOLO("yolov8n.pt")

        # Run detection
        with st.spinner("Detecting..."):
            results = model(temp_path)
            result_img = results[0].plot()

        # Show result
        st.image(result_img, caption="🎯 Detection Result", use_container_width=True)

        # Cleanup (optional)
        os.remove(temp_path)
