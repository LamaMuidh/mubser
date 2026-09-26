import cv2
import streamlit as st
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Mubsir | Face Detection",
    layout="wide"
)

# Logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo1.jpeg", width=300)

st.title("👁️ Mubsir | Face Detection")
st.write(
    "Computer vision prototype for face detection with "
    "normal and thermal-style visualization."
)

# Load Haar Cascade safely
cascade_path = Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"

if not cascade_path.exists():
    st.error("Face detection model could not be loaded.")
    st.stop()

face_cascade = cv2.CascadeClassifier(str(cascade_path))

# Browser camera
camera_image = st.camera_input("Take a photo")

if camera_image is not None:

    # Convert uploaded camera image to OpenCV format
    file_bytes = np.asarray(
        bytearray(camera_image.getvalue()),
        dtype=np.uint8
    )

    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Normal view
    normal_frame = frame.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(
            normal_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Thermal-style visualization
    thermal_frame = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_JET
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            thermal_frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2
        )

    st.success(f"Detected {len(faces)} face(s)")

    left, right = st.columns(2)

    with left:
        st.subheader("Normal View")
        st.image(
            cv2.cvtColor(normal_frame, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    with right:
        st.subheader("Thermal-Style View")
        st.image(
            cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

st.caption(
    "Mubsir — Computer Vision Prototype | "
    "Built with Python, OpenCV and Streamlit"
)
