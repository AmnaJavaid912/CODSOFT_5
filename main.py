# Import necessary libraries
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile

# Initialize the streamlit app
st.title("Face Detection App")
st.write("This app detects faces in uploaded images using Haar cascades.")

# Load pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces
def detect_faces(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)
    return image, len(faces)

# Upload picture
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        fp = np.fromstring(uploaded_file.read(), np.uint8)
        uploaded_image = cv2.imdecode(fp, cv2.IMREAD_COLOR)
        st.image(uploaded_image, channels="BGR", caption="Uploaded Image")

    # Detect face button
    if st.button('Detect Face'):
        if uploaded_image is not None:
            # Convert to grayscale for the face detection
            gray_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
            # Detect faces
            detected_image, faces_found = detect_faces(uploaded_image)
            
            # Display the image with face detections
            if faces_found > 0:
                st.image(detected_image, channels="BGR", caption="Detected Faces")
            else:
                st.warning("No faces found in the picture.")
        else:
            st.error("Please upload an image first.")
