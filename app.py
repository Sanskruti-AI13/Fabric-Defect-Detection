import streamlit as st
import cv2
import numpy as np
import pickle

with open("models/decision_tree.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Fabric Defect Detection",
    page_icon="🧵",
    layout="wide"
)

st.title("Fabric Defect Detection")
st.write("AI-based fabric inspection using image processing and a Decision Tree.")

uploaded_file = st.file_uploader(
    "Upload a fabric image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    data = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # Image Processing
    resized = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    filtered = cv2.medianBlur(gray, 3)
    enhanced = cv2.equalizeHist(filtered)

    # Edge Detection
    edges = cv2.Canny(enhanced, 50, 150)

    # Segmentation
    _, threshold = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Defect Highlight
    highlighted = resized.copy()
    highlighted[threshold == 0] = [0, 0, 255]

    st.subheader("Image Processing")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Original")
        st.image(image)

    with col2:
        st.write("Grayscale")
        st.image(gray)

    with col3:
        st.write("Noise Removed")
        st.image(filtered)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Contrast Enhanced")
        st.image(enhanced)

    with col2:
        st.write("Edge Detection")
        st.image(edges)

    with col3:
        st.write("Segmentation")
        st.image(threshold)

    # Before / After
    st.subheader("Before / After Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Before")
        st.image(resized)

    with col2:
        st.write("After - Defect Highlight")
        st.image(highlighted)

    # Prediction
    features = enhanced.flatten().reshape(1, -1)
    prediction = model.predict(features)

    st.subheader("AI Prediction")

    if prediction[0] == 1:
        st.error("Prediction: STAIN")
    else:
        st.success("Prediction: DEFECT FREE")

    # Image Information
    st.subheader("Image Information")

    brightness = np.mean(enhanced)
    edge_pixels = np.sum(edges > 0)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Average Brightness", round(brightness, 2))

    with col2:
        st.metric("Detected Edge Pixels", int(edge_pixels))

st.divider()

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Images", "466")

with col2:
    st.metric("Stain Images", "398")

with col3:
    st.metric("Defect-Free Images", "68")

st.write("Algorithm: Decision Tree Classifier")
st.write("Test Accuracy: 72.34%")