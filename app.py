import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

@st.cache_resource(show_spinner=False)
def load_models():
    # Load the mask detection model
    mask_model = load_model("/Users/cbprasad/Downloads/vscode_cb/facemask_detector/models/mask_detector_mobilenet.h5")
    mask_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # This removes the warning
    # Load the face detector
    faceNet = cv2.dnn.readNet("/Users/cbprasad/Downloads/vscode_cb/facemask_detector/models/deploy.prototxt", "/Users/cbprasad/Downloads/vscode_cb/facemask_detector/models/res10_300x300_ssd_iter_140000.caffemodel")
    return mask_model, faceNet

mask_model, faceNet = load_models()

def detect_and_predict_mask(frame, faceNet, mask_model):
    (h, w) = frame.shape[:2]
    # Preprocess the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # Loop over detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = face.astype("float") / 255.0
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        preds = mask_model.predict(np.vstack(faces), batch_size=32)

    return (locs, preds)


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the incoming video frame to an OpenCV image
        img = frame.to_ndarray(format="bgr24")
        (locs, preds) = detect_and_predict_mask(img, faceNet, mask_model)
        classes = ["Incorrect", "Mask", "No Mask"]

        # Annotate the frame with predictions
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            idx = np.argmax(pred)
            label = classes[idx]
            if label == "Mask":
                color = (0, 255, 0)
            elif label == "Incorrect":
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            label_text = f"{label}: {pred[idx]*100:.2f}%"
            cv2.putText(img, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Real-Time Face Mask Detection")
st.write("Upload an image or use your webcam for real-time detection.")

# Sidebar option to choose mode
mode = st.sidebar.selectbox("Select Mode", ["Image Upload", "Webcam"])

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read and decode the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        output_image = image.copy()
        (locs, preds) = detect_and_predict_mask(image, faceNet, mask_model)
        classes = ["Incorrect", "Mask", "No Mask"]
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            idx = np.argmax(pred)
            label = classes[idx]
            if label == "Mask":
                color = (0, 255, 0)
            elif label == "Incorrect":
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            label_text = f"{label}: {pred[idx]*100:.2f}%"
            cv2.putText(output_image, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(output_image, (startX, startY), (endX, endY), color, 2)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        st.image(output_image, caption="Processed Image", use_column_width=True)
else:
    st.write("Starting webcam for real-time detection...")
    webrtc_streamer(key="mask-detection", video_processor_factory=VideoProcessor)