import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained face detector model
prototxt_path = "models/deploy.prototxt"
weights_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

# Load the trained mask detection model
mask_model = tf.keras.models.load_model("models/mask_detector_model.h5")

# Function to detect faces and predict masks
def detect_mask(frame):
    # Convert image to blob for OpenCV processing
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Detect faces
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
            
            # Preprocess face for mask model
            face = cv2.resize(face, (224, 224))
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = np.expand_dims(face, axis=0) / 255.0

            # Predict mask
            preds.append(mask_model.predict(face)[0])
            locs.append((startX, startY, endX, endY))

    return locs, preds
