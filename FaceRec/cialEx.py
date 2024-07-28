from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import cv2
import face_recognition

# Load emotion recognition model
json_file = open('facialRecognition/fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("facialRecognition/fer.h5")
print("Loaded emotion recognition model from disk")

# Load face recognition known image
known_image = face_recognition.load_image_file("FaceRec/known_image/Vincent.jpg")
vincent_encoding = face_recognition.face_encodings(known_image)[0]

# Load unknown image for face and emotion recognition
unknown_image = cv2.imread("FaceRec/unknown_image/Unknown3.jpg")
full_size_image = unknown_image.copy()
gray = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2GRAY)
face_cascade = cv2.CascadeClassifier('facialRecognition/haarcascade_frontalface_default.xml')

# Detect faces for both face recognition and emotion recognition
face_locations = face_recognition.face_locations(unknown_image)
faces = face_cascade.detectMultiScale(gray, 1.3, 10)

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Resize the image for better display
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

resize = ResizeWithAspectRatio(unknown_image, height=720)

for (top, right, bottom, left) in face_locations:
    # Face recognition
    face_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=[(top, right, bottom, left)])[0]
    match = face_recognition.compare_faces([vincent_encoding], face_encoding)
    name = "Recognized" if match[0] else "NotRecognized"
    color = (0, 255, 0) if match[0] else (0, 0, 255)
    cv2.rectangle(full_size_image, (left, top), (right, bottom), color, 2)
    cv2.putText(full_size_image, name, (left + 6, bottom + 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)

    # Emotion recognition
    roi_gray = gray[top:bottom, left:right]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
    yhat = loaded_model.predict(cropped_img)
    emotion_label = labels[int(np.argmax(yhat))]
    print("Emotion: " + emotion_label)
    cv2.putText(full_size_image, emotion_label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)


# Display the resulting images
cv2.imshow('Combined Emotion and Face Recognition', full_size_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
