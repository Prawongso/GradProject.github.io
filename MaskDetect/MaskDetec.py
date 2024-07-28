from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="MaskDetect/face_detector", help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="MaskDetect/mask_detector.model", help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


print("loading face detector model")
prototxtPath = os.path.sep.join([args['face'],"deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

print("loading face mask detector model...")
model = load_model("MaskDetect/model.h5")

image = cv2.imread("MaskDetect/input/input1.jpg")
image2 = cv2.imread("MaskDetect/input/input1.jpg")
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

resize = ResizeWithAspectRatio(image, height= 720)
orig = resize.copy()
(h, w) = resize.shape[:2]

blob = cv2.dnn.blobFromImage(resize, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

print("computing face detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0,detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > args['confidence']:
        box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        
        face = resize[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        
        (mask, withoutMask) = model.predict(face)[0]
        
        label = "Mask Detected" if mask > withoutMask else "No Mask Detected"
        color = (0,255,0) if label == "Mask Detected" else (0,0,255)
    
        #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        #cv2.putText(image, label, (w-110, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

        cv2.rectangle(resize, (startX, startY), (endX, endY), color, 2)
        cv2.putText(resize, label, (startX+10,endY+25), cv2.FONT_HERSHEY_DUPLEX , 1, color)

cv2.imshow("Output", resize)
cv2.waitKey(0)
cv2.destroyWindow("Output") 
if label == "No Mask Detected":
    cv2.imwrite("FaceRec/unknown_image/Unknown3.jpg", image2) 
    print("Loading face recognition...") 
    exec(open("FaceRec/cialEx.py").read())  
else: 
    cv2.imwrite("faceUnmask/input/input1.jpg", image2) 
    print("Loading AI face unmasking...") 
    exec(open("faceUnmask/faceUnmask.py").read()) 


