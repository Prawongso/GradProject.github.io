import face_recognition
import cv2

# Load the known image
known_image = face_recognition.load_image_file("FaceRec/known_image/biden.jpg")
biden_encoding = face_recognition.face_encodings(known_image)[0]

# Load the unknown image
unknown_image = cv2.imread("FaceRec/unknown_image/unknown.jpg")
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = unknown_image.shape[:2]

    if width is None and height is None:
        return unknown_image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(unknown_image, dim, interpolation=inter)

resize = ResizeWithAspectRatio(unknown_image, height= 720)

# If a face is found in the unknown image
if True in results:
    # Get face locations
    face_locations = face_recognition.face_locations(resize)

    # Loop through each detected face
    for (top, right, bottom, left) in face_locations:
        # Draw a box around the face
        cv2.rectangle(resize, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(resize, 'Recognized', (left + 6, bottom + 25), font, 0.8, (0, 255, 0), 1)


    # Display the resulting image
    cv2.imshow('Unknown Image', resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matching face found in the unknown image.")
    face_locations = face_recognition.face_locations(resize)

    # Loop through each detected face
    for (top, right, bottom, left) in face_locations:
        # Draw a box around the face
        cv2.rectangle(resize, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(resize, 'NotRecognized', (left + 6, bottom + 25), font, 0.8, (0, 0, 255), 1)


    # Display the resulting image
    cv2.imshow('Unknown Image', resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
