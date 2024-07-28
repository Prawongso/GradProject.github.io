import cv2 as cv

cam_port = 0
cap = cv.VideoCapture(0)
print("Press enter when ready to scan!") 
while(True):       
    ret, frame = cap.read() 
    
    cv.imshow('frame', frame) 

    if cv.waitKey(1) == 13:
        break
cap.release()
cv.destroyAllWindows()

cam = cv.VideoCapture(cam_port) 

result, image = cam.read() 

if result: 

    cv.imshow("input1", image) 

    cv.imwrite("MaskDetect\input\input1.jpg", image) 

    cv.waitKey(0) 
    cv.destroyWindow("input1") 
    cap.release()
    cv.destroyAllWindows()
    print("Loading face mask detection...") 
    exec(open("MaskDetect/MaskDetec.py").read())

else: 
    print("No image detected. Please! try again") 
