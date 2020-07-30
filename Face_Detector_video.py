import cv2
from random import randrange

# frontel_faces_default trained on a bunch of frontel_faces
# now loading or importing that xml file, which is pre-trained data on some face--frontols from opencv (haar cascade alogo)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# creating a classifier


# tocapture video from wenbcam 0 means default camera i.e webcam or we can add a video file
webcam = cv2.VideoCapture(0)
cv2.waitKey()

# interate over frames forever until the webcam is interrupted
while True:
    #reading the current frames
    # it returns 2 things
    # 1) If reading the frame was successful or not(boolean) 
    # 2) Actual frame or image  
    successfull_frame_read, frame = webcam.read()

    # converting the current frame into grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # the detected objects are returned as a list of rectangles surrounding the face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h) , (randrange(256),randrange(256),randrange(256)), 2)

    cv2.imshow('Piyush Face Detector',frame)
    key = cv2.waitKey(1)
    
    #stop if q is pressed
    if key==81 or key==113:
        break

print('Code Completed')