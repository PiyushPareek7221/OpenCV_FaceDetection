import cv2
from random import randrange

# frontel_faces_default trained on a bunch of frontel_faces
# now loading or importing that xml file, which is pre-trained data on some face--frontols from opencv (haar cascade alogo)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# creating a classifier

# choose an image to detect faces in
img= cv2.imread('new.jpg')

# making the image black and white using grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# now detecting faces
# detect multiscale detects objects of different sizes in the input image
# the detected objects are returned as a list of rectangles surrounding the face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# location of face upper-left, bottom right coordinates as a rectangle
# print(face_coordinates)
# [[270 109 431 431]]

#taking the coordinates
#drwaing rectangles around the faces 
#cv2.rectangle(img, top-left coordinates, width and hieght of img, color, thickness OR
# cv2.rectangle(img, (270, 109), (431+270,431+109) , (0,255,0), 2)
for (x,y,w,h) in face_coordinates:
    #(x,y,w,h) = face_coordinates[0] taken care by loop
    cv2.rectangle(img, (x,y), (x+w,y+h) , (randrange(128,256),randrange(128,256),randrange(128,256)), 2)

#to show an image and to hold the img window using wait_key
cv2.imshow('Piyush Face Detector',img)
cv2.waitKey()

print('Code Completed')