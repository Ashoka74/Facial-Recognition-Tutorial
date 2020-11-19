
# This code allows to fine-tune the minNeighbor parameter of haar_cascade face classifier function
# It returns the best parameters --> Showing the most faces


import numpy as np
import cv2 as cv
import caer

img = cv.imread(img, "PATH")
# cv.imshow('img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)



haar_cascade = cv.CascadeClassifier('haar_face_detection.xml')

dict = {}

# iterates minNeighbors from 1 to 5 to find best parameters

for i in range(1, 5):  
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=i)
    dict.update({i : len(faces_rect)})
    # print('The number of faces found for', i,  f'minNeighbors parameters is {len(faces_rect)} ')
best_parameter = max(dict, key= dict.get)

 # Return the minNeighbors corresponding to the highest number of detected faces

print("The picture contains", max(dict.values()), "faces with", best_parameter, "minNeighbors.")

# Return picture corresponding to the highest number of faces

best_faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors= best_parameter)
for (x,y,w,h) in best_faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness = 2)
cv.imshow("Detected Faces", img)


cv.waitKey(0)