import numpy as np
import cv2 as cv

people = ['', '']


haar_cascade = cv.CascadeClassifier('haar_face_detection.xml')

features = np.load("features.npy", allow_pickle = True)
labels = np.load("labels.npy", allow_pickle = True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


img = cv.imread(r'C:\Users\User\Desktop\Validation Set\10934095_698403936939215_3035775245255219715_o.jpg')
img = cv.resize(img, (int(img.shape[1]/2), (int(img.shape[0]/2))))

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person", gray)

# Detect Face

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 3)

for(x,y,w,h) in faces_rect:
    faces_crop = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_crop)
    print(f'label = {people[label]} with a confidence of {confidence}')


    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)

    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness = 2)

cv.imshow("Detect Face", img)
cv.waitKey(0)

