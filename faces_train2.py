
# For this work I followed a tutorial available at the following link 
# https://www.youtube.com/watch?v=oXlwWbU8l2o&t=11891s

# Define DIR as the path to the folder containing the list of people (training set)
# Define People as the name of these folders

import os
import cv2 as cv
import numpy as np


people = ['', '']
 
DIR = r''

features = []
labels = []

# Creates the training function

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        # For each folder, read contained images

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Apply face_recognizer function

            haar_cascade = cv.CascadeClassifier('haar_face_detection.xml')
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_crop = gray[y:y+h, x:x+w]
                features.append(faces_crop)
                labels.append(label)


create_train()     

features = np.array(features)
labels = np.array(labels)



print(f'Length of features list = {len(features)}')
print(f'Length of labels list = {len(labels)}')

face_recognizer = cv.face.LBPHFaceRecognizer_create()


face_recognizer.train(features, labels)

# Create corresponding training files

np.save('features.npy', features)
np.save('labels.npy', labels)
face_recognizer.save('face_trained.yml')
