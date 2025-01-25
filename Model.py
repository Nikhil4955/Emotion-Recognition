import os
import cv2 as cv
import numpy as np
import time

DIR = r'C:\Users\Akhil Anand\Desktop\Last_recog\emotion_train'

people = []
for i in os.listdir(DIR) :       
    people.append(i) 
print(people)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train() :
    start_time = time.time()
    for person in people :
        path = os.path.join(DIR, person)
        label = people.index(person)
        print(f"Processing images for: {person} with label {label}")

        for img in os.listdir(path) :
            img_path = os.path.join(path, img)

            if not os.path.isfile(img_path) :
                print(f"Warning: File {img_path} is not a valid file.")
                continue

            img_array = cv.imread(img_path)

            if img_array is None:  
                print(f"Warning: Could not read image {img_path}")
                continue

            grey = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(img_array, scaleFactor = 1.1, minNeighbors = 5)

            for (x, y, w, h) in  faces_rect :
                faces_roi = cv.resize(grey[y:y+h, x:x+w], (100, 100))
                features.append(faces_roi)
                labels.append(label)
                
    end_time = time.time()
    print(f"Time taken to load dataset : {end_time - start_time} seconds")


create_train()
print('Training done ---------------')

features = np.array(features, dtype=np.uint8)  
labels = np.array(labels, dtype=np.int32)   

print(f"Features shape: {features.shape}") 
print(f"Labels shape: {labels.shape}") 

face_recognizer = cv.face.LBPHFaceRecognizer_create()
try:
    face_recognizer.train(features, labels)
    print("Training successful!")
except Exception as e:
    print(f"Error during training: {e}")

print(f"Lenght of the features = {len(features)}")
print(f"Lenght of the labels = {len(labels)}")

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)