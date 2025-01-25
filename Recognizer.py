import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = []
for i in os.listdir(r'C:\Users\Akhil Anand\Desktop\Last_recog\emotion_train') :  
    people.append(i) 

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.read('face_trained.yml')

img_path = r'C:\Users\Akhil Anand\Desktop\Last_recog\emotion_val\sad_1.jpg'

if not os.path.isfile(img_path) :
    print(f"Error: File not found at {img_path}")
else :
    img = cv.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
    else:

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('Person', gray)   

        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces_rect :
            faces_roi = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(faces_roi)

            print(f'Label = {people[label]} with a confidence of {confidence}')
            cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), thickness=2)
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)


        cv.imshow('Detected Face', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        