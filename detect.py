import cv2
from cv2 import VideoCapture
import numpy as np
import face_recognition
import os

path = 'Images'
images = []
names = []

mylist = os.listdir(path)
print(mylist)

for cls in mylist:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    names.append(os.path.splitext(cls)[0])
print(names)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encod = face_recognition.face_encodings(img)[0]
        encodelist.append(encod)
    return encodelist

encodelistknown = findEncodings(images)

print('Encoding Complete')

cap = VideoCapture('./L&P.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)


while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    facecurF = face_recognition.face_locations(imgS)
    encodcurF = face_recognition.face_encodings(imgS,facecurF)

    for encodeFace,faceLoc in zip(encodcurF,facecurF):
        matches = face_recognition.compare_faces(encodelistknown,encodeFace)
        faceDis = face_recognition.face_distance(encodelistknown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = names[matchIndex]
            print(name)
            y1,x2,y2,x1 = faceLoc
            # y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
            cv2.putText(img,name,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
            
            
        result.write(img)
        cv2.imshow('webcam',img)
        cv2.waitKey(1)

result.release()   
cv2.destroyAllWindows()
   
print("The video was successfully saved")