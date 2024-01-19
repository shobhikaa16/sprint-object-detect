import cv2
import  os
import time
import numpy as np # contains all features
haar_file='haarcascade_frontalface_default.xml'
dataset="C:/Users/shobh/PycharmProjects/facedetect/Dataset"
(image,labels,names,id)= ([],[],{},0)
for(subdirs,dirs,files) in os.walk(dataset):
    for subdirs in dirs:
        names[id]=subdirs
        subjectpath= os.path.join(dataset,subdirs)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath,filename)
            label = id
            image.append(cv2.imread(path,0))
            labels.append(int(label))
        id = id+1

(width, height) = (640, 480)
(images,labels) = [np.array(lists) for lists in [image,labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images,labels)
face_cascade=cv2.CascadeClassifier(haar_file)
cap=cv2.VideoCapture(0)
print("Webcam is open?",cap.isOpened())

while True:
    ret,frame=cap.read();
    if ret==True:
        img_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(img_gray,1.4,4)
        #detectMultiplescel(source_image,scale,mon,neighbour)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            face=img_gray[y:y+h,x:x+w]
            face_resize=cv2.resize(face,(width,height))
            prediction = model.predict(face_resize)
            if prediction[1]<74:
                cv2.putText(frame,'%s'%(names[prediction[0]].strip()),(x+5,(y+25+h)),cv2.FONT_HERSHEY_PLAIN,1.5,(20,185,20),2)
                cv2.imshow("Frame Recognition",frame)
                if cv2.waitKey(1) == ord("q"):
                    break
cap.release()
cv2.destroyAllWindows()







