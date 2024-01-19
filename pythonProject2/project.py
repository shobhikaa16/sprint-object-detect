# today's objective is to detect the image
#take input of the image  then blurr it then use canny then detect it
import cv2
import numpy as np
framewidth = 640
frameheight = 488
cap = cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(4,frameheight)
def empty(x):
    pass
cv2.namedWindow("parameters")
cv2.resizeWindow('parameters',640,480)
cv2.createTrackbar("threshold1","parameters",100,255,empty)
cv2.createTrackbar("threshold2","parameters",200,255,empty)
cv2.createTrackbar("Area","parameters",1000,20000,empty)

def get_contours(img_dilate,img_contours):

    contours,hierarchy = cv2.findContours(cv2.cvtColor(img_dilate,cv2.COLOR_BGR2GRAY),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  # it will return list of contours and hierarchy
    for cnt in contours:
        area_min = cv2.getTrackbarPos('Area','parameters')
        area = cv2.contourArea(cnt)
        print(area)
        if area > area_min:


            cv2.drawContours(img_contours,[cnt],-1,(0,255,0),3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(img_contours,(x,y),(x+w,y+h),(255,0,255),3)
            cv2.putText(img_contours,"Points"+str(len(approx)),(x+w+20, y+h+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2,0)
            cv2.putText(img_contours,"Area"+str(int(area)),(x+w+45, y+h+45),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2,0)



while True:
    ret,img = cap.read()
    img_contours = img.copy()
    img_blur=cv2.GaussianBlur(img,(7,7),1)

    img_grey=cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
    img_grey=cv2.cvtColor(img_grey,cv2.COLOR_GRAY2BGR)

    #img_canny= cv2.Canny(img_grey,100,200)
    t1 = cv2.getTrackbarPos("threshold1","parameters")
    t2 = cv2.getTrackbarPos("threshold2","parameters")
    img_canny = cv2.Canny(img_grey,t1,t2)
    img_canny = cv2.cvtColor(img_canny,cv2.COLOR_GRAY2BGR)



   # output = np.hstack([img_canny])
    kernel= np.ones((5,5),np.uint8)
    img_dilate = cv2.dilate(img_canny,kernel,iterations=1)  #kernel tells neighbour point and iteration how many times
    #output = np.hstack([img_dilate])
    #output= np.hstack([img_grey])#only one dim
    #cv2.imshow("Image Blur ",img_blur)
    #output = np.hstack([img_grey,img_canny,img_dilate])
    get_contours(img_dilate,img_contours=img_contours)
    output = np.hstack([img_dilate,img_contours])
    #output= np.hstack([img,img_blur]) # this has two dim so this and grey can't be stack all once
    cv2.imshow("output",output)

    #cv2.imshow()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()


