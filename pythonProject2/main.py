import cv2
import numpy as np
#img= cv2.imread("modi ji.jpg")
# cv2.imshow("Image",img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(type(img))
# print(img.shape)
# cap= cv2.VideoCapture(0)
# while True:
#     ret,frame = cap.read() # this is taking the images continuesly
#
#     cv2.imshow("Webcame",frame)
#     if cv2.waitKey(1) == ord("q"):
#         break
# cap.release()
# cv2.destroyAllWindows()
img=cv2.imread("modi ji.jpg")
img2=cv2.resize(img,(540,480))
cv2.imshow("Image",img2)
# cv2.imshow("Image_original",img)
# cv2.imshow("Image",img2)
# print(img2.shape)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img4=cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
# cv2.imshow("Image1",img4)
# imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Image_grayscale",imggray)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img = cv2.imread("modi ji.jpg")
img=cv2.resize(img,(480,480))

# imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgblur=cv2.GaussianBlur(imggray,(9,9),0) # 9,9 shows the blurr value and 0 makes the more no. of lines
# # imgblur3=cv2.GaussianBlur(imggray,(3,3),0)
# imgcanny=cv2.Canny(imggray,100,100) # this canny detects the thin lines if the blurr pic
#
# cv2.imshow("Image Gray",imggray)
# cv2.imshow("Image Blur 3",imgblur)
# # cv2.imshow("Image Blur 9",imgblur) # increasing number will increase blurness
# cv2.imshow("Canny Image",imgcanny)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#import cv2

#img=np.zeros((512,512,3),np.uint8)
#cv2.line(img,(0,0),(512,512),(255,255,0),2)# this is drawing a line and giving the scale at 0,0 and mesurements 512
#cv2.line(source image, (starting coordinate), (ending coordinate), (color of line), (thickness of line))
#cv2.line(img,(0,512),(512,0),(0,0,255),40)
#cv2.rectangle(img,(100,100),(300,300),(0,255,0),cv2.FILLED)
#cv2.rectangle(img,(100,100),(300,300),(0,255,0),2)

#cv2.circle(img,(400,400),50,(203, 192, 255),-1) # thickness is there for the thickness of the radius
#cv2.putText(img,"OPENCV",(200,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
#putText(source,Text,Starting coordinate,font,font_scale,color,thickness)


#cv2.imshow("Image",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# import cv2
# import datetime as dt
#
# cap=cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read() #ret is a boolean value , frame is giving image in every milisecond
#     date_time=str(dt.datetime.now())
#     frame=cv2.putText(frame,date_time,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
#     cv2.imshow("Video",frame)
#     if cv2.waitKey(1)==ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

#import cv2
import numpy as np

# Read two images
image1 = cv2.imread('modi ji.jpg')
image2 = cv2.imread('meloni.jpg')

# Ensure that both images have the same height
rows, cols, channels = image1.shape
image2 = cv2.resize(image2, (int(image2.shape[1] * rows / image2.shape[0]), rows)) #here it resize it acc to image 1

# Resize the first image to match the width of the second image
image1 = cv2.resize(image1, (image2.shape[1], rows)) #shape returns the three dim x,y,z

# Merge images
merged_image = np.concatenate((image1, image2), axis=1)

# Display the merged image
cv2.imshow('Merged Image', merged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# how to do canning , graypicture  , blurr
#assignment do video capture and give date time and give your name there and do screen recording of that
#Tomorrow object detection, detect the human