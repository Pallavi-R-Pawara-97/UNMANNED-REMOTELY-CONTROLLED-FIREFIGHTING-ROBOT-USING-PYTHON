import cv2
import numpy as np

def nothing(x):
    pass
# create trackbars for color change
cv2.namedWindow('Trackbar')
#img2=cv2.resize('Trackbar',(150,150))

cv2.createTrackbar('L-H','Trackbar',50,179,nothing)
cv2.createTrackbar('L-S','Trackbar',0,255,nothing)
cv2.createTrackbar('L-V','Trackbar',0,255,nothing)
cv2.createTrackbar('U-H','Trackbar',179,179,nothing)
cv2.createTrackbar('U-S','Trackbar',255,255,nothing)
cv2.createTrackbar('U-V','Trackbar',255,255,nothing)

#img=cv2.imread('1.JPG')
#img = cv2.resize(img,(256,256))
cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    img = cv2.resize(frame,(256,256))
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imshow('HSVImage', img1)
       
    # get current positions of four trackbars
    l_h = cv2.getTrackbarPos('L-H','Trackbar')
    l_s = cv2.getTrackbarPos('L-S','Trackbar')
    l_v = cv2.getTrackbarPos('L-V','Trackbar')
    
    u_h = cv2.getTrackbarPos('U-H','Trackbar')
    u_s = cv2.getTrackbarPos('U-S','Trackbar')
    u_v = cv2.getTrackbarPos('U-V','Trackbar')
    
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])

# Threshold the HSV image to get only blue colors
    mask = cv2.inRange(img1, lower_blue, upper_blue)
    res = cv2.bitwise_and(img1, img1, mask=mask)
   
    cv2.imshow("o_Iamge", img)   
    cv2.imshow('mask', mask)
    cv2.imshow('res1', res)


    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()