import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    count_fire=0
    ret, frame = cap.read()
    frame = cv2.resize(frame,(256,256))
    cv2.imshow('input1',frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV',hsv)
    lower_green = np.array([0,60,200])
    upper_green = np.array([160,140,255])
    
    mask = cv2.inRange(frame, lower_green, upper_green)
    ret, thresh = cv2.threshold(mask, 50,255,cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    img4 = cv2.dilate(mask,kernel,iterations = 3)
    closing = cv2.morphologyEx(img4, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closing',closing)    
    count_fire = closing.sum()
    
    if (count_fire>1000):
        print('Fire Detected')
    else:
        print('No Fire')

    cv2.waitKey(1)
cv2.destroyAllWindows()    
