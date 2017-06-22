import cv2                              
import numpy as np
import scipy
import sys
import cv2.cv as cv 
import os
cap = cv2.VideoCapture(0)                
while( cap.isOpened() ) :
    ret,img = cap.read()

    #reading the frames

    cv2.imshow('input',img)
    firs=cv.RetrieveFrame(capture)
    cv.SaveImage("first.jpg",firs)

    #displaying the frames

    k = cv2.waitKey(10)
    if k == 27:
        cv2.destroyAllWindows()
        break
