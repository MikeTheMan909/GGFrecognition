import cv2 
import numpy as np
import os
dirname = os.path.dirname(__file__)
fileName =input("Input file name: ")
filePath = dirname + "\\" + fileName
if not os.path.exists(filePath):
    os.makedirs(filePath)
url = 'http://192.168.178.202:8080/video'
cap = cv2.VideoCapture(url)

count = 0
while(True):
    ret, frame = cap.read()
    if frame is not None:
        cv2.imshow('frame',frame)
    q = cv2.waitKey(1)
    if q == ord("q"):
        break
    if q == ord("c"):
        print(filePath + "\\"+ fileName+ str(count) +".jpg")
        cv2.imwrite(filePath + "\\"+ fileName+ str(count) +".jpg", frame)
        count = count +1
        print(count)
cv2.destroyAllWindows()