import qualifiers as ql
import argparse
import imutils
import cv2 as cv
import glob
import numpy as np
import math

# Function to detect the shape of the object in the image
def detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"
    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"
    # return the name of the shape
    return shape

# Function to detect the shape of the object in the image
def ShapeDect_RDP(map_directory):

    shapes = []
    cX_all = []
    cY_all = []

    for name in glob.glob(map_directory):
        
        img_array = cv.imread(name, cv.IMREAD_UNCHANGED)
        resized = imutils.resize(img_array, width=300)
        ratio = img_array.shape[0] / float(resized.shape[0])

        #gets the object from the image
        object_cutter = ql.object_cutter(resized)
        #converts the object to grayscale
        gray = cv.cvtColor(object_cutter, cv.COLOR_BGR2GRAY)
        #thresholds the image
        thresh = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)[1]

        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        
        c = max(cnts, key=cv.contourArea)
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv.moments(c)
        if(M["m00"]==0):
            shapes.append("line")
            continue
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = detect(c)

        shapes.append(shape)
        cX_all.append(cX)
        cY_all.append(cY)
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        # print_shape_on_image(c, shape, img_array, ratio, cX, cY)
    
    centerpoint = [[cX_all[i], cY_all[i]] for i in range(len(cX_all))]
    return shapes, centerpoint

# Function to print the shape on the image for RDP
def print_shape_on_image(c, shape, img_array, ratio, cX, cY):
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv.drawContours(img_array, [c], -1, (0, 255, 0), 2)
    cv.putText(img_array, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2)
    # show the output image
    cv.imshow("Image", img_array)
    cv.waitKey(0)





