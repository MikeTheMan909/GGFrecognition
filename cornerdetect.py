import numpy as np
import cv2 as cv
import qualifiers as ql
import imutils
import glob

def on_blockSize_trackbar(val):
    global blockSize
    if (val%2)==0:
        blockSize = val+1
        cv.setTrackbarPos('BlockSize', 'dst', blockSize)
    else:
        blockSize=val
        cv.setTrackbarPos('BlockSize', 'dst', blockSize)
    blockSize = max(blockSize, 1)

def on_Ksize_trackbar(val):
    global Ksize
    if (val%2)==0:
        Ksize = val+1
        cv.setTrackbarPos('Ksize', 'dst', Ksize)
    else:
        Ksize=val
        cv.setTrackbarPos('Ksize', 'dst', Ksize)
    Ksize = max(Ksize, 1)

def nothing(x):
    pass

def cornerdetect(map_directory):
    for name in glob.glob(map_directory):
        img_array = cv.imread(name, cv.IMREAD_UNCHANGED)
        resized = imutils.resize(img_array, width=300)
        ratio = img_array.shape[0] / float(resized.shape[0])

        #gets the object from the image
        object_cutter = ql.object_cutter(resized)
        #converts the object to grayscale
        gray = cv.cvtColor(object_cutter, cv.COLOR_BGR2GRAY)

        gray = np.float32(gray)

        thresh = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)[1]

        cv.namedWindow('dst')
        cv.createTrackbar('BlockSize','dst',2,31,on_blockSize_trackbar)
        cv.createTrackbar('Ksize','dst',3,31,on_Ksize_trackbar)
        cv.createTrackbar('K','dst',4,10,nothing)

        while(1):

            new = resized.copy()
            BlockSize = cv.getTrackbarPos('BlockSize','dst')
            Ksize = cv.getTrackbarPos('Ksize','dst')
            K = cv.getTrackbarPos('K','dst')
            K = K/100
            
            dst = cv.cornerHarris(thresh,BlockSize,Ksize,K)
            


            #result is dilated for marking the corners, not important
            dst = cv.dilate(dst,None)
            # Threshold for an optimal value, it may vary depending on the image.
            new[dst>0.01*dst.max()]=[0,0,255]

            cv.imshow('dst',new)

            if cv.waitKey(1) & 0xff == 113:
                
                cv.destroyAllWindows()
                break


