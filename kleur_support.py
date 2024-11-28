import cv2 as cv                 #Lib for image processing
import numpy as np
import matplotlib.pyplot as plt  #Lib for plotting images
from scipy import stats
import qualifiers


#RGB test codes
def kleurtest_rood(img_array,methode):
    img = qualifiers.object_cutter(img_array)
    B, G, R= cv.split(img)  # creates three seperate arrays for blue green and red
    rood_ar = R.flatten()  # Converts the 2D array to a 1D array
    rood_waardes = rood_ar[rood_ar != 0]  # Filter out zeros
    if methode == 1:
        return_val = np.mean(rood_waardes)  # takes the mean value of red
    elif methode == 0:
        return_val = stats.mode(rood_waardes)[0]  # takes the mean value of red
    return return_val/255 

def kleurtest_groen(img_array,methode):
    img = qualifiers.object_cutter(img_array)

    B, G, R= cv.split(img)
    groen_ar = G.flatten()  # Converts the 2D array to a 1D array

    groen_waardes = groen_ar[groen_ar != 0]  # Filter out zeros

    if methode == 1:
        return_val = np.mean(groen_waardes)  # takes the mean value of red
    elif methode == 0:
        return_val = stats.mode(groen_waardes)[0]  # takes the mean value of red
    return return_val/255

def kleurtest_blauw(img_array,methode):
    img = qualifiers.object_cutter(img_array)

    B, G, R= cv.split(img)
    blauw_ar = B.flatten()  # Converts the 2D array to a 1D array

    blauw_waardes = blauw_ar[blauw_ar != 0]  # Filter out zeros

    if methode == 1:
        return_val = np.mean(blauw_waardes)  # takes the mean value of red
    elif methode == 0:
        return_val = stats.mode(blauw_waardes)[0]  # takes the mean value of red    
    return return_val/255


#HSV test codes
def kleurtest_hue(img_array,methode):
    img = qualifiers.object_cutter(img_array)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(hsv)
    unfil_ar = H.flatten()  # Converts the 2D array to a 1D array

    fil_waardes = unfil_ar[unfil_ar != 0]  # Filter out zeros

    if methode == 1:
        return_val = np.mean(fil_waardes)  # takes the mean value of red
    elif methode == 0:
        return_val = stats.mode(fil_waardes)[0]  # takes the mean value of red
    return return_val/255     

def kleurtest_S(img_array,methode):
    img = qualifiers.object_cutter(img_array)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(hsv)
    unfil_ar = S.flatten()  # Converts the 2D array to a 1D array

    fil_waardes = unfil_ar[unfil_ar != 0]  # Filter out zeros

    if methode == 1:
        return_val = np.mean(fil_waardes)
    elif methode == 0:
        return_val = stats.mode(fil_waardes)[0]    
    return return_val/255  

def kleurtest_V(img_array,methode):
    img = qualifiers.object_cutter(img_array)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(hsv)
    unfil_ar = V.flatten()  # Converts the 2D array to a 1D array

    fil_waardes = unfil_ar[unfil_ar != 0]  # Filter out zeros

    if methode == 1:
        return_val = np.mean(fil_waardes)
    elif methode == 0:
        return_val = stats.mode(fil_waardes)[0]
    return return_val/255 