#%%

import cv2 as cv                 #Lib for image processing
import glob
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt  #Lib for plotting images
import kleur_support

lower_bound_hsv = [30,35,20]
upper_bound_hsv = [75, 255, 255]
min_contour_area = 500


#     return new_img
def object_cutter(img_array):
    # Step 1: Convert image to HSV
    hsv = cv.cvtColor(img_array, cv.COLOR_BGR2HSV)

    # Step 2: Create a mask for the green color
    lower_green = np.array(lower_bound_hsv)
    upper_green = np.array(upper_bound_hsv)
    mask = cv.inRange(hsv, lower_green, upper_green)

    # Step 3: Find contours from the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Check if no contours were found
    if len(contours) == 0:
        return mask

    # Step 4: Initialize variables to store the brightest contour
    brightest_contour = None
    max_brightness = -1

    # Step 5: Iterate over all contours to find the brightest one
    for i, contour in enumerate(contours):
        contour_area = cv.contourArea(contour)
        if contour_area < min_contour_area:
            continue

        # Create a mask for this specific contour
        contour_mask = np.zeros_like(mask)
        cv.drawContours(contour_mask, [contour], -1, 255, thickness=cv.FILLED)

        # Visualize the contour mask

        # Apply the mask to the original image to get the region of interest (ROI)
        roi = cv.bitwise_and(img_array, img_array, mask=contour_mask)

        # Convert the ROI to grayscale to measure brightness
        roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        # Calculate the average brightness in the ROI
        avg_brightness = cv.mean(roi_gray, mask=contour_mask)[0]

        # Step 6: Keep track of the contour with the highest brightness
        if avg_brightness > max_brightness:
            max_brightness = avg_brightness
            brightest_contour = contour

    # Step 7: If we found the brightest contour, create a mask for it
    if brightest_contour is not None:
        # Create a mask for the brightest contour
        final_mask = np.zeros_like(mask)
        cv.drawContours(final_mask, [brightest_contour], -1, 255, thickness=cv.FILLED)

        # Use the final mask to cut the brightest object out of the original image
        brightest_object = cv.bitwise_and(img_array, img_array, mask=final_mask)

        # Convert to RGB for display (optional)
        brightest_object_rgb = cv.cvtColor(brightest_object, cv.COLOR_BGR2RGB)

        # Return the image with only the brightest object
        return brightest_object
    else:
        #print("No bright object found!")
        return mask

def kleur(map_directory,methode):

    img_size = 200
    names=[]
    labels=[]
    rood_outcome=[]
    groen_outcome=[]
    blauw_outcome=[]
    hue_outcome=[]
    S_outcome=[]
    V_outcome=[]

    nummer=[]
    i=0
    for name in glob.glob(map_directory):
        img_array = cv.imread(name, cv.IMREAD_UNCHANGED)
        names.append(name)
        label=name.split('_')[0] # we splitten de string en pakken het eerste stukje

        #de evt resize pre echte testen
        img_array = cv.resize(img_array, (img_size,img_size))
        
        rood_outcome.append(kleur_support.kleurtest_rood(img_array,methode))
        groen_outcome.append(kleur_support.kleurtest_groen(img_array,methode))
        blauw_outcome.append(kleur_support.kleurtest_blauw(img_array,methode))
        hue_outcome.append(kleur_support.kleurtest_hue(img_array,methode))
        S_outcome.append(kleur_support.kleurtest_S(img_array,methode))
        V_outcome.append(kleur_support.kleurtest_V(img_array,methode))

        nummer.append(i)
        i=i+1
        labels.append(label)
    values = [rood_outcome,groen_outcome,blauw_outcome,hue_outcome,S_outcome,V_outcome]
    return values
# %%
