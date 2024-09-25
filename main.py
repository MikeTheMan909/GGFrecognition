#%%

import cv2 as cv                 #Lib for image processing
import glob
import numpy as np
import matplotlib.pyplot as plt  #Lib for plotting images
import random
import qualifiers

#values_kleuren = [rood_outcome,groen_outcome,blauw_outcome,hue_outcome,S_outcome,V_outcome]


def init():
    # Define a method for selecting modus (1) or average (0)
    methode = 1
    
    # Get cucumber color values using qualifiers (e.g., RGB, HSV values)
    komkommer_kleuren_avg = qualifiers.kleur('photos/Trainingdata/cucumber/*.jpg', methode)
    
    # Unpack the color outcomes for the cucumber images
    rood_outcome, groen_outcome, blauw_outcome, hue_outcome, S_outcome, V_outcome = komkommer_kleuren_avg
    
    # Generate a single color for cucumber points (let's use green here as an example)
    cucumber_color = 'green'
    
    # Number of images (assuming the length of one outcome array applies to all)
    num_images = len(rood_outcome)
    
    # X-axis: Photo number (index)
    photo_numbers = np.arange(1, num_images + 1)

    # Plot 1: Red value vs Photo number
    plt.figure(figsize=(10, 6))
    plt.plot(photo_numbers, rood_outcome, marker='o', color=cucumber_color, label='Cucumber')
    plt.xlabel("Photo number")
    plt.ylabel("Red Value")
    plt.title("Red Value across Cucumber Photos")
    plt.legend()
    plt.show()

    # Plot 2: Green value vs Photo number
    plt.figure(figsize=(10, 6))
    plt.plot(photo_numbers, groen_outcome, marker='o', color=cucumber_color, label='Cucumber')
    plt.xlabel("Photo number")
    plt.ylabel("Green Value")
    plt.title("Green Value across Cucumber Photos")
    plt.legend()
    plt.show()

    # Plot 3: Blue value vs Photo number
    plt.figure(figsize=(10, 6))
    plt.plot(photo_numbers, blauw_outcome, marker='o', color=cucumber_color, label='Cucumber')
    plt.xlabel("Photo number")
    plt.ylabel("Blue Value")
    plt.title("Blue Value across Cucumber Photos")
    plt.legend()
    plt.show()

    # Plot 4: Hue value vs Photo number
    plt.figure(figsize=(10, 6))
    plt.plot(photo_numbers, hue_outcome, marker='o', color=cucumber_color, label='Cucumber')
    plt.xlabel("Photo number")
    plt.ylabel("Hue Value")
    plt.title("Hue Value across Cucumber Photos")
    plt.legend()
    plt.show()

    # Plot 5: Saturation value vs Photo number
    plt.figure(figsize=(10, 6))
    plt.plot(photo_numbers, S_outcome, marker='o', color=cucumber_color, label='Cucumber')
    plt.xlabel("Photo number")
    plt.ylabel("Saturation Value")
    plt.title("Saturation Value across Cucumber Photos")
    plt.legend()
    plt.show()

    # Plot 6: Value (Brightness) vs Photo number
    plt.figure(figsize=(10, 6))
    plt.plot(photo_numbers, V_outcome, marker='o', color=cucumber_color, label='Cucumber')
    plt.xlabel("Photo number")
    plt.ylabel("Value (Brightness)")
    plt.title("Brightness Value across Cucumber Photos")
    plt.legend()
    plt.show()

init()
# You can later extend this to add new fruits/vegetables with different colors