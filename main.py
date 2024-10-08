#%%

import cv2 as cv                 #Lib for image processing
import glob
import numpy as np
import matplotlib.pyplot as plt  #Lib for plotting images
import random
import qualifiers

#values_kleuren = [rood_outcome,groen_outcome,blauw_outcome,hue_outcome,S_outcome,V_outcome]


# Define a method for selecting modus (1) or average (0)
average = 1
mode = 0
    
# Get cucumber color values using qualifiers (e.g., RGB, HSV values)
paprika_kleuren_avg = qualifiers.kleur('photos/Trainingdata/greenpepper/*.jpg', average)
komkommer_kleuren_avg = qualifiers.kleur('photos/Trainingdata/cucumber/*.jpg', average)
appels_kleuren_avg = qualifiers.kleur('photos/Trainingdata/greenapple/*.jpg', average)
# Unpack the color outcomes for the cucumber images

   


def plotting(data_arrays, labels, colors):    
    num_datasets = len(data_arrays)  # Number of datasets
    
    # Titles for each plot (Red, Green, Blue, Hue, Saturation, Brightness)
    titles = ['Red Value', 'Green Value', 'Blue Value', 'Hue Value', 'Saturation Value', 'Brightness Value']
    
    # Loop through each color channel (red, green, blue, hue, saturation, value)
    for i in range(6):
        plt.figure(figsize=(10, 6))
        
        # Plot each dataset's corresponding color channel
        for j in range(num_datasets):
            num_images = len(data_arrays[j][i])  # Get the length of the current dataset's color channel
            photo_numbers = np.arange(1, num_images + 1)  # Create x-axis values corresponding to the number of images
            
            plt.plot(photo_numbers, data_arrays[j][i], marker='o', color=colors[j], label=labels[j])
        
        plt.xlabel("Photo number")
        plt.ylabel(titles[i])
        plt.title(f"{titles[i]} across Photos")
        plt.legend()
        plt.show()


# Call the function with multiple arrays
plotting(
    data_arrays=[komkommer_kleuren_avg, appels_kleuren_avg, paprika_kleuren_avg],  # Multiple datasets
    labels=['komkommer_avg', 'appel_avg','paprika_avg'],                            # Labels for each dataset
    colors=['green', 'yellow','brown']                                   # Colors for each dataset
)