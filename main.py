#%%

import cv2 as cv                 #Lib for image processing
import glob
import numpy as np
import matplotlib.pyplot as plt  #Lib for plotting images
import random
import qualifiers
import shapedetect
import plotting
import photopath as pp

#values_kleuren = [rood_outcome,groen_outcome,blauw_outcome,hue_outcome,S_outcome,V_outcome]


# Define a method for selecting modus (1) or average (0)
average = 1
mode = 0
    
# Get cucumber color values using qualifiers (e.g., RGB, HSV values)
paprika_kleuren_avg = qualifiers.kleur('photos/Trainingdata/greenpepper/*.jpg', average)
komkommer_kleuren_avg = qualifiers.kleur('photos/Trainingdata/cucumber/*.jpg', average)
appels_kleuren_avg = qualifiers.kleur('photos/Trainingdata/greenapple/*.jpg', average)
shapes_cucumber, center  = shapedetect.ShapeDect_RDP(pp.cucumber)
shapes_appel, center  = shapedetect.ShapeDect_RDP(pp.greenapple)
shapes_greenpepper, center  = shapedetect.ShapeDect_RDP(pp.greenpepper)
shapes_advocado, center  = shapedetect.ShapeDect_RDP(pp.advocado)
shapes_greengrape, center  = shapedetect.ShapeDect_RDP(pp.greengrape)
shapes_greenpear, center  = shapedetect.ShapeDect_RDP(pp.greenpear)
shapes_lemon, center  = shapedetect.ShapeDect_RDP(pp.lemon)
shapes_zucchini, center  = shapedetect.ShapeDect_RDP(pp.zucchini)

def most_frequent(List):
    return max(set(List), key=List.count)

# print("Cucumber: "+ most_frequent(shapes_cucumber))
# print("Appel: "+ most_frequent(shapes_appel))
# print("Advocado: "+ most_frequent(shapes_advocado))
# print("Greenpepper: "+ most_frequent(shapes_greenpepper))
# print("Greengrape: "+ most_frequent(shapes_greengrape))
# print("Greenpear: "+ most_frequent(shapes_greenpear))
# print("Lemon: "+ most_frequent(shapes_lemon))
# print("Zucchini: "+ most_frequent(shapes_zucchini))


# Unpack the color outcomes for the cucumber images

# plotting.plott_shapes(
#     data_arrays=[
#         shapes_appel, shapes_cucumber, 
#         shapes_zucchini, shapes_lemon, shapes_greenpear, 
#         shapes_greengrape, shapes_greenpepper, shapes_advocado
#     ],  # Multiple datasets
#     labels=[
#         'appel', 'komkommer', 
#         'zucchini', 'lemon', 'greenpear', 
#         'greengrape', 'greenpepper', 'advocado'
#     ],  # Labels for each dataset
#     colors=[
#         , 'yellow', 'red', 
#         'lightgreen', 'blue', 'magenta', 
#         'lightyellow', 'lightblue', 'pink'
#     ]  # Colors for each dataset
# )

# Call the function with multiple arrays
# plotting(
#     data_arrays=[shapes,],  # Multiple datasets
#     labels=['komkommer_avg', 'appel_avg'],                            # Labels for each dataset
#     colors=['green', 'yellow']                                   # Colors for each dataset
# )