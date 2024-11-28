#%%
import cv2 as cv                 #Lib for image processing
import glob
import numpy as np
from math import sqrt
import random
import qualifiers
import shapedetect
import cornerdetect
from enum import Enum

#Plotting lib
import plotting
import seaborn as sns
import photopath as pp
import matplotlib.pyplot as plt  #Lib for plotting images

#KNN model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

class types (Enum):
    cucumber = 0
    greenapple = 1
    greenpepper = 2
    advocado = 3
    greengrape = 4
    greenpear = 5
    lemon = 6
    zucchini = 7

#values_kleuren = [rood_outcome,groen_outcome,blauw_outcome,hue_outcome,S_outcome,V_outcome]


# Define a method for selecting modus (1) or average (0)
average = 1
mode = 0
    

# Get cucumber color values using qualifiers (e.g., RGB, HSV values)
advocado_kleuren_avg = qualifiers.kleur(pp.advocado, average)
greengrape_kleuren_avg = qualifiers.kleur(pp.greengrape, average)
greenpear_kleuren_avg = qualifiers.kleur(pp.greenpear, average)
lemon_kleuren_avg = qualifiers.kleur(pp.lemon, average)	
komkommer_kleuren_avg = qualifiers.kleur(pp.cucumber, average)
greenapple_kleuren_avg = qualifiers.kleur(pp.greenapple, average)
greenpepper_kleuren_avg = qualifiers.kleur(pp.greenpepper, average)
zucchini_kleuren_avg = qualifiers.kleur(pp.zucchini, average)

# paprika_kleuren_avg = qualifiers.kleur('photos/Trainingdata/greenpepper/*.jpg', average)
# komkommer_kleuren_avg = qualifiers.kleur('photos/Trainingdata/cucumber/*.jpg', average)
# appels_kleuren_avg = qualifiers.kleur('photos/Trainingdata/greenapple/*.jpg', average)
shapes_cucumber, center  = shapedetect.ShapeDect_RDP(pp.cucumber)
shapes_appel, center  = shapedetect.ShapeDect_RDP(pp.greenapple)
shapes_greenpepper, center  = shapedetect.ShapeDect_RDP(pp.greenpepper)
shapes_advocado, center  = shapedetect.ShapeDect_RDP(pp.advocado)
shapes_greengrape, center  = shapedetect.ShapeDect_RDP(pp.greengrape)
shapes_greenpear, center  = shapedetect.ShapeDect_RDP(pp.greenpear)
shapes_lemon, center  = shapedetect.ShapeDect_RDP(pp.lemon)
shapes_zucchini, center  = shapedetect.ShapeDect_RDP(pp.zucchini)

#cornerdetect.cornerdetect(pp.cucumber)

# def most_frequent(List):
#     return max(set(List), key=List.count)

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
# 


# insert all weak qualifiers for the model to predict the type of fruit
WeakQualifier1 = 0
WeakQualifier2 = 0
WeakQualifier3 = 0

# Insert all types per qualifier
classes = types.cucumber, types.greenapple, types.greenpepper, types.advocado, types.greengrape, types.greenpear, types.lemon, types.zucchini

# Insert weak qualifiers and type of fruit for the model to predict the type of fruit
new_data = list(zip(WeakQualifier1, WeakQualifier2, WeakQualifier3))
new_data_class = types.cucumber

# Create a KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)

data = list(zip(WeakQualifier1, WeakQualifier2, WeakQualifier3))
# Fit the model to the training data
knn_model.fit(data, classes)

# Make predictions on the test data
y_pred = knn_model.predict(new_data)

# Calculate the mean squared error
acc = accuracy_score(new_data_class, y_pred)
# Makes the plot colorful
cmap = sns.cubehelix_palette(as_cmap=True)

# Create a scatter plot
f, ax = plt.subplots()
points = ax.scatter(data[:, 0], data[:, 1], c=classes, s=50, cmap=cmap)

# Add a colorbar
f.colorbar(points)

# Show the plot
plt.show()

