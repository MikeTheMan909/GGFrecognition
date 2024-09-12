import cv2 as cv                 #Lib for image processing
import glob
import numpy as np
import matplotlib.pyplot as plt  #Lib for plotting images
import random
from scipy import stats


#komkommer uitsnijder
def kleursnijder(img_array):
    hsv = cv.cvtColor(img_array, cv.COLOR_BGR2HSV)

    lower_green = np.array([20, 60, 20])  # Lower bound for green
    upper_green = np.array([85, 255, 255])  # Upper bound for green

    mask = cv.inRange(hsv, lower_green, upper_green)

    return cv.bitwise_and(img_array, img_array, mask=mask)


#RGB test codes
def kleurtest_rood(img_array):
    img = kleursnijder(img_array)

    B, G, R= cv.split(img)
    rood_ar = R.flatten()  # Converts the 2D array to a 1D array

    rood_waardes = rood_ar[rood_ar != 0]  # Filter out zeros

    return_val, count = stats.mode(rood_waardes)    # Step 4: Merge the channels back together


    # Step 3: Zero out the Red and Blue channels
    G[:] = 0
    B[:] = 0

    # Step 4: Merge the channels back together
    green_img = cv.merge([R, G, B])  # The green channel is kept, R and B are now 0
    img_array_new = cv.resize(green_img, (200,200))
    plt.imshow(img_array_new, cmap="gray")
    plt.show()
    return return_val 

def kleurtest_groen(img_array):
    img = kleursnijder(img_array)

    B, G, R= cv.split(img)
    groen_ar = G.flatten()  # Converts the 2D array to a 1D array

    groen_waardes = groen_ar[groen_ar != 0]  # Filter out zeros

    return_val, count = stats.mode(groen_waardes)    # Step 4: Merge the channels back together
    return return_val 

def kleurtest_blauw(img_array):
    img = kleursnijder(img_array)

    B, G, R= cv.split(img)
    groen_ar = B.flatten()  # Converts the 2D array to a 1D array

    groen_waardes = groen_ar[groen_ar != 0]  # Filter out zeros

    return_val, count = stats.mode(groen_waardes)    # Step 4: Merge the channels back together
    return return_val 


#HSV test codes
def kleurtest_hue(img_array):
    img = kleursnijder(img_array)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(hsv)
    unfil_ar = H.flatten()  # Converts the 2D array to a 1D array

    fil_waardes = unfil_ar[unfil_ar != 0]  # Filter out zeros

    return_val, count = stats.mode(fil_waardes)    # Step 4: Merge the channels back together
    return return_val     

def kleurtest_S(img_array):
    img = kleursnijder(img_array)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(hsv)
    unfil_ar = S.flatten()  # Converts the 2D array to a 1D array

    fil_waardes = unfil_ar[unfil_ar != 0]  # Filter out zeros

    return_val, count = stats.mode(fil_waardes)    # Step 4: Merge the channels back together
    return return_val  

def kleurtest_V(img_array):
    img = kleursnijder(img_array)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(hsv)
    unfil_ar = V.flatten()  # Converts the 2D array to a 1D array

    fil_waardes = unfil_ar[unfil_ar != 0]  # Filter out zeros

    return_val, count = stats.mode(fil_waardes)    # Step 4: Merge the channels back together
    return return_val   



def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def komkommer():

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
    for name in glob.glob('photos/trainingdata/voorbeeldkomkommers/*.[jJ][pP][eE][gG]'):
        img_array = cv.imread(name, cv.IMREAD_UNCHANGED)
        names.append(name)
        label=name.split('_')[0] # we splitten de string en pakken het eerste stukje
        rood_outcome.append(kleurtest_rood(img_array))
        groen_outcome.append(kleurtest_groen(img_array))
        blauw_outcome.append(kleurtest_blauw(img_array))
        hue_outcome.append(kleurtest_hue(img_array))
        S_outcome.append(kleurtest_S(img_array))
        V_outcome.append(kleurtest_V(img_array))

        nummer.append(i)
        i=i+1
        labels.append(label)

    #auto coloring of dots
    name_label_uniq = unique(labels)
    num_labels = len(name_label_uniq)
    cmap = plt.get_cmap('hsv', num_labels)  # 'hsv' is an example; you can use other colormaps
    colors = [cmap(i) for i in range(num_labels)]  # Get color for each label

    for i in range(len(labels)):
        plt.scatter(rood_outcome[i],groen_outcome[i],color=colors[name_label_uniq.index(labels[i])])
        plt.xlabel("rood waarde")
        plt.ylabel("groen waarde")
    plt.show()
    for i in range(len(labels)):
        plt.scatter(blauw_outcome[i],hue_outcome[i],color=colors[name_label_uniq.index(labels[i])])
        plt.xlabel("blauw waarde")
        plt.ylabel("hue waarde")
    plt.show()
    for i in range(len(labels)):
        plt.scatter(S_outcome[i],V_outcome[i],color=colors[name_label_uniq.index(labels[i])])
        plt.xlabel("S waarde")
        plt.ylabel("V waarde")
    plt.show()    
