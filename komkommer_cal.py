#%%

import cv2 as cv                 #Lib for image processing
import glob
import numpy as np
import matplotlib.pyplot as plt  #Lib for plotting images
import random
from scipy import stats

img_size = 200


#komkommer uitsnijder
def kleursnijder(img_array,wit):
    hsv = cv.cvtColor(img_array, cv.COLOR_BGR2HSV)
    img_arr = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)

    lower_green = np.array([30, 10, 10])  # Lower bound for green
    upper_green = np.array([75, 255, 255])  # Upper bound for green

    mask = cv.inRange(hsv, lower_green, upper_green)

    if wit:
        # Maak een witte achtergrond
        white_background = np.full_like(img_array, 255)
        # Zet de groene delen zwart (0, 0, 0) en andere delen wit
        new_img = np.where(mask[:, :, np.newaxis] != 0, [0, 0, 0], white_background)
        new_img = new_img.astype(np.uint8)
    else:
        # Alleen de groene delen behouden en andere verwijderen
        new_img = cv.bitwise_and(img_array, img_array, mask=mask)

    return new_img


#RGB test codes
def kleurtest_rood(img_array):
    img = kleursnijder(img_array,False)
    B, G, R= cv.split(img)  # creates three seperate arrays for blue green and red
    rood_ar = R.flatten()  # Converts the 2D array to a 1D array
    rood_waardes = rood_ar[rood_ar != 0]  # Filter out zeros
    return_val = np.mean(rood_waardes)  # takes the mean value of red

    #for printing a red picture
    G[:] = 0
    B[:] = 0
    green_img = cv.merge([R, G, B])  
    img_array_new = cv.resize(green_img, (img_size,img_size))
    plt.imshow(img_array_new, cmap="gray")
    plt.show()
    return return_val 

def kleurtest_groen(img_array):
    img = kleursnijder(img_array,False)

    B, G, R= cv.split(img)
    groen_ar = G.flatten()  # Converts the 2D array to a 1D array

    groen_waardes = groen_ar[groen_ar != 0]  # Filter out zeros

    return_val  = np.mean(groen_waardes)    # Step 4: Merge the channels back together
    return return_val 

def kleurtest_blauw(img_array):
    img = kleursnijder(img_array,False)

    B, G, R= cv.split(img)
    groen_ar = B.flatten()  # Converts the 2D array to a 1D array

    groen_waardes = groen_ar[groen_ar != 0]  # Filter out zeros

    return_val  = np.mean(groen_waardes)    # Step 4: Merge the channels back together
    return return_val 


#HSV test codes
def kleurtest_hue(img_array):
    img = kleursnijder(img_array,False)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(hsv)
    unfil_ar = H.flatten()  # Converts the 2D array to a 1D array

    fil_waardes = unfil_ar[unfil_ar != 0]  # Filter out zeros

    return_val  = np.mean(fil_waardes)    # Step 4: Merge the channels back together
    return return_val     

def kleurtest_S(img_array):
    img = kleursnijder(img_array,False)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(hsv)
    unfil_ar = S.flatten()  # Converts the 2D array to a 1D array

    fil_waardes = unfil_ar[unfil_ar != 0]  # Filter out zeros

    return_val  = np.mean(fil_waardes)    # Step 4: Merge the channels back together
    return return_val  

def kleurtest_V(img_array):
    img = kleursnijder(img_array,False)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(hsv)
    unfil_ar = V.flatten()  # Converts the 2D array to a 1D array

    fil_waardes = unfil_ar[unfil_ar != 0]  # Filter out zeros

    return_val  = np.mean(fil_waardes)    # Step 4: Merge the channels back together
    return return_val   

def vormherkenning(img_array):
    # Stap 1: Converteer de afbeelding naar HSV en snijd op kleur
    img = kleursnijder(img_array, True)

    # Stap 2: Converteer de afbeelding naar grijswaarden
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Stap 3: Verwijder ruis met een Gaussian blur
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Stap 4: Pas drempelwaarde toe om de vorm te isoleren
    _, threshold = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY_INV)

    # Stap 5: Vind de contouren in de afbeelding
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Stap 6: Vind de contour met het grootste oppervlak
    if contours:
        grootste_contour = max(contours, key=cv.contourArea)  # Selecteer het grootste contour

        # Optioneel: Filter contouren die te klein zijn
        if cv.contourArea(grootste_contour) > 100:  # Minimale grootte om kleine ruis te filteren
            # Maak een leeg masker om het grootste contour op te tekenen
            mask = np.zeros_like(gray)
            
            # Teken het grootste contour in het masker
            cv.drawContours(mask, [grootste_contour], -1, (255), thickness=cv.FILLED)
            
            # Toon het resultaat waarbij alleen het grootste contour zichtbaar is
            result = cv.bitwise_and(img_array, img_array, mask=mask)

            # Toon de afbeelding met het grootste object
            plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
            plt.show()
            

            #returnd nu een afbeelding
            return result
    else:
        print("Geen contouren gevonden.")
        return -1



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

    names=[]
    labels=[]
    rood_outcome=[]
    groen_outcome=[]
    blauw_outcome=[]
    hue_outcome=[]
    S_outcome=[]
    V_outcome=[]
    vormen=[]   # Voor het opslaan van de vormen

    nummer=[]
    i=0
    for name in glob.glob('photos/Trainingdata/cucumber/*.jpg'):
        img_array = cv.imread(name, cv.IMREAD_UNCHANGED)
        names.append(name)
        label=name.split('_')[0] # we splitten de string en pakken het eerste stukje
        img_array = cv.resize(img_array, (img_size,img_size))
        vormen.append(vormherkenning(img_array))  # Sla de vorm op voor latere analyse
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
        plt.scatter(vormen[i],V_outcome[i],color=colors[name_label_uniq.index(labels[i])])
        plt.xlabel("vorm")
        plt.ylabel("V waarde")
    plt.show()    
# %%
