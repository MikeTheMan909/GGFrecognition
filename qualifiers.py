#%%

import cv2 as cv                 #Lib for image processing
import glob
import numpy as np
from scipy import stats
import kleur_support

lower_bound_hsv = [40,40,20]
upper_bound_hsv = [70, 255, 255]


#komkommer uitsnijder
def object_cutter(img_array):
    hsv = cv.cvtColor(img_array, cv.COLOR_BGR2HSV)

    lower_green = np.array(lower_bound_hsv)  # Lower bound for green
    upper_green = np.array(upper_bound_hsv)  # Upper bound for green

    mask = cv.inRange(hsv, lower_green, upper_green)
    new_img = cv.bitwise_and(img_array, img_array, mask=mask)
    new_img = cv.cvtColor(new_img, cv.COLOR_BGR2RGB)

    return new_img

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
