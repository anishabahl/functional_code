# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:29:16 2020

@author: ab20
"""

import cv2
import rawpy as raw
import imageio
import numpy as np
import matplotlib.pyplot as plt
import spectral
from spectral import imshow
from spectral import envi
import xml.etree.ElementTree as ET
########INPUTS
data = '/home/ab20/Data/System_Paper/raw_data/' #path to data
filename = 'A1'
whitename = 'ref_white' #white reference image name
darkname = 'ref_dark' #dark reference image name
filetype = "png" #either 'raw' or 'png' must be the same for all 3 images
whitebalance = 2 #0= 'off', 1= (rawimg-dark)/(white-dark) then clip 0 to 1, 2= max of values in division then clip
newfilename = 'A1demosaiced' #name of new hdr file created
bilinear_interpolation = 'ON'
########IMPORTING DATA
if filetype == "raw":
    img = np.fromfile(data+filename+'.raw', dtype=np.uint8, sep="")
    rawimg = img.reshape(1088,2048)
    whiteimg = np.fromfile(data+whitename+'.raw', dtype=np.uint8, sep="")
    white = whiteimg.reshape(1088,2048)
    darkimg = np.fromfile(data+darkname+'.raw', dtype=np.uint8, sep="")
    dark = darkimg.reshape(1088,2048)
if filetype == "png":
    rawimg = cv2.imread(data+filename+'.png', cv2.IMREAD_UNCHANGED)
    white = cv2.imread(data+whitename+'.png', cv2.IMREAD_UNCHANGED)
    dark = cv2.imread(data+darkname+'.png', cv2.IMREAD_UNCHANGED)
    
########WHITE BALANCE
##if whitebalance == 'ON':
##    print(rawimg)
rawimg = rawimg.astype('float32')
dark = dark.astype('float32')
white = white.astype('float32')
if whitebalance == 1:
    rawimg = (rawimg-dark)/(white-dark + np.finfo(float).eps)
    rawimg = np.clip(rawimg, 0, 1)
if whitebalance == 2:
    zeros = np.zeros(rawimg.shape)
    epsilons = zeros + np.finfo(float).eps
    rawimg = np.maximum((rawimg-dark), zeros)/np.maximum((white-dark), epsilons)
    rawimg = np.clip(rawimg, 0, 1)
##    rawimg = np.clip(rawimg, 0, 1)
##    rawimg = np.nan_to_num(rawimg)
##    np.putmask(rawimg, rawimg>256, 256)
##    rawimg = 255*(rawimg - rawimg.min())/(rawimg.max()-rawimg.min())
    #test by showing
    view = imshow(rawimg)
##    rawimg = np.subtract(rawimg, dark)/np.subtract(white, dark)
###clip values above 256 as 8-bit image
##for i in range(rawimg.shape[0]):
##	for j in range(rawimg.shape[1]):
##		if rawimg[i][j]>256:
##			rawimg[i][j] = 256
    
########SEPARATING MOSAIC INTO BANDS
Hypercube = np.zeros((1088,2048,25))
y=rawimg.shape[0]
x = rawimg.shape[1]
Hypercube[4:y:5, 1:x:5, 0] = rawimg[4:y:5, 1:x:5]
Hypercube[4:y:5, 2:x:5, 1] = rawimg[4:y:5, 2:x:5]
Hypercube[4:y:5, 3:x:5, 2] = rawimg[4:y:5, 3:x:5]
Hypercube[4:y:5, 4:x:5, 3] = rawimg[4:y:5, 4:x:5]
Hypercube[4:y:5, 0:x:5, 4] = rawimg[4:y:5, 0:x:5]
Hypercube[3:y:5, 0:x:5, 5] = rawimg[3:y:5, 0:x:5]
Hypercube[3:y:5, 1:x:5, 6] = rawimg[3:y:5, 1:x:5]
Hypercube[3:y:5, 2:x:5, 7] = rawimg[3:y:5, 2:x:5]
Hypercube[3:y:5, 3:x:5, 8] = rawimg[3:y:5, 3:x:5]
Hypercube[3:y:5, 4:x:5, 9] = rawimg[3:y:5, 4:x:5]
Hypercube[2:y:5, 0:x:5, 10] = rawimg[2:y:5, 0:x:5]
Hypercube[2:y:5, 1:x:5, 11] = rawimg[2:y:5, 1:x:5]
Hypercube[2:y:5, 2:x:5, 12] = rawimg[2:y:5, 2:x:5]
Hypercube[2:y:5, 3:x:5, 13] = rawimg[2:y:5, 3:x:5]
Hypercube[2:y:5, 4:x:5, 14] = rawimg[2:y:5, 4:x:5]
Hypercube[1:y:5, 0:x:5, 15] = rawimg[1:y:5, 0:x:5]
Hypercube[1:y:5, 1:x:5, 16] = rawimg[1:y:5, 1:x:5]
Hypercube[1:y:5, 2:x:5, 17] = rawimg[1:y:5, 2:x:5]
Hypercube[1:y:5, 3:x:5, 18] = rawimg[1:y:5, 3:x:5]
Hypercube[1:y:5, 4:x:5, 19] = rawimg[1:y:5, 4:x:5]
Hypercube[0:y:5, 0:x:5, 20] = rawimg[0:y:5, 0:x:5]
Hypercube[0:y:5, 1:x:5, 21] = rawimg[0:y:5, 1:x:5]
Hypercube[0:y:5, 2:x:5, 22] = rawimg[0:y:5, 2:x:5]
Hypercube[0:y:5, 3:x:5, 23] = rawimg[0:y:5, 3:x:5]
Hypercube[0:y:5, 4:x:5, 24] = rawimg[0:y:5, 4:x:5]
########INTERPOLATING TO FILL GAPS -- using bilinear interpolation
##y = ((rawimg.shape[0] - 5)//11)*11
##x = ((rawimg.shape[1] - 5)//11)*11
Hypercube= np.pad(Hypercube,((5,5), (5,5), (0,0))) #pad but will have edge effects as only padded with 0s
y = (rawimg.shape[0]//5)*5
x = (rawimg.shape[1]//5)*5

##Hypercube[9:y:5, 5:x:5, 0] = 0.01*(5*6*Hypercube[19:y+10:5, 6:x:5, 0] + 4*5*Hypercube[19:y+10:5, 16:x+10:5, 0] +6*5*Hypercube[9:y:5, 6:x:5, 0] + 4*5*Hypercube[9:y:5, 16:x+10:5, 0])
####Hypercube[9:y-5:5, 5:x-5:5, 0] = 0.01*(5*6*Hypercube[19:y:5, 6:x-5:5, 0] + 4*5*Hypercube[19:y:5, 16:x:5, 0] +6*5*Hypercube[9:y-5:5, 6:x-5:5, 0] + 4*5*Hypercube[9:y-5:5, 16:x:5, 0]) #tried to do x and y in terms of hypercube.shape but didn't work
##Hypercube[9:y:5, 7:x:5, 0] = 0.01*(5*4*Hypercube[19:y+10:5, 6:x:5, 0] + 6*5*Hypercube[19:y+10:5, 16:x+10:5, 0] +4*5*Hypercube[9:y:5, 6:x:5, 0] + 6*5*Hypercube[9:y:5, 16:x+10:5, 0])
##Hypercube[9:y:5, 8:x:5, 0] = 0.01*(5*7*Hypercube[19:y+10:5, 6:x:5, 0] + 3*5*Hypercube[19:y+10:5, 16:x+10:5, 0] +7*5*Hypercube[9:y:5, 6:x:5, 0] + 3*5*Hypercube[9:y:5, 16:x+10:5, 0])
##Hypercube[9:y:5, 9:x:5, 0] = 0.01*(5*8*Hypercube[19:y+10:5, 6:x:5, 0] + 2*5*Hypercube[19:y+10:5, 16:x+10:5, 0] +8*5*Hypercube[9:y:5, 6:x:5, 0] + 2*5*Hypercube[9:y:5, 16:x+10:5, 0])
##Hypercube[8:y:5, 5:x:5, 0] = 0.01*(4*6*Hypercube[19:y+10:5, 6:x:5, 0] + 4*4*Hypercube[19:y+10:5, 16:x+10:5, 0] +6*6*Hypercube[9:y:5, 6:x:5, 0] + 4*6*Hypercube[9:y:5, 16:x+10:5, 0])
##Hypercube[8:y:5, 6:x:5, 0] = 0.01*(4*5*Hypercube[19:y+10:5, 6:x:5, 0] + 5*4*Hypercube[19:y+10:5, 16:x+10:5, 0] +5*6*Hypercube[9:y:5, 6:x:5, 0] + 5*6*Hypercube[9:y:5, 16:x+10:5, 0])
##Hypercube[8:y:5, 7:x:5, 0] = 0.01*(4*4*Hypercube[19:y+10:5, 6:x:5, 0] + 6*4*Hypercube[19:y+10:5, 16:x+10:5, 0] +4*6*Hypercube[9:y:5, 6:x:5, 0] + 6*6*Hypercube[9:y:5, 16:x+10:5, 0])
##Hypercube[8:y:5, 8:x:5, 0] = 0.01*(4*3*Hypercube[19:y+10:5, 6:x:5, 0] + 7*4*Hypercube[19:y+10:5, 16:x+10:5, 0] +3*6*Hypercube[9:y:5, 6:x:5, 0] + 7*6*Hypercube[9:y:5, 16:x+10:5, 0])
##Hypercube[8:y:5, 9:x:5, 0] = 0.01*(4*2*Hypercube[19:y+10:5, 6:x:5, 0] + 8*4*Hypercube[19:y+10:5, 16:x+10:5, 0] +2*6*Hypercube[9:y:5, 6:x:5, 0] + 8*6*Hypercube[9:y:5, 16:x+10:5, 0])
##Hypercube[7:y:5, 5:x:5, 0] = 0.01*(3*6*Hypercube[19:y+10:5, 6:x:5, 0] + 4*3*Hypercube[19:y+10:5, 16:x+10:5, 0] +6*7*Hypercube[9:y:5, 6:x:5, 0] + 4*7*Hypercube[9:y:5, 16:x+10:5, 0])
##Hypercube[7:y:5, 6:x:5, 0] = 0.01*(3*5*Hypercube[19:y+10:5, 6:x:5, 0] + 5*3*Hypercube[19:y+10:5, 16:x+10:5, 0] +5*7*Hypercube[9:y:5, 6:x:5, 0] + 5*7*Hypercube[9:y:5, 16:x+10:5, 0])
if bilinear_interpolation == 'ON':
    #creating loops instead of 24*24 commands
    v = [1,2,3,4,0,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
    w = [4,4,4,4,4,3,3,3,3,3,2,2,2,2,2,1,1,1,1,1,0,0,0,0,0]
    for b in range(0,25):
        for i in range(0,5) :
            for j in range(0,5):
                if Hypercube[j+5][i+5][b] == 0:
                    k=i+5-v[b]
                    l = j+5-w[b]
                    Hypercube[j+5:y:5, i+5:x:5, b] = 0.01*((10-l)*(10-k)*Hypercube[w[b]+5:y:5, v[b]+5:x:5, b] + (10-l)*k*Hypercube[w[b]+5:y:5, v[b]+15:x+10:5, b] + l*(10-k)*Hypercube[w[b]+15:y+10:5, v[b]+5:x:5, b] + k*l*Hypercube[w[b]+15:y+10:5, v[b]+15:x+10:5, b])
                else:
                    pass
##Hypercube[9:y:5, 5:x:5, 0] = 0.01*(5*6*Hypercube[9:y:5, 6:x:5, 0] + 4*5*Hypercube[9:y:5, 16:x+10:5, 0] +6*5*Hypercube[19:y+10:5, 6:x:5, 0] + 4*5*Hypercube[19:y+10:5, 16:x+10:5, 0]) #tried to change y direction so same as np indexing to create loops
#Normalise values
Hypercube = (Hypercube -Hypercube.min())/(Hypercube.max()-Hypercube.min())
Hypercube = 255*Hypercube
#Remove padding
Hypercube = np.delete(Hypercube, np.s_[0:5], 0)
Hypercube = np.delete(Hypercube, np.s_[-5:], 0)
Hypercube = np.delete(Hypercube, np.s_[0:5], 1)
Hypercube = np.delete(Hypercube, np.s_[-5:], 1)
#Test by visualising pseudo-RGB image
view2 = imshow(Hypercube, bands = (1,11,23)) #Does not appear quite as expected not sure why
#############SAVE DATA
##save_image(newfilename+'.hdr', Hypercube)
##md = {'lines': Hypercube.shape[0]
##      'samples':Hypercube.shape[1]
##      'bands': Hypercube.shape[2]
##      'data type': 12}
img = envi.save_image(data+newfilename+'.hdr', Hypercube, shape = Hypercube.shape, dtype=np.float32, force=True)
