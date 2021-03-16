import os
import cv2
import rawpy as raw
import imageio
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import spectral
from spectral import imshow
from spectral import envi
import xml.etree.ElementTree as ET
#######INPUTS
data = '/home/ab20/Data/Kuka/2021_03_15/NIR/v0/' #path to data
filetype = '.png'
whitename = 'ref_white' #white reference image name
darkname = 'ref_dark' #dark reference image name
newlocation = '/home/ab20/Data/Kuka/2021_03_15/NIR/v0_demosaiced/' #name of new location to put new files (must already exist)
x = '/home/ab20/Data/Calibration_file/fullxaxis25.csv' #all wavelengths

wavelengths = genfromtxt(x, delimiter=',')
wavelengths = np.delete(wavelengths, 0,0)
wavelengths = wavelengths.reshape(wavelengths.shape[0],)
#print(wavelengths[:])
wavelengths = [str(w) for w in wavelengths]
#print(wavelengths)
for file in sorted(os.listdir(data)):
    if file.endswith(filetype):
        if not file.endswith('label'+filetype):
            if str(file[:-4]) != whitename and str(file[:-4]) != darkname:
                print(file[:-4])
                filename = file[:-4]
                newfilename = filename+'_demosaiced'
                #######IMPORTING DATA
                if filetype == ".raw":
                    img = np.fromfile(data+filename+'.raw', dtype=np.uint8, sep="")
                    rawimg = img.reshape(1088,2048)
                    whiteimg = np.fromfile(data+whitename+'.raw', dtype=np.uint8, sep="")
                    white = whiteimg.reshape(1088,2048)
                    darkimg = np.fromfile(data+darkname+'.raw', dtype=np.uint8, sep="")
                    dark = darkimg.reshape(1088,2048)
                if filetype == ".png":
                    rawimg = cv2.imread(data+filename+'.png', cv2.IMREAD_UNCHANGED)
                    white = cv2.imread(data+whitename+'.png', cv2.IMREAD_UNCHANGED)
                    dark = cv2.imread(data+darkname+'.png', cv2.IMREAD_UNCHANGED)
                #######WHITE BALANCING
                rawimg = rawimg.astype('float32')
                dark = dark.astype('float32')
                white = white.astype('float32')
                zeros = np.zeros(rawimg.shape)
                epsilons = zeros + np.finfo(float).eps
                rawimg = np.maximum((rawimg-dark), zeros)/np.maximum((white-dark), epsilons)
                rawimg = np.clip(rawimg, 0, 1)
##                view = imshow(rawimg)
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
                Hypercube= np.pad(Hypercube,((5,5), (5,5), (0,0))) #pad but will have edge effects as only padded with 0s
                y = (rawimg.shape[0]//5)*5
                x = (rawimg.shape[1]//5)*5
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
                #Normalise values
                Hypercube = (Hypercube -Hypercube.min())/(Hypercube.max()-Hypercube.min())
                Hypercube = 255*Hypercube
                #Remove padding
                Hypercube = np.delete(Hypercube, np.s_[0:5], 0)
                Hypercube = np.delete(Hypercube, np.s_[-5:], 0)
                Hypercube = np.delete(Hypercube, np.s_[0:5], 1)
                Hypercube = np.delete(Hypercube, np.s_[-5:], 1)

            #reorder
                Data = np.zeros((Hypercube.shape))
                Data[:, :, 0] = Hypercube[:, :, 20]
                Data[:, :, 1] = Hypercube[:, :, 21]
                Data[:, :, 2] = Hypercube[:, :, 22]
                Data[:, :, 3] = Hypercube[:, :, 23]
                Data[:, :, 4] = Hypercube[:, :, 24]
                Data[:, :, 5] = Hypercube[:, :, 15]
                Data[:, :, 6] = Hypercube[:, :, 16]
                Data[:, :, 7] = Hypercube[:, :, 17]
                Data[:, :, 8] = Hypercube[:, :, 18]
                Data[:, :, 9] = Hypercube[:, :, 19]
                Data[:, :, 10] = Hypercube[:, :, 10]
                Data[:, :, 11] = Hypercube[:, :, 11]
                Data[:, :, 12] = Hypercube[:, :, 12]
                Data[:, :, 13] = Hypercube[:, :, 13]
                Data[:, :, 14] = Hypercube[:, :, 14]
                Data[:, :, 15] = Hypercube[:, :, 5]
                Data[:, :, 16] = Hypercube[:, :, 6]
                Data[:, :, 17] = Hypercube[:, :, 7]
                Data[:, :, 18] = Hypercube[:, :, 8]
                Data[:, :, 19] = Hypercube[:, :, 9]
                Data[:, :, 20] = Hypercube[:, :, 4]
                Data[:, :, 21] = Hypercube[:, :, 0]
                Data[:, :, 22] = Hypercube[:, :, 1]
                Data[:, :, 23] = Hypercube[:, :, 2]
                Data[:, :, 24] = Hypercube[:, :, 3]
                #Test by visualising pseudo-RGB image
##                view2 = imshow(Hypercube, bands = (1,11,23)) #Does not appear quite as expected not sure why
                #############SAVE DATA
                metadata = {}
                metadata["wavelength"] = wavelengths
                metadata["wavelength units"] = "nm"
                metadata["bands"] = 25
                #metadata["height"] = 1088
                #metadata["width"] = 2048
                img = envi.save_image(newlocation+newfilename+'.hdr', Data, shape = Data.shape, dtype=np.float32, force=True, metadata=metadata)


