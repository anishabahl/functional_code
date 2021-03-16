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
data = '/home/ab20/Data/Kuka/2021_03_15/vis/v11/' #path to data
filetype = '.png'
whitename = 'ref_white' #white reference image name
darkname = 'ref_dark' #dark reference image name
newlocation = '/home/ab20/Data/Kuka/2021_03_15/vis/v11_demosaiced/' #name of new location to put new files (must already exist)
x = '/home/ab20/Data/Calibration_file/visfullxaxis16.csv' #all wavelengths

wavelengths = genfromtxt(x, delimiter=',')
wavelengths = np.delete(wavelengths, 0,0)
wavelengths = wavelengths.reshape(wavelengths.shape[0],)
wavelengths = [str(w) for w in wavelengths]
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
                Hypercube = np.zeros((1088,2048,16))
                y=rawimg.shape[0]
                x = rawimg.shape[1]
                Hypercube[0:y:4, 0:x:4, 0] = rawimg[0:y:4, 0:x:4]
                Hypercube[0:y:4, 1:x:4, 1] = rawimg[0:y:4, 1:x:4]
                Hypercube[0:y:4, 2:x:4, 2] = rawimg[0:y:4, 2:x:4]
                Hypercube[0:y:4, 3:x:4, 3] = rawimg[0:y:4, 3:x:4]
                Hypercube[1:y:4, 0:x:4, 4] = rawimg[1:y:4, 0:x:4]
                Hypercube[1:y:4, 1:x:4, 5] = rawimg[1:y:4, 1:x:4]
                Hypercube[1:y:4, 2:x:4, 6] = rawimg[1:y:4, 2:x:4]
                Hypercube[1:y:4, 3:x:4, 7] = rawimg[1:y:4, 3:x:4]
                Hypercube[2:y:4, 0:x:4, 8] = rawimg[2:y:4, 0:x:4]
                Hypercube[2:y:4, 1:x:4, 9] = rawimg[2:y:4, 1:x:4]
                Hypercube[2:y:4, 2:x:4, 10] = rawimg[2:y:4, 2:x:4]
                Hypercube[2:y:4, 3:x:4, 11] = rawimg[2:y:4, 3:x:4]
                Hypercube[3:y:4, 0:x:4, 12] = rawimg[3:y:4, 0:x:4]
                Hypercube[3:y:4, 1:x:4, 13] = rawimg[3:y:4, 1:x:4]
                Hypercube[3:y:4, 2:x:4, 14] = rawimg[3:y:4, 2:x:4]
                Hypercube[3:y:4, 3:x:4, 15] = rawimg[3:y:4, 3:x:4]
                ########INTERPOLATING TO FILL GAPS -- using bilinear interpolation
                Hypercube= np.pad(Hypercube,((4,4), (4,4), (0,0))) #pad but will have edge effects as only padded with 0s
                y = (rawimg.shape[0]//4)*4
                x = (rawimg.shape[1]//4)*4
                #creating loops instead of 24*24 commands
                v = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
                w = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
                for b in range(0,16):
                    for i in range(0,4) :
                        for j in range(0,4):
                            if Hypercube[j+4][i+4][b] == 0:
                                k=i+4-v[b]
                                l = j+4-w[b]
                                Hypercube[j+4:y:4, i+4:x:4, b] = (1/64)*((8-l)*(8-k)*Hypercube[w[b]+4:y:4, v[b]+4:x:4, b] + (8-l)*k*Hypercube[w[b]+4:y:4, v[b]+12:x+8:4, b] + l*(8-k)*Hypercube[w[b]+12:y+8:4, v[b]+4:x:4, b] + k*l*Hypercube[w[b]+12:y+8:4, v[b]+12:x+8:4, b])
                            else:
                                pass
                #print(Hypercube[4:-4,4:-4,1])
                #Normalise values
                Hypercube = (Hypercube -Hypercube.min())/(Hypercube.max()-Hypercube.min())
                Hypercube = 255*Hypercube
                #Remove padding
                Hypercube = np.delete(Hypercube, np.s_[0:4], 0)
                Hypercube = np.delete(Hypercube, np.s_[-4:], 0)
                Hypercube = np.delete(Hypercube, np.s_[0:4], 1)
                Hypercube = np.delete(Hypercube, np.s_[-4:], 1)
                Data = Hypercube
                #############SAVE DATA
                metadata = {}
                metadata["wavelength"] = wavelengths
                metadata["wavelength units"] = "nm"
                metadata["bands"] = 16
                #metadata["height"] = 1088
                #metadata["width"] = 2048
                img = envi.save_image(newlocation+newfilename+'.hdr', Data, shape = Data.shape, dtype=np.float32, force=True, metadata=metadata)


