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
import spectral.io.aviris as aviris
##########INPUTS
datapath = '/home/ab20/Data/System_Paper/raw_data/' #path to demosaiced data
filename = 'A1demosaiced' #name of data file
segmentsfile = 'A1_label' #name of file with segments
calibrationpath = '/home/ab20/Data/Calibration_file/'
calibrationfile = 'calibrationmatrix' #name of calibration csv file
bandwavelengths = 'bandwavelengths'
newfilename = 'A1calibrated' #name of new file
newpath = '/home/ab20/Data/System_Paper/calibrated_data/'
Spectra = 'ON' #'ON' if want to plot spectra similar to system paper
tile = 1 #counting on checkerboard left to right top to bottom like a book with A1 = 1
spectrometer = 'spydercheckr_spectra_spectrometer' #file with spectrometer data from checkerboard
prereorder = 'ON' #Put on if necessary to reorder hypercube prior to calibration
postreorder = '0' #reorder calibrated hypercube,not sure this should happen, 0=none, 1=simple reversal of order of bands, 2=reordering as if in mosaic with 2 tiles removed counting from bottom instead of top
###########IMPORT DATA
data = envi.open(datapath+filename+'.hdr', datapath + filename + '.img')
Hypercubedata = data[:,:,:]
calibration = genfromtxt(calibrationpath+calibrationfile+'.csv', delimiter=',')
segments = cv2.imread(datapath+segmentsfile+'.png', cv2.IMREAD_UNCHANGED)
xaxis = genfromtxt(calibrationpath+bandwavelengths+'.csv', delimiter=',')
spectrometerdata = genfromtxt(calibrationpath+spectrometer+'.csv', delimiter=',')
spectrometerdata = np.delete(spectrometerdata, 0, 0)
xaxis = np.delete(xaxis,0,0)
xaxis = xaxis.reshape(xaxis.shape[0],1)
##imec = genfromtxt(calibrationpath + 'spydercheckr_spectra_imec.csv', delimiter=',')
##imec = np.delete(imec, 0, 0)
##calibration = np.delete(calibration,0,0)
calibration = np.delete(calibration,0,0)
calibration = np.transpose(calibration)
if prereorder == 'ON':
    Data = np.zeros((Hypercubedata.shape))
    Data[:,:,0] = Hypercubedata[:,:,20]
    Data[:,:,1] = Hypercubedata[:,:,21]
    Data[:,:,2] = Hypercubedata[:,:,22]
    Data[:,:,3] = Hypercubedata[:,:,23]
    Data[:,:,4] = Hypercubedata[:,:,24]
    Data[:,:,5] = Hypercubedata[:,:,15]
    Data[:,:,6] = Hypercubedata[:,:,16]
    Data[:,:,7] = Hypercubedata[:,:,17]
    Data[:,:,8] = Hypercubedata[:,:,18]
    Data[:,:,9] = Hypercubedata[:,:,19]
    Data[:,:,10] = Hypercubedata[:,:,10]
    Data[:,:,11] = Hypercubedata[:,:,11]
    Data[:,:,12] = Hypercubedata[:,:,12]
    Data[:,:,13] = Hypercubedata[:,:,13]
    Data[:,:,14] = Hypercubedata[:,:,14]
    Data[:,:,15] = Hypercubedata[:,:,5]
    Data[:,:,16] = Hypercubedata[:,:,6]
    Data[:,:,17] = Hypercubedata[:,:,7]
    Data[:,:,18] = Hypercubedata[:,:,8]
    Data[:,:,19] = Hypercubedata[:,:,9]
    Data[:,:,20] = Hypercubedata[:,:,4]
    Data[:,:,21] = Hypercubedata[:,:,0]
    Data[:,:,22] = Hypercubedata[:,:,1]
    Data[:,:,23] = Hypercubedata[:,:,2]
    Data[:,:,24] = Hypercubedata[:,:,3]
else:
    Data = data[:,:,:]
###########CALIBRATION
calibrateddata = np.zeros((Data.shape[0], Data.shape[1], calibration.shape[1]))
Data = (Data - Data.min())/(Data.max() - Data.min())
calibrateddata[0:calibrateddata.shape[0]:1, 0:calibrateddata.shape[1]:1] = np.dot(Data[0:Data.shape[0]:1, 0:Data.shape[1]:1], calibration)
##calibrateddata = (calibrateddata - calibrateddata.min())/(calibrateddata.max() - calibrateddata.min())
if postreorder == '1':
    calibrateddata = np.flip(calibrateddata, axis = 2)
if postreorder == '2': 
    Calibrateddata = calibrateddata
    calibrateddata = np.zeros((Calibrateddata.shape))
    calibrateddata[:,:,0] = Calibrateddata[:,:,20]
    calibrateddata[:,:,1] = Calibrateddata[:,:,21]
    calibrateddata[:,:,2] = Calibrateddata[:,:,22]
    calibrateddata[:,:,3] = Calibrateddata[:,:,15]
    calibrateddata[:,:,4] = Calibrateddata[:,:,16]
    calibrateddata[:,:,5] = Calibrateddata[:,:,17]
    calibrateddata[:,:,6] = Calibrateddata[:,:,18]
    calibrateddata[:,:,7] = Calibrateddata[:,:,19]
    calibrateddata[:,:,8] = Calibrateddata[:,:,10]
    calibrateddata[:,:,9] = Calibrateddata[:,:,11]
    calibrateddata[:,:,10] = Calibrateddata[:,:,12]
    calibrateddata[:,:,11] = Calibrateddata[:,:,13]
    calibrateddata[:,:,12] = Calibrateddata[:,:,14]
    calibrateddata[:,:,13] = Calibrateddata[:,:,5]
    calibrateddata[:,:,14] = Calibrateddata[:,:,6]
    calibrateddata[:,:,15] = Calibrateddata[:,:,7]
    calibrateddata[:,:,16] = Calibrateddata[:,:,8]
    calibrateddata[:,:,17] = Calibrateddata[:,:,9]
    calibrateddata[:,:,18] = Calibrateddata[:,:,0]
    calibrateddata[:,:,19] = Calibrateddata[:,:,1]
    calibrateddata[:,:,20] = Calibrateddata[:,:,2]
    calibrateddata[:,:,21] = Calibrateddata[:,:,3]
    calibrateddata[:,:,22] = Calibrateddata[:,:,4]
###########SAVE AS CALIBRATED IMAGE
img = envi.save_image(newpath+newfilename+'.hdr', calibrateddata, shape = calibrateddata.shape, dtype=np.float32, force=True)
###########TEST BY PSEUDO_RGB IMAGE
##img.bands = aviris.read_aviris_bands(calibrationpath+calibrationfile+'.csv')
view4 = imshow(calibrateddata, bands=(0,11,22))
###########PLOTTING SPECTRA
if Spectra == 'ON':
    segments3D = np.repeat(segments[:, :, np.newaxis], calibrateddata.shape[2], axis=2)
    segmenteddata = np.multiply(segments3D, calibrateddata) # produces matrix of 0s everywhere except in segmented region where calibrated spectral data
    #TEST WITH PSEUDO RGB IMAGE
    view5 = imshow(segmenteddata, bands=(0,11,22))
    n = np.count_nonzero(segmenteddata, axis=(0,1))
    spectrum = np.zeros((calibrateddata.shape[2], 1))
    for i in range(spectrum.shape[0]):
        spectrum[i] = np.sum(segmenteddata[:,:,i])/n[i] #creates y axis of spectrum
##    Spectrum = np.hstack((xaxis, spectrum))
    plt.figure('Spectrum')
    plt.plot(xaxis, spectrum, label='Photonfocus')
    plt.plot(spectrometerdata[:,0], spectrometerdata[:,tile], label='Spectrometer')
##    plt.plot(imec[:,0], imec[:,1], label='Imec')
    plt.xlabel('wavelength/nm')
    plt.ylabel('intensity/arb')
    plt.legend(loc = 'best')
    plt.show()
