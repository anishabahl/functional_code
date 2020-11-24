import cv2
import rawpy as raw
import imageio
import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import spectral
from spectral import imshow
from spectral import envi
import xml.etree.ElementTree as ET
import spectral.io.aviris as aviris
import h5py
##########INPUTS
datapath = '/home/ab20/Data/System_Paper/imec/A/' #path to demosaiced data
filename = 'A1_corrected' #name of data file
filetype = '.raw' #type of raw data file ie raw for imec and img for photonfocus
segmentsfile = 'A1_corrected_label' #name of file with segments
tile = 1 #counting on checkerboard left to right top to bottom like a book with A1 = 1
tilename = '1A'
camera = 'imec'
calibrationpath = '/home/ab20/Data/Calibration_file/' #only necessary for calibration and spectrometer data so if turned off does not matter

calibratedata = 'OFF' #currently should always be off for imec
prereorder = 'ON' #Put on if necessary to reorder hypercube prior to calibration
calibrationfile = 'calibrationmatrix' #name of calibration csv file
newfilename = 'A1calibrated' #name of new file
newpath = '/home/ab20/Data/System_Paper/calibrated_data/'

Spectra = 'ON' #'ON' if want to plot spectra similar to system paper
spectrometer = 'spydercheckr_spectra_spectrometer' #file with spectrometer data from checkerboard
plotcomparison = 'ON' #if want to plot previous data from .h5 file
comparepath = '/home/ab20/Data/analysis/' #path to existing photonfocus and imec data for comparison
comparedata = 'checkerboard_imec_v1.h5' #file with existing photonfocus or imec data for comparison
bandwavelengths = 'bandwavelengths'

###########IMPORT DATA
data = envi.open(datapath+filename+'.hdr', datapath + filename + filetype)
Hypercubedata = data[:,:,:]
if calibratedata == 'ON':
    calibration = genfromtxt(calibrationpath+calibrationfile+'.csv', delimiter=',')
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
else:
    Data = Hypercubedata
if Spectra == 'ON':
    segments = cv2.imread(datapath+segmentsfile+'.png', cv2.IMREAD_UNCHANGED)
    spectrometerdata = genfromtxt(calibrationpath+spectrometer+'.csv', delimiter=',')
    spectrometerdata = np.delete(spectrometerdata, 0, 0)
    xaxis = genfromtxt(calibrationpath+bandwavelengths+'.csv', delimiter=',')
    xaxis = np.delete(xaxis,0,0)
    xaxis = xaxis.reshape(xaxis.shape[0],1)
    if plotcomparison == 'ON':
        importfile = h5py.File(comparepath +comparedata, 'r')
        compare = pd.read_hdf(comparepath+comparedata)
        meanscompare = compare.groupby(['patch', 'wavelength']).mean()
        compareplot = meanscompare.loc[tilename]

###########CALIBRATION
if calibratedata == 'ON':
    calibrateddata = np.zeros((Data.shape[0], Data.shape[1], calibration.shape[1]))
    Data = (Data - Data.min())/(Data.max() - Data.min())
    calibrateddata[0:calibrateddata.shape[0]:1, 0:calibrateddata.shape[1]:1] = np.dot(Data[0:Data.shape[0]:1, 0:Data.shape[1]:1], calibration)
###########SAVE AS CALIBRATED IMAGE
    img = envi.save_image(newpath+newfilename+'.hdr', calibrateddata, shape = calibrateddata.shape, dtype=np.float32, force=True)
###########TEST BY PSEUDO_RGB IMAGE
    view4 = imshow(calibrateddata, bands=(0,11,22))
else:
    calibrateddata = Data
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
    plt.figure('Spectrum')
##    plt.plot(xaxis, spectrum, label='Photonfocus')
    plt.plot(spectrum, label = camera)
    plt.plot(spectrometerdata[:,0], spectrometerdata[:,tile], label='Spectrometer')
    plt.plot(compareplot, label = 'Previous '+camera+' data')
    plt.xlabel('wavelength/nm')
    plt.ylabel('intensity/arb')
    plt.legend(loc = 'best')
    plt.show()
