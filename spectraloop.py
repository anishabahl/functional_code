import os
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
#######INPUTS
#currently for photonfocus: filetype = .img, calibratedata = on, prereorder = on, separatexaxis = on
#imec is opposite
datapath = '/home/ab20/Data/System_Paper/Photonfocus/v2demosaiced/' #path to data
filetype = '.img' #.img or .raw
camera = 'photonfocus'
calibrationpath = '/home/ab20/Data/Calibration_file/' #only necessary for calibration and spectrometer data so if turned off does not matter
calibratedata = 'ON' #currently should always be off for imec
prereorder = 'OFF' #Put on if necessary to reorder hypercube prior to calibration
calibrationfile = 'calibrationmatrix' #name of calibration csv file
newlocation = '/home/ab20/Data/Pichette/outofbox/onv2/' #name of new location to put new files (must already exist)
Spectra = 'ON' #'ON' if want to plot spectra similar to system paper
plottype = 'collective' #options are individual or collective
spectrometer = 'spydercheckr_spectra_spectrometer' #file with spectrometer data from checkerboard
plotcomparison = 'OFF' #if want to plot previous data from .h5 file
comparepath = '/home/ab20/Data/analysis/' #path to existing photonfocus and imec data for comparison
comparedata = 'checkerboard_photonfocus_v1.h5' #file with existing photonfocus or imec data for comparison
separatexaxis = 'ON' #on for photonfocus off for imec at the moment
bandwavelengths = 'bandwavelengths' #file with xaxis if separate
#######list to identify correct tile data in spectrometer and .h5 files
tiles = ['0', '1A', '2A', '3A', '4A', '5A', '6A', '1B', '2B', '3B', '4B', '5B', '6B', '1C', '2C', '3C', '4C', '5C', '6C', '1D', '2D', '3D', '4D', '5D', '6D', '1E', '2E', '3E', '4E', '5E', '6E', '1F', '2F', '3F', '4F', '5F', '6F', '1G', '2G', '3G', '4G', '5G', '6G', '1H', '2H', '3H', '4H', '5H', '6H']
tilearray = np.array([['1A', '2A', '3A', '4A', '5A', '6A'], ['1B', '2B', '3B', '4B', '5B', '6B'], ['1C', '2C', '3C', '4C', '5C', '6C'], ['1D', '2D', '3D', '4D', '5D', '6D'], ['1E', '2E', '3E', '4E', '5E', '6E'], ['1F', '2F', '3F', '4F', '5F', '6F'], ['1G', '2G', '3G', '4G', '5G', '6G'], ['1H', '2H', '3H', '4H', '5H', '6H']])
if plottype == 'collective':
    fig, axs = plt.subplots(tilearray.shape[0], tilearray.shape[1], sharex=True, sharey=True, figsize = [12, 12])
    plt.subplots_adjust(hspace = 0.25)
if calibratedata == 'ON':
    tosave = np.zeros((23, len(tiles))) #need to find way of counting bands here for 23
if calibratedata == 'OFF':
    tosave = np.zeros((25, len(tiles)))
for file in sorted(os.listdir(datapath)):
    if file.endswith(filetype):
        #setting file specific variables
        print(file[:-4])
        filename = file[:-4]
        segmentsfile = file[:2]+'_label'
        newfilename = file[:2] + '_calibrated'
        tilename = file[1]+file[0] 
        tile = tiles.index(tilename)
        coordindices = np.where(tilearray == tilename)
        coord = list(zip(coordindices[0], coordindices[1]))[0]
        #######IMPORTING DATA
        data = envi.open(datapath+filename+'.hdr', datapath + filename + filetype)
        Hypercubedata = data[:,:,:]
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
        if calibratedata == 'ON':
            calibration = genfromtxt(calibrationpath+calibrationfile+'.csv', delimiter=',')
            calibration = np.delete(calibration,0,0)
            calibration = np.transpose(calibration)
        if Spectra == 'ON':
            segments = cv2.imread(datapath+segmentsfile+'.png', cv2.IMREAD_UNCHANGED)
            spectrometerdata = genfromtxt(calibrationpath+spectrometer+'.csv', delimiter=',')
            spectrometerdata = np.delete(spectrometerdata, 0, 0)
        if separatexaxis == 'ON': 
            xaxis = genfromtxt(calibrationpath+bandwavelengths+'.csv', delimiter=',')
            xaxis = np.delete(xaxis,0,0)
            xaxis = xaxis.reshape(xaxis.shape[0],1)
        else:
            band = envi.open(datapath + filename + '.hdr')
            xaxis = band.bands.centers[:]
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
            img = envi.save_image(newlocation+newfilename+'.hdr', calibrateddata, shape = calibrateddata.shape, dtype=np.float32, force=True)
        else:
            calibrateddata = (Data - Data.min())/(Data.max() - Data.min())
        ###########PLOTTING SPECTRA
        if Spectra == 'ON':
            segments3D = np.repeat(segments[:, :, np.newaxis], calibrateddata.shape[2], axis=2)
            segmenteddata = np.multiply(segments3D, calibrateddata) # produces matrix of 0s everywhere except in segmented region where calibrated spectral data
            n = np.count_nonzero(segmenteddata, axis=(0,1))
            spectrum = np.zeros((calibrateddata.shape[2], 1))
            for i in range(spectrum.shape[0]):
                spectrum[i] = np.sum(segmenteddata[:,:,i])/n[i] #creates y axis of spectrum
            tosave[:, tile] = spectrum[:, 0]
            spectrum = np.hstack((xaxis, spectrum))
            spectrum = spectrum[np.argsort(spectrum[:, 0])]
            if plottype == 'individual':
                plt.figure('Spectrum'+tilename)
##                plt.plot(xaxis, spectrum, label = camera)
                plt.plot(spectrum[:, 0], spectrum[:, 1], label = camera)
                plt.plot(spectrometerdata[:,0], spectrometerdata[:,tile], label='Spectrometer')
                if plotcomparison == 'ON':
                    plt.plot(compareplot, label = 'Previous '+camera+' data')
                plt.xlabel('wavelength/nm')
                plt.ylabel('intensity/arb')
                plt.title(tilename)
                plt.legend(loc = 'best')
                plt.savefig(newlocation + ' ' + tilename)
                plt.close()
            if plottype == 'collective':
##                axs[coord[0], coord[1]].plot(xaxis, spectrum, label = camera)
                axs[coord[0], coord[1]].plot(spectrum[:, 0], spectrum[:, 1], label = camera)
                axs[coord[0], coord[1]].plot(spectrometerdata[:,0], spectrometerdata[:,tile], label='Spectrometer')
                if plotcomparison == 'ON':
                    axs[coord[0], coord[1]].plot(compareplot, label = 'Previous '+camera+' data')
                axs[coord[0], coord[1]].set_title(tilename, fontsize = 8)        
        continue
if plottype == 'collective':
##    plt.xlabel('wavelength/nm')
##    plt.ylabel('intensity/arb')
##    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
##    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
##    plt.legend(lines, labels, loc = 'best')
    fig.text(0.5, 0.04, 'wavelength/nm', ha='center', fontsize = 12)
    fig.text(0.04, 0.5, 'intensity/arb', va='center', rotation='vertical', fontsize = 12)
##    plt.legend(loc = 'lower center')
    plt.show(block=False)
    plt.savefig(newlocation + ' spectrum')
Array = pd.DataFrame(tosave)
Array.to_csv(newlocation + 'data.csv', index=False)
