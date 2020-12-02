import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy.linalg import lstsq
import math
######INPUTS
bands = 'bandresponses.csv'
path = '/home/ab20/Data/Calibration_file/'
band_parameters = 'idealbandparameters.csv' 
lightspectrum = 'IR light with LP UV filter thorugh exoscope and adaptor.txt'
filterdata = 'AsahiSpectra_XVL0670.csv' 
newname = 'Pichettematrix'
newpath = '/home/ab20/Data/Calibration_file/'
######DEFINE FUNCTIONS
def approx_gaus(x, QE, fwhm, centre):
    sigma = 0.5*fwhm/np.sqrt(np.log(2))
    return QE*np.exp(-np.square((x-centre)/sigma))
def findC(C, A, Aideal):
    C = np.reshape(C, (bandresponses.shape[1]-1, bandresponses.shape[1]-1))
    D = Aideal - np.dot(A,C)
    return D.flatten()
##    return np.sqrt(np.multiply(D, D))
######IMPORT DATA
bandresponses = genfromtxt(path+bands, delimiter=',')
bandresponses = np.delete(bandresponses, 0, 0)
wavelengths = bandresponses[:, 0]
bandparameters = genfromtxt(path+band_parameters, delimiter=',')
bandparameters = np.delete(bandparameters,0, 0)
lightsource = genfromtxt(path+lightspectrum, skip_header = 14, delimiter = '\t')
filterspectrum = genfromtxt(path+filterdata, delimiter=',')
filterspectrum = np.delete(filterspectrum,0,0)
######Light source spectrum = spectrum x filterspectrum
#sort so both ascending wavelengths
sortedlight = lightsource[np.argsort(lightsource[:,0])]
sortedfilter = filterspectrum[np.argsort(filterspectrum[:,0])]
#normalise intensities
sortedlight[:,1] = (sortedlight[:,1]-sortedlight[:,1].min())/(sortedlight[:,1].max()-sortedlight[:,1].min())
plt.figure("light spectrum")
plt.plot(sortedlight[:, 0], sortedlight[:, 1], label='original light spectrum')
plt.show(block=False)
sortedfilter[:,1] = (sortedfilter[:,1]-sortedfilter[:,1].min())/(sortedfilter[:,1].max()-sortedfilter[:,1].min())
plt.figure("filter spectrum")
plt.plot(sortedfilter[:, 0], sortedfilter[:, 1], label='original filter spectrum')
#expand filter array to be same shape as light source spectrum 
interpolatedfilter = np.zeros(sortedlight.shape)
interpolatedfilter[:,0] = sortedlight[:, 0]
begin = np.array([0, 0])
end = np.array([1200, 1])
sortedfilter = np.vstack((begin, sortedfilter, end))
f = interp1d(sortedfilter[:, 0], sortedfilter[:, 1])
interpolatedfilter[:, 1] = f(interpolatedfilter[:, 0])
plt.plot(interpolatedfilter[:, 0], interpolatedfilter[:, 1], label='interpolated filter')
plt.legend(loc='best')
plt.show(block=False) 
#multiply light source and filter spectra
sortedfilter = np.delete(sortedfilter, 0, 0)
sortedfilter = np.delete(sortedfilter, -1, 0)
correctedlight = np.zeros(sortedlight.shape)
correctedlight[:,0] = sortedlight[:, 0]
correctedlight[:, 1] = np.multiply(sortedlight[:, 1], interpolatedfilter[:,1])
plt.figure('light spectrum')
plt.plot(correctedlight[:, 0], correctedlight[:, 1], label='multiplied by filter')
#create less resolved light spectrum to match wavelength axis of all other data
smalllight = np.zeros((bandresponses.shape[0], 2))
smalllight[:, 0] = wavelengths
g = interp1d(correctedlight[:,0], correctedlight[:,1])
smalllight[:,1] = g(smalllight[:,0])
plt.plot(smalllight[:,0], smalllight[:,1], label='less resolved spectrum')
plt.legend(loc = 'best')
plt.show(block = False)
######Creating optical transmission but should measure?
optictrans = np.ones((bandresponses.shape[0], 1)) 
noise = np.random.normal(0, 0.5, optictrans.shape)
optictrans = optictrans + noise
######GENERATING IDEALS
optictransideal = np.ones((bandresponses.shape[0], 1)) #all 1 as ideally no loss of intensity
lightideal = np.copy(wavelengths)
lightideal = np.where(lightideal>1000, 0, lightideal)
lightideal = np.where(lightideal<670, 0, lightideal)
lightideal = np.where(lightideal!=0, 1, lightideal) #ideally box function where changes at 670 and 1000nm
plt.figure('band responses')
##Approximate Gaussians for band responses
bandideal = np.zeros((bandresponses.shape[0], bandresponses.shape[1]-1))
for i in range(bandresponses.shape[1]-1):
    bandideal[:, i] = approx_gaus(x=wavelengths, QE=bandparameters[2, i], fwhm=bandparameters[1, i], centre=bandparameters[0, i])
##    plt.figure(i+1)
    plt.plot(wavelengths, bandresponses[:, i+1], color='k',  label='real')
    plt.plot(wavelengths, bandideal[:, i], color = 'r', label = 'ideal')
    if i ==0:
        plt.legend(loc='best')
    plt.show(block=False)
#######GENERATING A AND Aideal MATRICES
#may need to reshape all wavelength axes similarly to light source x filter calc
A = np.zeros(bandideal.shape)
Aideal = np.zeros(bandideal.shape)
for i in range(bandresponses.shape[1]-1):
    A[:, i] = np.multiply(np.multiply(bandresponses[:, i+1], optictrans[:,0]), smalllight[:,1])
    Aideal[:, i] = np.multiply(np.multiply(lightideal, optictransideal[:,0]), bandideal[:, i])
for i in range(bandresponses.shape[1]-1):
    A[:, i] = A[:, i]/A.sum(axis=0)[i]
    Aideal[:, i] = Aideal[:, i]/Aideal.sum(axis=0)[i]
#######FIND C BY MINIMISATION
C, flag = leastsq(findC, x0 = np.ones(((bandresponses.shape[1]-1)**2, 1)), args=(A, Aideal))
C = np.reshape(C, (bandresponses.shape[1]-1, bandresponses.shape[1]-1))
##C, res, rnk, s = lstsq(Aideal, A)
print(C)
