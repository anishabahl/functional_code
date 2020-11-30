import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
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
plt.plot(sortedlight[:, 0], sortedlight[:, 1])
plt.show(block=False)
sortedfilter[:,1] = (sortedfilter[:,1]-sortedfilter[:,1].min())/(sortedfilter[:,1].max()-sortedfilter[:,1].min())
plt.figure("filter spectrum")
plt.plot(sortedfilter[:, 0], sortedfilter[:, 1])
#expand filter array to be same shape as light source spectrum 
interpolatedfilter = np.zeros(sortedlight.shape)
interpolatedfilter[:,0] = sortedlight[:, 0]
begin = np.array([0, 0])
end = np.array([1200, 1])
interpolatedfilter = np.vstack((begin, interpolatedfilter, end))
interpolatedfilterdf = pd.DataFrame(interpolatedfilter)
sortedfilterdf = pd.DataFrame(sortedfilter)
roundwavelength1 = np.round(interpolatedfilter[:,0])
interpolatedfilterdf['Rounded'] = roundwavelength1
roundwavelength2 = np.round(sortedfilter[:,0])
sortedfilterdf['Rounded'] = roundwavelength2
interpolatedfilterdf = pd.merge(interpolatedfilterdf, sortedfilterdf, how='left', on=['Rounded'])
interpolatedfilterdf = interpolatedfilterdf.drop(['1_x', 'Rounded', '0_y'], axis=1)
interpolatedfilterdf.iloc[0]['1_y'] = 0
interpolatedfilterdf.iloc[-1]['1-y'] = 1
plt.plot(interpolatedfilterdf.iloc[:]['0_x'], interpolatedfilterdf.iloc[:]['1_y']) #check spectrum has not changed
plt.show(block=False)
#interpolate tails of filter spectrum so no more NaN
#multiply light source and filter spectra 
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
for i in range(25):
    bandideal[:, i] = approx_gaus(x=wavelengths, QE=bandparameters[2, i], fwhm=bandparameters[1, i], centre=bandparameters[0, i])
##    plt.figure(i+1)
    plt.plot(wavelengths, bandresponses[:, i+1], color='k',  label='real')
    plt.plot(wavelengths, bandideal[:, i], color = 'r', label = 'ideal')
    if i ==0:
        plt.legend(loc='best')
    plt.show(block=False)
