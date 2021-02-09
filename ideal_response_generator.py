import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.linalg import lstsq
from scipy.optimize import fmin_slsqp
from scipy.optimize import minimize
from scipy.sparse.linalg import lsmr
from scipy.optimize import lsq_linear
from scipy.optimize import curve_fit
import sklearn
from sklearn.linear_model import Ridge
import math
import os
import glob
import spectral
from spectral import imshow
from spectral import envi
import cv2
import leastsqbound
######INPUTS
bands = 'bandresponses.csv' #measured unfiltered band responses from imec calibration file
path = '/home/ab20/Data/Calibration_file/' #location of file
filterdata = 'AsahiSpectra_XVL0670.csv' #filter spectrum
band_parameters = 'idealbandparameters.csv' #imec parameters imported for initial guess
newlocation = '/home/ab20/Data/Calibration_file/' #location of new file
newname = 'fittedparameters.csv'
######DEFINE FUNCTIONS
def lorentz(x, QE, fwhm, centre):
    p = math.sqrt(4*centre**2 + fwhm**2) - 2*centre
    y = (p*x)/((x-centre)**2 + p*x)
    y2 = QE*y
    return y2
######IMPORT DATA
bandresponses = genfromtxt(path+bands, delimiter=',')
bandresponses = np.delete(bandresponses, 0, 0)
filterspectrum = genfromtxt(path+filterdata, delimiter=',')
filterspectrum = np.delete(filterspectrum,0,0)
bandparameters = genfromtxt(path+band_parameters, delimiter=',')
bandparameters = np.delete(bandparameters,0, 0)
######MULTIPLYING BAND RESPONSE BY FILTER
sortedfilter = filterspectrum[np.argsort(filterspectrum[:,0])]
sortedfilter[:,1] = (sortedfilter[:,1]-sortedfilter[:,1].min())/(sortedfilter[:,1].max()-sortedfilter[:,1].min())
interpolatedfilter = np.zeros((bandresponses.shape[0], 2))
interpolatedfilter[:,0] = bandresponses[:, 0]
b1 = np.array([399.998, 0])
e1 = np.array([1000.0, 1])
sortedfilter = np.vstack((b1, sortedfilter, e1))
f1 = interp1d(sortedfilter[:, 0], sortedfilter[:, 1])
interpolatedfilter[:, 1] = f1(interpolatedfilter[:, 0])
#interpolatedfilter = interpolatedfilter[1:-1]
plt.figure("filter spectrum")
plt.plot(interpolatedfilter[:, 0], interpolatedfilter[:, 1], label='filter spectrum')
plt.legend(loc='best')
plt.show(block=False)
plt.figure("band responses")
filteredbandresponses = np.zeros(bandresponses.shape)
filteredbandresponses[:, 0] = bandresponses[:, 0]
i = 0
for i in range(bandresponses.shape[1] - 1):
    filteredbandresponses[:, i + 1] = np.multiply(bandresponses[:, i + 1], interpolatedfilter[:, 1])
    plt.plot(bandresponses[:, 0], bandresponses[:, i+1], label='band responses', color='black')
    plt.plot(filteredbandresponses[:, 0], filteredbandresponses[:, i+1], label='filtered band responses', color='red')
    i+=1
######FIT LORENTZIAN FUNCTIONS
j = 0
plt.figure('Fitted response')
fittedparameters = np.zeros(bandparameters.shape)
for j in range(bandresponses.shape[1] - 1):
    guess = (bandparameters[2, j], bandparameters[1, j], bandparameters[0, j])
    popt, pcov = curve_fit(lorentz, filteredbandresponses[:, 0], filteredbandresponses[:, j+1], p0=guess, bounds=((0,0,400), (1, 1000, 1000)))
    fittedparameters[2, j] = popt[0]
    fittedparameters[1, j] = popt[1]
    fittedparameters[0, j] = popt[2]
    ###need to get errors and contributions
    plt.figure('Fitted response')
    plt.plot(filteredbandresponses[:, 0], filteredbandresponses[:, j+1], color = 'red')
    plt.plot(filteredbandresponses[:, 0], lorentz(filteredbandresponses[:, 0], *popt), color='blue')
    plt.figure(str(j) + 'th band response')
    plt.plot(filteredbandresponses[:, 0], filteredbandresponses[:, j + 1], color='red')
    plt.plot(filteredbandresponses[:, 0], lorentz(filteredbandresponses[:, 0], *popt), color='blue')
    plt.show(block=False)
    j += 1
plt.show(block=False)
######EXPORT PARAMETERS
Array = pd.DataFrame(fittedparameters)
Array.to_csv(newlocation+newname, index=False)