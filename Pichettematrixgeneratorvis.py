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
bands = 'visbandresponses.csv'
path = '/home/ab20/Data/Calibration_file/'
band_parameters = 'visidealbandparameters.csv'
lightspectrum = 'IR with no UV filter directly from source.txt'
filterdata = 'vis_filter.csv'
spectrometer = 'spydercheckr_spectra_spectrometer'
datapath = '/home/ab20/Data/Kuka/2021_03_15/vis/v0_demosaiced/' #must have full hypercube data
filetype = '.img'
newname = 'vis_v0'
newxname = 'fullxaxis16'
newpath = '/home/ab20/Data/Calibration_file/'
imagepath = '/home/ab20/Data/Kuka/2021_03_15/vis/calibration_images/' #path to save new images
lam = 0.005 #parameter for regularisation
optCmethod = 'lsmr' #options are scikit 'Ridge' and 'lsmr'
optRmethod = 'SLSQP' #options are scipy minimise 'SLSQP' and 'least_sq'
alpha = 0.05 #bounds values in R matrix
filterbands = 'ON'
filterlight = 'OFF'
######DEFINE FUNCTIONS
def findR(R, C, Aideal, r, rref):
    R = np.reshape(R, (bandresponses.shape[1]-1, bandresponses.shape[1]-3))
    D = np.dot(np.transpose(C+R), r) - np.dot(np.transpose(Aideal), rref)
    return D.flatten()
def findRcost(R, C, Aideal, r, rref):
    R = np.reshape(R, (bandresponses.shape[1]-1, bandresponses.shape[1]-3))
    D = np.dot(np.transpose(C+R), r) - np.dot(np.transpose(Aideal), rref)
    return np.sum(np.square(D))
def lorentz(x, QE, fwhm, centre):
    p = math.sqrt(4*centre**2 + fwhm**2) - 2*centre
    y = (p*x)/((x-centre)**2 + p*x)
    y2 = QE*y
    return y2
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
#plt.plot(sortedfilter[:, 0], sortedfilter[:, 1], label='original filter spectrum')
#expand filter array to be same shape as light source spectrum
interpolatedfilter = np.zeros(sortedlight.shape)
interpolatedfilter[:,0] = sortedlight[:, 0]
begin = np.array([0, 0])
end = np.array([1200, 1])
sortedfilter = np.vstack((begin, sortedfilter, end))
f = interp1d(sortedfilter[:, 0], sortedfilter[:, 1])
interpolatedfilter[:, 1] = f(interpolatedfilter[:, 0])
interpolatedfilter2 = np.zeros((bandresponses.shape[0], 2))
interpolatedfilter2[:,0] = bandresponses[:, 0]
b1 = np.array([399.998, 0])
e1 = np.array([1000.0, 1])
sortedfilter2 = np.vstack((b1, sortedfilter, e1))
f1 = interp1d(sortedfilter2[:, 0], sortedfilter2[:, 1])
interpolatedfilter2[:, 1] = f1(interpolatedfilter2[:, 0])
plt.plot(interpolatedfilter[:, 0], interpolatedfilter[:, 1], label='filter spectrum')
plt.plot(interpolatedfilter2[:, 0], interpolatedfilter2[:, 1], label='filter spectrum interpolated differently')
plt.legend(loc='best')
plt.show(block=False)
plt.savefig(imagepath+'Filter_Spectrum')
#multiply light source and filter spectra
sortedfilter = np.delete(sortedfilter, 0, 0)
sortedfilter = np.delete(sortedfilter, -1, 0)
correctedlight = np.zeros(sortedlight.shape)
correctedlight[:,0] = sortedlight[:, 0]
if filterlight == 'ON':
    correctedlight[:, 1] = np.multiply(sortedlight[:, 1], interpolatedfilter[:,1])
else:
    correctedlight[:, 1] = sortedlight[:, 1]
plt.figure('light spectrum')
#plt.plot(correctedlight[:, 0], correctedlight[:, 1], label='multiplied by filter')
#create less resolved light spectrum to match wavelength axis of all other data
smalllight = np.zeros((bandresponses.shape[0], 2))
smalllight[:, 0] = wavelengths
g = interp1d(correctedlight[:,0], correctedlight[:,1])
smalllight[:,1] = g(smalllight[:,0])
plt.plot(smalllight[:,0], smalllight[:,1], label='multiplied by filter')
plt.legend(loc = 'best')
plt.show(block = False)
plt.savefig(imagepath+'Light_Spectrum')
# multiplying band responses by filter
#filteredbandresponses = np.zeros(bandresponses.shape)
#filteredbandresponses[:, 0] = bandresponses[:, 0]
if filterbands == 'ON':
    i = 0
    for i in range(bandresponses.shape[1] - 1):
        bandresponses[:, i + 1] = np.multiply(bandresponses[:, i + 1], interpolatedfilter2[:, 1])
        i+=1
######Creating optical transmission but should measure?
optictrans = np.ones((bandresponses.shape[0], 1))
noise = np.random.normal(0, 0.5, optictrans.shape)
optictrans = optictrans + noise
#####importing reference data
##spectrometerdata = genfromtxt(path+spectrometer+'.csv', delimiter=',')
spectrometerdata = pd.read_csv(path+spectrometer+'.csv')
cols = list(spectrometerdata.columns)
x = cols.pop(0)
dict = {x:x}
for i in range(len(cols)):
	new = cols[i][1]+cols[i][0]
	dict[cols[i]] = new
spectrometerdata.rename(columns = dict, inplace=True)
spectrometerdata = spectrometerdata.to_numpy()
#interpolate to make sure spectrometer data has same wavelength axis as the rest
spectra = np.zeros((bandresponses.shape[0], spectrometerdata.shape[1]-1))
for i in range(1, spectrometerdata.shape[1]):
    h = interp1d(spectrometerdata[:, 0], spectrometerdata[:, i], fill_value="extrapolate")
    spectra[:, i-1]=h(wavelengths)
##spectrometerdata = np.delete(spectrometerdata, 0, 0)
##print(str(spectrometerdata[0][:]))
n_files = len(glob.glob1(datapath,"*" + filetype))
n_bands = bandresponses.shape[1] - 1
Rdata = np.zeros((n_bands, n_files))
a = 0
for file in sorted(os.listdir(datapath)):
    if file.endswith(filetype):
        #setting file specific variables
        print(file[:-4])
        filename = file[:-4]
        segmentsfile = file[:2]+'_label'
        data = envi.open(datapath+filename+'.hdr', datapath + filename + filetype)
        Hypercubedata = data[:,:,:]
        Data = data[:,:,:]
        Data = (Data - Data.min())/(Data.max() - Data.min())
        segments = cv2.imread(datapath+segmentsfile+'.png', cv2.IMREAD_UNCHANGED)
        segments3D = np.repeat(segments[:, :, np.newaxis], Data.shape[2], axis=2)
        segmenteddata = np.multiply(segments3D, Data) # produces matrix of 0s everywhere except in segmented region where calibrated spectral data
        n = np.count_nonzero(segmenteddata, axis=(0,1))
        refdata = np.zeros((segmenteddata.shape[2],1))
        for i in range(refdata.shape[0]):
            refdata[i] = np.sum(segmenteddata[:,:,i])/n[i]
        Rdata[:, a:a+1] = refdata
        a = a+1
spectra = spectra[:, 0:a]
######GENERATING IDEALS
optictransideal = np.ones((bandresponses.shape[0], 1)) #all 1 as ideally no loss of intensity
lightideal = np.copy(wavelengths)
lightideal = np.where(lightideal>1000, 0, lightideal)
lightideal = np.where(lightideal<670, 0, lightideal)
lightideal = np.where(lightideal!=0, 1, lightideal) #ideally box function where changes at 670 and 1000nm
plt.figure('band responses')
##Approximate Gaussians for band responses
bandideal = np.zeros((bandresponses.shape[0], bandresponses.shape[1]-1))
cmap = plt.cm.Greys
colors = [cmap(i) for i in range(cmap.N)]
cmap2 = plt.cm.Greens
colors2 = [cmap2(i) for i in range(cmap2.N)]
for i in range(bandresponses.shape[1]-1):
    bandideal[:, i] = lorentz(x=wavelengths, QE=bandparameters[2, i], fwhm=bandparameters[1, i], centre=bandparameters[0, i])
    plt.plot(wavelengths, bandresponses[:, i+1], color=colors[-i*7],  label='filtered measured')
    plt.plot(wavelengths, bandideal[:, i], color = colors2[-i*7], label = 'ideal')
    if i ==0:
        plt.legend(loc='best')
    plt.show(block=False)
    plt.savefig(imagepath+'Band_responses')
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
#######FIND C BY MINIMISATION WITH REGULARISATION
##C, flag = leastsq(findC, x0 = np.ones(((bandresponses.shape[1]-1)**2, 1)), args=(A, Aideal))
##C = np.reshape(C, (bandresponses.shape[1]-1, bandresponses.shape[1]-1))
##regC, regflag = leastsq(regfindC, x0 = np.ones(((bandresponses.shape[1]-1)**2, 1)), args=(A, Aideal, 0))
if optCmethod == 'Ridge':
    regC = Ridge(alpha = lam)
    regC.fit(A, Aideal)
    regCresult = regC.coef_
    regCstats = regC.score(A, Aideal)
    print(regCstats)
    regCresult = np.transpose(regCresult)
if optCmethod == 'lsmr':
    C = np.zeros((bandresponses.shape[1]-1, Aideal.shape[1]))
    resid = np.zeros((Aideal.shape[1], 1))
    for i in range(Aideal.shape[1]):
        a,b,c,d = lsmr(A, Aideal[:, i], damp = lam)[:4]
        C[:, i] = a
        regCresult = C
        resid[i] = d
    #print(resid)
    print('Sum of resid of rows of C = ' + str(resid.sum()))
#regCresult should be same as C when lam=0 but is transpose probably because of all flattening and reshaping
##print(regC)
##F = C-Ctest
##G = C - regC
#######FIND R BY REFINEMENT
##R, flag = leastsq(findR, x0 = np.ones(((bandresponses.shape[1]-1)**2, 1)), args=(regCresult, Aideal, Rdata, spectra))
#originally used leastsq but not able to bound values so use SLSQP now which gives different results with same input but still gives mostly good fit
if optRmethod == 'SLSQP':
    a = (-alpha, alpha)
    bounds = ((a, )*((bandresponses.shape[1]-1)*(bandresponses.shape[1]-3)))
    resR = minimize(findRcost, x0 = np.zeros(((bandresponses.shape[1]-1)*(bandresponses.shape[1]-3), )), bounds=bounds, args=(regCresult, Aideal, Rdata, spectra), method='SLSQP', options={'disp': True})
    altR = resR.x
    altR = np.reshape(altR, (bandresponses.shape[1]-1, bandresponses.shape[1]-3))
if optRmethod == 'least_sq':
    residR = np.zeros((Aideal.shape[1]))
    R = np.zeros((bandresponses.shape[1]-1, Aideal.shape[1]))
    for i in range(Aideal.shape[1]):
        res = lsq_linear(A, Aideal[:, i], bounds = (-alpha, alpha), verbose = 1) #not sure what to put for A and Aideal
        R[:, i] = res.x
        residR[i] = res.cost
##R, flag = leastsq(findR, x0 = np.ones((regCresult.shape)), args=(regCresult, Aideal, Rdata, spectra))
##R = np.reshape(R, (bandresponses.shape[1]-1, bandresponses.shape[1]-1))
##R = np.transpose(R)
regCresult = np.transpose(regCresult)
altR = np.transpose(altR)
CR = regCresult + altR
##X = R - altR
plt.figure('C+R')
figCR, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.title.set_text('C')
ax1.imshow(regCresult, cmap='Blues')
##ax1.imshow(R, cmap='Blues')
ax2.title.set_text('R')
ax2.imshow(altR, cmap='Reds')
ax3.title.set_text('C+R')
ax3.imshow(CR, cmap='Purples')
figCR.show()
name = 'C+R_lambda=' + str(lam) + '_alpha=' + str(alpha)+'_halfv1'
figCR.savefig(str(imagepath)+newname+'.png')
Array = pd.DataFrame(CR)
Array.to_csv(newpath+newname+'.csv', index=False)
Array2 = pd.DataFrame(bandparameters[0, :])
Array2 = Array2.drop([20,21])
Array2.to_csv(newpath+newxname+'.csv', index=False)
