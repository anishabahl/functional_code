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
bands = 'bandresponses.csv'
path = '/home/ab20/Data/Calibration_file/'
band_parameters = 'idealbandparameters.csv'
secondpeak_parameters = 'idealbandparameterssecondpeak.csv'
lightspectrum = 'IR light with LP UV filter thorugh exoscope and adaptor.txt'
filterdata = 'AsahiSpectra_XVL0670.csv'
imecfilter_1 = 'longpass_filter.csv'
imecfilter_2 = 'shortpass_filter.csv'
spectrometer = 'spydercheckr_spectra_spectrometer'
datapath = '/home/ab20/Data/System_Paper/Photonfocus/halfv1demosaiced/' #must have full hypercube data
filetype = '.img'
newname = 'Pichettematrix'
newxname = 'fullxaxis'
newpath = '/home/ab20/Data/Calibration_file/'
imagepath = '/home/ab20/Data/Pichette/bideal/Lorentzian/secondary_imecfilter/' #path to save new images
prereorder = 'OFF' #should be off for PF at the moment
idealmethod = 'Lorentzian' #Lorentzian or Gaussian or ApproxLorentzian
compareGauss = 'OFF'
compareApproxLorentz = 'OFF'
lam = 0.005 #parameter for regularisation
optCmethod = 'lsmr' #options are scikit 'Ridge' and 'lsmr'
optRmethod = 'SLSQP' #options are scipy minimise 'SLSQP' and 'least_sq'
alpha = 0.05 #bounds values in R matrix
filterbandresponses = 'ON' #uses imec filters on measured band responses
secondarypeaks = 'ON' #fits and plots secondary peaks
errmethods = ['MSEL', 'RMSEL', 'MSEM', 'RMSEM', 'MSSM', 'RMSSM'] #number of methods of error calculation
######DEFINE FUNCTIONS
def approx_gaus(x, QE, fwhm, centre):
    sigma = 0.5*fwhm/np.sqrt(np.log(2))
    return QE*np.exp(-np.square((x-centre)/sigma))
##def findC(C, A, Aideal):
##    C = np.reshape(C, (bandresponses.shape[1]-1, bandresponses.shape[1]-1))
##    D = Aideal - np.dot(A,C)
##    return D.flatten()
##    return np.sqrt(np.multiply(D, D))
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
def approx_lorentz(x, QE, fwhm, centre):
    x2 = scipy.constants.c/x
    fwhm2 = (-scipy.constants.c/centre**2)*fwhm
    centre2 = scipy.constants.c/centre
    y = fwhm2**2/(fwhm2**2 + 4*(x2-centre2)**2)
    y2 = QE*y
    return y2
######IMPORT DATA
bandresponses = genfromtxt(path+bands, delimiter=',')
bandresponses = np.delete(bandresponses, 0, 0)
wavelengths = bandresponses[:, 0]
bandparameters = genfromtxt(path+band_parameters, delimiter=',')
bandparameters = np.delete(bandparameters,0, 0)
bandparameters2 = genfromtxt(path+secondpeak_parameters, delimiter=',')
bandparameters2 = np.delete(bandparameters2,0,0)
lightsource = genfromtxt(path+lightspectrum, skip_header = 14, delimiter = '\t')
filterspectrum = genfromtxt(path+filterdata, delimiter=',')
filterspectrum = np.delete(filterspectrum,0,0)
imecfilter1 = genfromtxt(path+imecfilter_1, delimiter=',')
imecfilter1 = np.delete(imecfilter1, 0, 0)
imecfilter2 = genfromtxt(path+imecfilter_2, delimiter=',')
imecfilter2 = np.delete(imecfilter2, 0,0)
######Light source spectrum = spectrum x filterspectrum
#sort so both ascending wavelengths
sortedlight = lightsource[np.argsort(lightsource[:,0])]
sortedfilter = filterspectrum[np.argsort(filterspectrum[:,0])]
sortedimecfilter1 = imecfilter1[np.argsort(imecfilter1[:, 0])]
sortedimecfilter2 = imecfilter2[np.argsort(imecfilter2[:, 0])]
#normalise intensities
sortedlight[:,1] = (sortedlight[:,1]-sortedlight[:,1].min())/(sortedlight[:,1].max()-sortedlight[:,1].min())
plt.figure("light spectrum")
plt.plot(sortedlight[:, 0], sortedlight[:, 1], label='original light spectrum')
plt.show(block=False)
sortedfilter[:,1] = (sortedfilter[:,1]-sortedfilter[:,1].min())/(sortedfilter[:,1].max()-sortedfilter[:,1].min())
plt.figure("filter spectrum")
sortedimecfilter1[:,1] = (sortedimecfilter1[:,1]-sortedimecfilter1[:,1].min())/(sortedimecfilter1[:,1].max()-sortedimecfilter1[:,1].min())
sortedimecfilter2[:,1] = (sortedimecfilter2[:,1]-sortedimecfilter2[:,1].min())/(sortedimecfilter2[:,1].max()-sortedimecfilter2[:,1].min())
#plt.plot(sortedfilter[:, 0], sortedfilter[:, 1], label='original filter spectrum')
#expand filter array to be same shape as light source spectrum 
interpolatedfilter = np.zeros(sortedlight.shape)
interpolatedfilter[:,0] = sortedlight[:, 0]
begin = np.array([0, 0])
end = np.array([1200, 1])
sortedfilter = np.vstack((begin, sortedfilter, end))
f = interp1d(sortedfilter[:, 0], sortedfilter[:, 1])
interpolatedfilter[:, 1] = f(interpolatedfilter[:, 0])
plt.figure("filter spectrum")
plt.plot(interpolatedfilter[:, 0], interpolatedfilter[:, 1], label='filter spectrum')
plt.legend(loc='best')
plt.show(block=False)
plt.savefig(imagepath+'Filter_Spectrum')
#expand filter to be same shape as band response spectrum
interpolatedimecfilter1 = np.zeros((bandresponses.shape[0], 2))
interpolatedimecfilter1[:,0] = bandresponses[:, 0]
b1 = np.array([399.998, 0])
e1 = np.array([1000.0, 1])
sortedimecfilter1 = np.vstack((b1, sortedimecfilter1, e1))
imecf1 = interp1d(sortedimecfilter1[:, 0], sortedimecfilter1[:, 1])
interpolatedimecfilter1[:, 1] = imecf1(interpolatedimecfilter1[:, 0])

interpolatedimecfilter2 = np.zeros((bandresponses.shape[0], 2))
interpolatedimecfilter2[:,0] = bandresponses[:, 0]
b2 = np.array([399.998, 0])
e2 = np.array([1000.0, 1])
sortedimecfilter2 = np.vstack((b2, sortedimecfilter2, e2))
imecf2 = interp1d(sortedimecfilter2[:, 0], sortedimecfilter2[:, 1])
interpolatedimecfilter2[:, 1] = imecf2(interpolatedimecfilter2[:, 0])
plt.figure("imec filter spectrum")
plt.plot(interpolatedimecfilter1[:, 0], interpolatedimecfilter1[:, 1], label='imec longpass filter spectrum')
plt.plot(interpolatedimecfilter2[:, 0], interpolatedimecfilter2[:, 1], label='imec shortpass filter spectrum')
plt.legend(loc='best')
plt.show(block=False)
plt.savefig(imagepath+'Imec_Filter_Spectrum')

#multiply light source and filter spectra
sortedfilter = np.delete(sortedfilter, 0, 0)
sortedfilter = np.delete(sortedfilter, -1, 0)
correctedlight = np.zeros(sortedlight.shape)
correctedlight[:,0] = sortedlight[:, 0]
correctedlight[:, 1] = np.multiply(sortedlight[:, 1], interpolatedfilter[:,1])
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
######Multiplying bandresponses by imec filters
if filterbandresponses == 'ON':
    for i in range(bandresponses.shape[1]-1):
        bandresponses[:, i+1] = np.multiply(bandresponses[:, i+1], interpolatedimecfilter1[:, 1])
        bandresponses[:, i+1] = np.multiply(bandresponses[:, i+1], interpolatedimecfilter2[:, 1])
######GENERATING IDEALS
optictransideal = np.ones((bandresponses.shape[0], 1)) #all 1 as ideally no loss of intensity
lightideal = np.copy(wavelengths)
lightideal = np.where(lightideal>1000, 0, lightideal)
lightideal = np.where(lightideal<670, 0, lightideal)
lightideal = np.where(lightideal!=0, 1, lightideal) #ideally box function where changes at 670 and 1000nm
if compareGauss == 'ON':
    plt.figure('Gaussian band responses')
plt.figure('Lorentzian band responses')
if compareApproxLorentz == 'ON':
    plt.figure('Approximate Lorentzian band responses')
##Approximate Gaussians for band responses
bandideal = np.zeros((bandresponses.shape[0], bandresponses.shape[1]-1))
if compareGauss == 'ON':
    Gaussbandideal = np.zeros((bandresponses.shape[0], bandresponses.shape[1]-1))
Lorentzbandideal = np.zeros((bandresponses.shape[0], bandresponses.shape[1]-1))
if compareApproxLorentz == 'ON':
    ApproxLorentzbandideal = np.zeros((bandresponses.shape[0], bandresponses.shape[1]-1))
if secondarypeaks == 'ON':
    if compareGauss == 'ON':
        Gaussbandideal2 = np.zeros((bandresponses.shape[0], bandresponses.shape[1]-1))
    Lorentzbandideal2 = np.zeros((bandresponses.shape[0], bandresponses.shape[1]-1))
    if compareApproxLorentz == 'ON':
        ApproxLorentzbandideal2 = np.zeros((bandresponses.shape[0], bandresponses.shape[1]-1))
cmap = plt.cm.Greys
colors = [cmap(i) for i in range(cmap.N)]
cmap2 = plt.cm.Greens
colors2 = [cmap2(i) for i in range(cmap2.N)]
cmap3 = plt.cm.Blues
colors3 = [cmap3(i) for i in range(cmap3.N)]
cmap4 = plt.cm.Reds
colors4 = [cmap4(i) for i in range(cmap4.N)]
if compareGauss or compareApproxLorentz == 'ON':
    methods = 1
if compareGauss and compareApproxLorentz == 'ON':
    methods = 2
Paramsarray = np.zeros((bandresponses.shape[1]-1,5+methods*(len(errmethods)+1)))
if secondarypeaks == 'ON':
    Paramsarray2 = np.zeros((bandresponses.shape[1]-1,5+methods*(len(errmethods)+1)))
for i in range(bandresponses.shape[1]-1):
    bandideal[:, i] = lorentz(x=wavelengths, QE=bandparameters[2, i], fwhm=bandparameters[1, i], centre=bandparameters[0, i])
    if compareGauss == 'ON':
        Gaussbandideal[:, i] = approx_gaus(x=wavelengths, QE=bandparameters[2, i], fwhm=bandparameters[1, i], centre=bandparameters[0, i])
        Gausscontrib = np.trapz(Gaussbandideal[:, i], wavelengths) / np.trapz(bandresponses[:, i + 1], wavelengths)
        GaussRSSerr = np.sum((bandresponses[:, i + 1] - Gaussbandideal[:, i]) ** 2)
        GaussMSEerr = GaussRSSerr/(bandresponses.shape[0])
        GaussRMSEerr = math.sqrt(GaussMSEerr)
        GaussAAE = np.sum(abs(bandresponses[:, i + 1] - Gaussbandideal[:, i]))
        GaussMAE = np.sum(abs(bandresponses[:, i + 1] - Gaussbandideal[:, i]))/(bandresponses.shape[0])
        GaussmaxAE = np.amax(abs(bandresponses[:, i + 1] - Gaussbandideal[:, i]))
    Lorentzbandideal[:, i] = lorentz(x=wavelengths, QE=bandparameters[2, i], fwhm=bandparameters[1, i], centre=bandparameters[0, i])
    if compareApproxLorentz == 'ON':
        ApproxLorentzbandideal[:, i] = approx_lorentz(x=wavelengths, QE=bandparameters[2, i], fwhm=bandparameters[1, i], centre=bandparameters[0, i])
        ApproxLorentzcontrib = np.trapz(ApproxLorentzbandideal[:, i], wavelengths) / np.trapz(bandresponses[:, i + 1],
                                                                                              wavelengths)
        ApproxLorentzRSSerr = np.sum((bandresponses[:, i + 1] - ApproxLorentzbandideal[:, i]) ** 2)
        ApproxLorentzMSEerr = ApproxLorentzRSSerr / (bandresponses.shape[0])
        ApproxLorentzRMSEerr = math.sqrt(ApproxLorentzMSEerr)
        ApproxLorentzAAE = np.sum(abs(bandresponses[:, i + 1] - ApproxLorentzbandideal[:, i]))
        ApproxLorentzMAE = np.sum(abs(bandresponses[:, i + 1] - ApproxLorentzbandideal[:, i])) / (bandresponses.shape[0])
        ApproxLorentzmaxAE = np.amax(abs(bandresponses[:, i + 1] - ApproxLorentzbandideal[:, i]))
    #Lorentzcontrib = np.trapz(Lorentzbandideal[:, i], wavelengths)/np.trapz(bandresponses[:, i+1], wavelengths)
    Lorentzcontrib = Lorentzbandideal.sum(axis=0)[i]/ bandresponses.sum(axis=0)[i+1]
    LorentzRSSerr = np.sum((bandresponses[:, i+1] - Lorentzbandideal[:, i])**2)
    #LorentzMSEerr = LorentzRSSerr / (bandresponses.shape[0])
    LorentzMSEL = LorentzRSSerr / np.trapz(Lorentzbandideal[:, i], wavelengths)
    LorentzRMSEL = math.sqrt(LorentzMSEL)
    LorentzMSEM = LorentzRSSerr / np.trapz(bandresponses[:, i+1], wavelengths)
    LorentzRMSEM = math.sqrt(LorentzMSEM)
    LorentzMSSM = LorentzRSSerr / bandresponses.sum(axis=0)[i+1]
    LorentzRMSSM = math.sqrt(LorentzMSSM)
    #LorentzAAE = np.sum(abs(bandresponses[:, i + 1] - Lorentzbandideal[:, i]))
    #LorentzMAE = np.sum(abs(bandresponses[:, i + 1] - Lorentzbandideal[:, i])) / (bandresponses.shape[0])
    #LorentzmaxAE = np.amax(abs(bandresponses[:, i + 1] - Lorentzbandideal[:, i]))
    if secondarypeaks == 'ON':
        if compareGauss == 'ON':
            Gaussbandideal2[:, i] = approx_gaus(x=wavelengths, QE=bandparameters2[2, i], fwhm=bandparameters2[1, i],
                                            centre=bandparameters2[0, i])
            Gausscontrib2 = np.trapz(Gaussbandideal2[:, i], wavelengths) / np.trapz(bandresponses[:, i + 1],
                                                                                    wavelengths)
            GaussRSSerr2 = np.sum((bandresponses[:, i + 1] - Gaussbandideal2[:, i]) ** 2)
            GaussMSEerr2 = GaussRSSerr2 / (bandresponses.shape[0])
            GaussRMSEerr2 = math.sqrt(GaussMSEerr2)
            GaussMAE2 = np.sum(abs(bandresponses[:, i + 1] - Gaussbandideal2[:, i])) / (bandresponses.shape[0])
            GaussAAE2 = np.sum(abs(bandresponses[:, i + 1] - Gaussbandideal2[:, i]))
            GaussmaxAE2 = np.amax(abs(bandresponses[:, i + 1] - Gaussbandideal2[:, i]))
        Lorentzbandideal2[:, i] = lorentz(x=wavelengths, QE=bandparameters2[2, i], fwhm=bandparameters2[1, i],
                                          centre=bandparameters2[0, i])
        if compareApproxLorentz == 'ON':
            ApproxLorentzbandideal2[:, i] = approx_lorentz(x=wavelengths, QE=bandparameters2[2, i],
                                                       fwhm=bandparameters2[1, i],
                                                       centre=bandparameters2[0, i])
            ApproxLorentzcontrib2 = np.trapz(ApproxLorentzbandideal2[:, i], wavelengths) / np.trapz(
                bandresponses[:, i + 1], wavelengths)
            ApproxLorentzRSSerr2 = np.sum((bandresponses[:, i + 1] - ApproxLorentzbandideal2[:, i]) ** 2)
            ApproxLorentzMSEerr2 = ApproxLorentzRSSerr2 / (bandresponses.shape[0])
            ApproxLorentzRMSEerr2 = math.sqrt(ApproxLorentzMSEerr2)
            ApproxLorentzMAE2 = np.sum(abs(bandresponses[:, i + 1] - ApproxLorentzbandideal2[:, i])) / (
            bandresponses.shape[0])
            ApproxLorentzAAE2 = np.sum(abs(bandresponses[:, i + 1] - ApproxLorentzbandideal2[:, i]))
            ApproxLorentzmaxAE2 = np.amax(abs(bandresponses[:, i + 1] - ApproxLorentzbandideal2[:, i]))
        #Lorentzcontrib2 = np.trapz(Lorentzbandideal2[:, i], wavelengths) / np.trapz(bandresponses[:, i + 1],wavelengths)
        Lorentzcontrib2 = Lorentzbandideal2.sum(axis=0)[i] / bandresponses.sum(axis=0)[i + 1]
        LorentzRSSerr2 = np.sum((bandresponses[:, i + 1] - Lorentzbandideal2[:, i]) ** 2)
        LorentzMSEL2 = LorentzRSSerr2 / np.trapz(Lorentzbandideal2[:, i], wavelengths)
        LorentzRMSEL2 = math.sqrt(LorentzMSEL2)
        LorentzMSEM2 = LorentzRSSerr2 / np.trapz(bandresponses[:, i+1], wavelengths)
        LorentzRMSEM2 = math.sqrt(LorentzMSEM2)
        LorentzMSSM2 = LorentzRSSerr2 / bandresponses.sum(axis=0)[i + 1]
        LorentzRMSSM2 = math.sqrt(LorentzMSSM2)
        #LorentzMSEerr2 = LorentzRSSerr2 / (bandresponses.shape[0])
        #LorentzMSEerr2 = LorentzRSSerr2 / np.trapz(Lorentzbandideal2[:, i], wavelengths)
        #LorentzRMSEerr2 = math.sqrt(LorentzMSEerr2)
        #LorentzMAE2 = np.sum(abs(bandresponses[:, i + 1] - Lorentzbandideal2[:, i])) / (bandresponses.shape[0])
        #LorentzAAE2 = np.sum(abs(bandresponses[:, i + 1] - Lorentzbandideal2[:, i]))
        #LorentzmaxAE2 = np.amax(abs(bandresponses[:, i + 1] - Lorentzbandideal2[:, i]))


    #Paramsarray[i, 0] = bandparameters[0, i]
    #Paramsarray[i, 1] = bandparameters[1, i]
    #Paramsarray[i, 2] = bandparameters[2, i]
    #Paramsarray[i, 3] = bandparameters[3, i]
    #Paramsarray[i, 4] = bandparameters[4, i]
    Paramsarray[i, 0:5] = bandparameters[0:5, i]
    Paramsarray[i, 5] = Lorentzcontrib
    Paramsarray[i, 5+errmethods.index('MSEL')+1] = LorentzMSEL
    Paramsarray[i, 5 + errmethods.index('RMSEL') + 1] = LorentzRMSEL
    Paramsarray[i, 5 + errmethods.index('MSEM') + 1] = LorentzMSEM
    Paramsarray[i, 5 + errmethods.index('RMSEM') + 1] = LorentzRMSEM
    Paramsarray[i, 5 + errmethods.index('MSSM') + 1] = LorentzMSSM
    Paramsarray[i, 5 + errmethods.index('RMSSM') + 1] = LorentzRMSSM
    #Paramsarray[i, 5 + errmethods.index('MAE') + 1] = LorentzMAE
    #Paramsarray[i, 5 + errmethods.index('Max AE') + 1] = LorentzmaxAE
    if compareGauss == 'ON' and compareApproxLorentz == 'ON':
        Paramsarray[i, 6+len(errmethods)] = Gausscontrib
        Paramsarray[i, 6+len(errmethods)+errmethods.index('RSS')+1] = GaussRSSerr
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('MSE') + 1] = GaussMSEerr
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('RMSE') + 1] = GaussRMSEerr
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('AAE') + 1] = GaussAAE
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('MAE') + 1] = GaussMAE
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('Max AE') + 1] = GaussmaxAE
        Paramsarray[i, 7+len(errmethods)*2] = ApproxLorentzcontrib
        Paramsarray[i, 7+len(errmethods)*2+errmethods.index('RSS')+1] = ApproxLorentzRSSerr
        Paramsarray[i, 7 + len(errmethods) * 2 + errmethods.index('MSE') + 1] = ApproxLorentzMSEerr
        Paramsarray[i, 7 + len(errmethods) * 2 + errmethods.index('RMSE') + 1] = ApproxLorentzRMSEerr
        Paramsarray[i, 7 + len(errmethods) * 2 + errmethods.index('AAE') + 1] = ApproxLorentzAAE
        Paramsarray[i, 7 + len(errmethods) * 2 + errmethods.index('MAE') + 1] = ApproxLorentzMAE
        Paramsarray[i, 7 + len(errmethods) * 2 + errmethods.index('Max AE') + 1] = ApproxLorentzmaxAE
    elif compareGauss == 'ON':
        Paramsarray[i, 6+len(errmethods)] = Gausscontrib
        Paramsarray[i, 6+len(errmethods)+errmethods.index('RSS')+1] = GaussRSSerr
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('MSE') + 1] = GaussMSEerr
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('RMSE') + 1] = GaussRMSEerr
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('AAE') + 1] = GaussAAE
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('MAE') + 1] = GaussMAE
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('Max AE') + 1] = GaussmaxAE
    elif compareApproxLorentz == 'ON':
        Paramsarray[i, 6+len(errmethods)] = ApproxLorentzcontrib
        Paramsarray[i, 6+len(errmethods)+errmethods.index('RSS')+1] = ApproxLorentzRSSerr
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('MSE') + 1] = ApproxLorentzMSEerr
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('RMSE') + 1] = ApproxLorentzRMSEerr
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('AAE') + 1] = ApproxLorentzAAE
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('MAE') + 1] = ApproxLorentzMAE
        Paramsarray[i, 6 + len(errmethods) + errmethods.index('Max AE') + 1] = ApproxLorentzmaxAE

    if secondarypeaks == 'ON':
        Paramsarray2[i, 0:5] = bandparameters2[0:5, i]
        Paramsarray2[i, 5] = Lorentzcontrib2
        Paramsarray2[i, 5 + errmethods.index('MSEL')+1] = LorentzMSEL2
        Paramsarray2[i, 5 + errmethods.index('RMSEL') + 1] = LorentzRMSEL2
        Paramsarray2[i, 5 + errmethods.index('MSEM') + 1] = LorentzMSEM2
        Paramsarray2[i, 5 + errmethods.index('RMSEM') + 1] = LorentzRMSEM2
        Paramsarray2[i, 5 + errmethods.index('MSSM') + 1] = LorentzMSSM2
        Paramsarray2[i, 5 + errmethods.index('RMSSM') + 1] = LorentzRMSSM2
        #Paramsarray2[i, 5 + errmethods.index('MAE') + 1] = LorentzMAE2
        #Paramsarray2[i, 5 + errmethods.index('Max AE') + 1] = LorentzmaxAE2
        if compareGauss == 'ON' and compareApproxLorentz == 'ON':
            Paramsarray2[i, 6 + len(errmethods)] = Gausscontrib2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('RSS')+1] = GaussRSSerr2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('MSE') + 1] = GaussMSEerr2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('RMSE') + 1] = GaussRMSEerr2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('AAE') + 1] = GaussAAE2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('MAE') + 1] = GaussMAE2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('Max AE') + 1] = GaussmaxAE2
            Paramsarray2[i, 7 + len(errmethods) * 2] = ApproxLorentzcontrib2
            Paramsarray2[i, 7 + len(errmethods) * 2 + errmethods.index('RSS')+1] = ApproxLorentzRSSerr2
            Paramsarray2[i, 7 + len(errmethods) * 2 + errmethods.index('MSE') + 1] = ApproxLorentzMSEerr2
            Paramsarray2[i, 7 + len(errmethods) * 2 + errmethods.index('RMSE') + 1] = ApproxLorentzRMSEerr2
            Paramsarray2[i, 7 + len(errmethods) * 2 + errmethods.index('AAE') + 1] = ApproxLorentzAAE2
            Paramsarray2[i, 7 + len(errmethods) * 2 + errmethods.index('MAE') + 1] = ApproxLorentzMAE2
            Paramsarray2[i, 7 + len(errmethods) * 2 + errmethods.index('Max AE') + 1] = ApproxLorentzmaxAE2
        elif compareGauss == 'ON':
            Paramsarray2[i, 6 + len(errmethods)] = Gausscontrib2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('RSS')+1] = GaussRSSerr2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('MSE') + 1] = GaussMSEerr2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('RMSE') + 1] = GaussRMSEerr2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('MAE') + 1] = GaussMAE2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('AAE') + 1] = GaussAAE2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('Max AE') + 1] = GaussmaxAE2
            Gausssum = Gaussbandideal[:, i] + Gaussbandideal2[:, i]
        elif compareApproxLorentz == 'ON':
            Paramsarray2[i, 6 + len(errmethods)] = ApproxLorentzcontrib2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('RSS')+1] = ApproxLorentzRSSerr2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('MSE') + 1] = ApproxLorentzMSEerr2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('RMSE') + 1] = ApproxLorentzRMSEerr2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('MAE') + 1] = ApproxLorentzMAE2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('AAE') + 1] = ApproxLorentzAAE2
            Paramsarray2[i, 6 + len(errmethods) + errmethods.index('Max AE') + 1] = ApproxLorentzmaxAE2
            ApproxLorentzsum = ApproxLorentzbandideal[:, i] + ApproxLorentzbandideal2[:, i]


        #Paramsarray2[i, 0:5] = bandparameters2[0:5, i]
        #if compareGauss == 'ON':
            #Paramsarray2[i, 6] = Gausscontrib2
            #Paramsarray2[i, 5] = Gausserr2
        #Paramsarray2[i, 8] = Lorentzcontrib2
        #Paramsarray2[i, 7] = Lorentzerr2
        #if compareApproxLorentz == 'ON':
            #Paramsarray2[i, 10] = ApproxLorentzcontrib2
            #Paramsarray2[i, 9] = ApproxLorentzerr2


        Lorentzsum = Lorentzbandideal[:, i] + Lorentzbandideal2[:, i]

    if compareGauss == 'ON':
        plt.figure('Gaussian band responses')
        plt.plot(wavelengths, bandresponses[:, i+1], color=colors[-i*7],  label='real')
        if secondarypeaks == 'ON':
            plt.plot(wavelengths, Gausssum, color=colors2[-i * 7], label="Gaussian ideal")
        else:
            plt.plot(wavelengths, Gaussbandideal[:, i], color = colors2[-i*7], label ="Gaussian ideal")
        if i ==0:
            plt.legend(loc='best')
    #plt.show(block=False)
        plt.savefig(imagepath+'Gaussian_band_responses')
    plt.figure('Lorentzian band responses')
    plt.plot(wavelengths, bandresponses[:, i + 1], color=colors[-i * 7], label='real')
    if secondarypeaks == 'ON':
        plt.plot(wavelengths, Lorentzsum, color=colors4[-i * 7], label="Lorentzian ideal")
    else:
        plt.plot(wavelengths, Lorentzbandideal[:, i], color=colors4[-i * 7], label="Lorentzian ideal")
    if i ==0:
        plt.legend(loc='best')
    #plt.show(block=False)
    plt.savefig(imagepath+'Lorentzian_band_responses')
    if compareApproxLorentz == 'ON':
        plt.figure('Approximate Lorentzian band responses')
        plt.plot(wavelengths, bandresponses[:, i + 1], color=colors[-i * 7], label='real')
        if secondarypeaks == 'ON':
            plt.plot(wavelengths, ApproxLorentzsum, color=colors3[-i * 7], label="Approximate Lorentzian ideal")
        else:
            plt.plot(wavelengths, ApproxLorentzbandideal[:, i], color=colors3[-i * 7], label="Approximate Lorentzian ideal")
    #plt.plot(wavelengths, ApproxLorentzbandideal2[:, i], color=colors4[-i * 7])
    #plt.plot(wavelengths, ApproxLorentzsum, color=colors4[-i * 7], label="Approximate Lorentzian ideal")
        if i ==0:
            plt.legend(loc='best')
    #plt.show(block=False)
        plt.savefig(imagepath+'Approximate_Lorentzian_band_responses')
    plt.figure(str(i+1)+'th_band_response')
    plt.plot(wavelengths, bandresponses[:, i + 1], color='black', label='real')
    if compareGauss == 'ON':
        plt.plot(wavelengths, Gaussbandideal[:, i], color='green', label="Gaussian ideal")
    plt.plot(wavelengths, Lorentzbandideal[:, i], color='maroon', label="Lorentzian ideal")
    if compareApproxLorentz == 'ON':
        plt.plot(wavelengths, ApproxLorentzbandideal[:, i], color='blue', label="Approximate Lorentzian ideal")
    if secondarypeaks == 'ON':
        if compareGauss == 'ON':
            plt.plot(wavelengths, Gaussbandideal2[:, i], color='palegreen', label='Gaussian ideal secondary peak')
            plt.plot(wavelengths, Gausssum, color='mediumseagreen', label='Gaussian ideal sum')
        if compareApproxLorentz == 'ON':
            plt.plot(wavelengths, ApproxLorentzbandideal2[:, i], color='lightsteelblue',
                     label='Approx Lorentzian ideal secondary peak')
            plt.plot(wavelengths, ApproxLorentzsum, color='royalblue', label='Approx Lorentzian ideal sum')
        plt.plot(wavelengths, Lorentzbandideal2[:, i], color='salmon', label='Lorentzian ideal secondary peak')
        plt.plot(wavelengths, Lorentzsum, color='red', label='Lorentzian ideal sum')

    plt.legend(loc='best')
    #plt.show(block=False)
    plt.savefig(imagepath + str(i+1)+'th_band_response')
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
Aideal = np.delete(Aideal, 20, 1)
Aideal = np.delete(Aideal, 21, 1)
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
figCR.savefig(str(imagepath)+name+'.png')
Array = pd.DataFrame(CR)
Array.to_csv(newpath+name+'.csv', index=False)
Array2 = pd.DataFrame(bandparameters[0, :])
Array2 = Array2.drop([20,21])
Array2.to_csv(newpath+newxname+'.csv', index=False)
ParamsArray = pd.DataFrame(Paramsarray)
columns = ['central wavelength', 'FWHM', 'QE', 'imec fit error', 'imec contribution', 'Lorentzian contribution']
for i in range(len(errmethods)):
    columns.append('Lorentzian '+errmethods[i])
if compareGauss == 'ON' and compareApproxLorentz == 'ON':
    columns.append('Gaussian contribution')
    for i in range(len(errmethods)):
        columns.append('Gaussian '+errmethods[i])
    columns.append('Approx Lorentz contribution')
    for i in range(len(errmethods)):
        columns.append('Approx Lorentz ' + errmethods[i])
elif compareGauss == 'ON':
    columns.append('Gaussian contribution')
    for i in range(len(errmethods)):
        columns.append('Gaussian '+errmethods[0])
elif compareApproxLorentz == 'ON':
    columns.append('Approx Lorentz contribution')
    for i in range(len(errmethods)):
        columns.append('Approx Lorentz ' + errmethods[0])
ParamsArray.columns = columns
#ParamsArray.columns = ['central wavelength', 'FWHM', 'QE', 'imec fit error', 'imec contribution', 'Gaussian fit error', 'Gaussian contribution', 'Lorentzian fit error', 'Lorentzian contribution', 'Approx Lorentzian fit error', 'Approx Lorentzian contribution']
ParamsArray.to_csv(imagepath+'Ideal_band_responses.csv', index=False)
if secondarypeaks == 'ON':
    ParamsArray2 = pd.DataFrame(Paramsarray2)
    columns2 = ['central wavelength', 'FWHM', 'QE', 'imec fit error', 'imec contribution',
                           'Lorentzian contribution']
    for i in range(len(errmethods)):
        columns2.append('Lorentzian ' + errmethods[i])
    if compareGauss == 'ON' and compareApproxLorentz == 'ON':
        columns2.append('Gaussian contribution')
        for i in range(len(errmethods)):
            columns2.append('Gaussian ' + errmethods[i])
        columns2.append('Approx Lorentz contribution')
        for i in range(len(errmethods)):
            columns2.append('Approx Lorentz ' + errmethods[i])
    elif compareGauss == 'ON':
        columns2.append('Gaussian contribution')
        for i in range(len(errmethods)):
            columns2.append('Gaussian ' + errmethods[i])
    elif compareApproxLorentz == 'ON':
        columns2.append('Approx Lorentz contribution')
        for i in range(len(errmethods)):
            columns2.append('Approx Lorentz ' + errmethods[i])
    ParamsArray2.columns = columns2
    #ParamsArray2.columns = ['central wavelength', 'FWHM', 'QE', 'imec fit error', 'imec contribution', 'Gaussian fit error', 'Gaussian contribution', 'Lorentzian fit error', 'Lorentzian contribution', 'Approx Lorentzian fit error', 'Approx Lorentzian contribution']
    ParamsArray2.to_csv(imagepath+'Ideal_band_responses_second_peak.csv', index=False)
