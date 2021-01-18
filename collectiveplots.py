import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
##########INPUTS
v1path = '/home/ab20/Data/Pichette/v1/onV2/' #change last section for which dataset plotting
v2path = '/home/ab20/Data/Pichette/v2/onV2/'
halfv1path = '/home/ab20/Data/Pichette/halfv2/onv2/'
paper = '/home/ab20/Data/Pichette/outofbox/onv2/'
nocalpath = '/home/ab20/Data/Pichette/nocalibration/onV2/'
calibrationpath = '/home/ab20/Data/Calibration_file/'
bandwavelengths = 'fullxaxis'
bandwavelengths3 = 'fullxaxis25'
paperwavelengths = 'bandwavelengths'
newlocation = '/home/ab20/Data/Pichette/collectivev2/'
spectrometer = 'spydercheckr_spectra_spectrometer'
##########IMPORT
v1data = genfromtxt(v1path + 'data.csv', delimiter = ',')
v1data = np.delete(v1data,0,0)
v1data = np.delete(v1data,0,1)
v2data = genfromtxt(v2path + 'data.csv', delimiter = ',')
v2data = np.delete(v2data,0,0)
v2data = np.delete(v2data,0,1)
halfv1data = genfromtxt(halfv1path + 'data.csv', delimiter = ',')
halfv1data = np.delete(halfv1data,0,0)
halfv1data = np.delete(halfv1data,0,1)
paperdata = genfromtxt(paper + 'data.csv', delimiter = ',')
paperdata = np.delete(paperdata,0,0)
paperdata = np.delete(paperdata,0,1)
x = genfromtxt(calibrationpath+bandwavelengths+'.csv', delimiter=',')
x = np.delete(x, 0,0)
x = x.reshape(x.shape[0],1)
x2 = genfromtxt(calibrationpath+paperwavelengths+'.csv', delimiter=',')
x2 = np.delete(x2, 0,0)
x2 = x2.reshape(x2.shape[0],1)
x3 = genfromtxt(calibrationpath+bandwavelengths3+'.csv', delimiter=',')
x3 = np.delete(x3, 0,0)
x3 = x3.reshape(x3.shape[0],1)
spectrometerdata = genfromtxt(calibrationpath+spectrometer+'.csv', delimiter=',')
spectrometerdata = np.delete(spectrometerdata, 0, 0)
nocaldata = genfromtxt(nocalpath + 'data.csv', delimiter = ',')
nocaldata = np.delete(nocaldata,0,0)
nocaldata = np.delete(nocaldata,0,1)
#########PLOT
tiles = ['1A', '2A', '3A', '4A', '5A', '6A', '1B', '2B', '3B', '4B', '5B', '6B', '1C', '2C', '3C', '4C', '5C', '6C', '1D', '2D', '3D', '4D', '5D', '6D', '1E', '2E', '3E', '4E', '5E', '6E', '1F', '2F', '3F', '4F', '5F', '6F', '1G', '2G', '3G', '4G', '5G', '6G', '1H', '2H', '3H', '4H', '5H', '6H']
tilearray = np.array([['1A', '2A', '3A', '4A', '5A', '6A'], ['1B', '2B', '3B', '4B', '5B', '6B'], ['1C', '2C', '3C', '4C', '5C', '6C'], ['1D', '2D', '3D', '4D', '5D', '6D'], ['1E', '2E', '3E', '4E', '5E', '6E'], ['1F', '2F', '3F', '4F', '5F', '6F'], ['1G', '2G', '3G', '4G', '5G', '6G'], ['1H', '2H', '3H', '4H', '5H', '6H']])
fig, axs = plt.subplots(tilearray.shape[0], tilearray.shape[1], sharex=True, sharey=True, figsize = [12, 12])
plt.subplots_adjust(hspace = 0.25)
for i in range(v1data.shape[1]):
    tile = tiles[i]
    nocalspec = np.hstack((x3, nocaldata[:, i].reshape(nocaldata[:, i].shape[0],1)))
    nocalspec = nocalspec[np.argsort(nocalspec[:, 0])]
    v1spec = np.hstack((x, v1data[:, i].reshape(v1data[:, i].shape[0],1)))
    v1spec = v1spec[np.argsort(v1spec[:,0])]
    v2spec = np.hstack((x, v2data[:, i].reshape(v2data[:, i].shape[0],1)))
    v2spec = v2spec[np.argsort(v2spec[:,0])]
    halfv1spec = np.hstack((x, halfv1data[:, i].reshape(halfv1data[:, i].shape[0],1)))
    halfv1spec = halfv1spec[np.argsort(halfv1spec[:,0])]
    paperspec = np.hstack((x2, paperdata[:, i].reshape(paperdata[:, i].shape[0],1)))
    paperspec = paperspec[np.argsort(paperspec[:,0])]
    coordindices = np.where(tilearray == tile)
    coord = list(zip(coordindices[0], coordindices[1]))[0]
    axs[coord[0], coord[1]].plot(paperspec[:, 0], paperspec[:, 1], label='using premade matrix')
    axs[coord[0], coord[1]].plot(v1spec[:, 0], v1spec[:, 1], label='fitted on v1')
    axs[coord[0], coord[1]].plot(v2spec[:, 0], v2spec[:, 1], label='fitted on v2')
    axs[coord[0], coord[1]].plot(halfv1spec[:, 0], halfv1spec[:, 1], label='fitted on half v2')
    axs[coord[0], coord[1]].plot(spectrometerdata[:, 0], spectrometerdata[:, i+1], label='spectrometer')
    axs[coord[0], coord[1]].plot(nocalspec[:, 0], nocalspec[:, 1], label='no cross-talk correction')
    axs[coord[0], coord[1]].set_title(tile, fontsize = 8)
    if i == 0:
        axs[coord[0], coord[1]].legend(loc = 'best')
fig.text(0.5, 0.04, 'wavelength/nm', ha='center', fontsize = 12)
fig.text(0.04, 0.5, 'intensity/arb', va='center', rotation='vertical', fontsize = 12)
plt.show(block=False)
plt.savefig(newlocation + 'collective spectrum')
