import cv2
import rawpy as raw
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spectral
from spectral import imshow
from spectral import envi
import xml.etree.ElementTree as ET
########INPUTS
calibrationpath = '/home/ab20/Data/Calibration_file/'
calibrationfile = 'CMV2K-SSM5x5-665_975-13.8.5.4.xml' 
newlocation = '/home/ab20/Data/Calibration_file/'
newname = 'calibrationmatrix.csv'
newnamewavelengths = 'bandwavelengths.csv'
########IMPORT DATA
tree = ET.parse(calibrationpath+calibrationfile)
root = tree.getroot()
rows_tree = root.findall("./system_info/spectral_correction_info/correction_matrices/correction_matrix/virtual_bands/virtual_band")
##bands = [None] * len(rows_tree)
array = np.zeros((len(rows_tree), 25))
array2 = np.zeros((len(rows_tree), 1))
for i, rows_tree in enumerate(rows_tree):
##    print(i, rows_tree)
   coefficients = np.fromstring(rows_tree.find("coefficients").text, sep=" , ")
   wavelengths = np.fromstring(rows_tree.find("wavelength_nm").text, sep=" , ")
##   bands[i] = coefficients
   array[i] = coefficients
   array2[i] = wavelengths
########WRITE TO CSV
Array = pd.DataFrame(array)
Array2 = pd.DataFrame(array2)
Array.to_csv(newlocation+newname, index=False)
Array2.to_csv(newlocation+newnamewavelengths, index=False)
