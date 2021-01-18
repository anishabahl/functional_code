import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
######INPUTS
calibrationpath = '/home/ab20/Data/Calibration_file/'
calibrationfile = 'CMV2K-SSM5x5-665_975-13.8.5.4.xml' 
newlocation = '/home/ab20/Data/Calibration_file/'
newname = 'bandresponses.csv'
parameternewname = 'idealbandparameters.csv'
######IMPORT DATA
tree = ET.parse(calibrationpath+calibrationfile)
root = tree.getroot()
wavelengths = root.find("./filter_info/calibration_info/sample_points_nm")
wcolumn = np.fromstring(wavelengths.text, sep= ',')
wcolumn = np.reshape(wcolumn, (wcolumn.shape[0], 1))
columns_tree = root.findall("./filter_info/filter_zones/filter_zone/bands/band")
parameter_tree = root.findall('./filter_info/filter_zones/filter_zone/bands/band/peaks/peak[@order="1"]')
array = np.zeros((wcolumn.shape[0], len(columns_tree)))
array2 = np.zeros((5, len(columns_tree)))
for i, columns_tree in enumerate(columns_tree):
    band = np.fromstring(columns_tree.find("response").text, sep=" , ")
    array[:, i] = band
    parameter0 = np.fromstring(parameter_tree[i].find("wavelength_nm").text, sep=" , ")
    array2[0, i] = parameter0
    parameter1 = np.fromstring(parameter_tree[i].find("fwhm_nm").text, sep=" , ")
    array2[1, i] = parameter1
    parameter2 = np.fromstring(parameter_tree[i].find("QE").text, sep=" , ")
    array2[2, i] = parameter2
    parameter3 = np.fromstring(parameter_tree[i].find("fit_error").text, sep=" , ")
    array2[3, i] = parameter3
    parameter4 = np.fromstring(parameter_tree[i].find("contribution").text, sep=" , ")
    array2[4, i] = parameter4
whole = np.hstack((wcolumn, array))
#######SAVE
Array = pd.DataFrame(whole)
Array.to_csv(newlocation+newname, index=False)
Array2 = pd.DataFrame(array2)
Array2.to_csv(newlocation+parameternewname, index=False)
