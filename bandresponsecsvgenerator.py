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
parameter2name = 'idealbandparameterssecondpeak.csv'
filter1name = 'longpass_filter.csv'
filter2name = 'shortpass_filter.csv'
######IMPORT DATA
tree = ET.parse(calibrationpath+calibrationfile)
root = tree.getroot()
wavelengths = root.find("./filter_info/calibration_info/sample_points_nm")
wcolumn = np.fromstring(wavelengths.text, sep= ',')
wcolumn = np.reshape(wcolumn, (wcolumn.shape[0], 1))
columns_tree = root.findall("./filter_info/filter_zones/filter_zone/bands/band")
parameter_tree = root.findall('./filter_info/filter_zones/filter_zone/bands/band/peaks/peak[@order="1"]')
parameter_tree2 = root.findall('./filter_info/filter_zones/filter_zone/bands/band/peaks/peak[@order="2"]')
optical_tree1 = root.findall('./system_info/optical_components/optical_component')
array = np.zeros((wcolumn.shape[0], len(columns_tree)))
array2 = np.zeros((5, len(columns_tree)))
array3 = np.zeros((5, len(columns_tree)))
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
    parameter02 = np.fromstring(parameter_tree2[i].find("wavelength_nm").text, sep=" , ")
    array3[0, i] = parameter02
    parameter12 = np.fromstring(parameter_tree2[i].find("fwhm_nm").text, sep=" , ")
    array3[1, i] = parameter12
    parameter22 = np.fromstring(parameter_tree2[i].find("QE").text, sep=" , ")
    array3[2, i] = parameter22
    parameter32 = np.fromstring(parameter_tree2[i].find("fit_error").text, sep=" , ")
    array3[3, i] = parameter32
    parameter42 = np.fromstring(parameter_tree2[i].find("contribution").text, sep=" , ")
    array3[4, i] = parameter42

wavelengthfilter1 = np.fromstring(optical_tree1[0].find('sample_points_nm').text, sep=' , ')
transmissionfilter1 = np.fromstring(optical_tree1[0].find('response').text, sep=' , ')
array4 = np.zeros((len(wavelengthfilter1), 2))
array4[:, 0] = wavelengthfilter1
array4[:, 1] = transmissionfilter1

wavelengthfilter2 = np.fromstring(optical_tree1[1].find('sample_points_nm[@nr_elements="601"]').text, sep=' , ')
transmissionfilter2 = np.fromstring(optical_tree1[1].find('response[@nr_elements="601"]').text, sep=' , ')
array5 = np.zeros((len(wavelengthfilter2), 2))
array5[:, 0] = wavelengthfilter2
array5[:, 1] = transmissionfilter2

whole = np.hstack((wcolumn, array))
#######SAVE
Array = pd.DataFrame(whole)
Array.to_csv(newlocation+newname, index=False)
Array2 = pd.DataFrame(array2)
Array2.to_csv(newlocation+parameternewname, index=False)
Array3 = pd.DataFrame(array3)
Array3.to_csv(newlocation+parameter2name, index=False)
Array4 = pd.DataFrame(array4)
Array4.to_csv(newlocation+filter1name, index=False)
Array5 = pd.DataFrame(array5)
Array5.to_csv(newlocation+filter2name, index=False)