import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

##Set variables
camera = "4x4 vis"
calibrationpath = '/home/ab20/Data/Calibration_file/'
calibrationfile = 'CMV2K-SSM4x4-470_620-9.6.13.13.xml'
newlocation = '/home/ab20/Data/Calibration_file/'

##Defining functions
def save_band_responses(root, newlocation, newname):
    """
    Function which obtains and saves csv of all band responses
    Requires:
        root = root of tree of xml file
        newlocation = absolute path of directory to save csv in
        newname = name of new csv
    """
    #find relevant values in xml
    columns_tree = root.findall("./filter_info/filter_zones/filter_zone/bands/band")
    wavelengths = root.find("./filter_info/calibration_info/sample_points_nm")
    #extract sample points wavelength column
    wcolumn = np.fromstring(wavelengths.text, sep=',')
    wcolumn = np.reshape(wcolumn, (wcolumn.shape[0], 1))
    #initialise array
    array = np.zeros((wcolumn.shape[0], len(columns_tree)))
    #find band responses and fill in array
    for i, columns_tree in enumerate(columns_tree):
        band = np.fromstring(columns_tree.find("response").text, sep=" , ")
        array[:, i] = band
    #add wavelengths column
    whole = np.hstack((wcolumn, array))
    Array = pd.DataFrame(whole)
    #save
    Array.to_csv(newlocation + newname + ".csv", index=False)
    return whole

def save_fit_parameters(root, order, newlocation, newname):
    """
    Function to save parameters from imec fitted band responses of the peak of order given
    Requires:
        root = root of tree of xml file
        order = peak order (primary, secondary, etc)
        newlocation = Where to save csv
        newname = Name of csv
    """
    #find relevant values in xml
    columns_tree = root.findall("./filter_info/filter_zones/filter_zone/bands/band")
    parameter_tree = root.findall('./filter_info/filter_zones/filter_zone/bands/band/peaks/peak[@order='+'"'+str(order)+'"'+')]')
    array = np.zeros((5, len(columns_tree)))
    for i, columns_tree in enumerate(columns_tree):
        parameter0 = np.fromstring(parameter_tree[i].find("wavelength_nm").text, sep=" , ")
        array[0, i] = parameter0
        parameter1 = np.fromstring(parameter_tree[i].find("fwhm_nm").text, sep=" , ")
        array[1, i] = parameter1
        parameter2 = np.fromstring(parameter_tree[i].find("QE").text, sep=" , ")
        array[2, i] = parameter2
        parameter3 = np.fromstring(parameter_tree[i].find("fit_error").text, sep=" , ")
        array[3, i] = parameter3
        parameter4 = np.fromstring(parameter_tree[i].find("contribution").text, sep=" , ")
        array[4, i] = parameter4
    Array = pd.DataFrame(array)
    Array.to_csv(newlocation + newname + ".csv", index=False)
    return array

def save_wavelengths(root, order, newlocation, newname):
    """
    Saves single column of central wavelengths for each band
    Requires:
        root = root of tree of xml file
        order = primary/secondary etc ie. order of peak to save central wavelength of
        newlocation = absolute path of directory to save csv in
        newname = name of new csv
    """
    columns_tree = root.findall("./filter_info/filter_zones/filter_zone/bands/band")
    parameter_tree = root.findall('./filter_info/filter_zones/filter_zone/bands/band/peaks/peak[@order='+'"'+str(order)+'"'+']')
    array = np.zeros((len(columns_tree), 1))
    for i, columns_tree in enumerate(columns_tree):
        parameter0 = np.fromstring(parameter_tree[i].find("wavelength_nm").text, sep=" , ")
        array[i, 0] = parameter0
    Array = pd.DataFrame(array)
    Array.to_csv(newlocation + newname + ".csv", index=False)
    return array

def save_filter(root, order, newlocation, newname):
    """
    Saves spectrum of filters imec uses
    Requires:
        root = root of tree of xml file
        order = first, second, third etc
        newlocation = absolute path of directory to save csv in
        newname = name of new csv
    """
    optical_tree = root.findall('./system_info/optical_components/optical_component')
    wavelengthfilter = np.fromstring(optical_tree1[order].find('sample_points_nm').text, sep=' , ')
    transmissionfilter = np.fromstring(optical_tree1[order].find('response').text, sep=' , ')
    array = np.zeros((len(wavelengthfilter), 2))
    array[:, 0] = wavelengthfilter
    array[:, 1] = transmissionfilter
    Array = pd.DataFrame(array)
    Array.to_csv(newlocation + newname + ".csv", index=False)
    return array

##Use functions
def main_vis():
    ##Import data
    tree = ET.parse(calibrationpath + calibrationfile)
    root = tree.getroot()

    optical_tree1 = root.findall('./system_info/optical_components/optical_component')
    array3 = np.zeros((5, len(columns_tree)))
    array6 = np.zeros((len(columns_tree), 1))

    save_band_responses(root = root, newlocation=newlocation, newname='visbandresponses')
    save_fit_parameters(root = root, order=1, newlocation=newlocation, newname='visidealparameters')
    save_wavelengths(root = root, order = 1, newlocation=newlocation, newname='visfullxaxis16')
    save_filter(root=root, order=1, newlocation=newlocation, newname='vis_filter')

def main_NIR():
    ##Import data
    tree = ET.parse(calibrationpath + calibrationfile)
    root = tree.getroot()

    optical_tree1 = root.findall('./system_info/optical_components/optical_component')
    array6 = np.zeros((len(columns_tree), 1))

    save_band_responses(root=root, newlocation=newlocation, newname='bandresponses')
    save_fit_parameters(root=root, order=1, newlocation=newlocation, newname='idealparameters')
    save_fit_parameters(root=root, order=2, newlocation=newlocation, newname='secondaryidealparameters')
    save_wavelengths(root=root, order=1, newlocation=newlocation, newname='fullxaxis25')
    save_filter(root=root, order=1, newlocation=newlocation, newname='longpass_filter')
    save_filter(root=root, order=2, newlocation=newlocation, newname='shortpass_filter')
    
##Run
if __name__ = "__main__":
    if camera = "4x4 vis":
        main_vis()
    if camera = "5x5 NIR":
        main_NIR()
