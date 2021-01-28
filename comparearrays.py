import os
import cv2
import rawpy as raw
import imageio
import numpy as np
import matplotlib.pyplot as plt
import spectral
from spectral import imshow
from spectral import envi
import xml.etree.ElementTree as ET

data8 = '/home/ab20/Data/compare_viewers/Mono8/' #path to Mono8
viewerim8 = 'pf_20210127_124326_2371_img.png'
myviewerim8 = 'pf_20210127_124358_873_img.png'
data10 = '/home/ab20/Data/compare_viewers/Mono10/' #path to Mono10
viewerim10 = 'pf_20210127_124701_2286_img.png'
myviewerim10 = 'pf_20210127_124743_3857_img.png'

viewer8 = cv2.imread(data8+viewerim8, cv2.IMREAD_UNCHANGED)
viewer8 = viewer8.astype(np.float32)
myviewer8 = cv2.imread(data8+myviewerim8, cv2.IMREAD_UNCHANGED)
myviewer8 = myviewer8.astype(np.float32)

viewer10 = cv2.imread(data10+viewerim10, cv2.IMREAD_UNCHANGED)
viewer10 = viewer10.astype(np.float32)
myviewer10 = cv2.imread(data10+myviewerim10, cv2.IMREAD_UNCHANGED)
myviewer10 = myviewer10.astype(np.float32)
norm10 = (myviewer10 - myviewer10.min())/(myviewer10.max() - myviewer10.min())
scale10 = (myviewer10/myviewer10.max() * 255).astype(np.uint8)
print(norm10.min())
print(norm10.max())

print(scale10.min())
print(scale10.max())

diff8 = viewer8 - myviewer8
print(diff8.min())
print(diff8.max())

diff10 = viewer10 - myviewer10
print(diff10.min())
print(diff10.max())