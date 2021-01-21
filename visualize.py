import numpy as np
import mahotas
import cv2
import os

from skimage import feature, filters
from skimage.io import imread
from pylab import imshow, show 
from matplotlib import pyplot as plt
import matplotlib

# ---------------------------------------
numPoints = 24
radius = 8

imNeg = imread('Data/Negative/00001.jpg')
imPos = imread('Data/Positive/00001.jpg')

# Haralick texture
ngray = cv2.cvtColor(imNeg, cv2.COLOR_BGR2GRAY)
pgray = cv2.cvtColor(imPos, cv2.COLOR_BGR2GRAY)
nh_gray = mahotas.features.haralick(ngray) 
ph_gray = mahotas.features.haralick(pgray) 
nh = mahotas.features.haralick(imNeg)
ph = mahotas.features.haralick(imPos)
fig, ((f1,f2), (f3,f4)) = plt.subplots(nrows=2, ncols=2,figsize=(9, 6))
f1.imshow(nh_gray)
f1.set_xlabel('Haralick (on grayscale) Negative')
f2.imshow(ph_gray)
f2.set_xlabel('Haralick (on grayscale) Positive')
f3.imshow(nh)
f3.set_xlabel('Haralick (on original) Negative')
f4.imshow(ph)
f4.set_xlabel('Haralick (on original) Positive')
plt.show()

# Color Histogram 
imHSV = cv2.cvtColor(imNeg,cv2.COLOR_BGR2HSV)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([imHSV],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.title('Color histogram for Negative HSV photo (without crack)')
plt.show()

imHSV_P = cv2.cvtColor(imPos,cv2.COLOR_BGR2HSV)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([imHSV_P],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.title('Color histogram for Positive HSV photo (with crack)')
plt.show()


# Local Binary Pattern (LBP)
lbpN = feature.local_binary_pattern(ngray, numPoints, radius, method="uniform")
lbpP = feature.local_binary_pattern(pgray, numPoints, radius, method="uniform")
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(9, 6))
plt.gray()
ax1.imshow(imNeg)
ax1.set_xlabel('Image without Crack')
ax2.imshow(imPos)
ax2.set_xlabel('Image with Crack')

ax3.hist(lbpN.ravel(), density=True, bins=numPoints + 2, range=(0, numPoints + 2))
ax3.set_ylabel('Percentage')
ax3.set_xlabel('Uniform LBP values')
ax4.hist(lbpP.ravel(), density=True, bins=numPoints + 2, range=(0, numPoints + 2))
ax4.set_ylabel('Percentage')
ax4.set_xlabel('Uniform LBP values')
plt.show()

# Canny edge detection
cannyN = feature.canny(ngray)
cannyP = feature.canny(pgray)
fig, (c1,c2) = plt.subplots(nrows=1, ncols=2,figsize=(9, 6))
c1.imshow(cannyN,cmap='gray')
c1.set_xlabel('Negative (without crack)')
fig.suptitle('Canny Edge detection', fontsize=16)
c2.imshow(cannyP,cmap='gray')
c2.set_xlabel('Positive (with crack)')
plt.show()

# Sobel Edge detection
sobelN = filters.sobel(ngray)
sobelP = filters.sobel(pgray)
fig, (s1,s2) = plt.subplots(nrows=1, ncols=2,figsize=(9, 6))
s1.imshow(sobelN,cmap='gray')
s1.set_xlabel('Negative (without crack)')
fig.suptitle('Sobel Edge detection', fontsize=16)
s2.imshow(sobelP,cmap='gray')
s2.set_xlabel('Positive (with crack)')
plt.show()