import numpy as np
import mahotas
import cv2
import os

from skimage import feature, filters, exposure
from skimage.io import imread
from skimage.transform import resize
from pylab import imshow, show 
from matplotlib import pyplot as plt
import matplotlib

# -------------------------------------------
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
f1.set_xlabel('Haralick on grayscale image (Negative)')
f2.imshow(ph_gray)
f2.set_xlabel('Haralick on grayscale image (Positive)')
f3.imshow(nh)
f3.set_xlabel('Haralick on original image (Negative)')
f4.imshow(ph)
f4.set_xlabel('Haralick on original image (Positive)')
plt.show()

# Color Histogram 
imHSV = cv2.cvtColor(imNeg,cv2.COLOR_BGR2HSV)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([imHSV],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.title('Color Histogram for image in HSV (Negative)')
plt.show()

imHSV_P = cv2.cvtColor(imPos,cv2.COLOR_BGR2HSV)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([imHSV_P],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.title('Color Histogram for image in HSV (Positive)')
plt.show()

# Local Binary Pattern (LBP)
lbpN = feature.local_binary_pattern(ngray, numPoints, radius, method="uniform")
lbpP = feature.local_binary_pattern(pgray, numPoints, radius, method="uniform")
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(9, 6))
plt.gray()
ax1.imshow(imNeg)
ax1.set_xlabel('Image without Crack (Negative)')
ax2.imshow(imPos)
ax2.set_xlabel('Image with Crack (Positive)')

ax3.hist(lbpN.ravel(), density=True, bins=numPoints + 2, range=(0, numPoints + 2))
ax3.set_ylabel('Percentage')
ax3.set_xlabel('Uniform LBP values')
ax4.hist(lbpP.ravel(), density=True, bins=numPoints + 2, range=(0, numPoints + 2))
ax4.set_ylabel('Percentage')
ax4.set_xlabel('Uniform LBP values')
plt.show()

# Histogram of Oriented Gradients
res_imgN = resize(imNeg, (128, 64))
res_imgP = resize(imPos, (128, 64))
fdN, hog_imageN = feature.hog(res_imgN, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
fdP, hog_imageP = feature.hog(res_imgP, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True) 
hog_image_rescaled = exposure.rescale_intensity(hog_imageN, in_range=(0, 10)) 
ax1.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
ax1.set_title('Negative') 
fig.suptitle('Histogram of Oriented Gradients', fontsize=16)
hog_image_rescaledP = exposure.rescale_intensity(hog_imageP, in_range=(0, 10)) 
ax2.imshow(hog_image_rescaledP, cmap=plt.cm.gray) 
ax2.set_title('Positive')

plt.show()

# Zernike Moments
filtN = imNeg[:,:,0]
filtP = imPos[:,:,0]
fig, (zer1, zer2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
zer1.imshow(filtN) 
zer1.set_title('Negative') 
fig.suptitle('Filterd Images for Zernike Moments', fontsize=18)
zer2.imshow(filtP) 
zer2.set_title('Positive')
plt.show()

# --------------------------------------------------
# Plotting accuracy as a function of sample size
# Hu Moments
x = np.linspace(1, 20, 20)
y = np.array([91.75,93.875,92,92.31,92,92,92.32,92.28,92.64,93.425,93.41,92.94,92.885,93.375,93.18,93.81,93.985,94.46,94.355,94.4])

plt.plot(x, y, 'o', color='blue'); 
plt.xlim([0,21])
plt.ylim([0,100])
plt.xticks(np.arange(0, 21, 1))
plt.grid(True)
plt.xlabel('Sample size (in images per class *1000)')
plt.ylabel('Accuracy (in %)')
plt.title('Accuracy as function of sample size - Hu Moments')
plt.show()

# Zernike Moments
x = np.linspace(1, 20, 20)
y = np.array([74.25,78.625,79.58,80.5,80.75,78.54,79.07,78.97,80.19,79.65,80,80.71,80.44,80.41,79.45,80.41,80.93,81.72,81.37,82.65])

plt.plot(x, y, 'o', color='green'); 
plt.xlim([0,21])
plt.ylim([0,100])
plt.xticks(np.arange(0, 21, 1))
plt.grid(True)
plt.xlabel('Sample size (in images per class *1000)')
plt.ylabel('Accuracy (in %)')
plt.title('Accuracy as function of sample size - Zernike Moments')
plt.show()