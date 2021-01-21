import numpy as np
import mahotas
import cv2
import os
import glob
import shutil

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from skimage import feature, filters
from skimage.io import imread
from pylab import imshow, show 
from matplotlib import pyplot as plt
import matplotlib
# XGBClassifier
# RandomForestClassifier
# DecisionTreeClassifier
# SGDClassifier

images_per_class = 20000
train_path = "Data"
bins = 8
num_trees = 100
test_size = 0.20
seed = 9
scoring = "accuracy"

# Feature Descriptors
# 1: Hu moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hu = cv2.HuMoments(cv2.moments(image)).flatten()
    return hu

# 2: Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray,ignore_zeros=True).mean(axis=0)
    return haralick

# 3: Color Histogram
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# 4: Local Binary Pattern (LBP)
def fd_local_binary_pattern(image, numPoints=24, radius=8, eps=1e-7):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist

# 5: Canny Edge Detector
def fd_canny(image): 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = feature.canny(image)
    return canny.flatten()
   
# 6: Sobel Edge Detector
def fd_sobel(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel = filters.sobel(image)
    return sobel.flatten()



# 
# class_labels = os.listdir(train_path)
# features_list = []
# labels = []

# # loop over the training data sub-folders
# for class_name in class_labels:
#     # join the training data path and each species training folder
#     dir = os.path.join(train_path, class_name)

#     # get the current training label
#     current_label = class_name

#     # loop over the images in each sub-folder
#     for x in range(1, images_per_class+1):

#         file = dir + "/" + str(x) + ".jpg"

#         image = cv2.imread(file)

#         # Global Feature extraction
#         fv_hu_moments = fd_hu_moments(image)
#         fv_haralick = fd_haralick(image)
#         fv_histogram = fd_histogram(image)

        
#         feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

#         # update the list of labels and feature vectors
#         labels.append(current_label)
#         features_list.append(feature)

#     print("[STATUS] processed folder: {}".format(current_label))

# print("[STATUS] completed Global Feature Extraction...")


# # encode the target labels
# le          = LabelEncoder()
# target      = le.fit_transform(labels)
# # scale features in the range (0-1)
# scaler            = MinMaxScaler(feature_range=(0, 1))
# rescaled_features = scaler.fit_transform(global_features)

# #initiate ML model
# clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

# #initiate 10-fold validation function
# kfold = KFold(n_splits=10, random_state=seed)

# #predict the images labels using model with input data and 10-fold validation
# cv_results = cross_val_score(clf, rescaled_features, target, cv=kfold, scoring=scoring)
# print(cv_results.mean())




