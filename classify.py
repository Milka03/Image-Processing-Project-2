import numpy as np
import mahotas
import cv2
import os
import glob
import shutil

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from skimage import feature, filters
from skimage.transform import resize
from skimage.io import imread
from pylab import imshow, show 
from matplotlib import pyplot as plt
import matplotlib

### Variables ###
images_per_class = 1000
train_path = "Data"
bins = 8
num_trees = 100
test_size = 0.20
seed = 9
scoring = "accuracy"

### Feature Descriptors ###
# 1: Hu Moments
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
def fd_lbp(image, numPoints=24, radius=8, eps=1e-7):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist

# 5: Histogram of Oriented Gradients (HOG)
def fd_hog(image):
    image = resize(image, (128, 64))
    fd = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=True)
    return fd

# 6: Zernike Moments
def fd_zernike(image):
    return mahotas.features.zernike_moments(image[:, :, 0], degree=20, radius=20)


### Run during first execution of program to ensure consistency in files' naming ###
# path = './Data/Positive'
# files = os.listdir(path)
# for f in files:
#     if '_1' in f:
#         newname = f.replace('_1','')
#         os.rename(os.path.join(path, f),os.path.join(path,newname))


### Classification ###
class_labels = os.listdir(train_path)
features_list = []
labels = []

# Iterate over training folders
for class_name in class_labels:
    dir = os.path.join(train_path, class_name)
    current_label = class_name
    print("\nExtracting features in folder: {} ...".format(current_label))
    for x in range(1, images_per_class+1):
        if x < 10:
            file = dir + "/0000" + str(x) + ".jpg"
        elif 9 < x < 100:
            file = dir + "/000" + str(x) + ".jpg"
        elif 99 < x < 1000:
            file = dir + "/00" + str(x) + ".jpg"
        elif 999 < x < 10000:
            file = dir + "/0" + str(x) + ".jpg"
        else:
            file = dir + "/" + str(x) + ".jpg"

        # All files are the same size so no need for resizing
        image = cv2.imread(file)
        # Feature Extraction
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)
        fv_lbp = fd_lbp(image)
        fv_hog = fd_hog(image)
        fv_zernike = fd_zernike(image)

        cumulative_feature = np.hstack([fv_haralick,fv_lbp,fv_hog])
        # cumulative_feature = np.hstack([fv_hu_moments])
        # cumulative_feature = np.hstack([fv_haralick])
        # cumulative_feature = np.hstack([fv_histogram])
        # cumulative_feature = np.hstack([fv_lbp])
        # cumulative_feature = np.hstack([fv_hog])
        # cumulative_feature = np.hstack([fv_zernike])

        # Update the list of labels and feature vectors
        labels.append(current_label)
        features_list.append(cumulative_feature)
    print("Processed folder: {}".format(current_label))
print("Feature Extraction Completed.\n")

# Encode the target labels
encoder = LabelEncoder()
target = encoder.fit_transform(labels)
# Scale features in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(features_list)
print("Encoded labels, scaled features.\nTraining the model...")

# Initiate ML model and divide dataset into training and testing part
clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
(trainFeatures, testFeatures, trainLabels, testLabels) = train_test_split(np.array(rescaled_features),np.array(target),test_size=test_size,random_state=seed)

# Train the model and evaluate prediction score
clf.fit(trainFeatures, trainLabels)
result = clf.score(testFeatures, testLabels)

print("\nImages per class used: {}".format(images_per_class))
print("Accuracy for Random Forest Classifier: {}\n".format(result))
