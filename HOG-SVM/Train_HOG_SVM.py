# Importing the necessary modules:
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, NuSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from numpy import *
import csv

# Define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

# Path for positive & negative dataset
pos_im_path = r"dataset_seg_big/positive_images" 
neg_im_path= r"dataset_seg_big/negative_images"

# read the image files:
pos_im_listing = os.listdir(pos_im_path)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing) 
num_neg_samples = size(neg_im_listing)
print("Positive data size : ", num_pos_samples)
print("Negative data size : ", num_neg_samples)
data= []
labels = []


# compute HOG features and label them:
for file in pos_im_listing: #this loop enables reading the files in the pos_im_listing variable one by one
    #print(file)
    img = Image.open(pos_im_path + '\\' + file) # open the file
    img = img.resize((64,128))
    
    gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
    # calculate HOG for positive features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
    data.append(fd)
    labels.append(1)

# Same for the negative images
for file in neg_im_listing:
    img= Image.open(neg_im_path + '\\' + file)
    img = img.resize((64,128))
    gray= img.convert('L')
    # Now we calculate the HOG for negative features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
    data.append(fd)
    labels.append(0)


    
# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels1 = le.fit_transform(labels)
print("Classes : ", le.classes_)


# Partitioning the data into training and testing splits, using 80%
# of the data for training and the remaining 20% for testing
print(" Constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.2, random_state=42)


print(len(trainData))
print(len(trainLabels))


#%% Train the linear SVM
print(" Training Linear SVM classifier...")
model = NuSVC(gamma="auto")
model.fit(trainData, trainLabels)

#%% Evaluate the classifier
print(" Evaluating classifier on test data ...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

# Save the model:
#%% Save the Model
joblib.dump(model, 'SVM_person.npy')


