"""
This code allows for creating a dataset of HOG_FEATURES of person images, the dataset is saved in
CSV file. The rows are the images (759 for the negative class and 759 for the positive class) 
and the colomns are the hog features for each image (3780 HOG feature).
"""

# Importing the necessary modules:
from skimage.feature import hog
import numpy as np
from numpy import *
import cv2, os, glob, csv
from PIL import Image 

# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

# Path for positive & negative dataset
pos_im_path = r"dataset/positive_images" 
neg_im_path= r"dataset/negative_images"

# read the image files:
pos_im_listing = os.listdir(pos_im_path)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing) 
num_neg_samples = size(neg_im_listing)
print("Positive data size : ", num_pos_samples)
print("Negative data size : ", num_neg_samples)
data = []
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

# Save to csv file
for i in range(len(labels)):
    data[i] = np.insert(data[i], 0, labels[i])

fields = []
for i in range(len(data[0])):
    if i == 0:
        fields += ['labels']
    else:
        fields += ["hf" + str(i-1)]

with open('HOG_dataset.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(data)






