import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import hog
from skimage import data, color, exposure
from PIL import Image
import cv2

# Read & resize two images
image1 = cv2.imread(r"test_images/segment114.jpg", 0)
image2 = cv2.imread(r"test_images/segment44.jpg", 0)
image1 = cv2.resize(image1, dsize=(200,200), interpolation = cv2.INTER_AREA)
image2 = cv2.resize(image2, dsize=(200,200), interpolation = cv2.INTER_AREA)

# Apply HOG to the previous Images
fd1, hog_image1 = hog(image1, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)
fd2, hog_image2 = hog(image2, orientations=9, pixels_per_cell=(8,8),
                    cells_per_block=(2,2), visualize=True)

# Rescale the intensity of the two images
hog_image_rescaled1 = exposure.rescale_intensity(hog_image1, in_range=(0, 10))
hog_image_rescaled2 = exposure.rescale_intensity(hog_image2, in_range=(0, 10))

# Show the hog feature images
images = [image1, hog_image1, hog_image_rescaled1, image2, hog_image2, hog_image_rescaled2]
labels = ['image', 'hog_image', 'hog_image_rescaled']

index = 0
fig, axs = plt.subplots(2, 3, figsize=(5, 3))
axs = axs.flatten()

for ax in axs:
    ax.imshow(images[index], cmap='gray')
    ax.set_title(labels[index % 3])
    index += 1

plt.show()

import numpy as np
dst = cv2.addWeighted(image2, 0.05, hog_image2, 0.95, 0, dtype=cv2.CV_32F).astype(np.uint8)
plt.title("HOG features added to the original image")
plt.imshow(dst, cmap='gray')
plt.show()