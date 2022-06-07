from skimage.feature import hog
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F 

orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)


classifier = "svm"

if classifier == "svm":
    # Load the SVM Model:
    model = joblib.load('SVM_person.npy')
elif classifier == "ann":
    # Load the ann model:
    class Net(nn.Module):
        def __init__(self, n_features):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(n_features, 1000)
            self.fc2 = nn.Linear(1000, 200)
            self.fc3 = nn.Linear(200, 1)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return torch.sigmoid(self.fc3(x))
    MODEL_PATH = 'model.pth'
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

################################################################################################################
############################### Test The classifier on a single image ##########################################
################################################################################################################
if 0:
    path = "C:/Users/elyousfh/OneDrive - STMicroelectronics/Desktop/Project/One Stage Detector/Depth Dataset/GAN Depth Images/Converted Images/00000416.jpg.tif"
    img = cv2.imread(path)
    img = color.rgb2gray(img)
    img = cv2.resize(img, dsize = (64,128))
    fds = hog(img, orientations, pixels_per_cell, cells_per_block, visualize=True)
    fds_array = fds[0].reshape(1, -1)
    pred = model.predict(fds_array)
    plt.imshow(img)
    plt.title("Person detected :" + str(pred[0]))
    plt.show()



################################################################################################################
############################## Test The classifier on a folder of  images ######################################
################################################################################################################
if 1:
    sum_true = 0
    sum_true_neg = 0

    all_preds = 0
    all_preds_neg =0


    y_true = list()
    y_pred = list()
    
    # Score on positive images
    for filename in os.listdir("C:/Users/elyousfh/OneDrive - STMicroelectronics/Desktop/Project/Segmentation/Segments_Dataset/person"):
        img= cv2.imread("C:/Users/elyousfh/OneDrive - STMicroelectronics/Desktop/Project/Segmentation/Segments_Dataset/person/" + str(filename))
        img = color.rgb2gray(img)
        img = cv2.resize(img, dsize = (64,128))
        fds = hog(img, orientations, pixels_per_cell, cells_per_block, visualize=True)
        fds_array = fds[0].reshape(1, -1)
        if classifier == "ann":
            pred = model(torch.Tensor(fds_array))
            pred = pred.cpu().detach().numpy().item()
            pred = int(pred > 0.5)
            #print(pred)
        elif classifier == "svm":
            pred = model.predict(fds_array)
            pred = pred[0]
        y_true.append("person")
        if pred == 1:
            sum_true = sum_true + 1
            y_pred.append("person")
        if pred == 0:
            y_pred.append("no person")
        all_preds = all_preds + 1

        
    

    # Score on negative images
    for filename in os.listdir("C:/Users/elyousfh/OneDrive - STMicroelectronics/Desktop/Project/Segmentation/Segments_Dataset/no person"):
        img= cv2.imread("C:/Users/elyousfh/OneDrive - STMicroelectronics/Desktop/Project/Segmentation/Segments_Dataset/no person/" + str(filename))
        img = color.rgb2gray(img)
        img = cv2.resize(img, dsize = (64,128))
        fds = hog(img, orientations, pixels_per_cell, cells_per_block, visualize=True)
        fds_array = fds[0].reshape(1, -1)
        if classifier == "ann":
            pred = model(torch.Tensor(fds_array))
            pred = pred.cpu().detach().numpy().item()
            pred = int(pred > 0.5)
            #print(pred)
        elif classifier == "svm":
            pred = model.predict(fds_array)
            pred = pred[0]
        
        y_true.append("no person")
        if pred == 0:
            sum_true_neg = sum_true_neg + 1
            y_pred.append("no person")
        if pred == 1:
            y_pred.append("person")
        all_preds_neg = all_preds_neg + 1

    print("Score on positive images = ",  sum_true/all_preds * 100)
    print("Score on negative images = ",  sum_true_neg/all_preds_neg * 100)


################################################################################################################
################################ Test The classifier on realtime images ########################################
################################################################################################################
if 0:
    import time 
    vid = cv2.VideoCapture(1)
    while(False):
        _, img = vid.read()
        imge = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        img = cv2.resize(imge, dsize = (64,128))

        # Calculate HOG descriptors
        time1 = time.time()
        fds = hog(img, orientations, pixels_per_cell, cells_per_block, visualize=True)
        fds_array = fds[0].reshape(1, -1)

        # Predict
        pred = model.predict(fds_array)
        
        print(time.time() - time1)
        window_name = 'Raw Detections after NMS'  
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (70, 70)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        imge = cv2.putText(imge, "Person detected : "+str(pred[0]), org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Raw Detections after NMS", imge)

        #### Save the images below
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
        


import seaborn as sebrn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
conf_matrix = confusion_matrix(y_true, y_pred, labels=["no person", "person"])

# Using Seaborn heatmap to create the plot
fx = sebrn.heatmap(conf_matrix, annot=True, cmap='turbo')

# labels the title and x, y axis of plot
fx.set_title('Confusion matrix on ambient dataset\n\n')
fx.set_xlabel('Predicted Values')
fx.set_ylabel('Actual Values ')

# labels the boxes
fx.xaxis.set_ticklabels(['No person','Person'])
fx.yaxis.set_ticklabels(['No person','Person'])

plt.show()