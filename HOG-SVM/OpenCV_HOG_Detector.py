import numpy as np
import cv2
import sys
from glob import glob
import itertools as it

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 0, 255), thickness)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

## On images
img = cv2.imread("rgb.jpg")
found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05) 
found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and inside(r, q):
            break
        else:
            found_filtered.append(r)
    draw_detections(img, found)
    draw_detections(img, found_filtered, 3) 
    print('%d (%d) found' % (len(found_filtered), len(found))) 

cv2.imshow('img', img) 
cv2.waitKey(0)


import sys
sys.exit(0)
cap= cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05) 
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
            else:
                found_filtered.append(r)
        draw_detections(img, found)
        draw_detections(img, found_filtered, 3) 
        print('%d (%d) found' % (len(found_filtered), len(found))) 
    cv2.imshow('img', img) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows() 
