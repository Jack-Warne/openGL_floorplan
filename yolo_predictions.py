import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader 
# load yaml file and yolo model
with open('yolo\pascal data\data.yaml') as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)

labels = data_yaml['names']

yolo = cv2.dnn.readNetFromONNX('yolov5/runs/train/exp/weights/last.onnx')
# set package to CPU 
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# load image
# from image get the detections
img = cv2.imread('floorplan.jpeg')
image = img.copy()

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# non maximum suppresion to make sure bounding boxes are correct

# draw the bounding box
