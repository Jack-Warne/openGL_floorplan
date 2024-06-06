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

#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

row, col, d = image.shape

# convert image into square matirix, this is because the yolo model wants square images
max_rc = max(row, col) # get the max value for rows and colums
input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8) # create a square image that is blank

input_image[0:row, 0:col] = image # overlay the two images to get a square image with the original image inside

#get predictions from square array 
blob = cv2.dnn.blobFromImage(input_image, 1/255, (640,640), swapRB=True, crop=False)
yolo.setInput(blob)
preds = yolo.forward() # get predictions

# non maximum suppresion to make sure bounding boxes are correct, removes duplicates and give only good probabality score
# filter on probablility score and confidence scorre i.e if confidence is less than 0.4 then filter it out
detections = preds[0]
boxes = []
confidences = []
classes = []

# calculate width and height of image
image_w, image_h = input_image.shape[:2]
x_factor = image_w/640
y_factor = image_h/640 # these are expected to be the same as the image should be square

for i in range(len(detections)):
    row = detections[i]
    confidence = row[4] # confidence of detection of object
    if confidence > 0.3:
        class_score = row[5:].max() # take maximum probability of object have 8 objects of all eight objects take the one it is most likely to be
        class_id = row[5:].argmax() # get index position at which maximum probability occurs
        if class_score > 0.25:
            cx, cy, w, h = row[0:4]
            #construct bounding box from four values
            # Left, Top, Width, Height
            left = int((cx - 0.5*w)*x_factor)
            top = int((cy - 0.5*h)*y_factor)
            width = int(w*x_factor)
            height = int(h*y_factor)
            
            box = np.array([left,top,width,height])

            # append values into the list
            confidences.append(confidence)
            boxes.append(box)
            classes.append(class_id)

# cleaning data
boxes_np = np.array(boxes).tolist()
confidences_np = np.array(confidences).tolist()

# apply NMS
index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.3).flatten() # this gives index positions

# draw the bounding box
for ind in index:
    # extract bounding boxes
    x,y,w,h = boxes_np[ind]
    # take confidences
    bb_conf = confidences_np[ind]
    # take classes
    classes_id  = classes[ind]
    class_name = labels[classes_id]

    text = f'{class_name}: {bb_conf}%'
    
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.rectangle(image, (x,y-30),(x+w,y),(255,255,255), -1)
    cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7,(0,0,0),1)

cv2.imshow('original', img)
cv2.imshow('yolo_prediction', image)

cv2.waitKey(0)
cv2.destroyAllWindows