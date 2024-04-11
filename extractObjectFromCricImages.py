"""
This script is created for:

Extracting Objects from the Images given in the CRIC dataset.

It stores the objects in a list of list for the whole dataset then writes the fetched object names in a text file named "objects_train.txt"
"""
import warnings
warnings.filterwarnings('ignore')
print('Importing packages...')

import os
import re
import torch
import spacy
import json
import math
import sys

# path = './obj_status.txt'
# sys.stdout = open(path, 'w')

from tqdm.auto import tqdm
from PIL import Image as img
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter as Count
    

# Setting Up YOLO for Object Detection

print('Importing YOLO...')
from ultralytics import YOLO
detector_model = YOLO('yolov8n.pt')  # load an official model

classToLabelMap = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 
                    7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 
                    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 
                    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 
                    28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 
                    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
                    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
                    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 
                    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
                    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
                    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
                    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 
                    79: 'toothbrush'}

def getClassLabels(classIndices):
    labels = []
    for idx in classIndices:
        labels.append(classToLabelMap[idx])
    return(set(labels))

def getResultsFromYoloDirectly(image):
    results =  detector_model.predict(image) 
    
    """
    # To Plot Bounding Boxes
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = img.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.thumbnail((500,500))
        im.show()  # show image
    """
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        confs = boxes.data[:, 5:6] # Confidence and class ID of the detected objects

    classIndices = confs.squeeze(1).int().tolist()
    return list(getClassLabels(classIndices))


# Fetching CRIC Dataset Now

print('\nFetching CRIC Data...')

train_file_path = '/home/aritra/cric/train_questions.json'
val_file_path = '/home/aritra/cric/val_questions.json'
test_file_path = '/home/aritra/cric/test_v1_questions.json'

# Training Set
with open(train_file_path, "r") as file:
     train_json = json.load(file)
        
# Validation Set
with open(val_file_path, "r") as file:
     val_json = json.load(file)
        
# Test Set
with open(test_file_path, "r") as file:
     test_json = json.load(file)


print('\nExtracting Training Data...')


# Extracting Data of Training Set

questionList, answerList, imgList, k_triplet = [],[],[],[]

# verifying
indexToExclude = []

with open('../text_files/error1.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)
        
with open('../text_files/error2.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)
        
with open('../text_files/error3.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)

# Storing Trainig Set into Py Lists

for i in tqdm(range(len(train_json))):
    
    if i in indexToExclude:
        continue
        
    pointer = train_json[i]
    
    questionList.append(pointer['question'])
    answerList.append(pointer['answer'])
    imgList.append(pointer['image_id'])
    k_triplet.append( ' '.join(pointer['sub_graph']['knowledge_items'][0]['triplet']) + '. ' )


questionList = questionList[:100000]
imgList = imgList[0:100000]


print('Extracting objects into Lists ...')
objectsList = []
objectDetectionFailureCount = 0

for i in tqdm(range(len(questionList))):
    
    try:
        filepath = '/home/aritra/cric/images/img/'
        imgName = str(imgList[i]) + '.jpg'
        concatedPath = os.path.join(filepath, imgName)
        currentImage = img.open(concatedPath)
        
        objectsInImage = getResultsFromYoloDirectly(currentImage)
        
        if objectsInImage == []:
            objectsList.append([])
        
        else:
            objectsList.append(objectsInImage)

    except:
        objectDetectionFailureCount += 1
        objectsList.append([])
        continue


filename = 'objects_train.txt' 
print(f'Writing into file {filename}')
with open(filename, 'w') as file:

    for objectsSubList in objectsList:

        for object_ in objectsSubList:
            file.write(f'{object_},')
        file.write('\n')

print(f'\nFile Stored at {filename}')
print(f'\nObject Detection Failed For {objectDetectionFailureCount} Images')
print(f'\nLength of the Objects List is {len(objectsList)}')
print('\n** Exiting Script **\n')