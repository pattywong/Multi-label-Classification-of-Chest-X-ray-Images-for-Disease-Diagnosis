import os, sys
from utils import *
from keras.utils import Sequence
import pandas as pd
import numpy as np
import math
import cv2
from termcolor import colored, cprint

def hist_equ(img):
    """
    Histogram equalization function for a 3-channel image.
    Attr/return: 3-channel image data 
    """
    for channel in range(3):
        img[:,:,channel] = cv2.equalizeHist(img[:,:,channel])
    return img

def std_imgNet(img):
    """
    Stardardization function for a 3-channel image with Imagenet mean and std values.
    Attr/return: 3-channel image data
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for channel in range(3):
        img[:,:,channel] = (img[:,:,channel] - mean[channel])/std[channel]
    return img

# Data Generator
class ChexpertSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, augmenter ):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augmenter = augmenter

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        
        # read images
        batch_x = self.x[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_x = np.array([cv2.imread(img_path) for img_path in batch_x])
        
        # augment
        batch_x = self.augmenter.augment_images(batch_x)
        
        # Histogram Equalization
        batch_x = np.array([hist_equ(img) for img in batch_x])                         
        # batch_x = np.array([np.stack((img,)*3, axis=-1) for img in batch_x])            
        
        # normalization
        batch_x = batch_x/255.
        
        # ImaageNet Standardization    
        batch_x = np.array([std_imgNet(img) for img in batch_x])                           
        
        # labels
        batch_y = np.array(self.y[idx * self.batch_size:(idx + 1) *self.batch_size])

        return batch_x , batch_y
    
# Data functions 

def load_data(annotation_path, annotation_file, img_dir, labels, dataset, mode):
    
    """
    This function:
        Reads a chestxray/chexpert annotation file.
        Stores image paths in x_set array and sets of labels in y_set array.
        Returns x_set and y_set arrays.
            x_set = ["/path/to/img1", "path/to/img2", ...]
            y_set = [[0,0,0,1,0], [0,0,0,0,0], ...]
    
    Attributes:
        annotation_path: directory of the annotation file.
        annotation_file: file name
        img_dir: directory of images
        labels: an array of labels
    """
    cprint('Loading data...', 'yellow', attrs=['reverse'])
    cprint("Annotation Path: " + os.path.join(annotation_path,annotation_file), "yellow")
    
    csv = pd.read_csv(os.path.join(annotation_path,annotation_file))
    image_index_col = csv['Image Index']
    x_set = []
    y_set = pd.DataFrame(csv)

    for index in range(len(image_index_col)):
        img_path = os.path.join(img_dir,image_index_col[index])
        x_set.append(img_path)

    x_set = np.array(x_set)
    y_set = y_set[labels]
            
    cprint(f"Number of samples in {annotation_file}: {len(x_set)}", "cyan")
    
    if mode == 'labelpowerset':
        # label-to-class-encoding
        class_label_generator(y_set)
        y_set = y_set['Classes']
        
    return x_set, y_set



