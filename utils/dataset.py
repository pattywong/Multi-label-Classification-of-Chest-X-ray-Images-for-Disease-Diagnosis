import pandas as pd
import numpy as np
import os, sys
from data import *
from augmenter import *

class Dataset:
    
    # Batch size
    BATCH_SIZE = 24
    
    # Select .csv file names
    ANNOTATION_TRAIN_FILE = "train.csv"
    ANNOTATION_VAL_FILE = "valid.csv"
    ANNOTATION_TEST_FILE = "valid.csv"
    
    def __init__(self):
        
        self.class_name = "Dataset"

        # Directories
        self.ANNOTATION_DIR = "/CheXpert-v1.0/csv"
        self.IMAGE_DIR = "/CheXpert-v1.0/images"
        
        # Labels
        self.LABELS = ['Cardiomegaly','Lung Opacity','Edema','Atelectasis','Pleural Effusion']
        
        # Load dataset
        (self.x_set_train, self.y_set_train) = self.load_data(
            self.ANNOTATION_DIR, self.ANNOTATION_TRAIN_FILE, 
            self.IMAGE_DIR, self.LABELS)
        
        (self.x_set_val, self.y_set_val) = self.load_data(
            self.ANNOTATION_DIR, self.ANNOTATION_VAL_FILE, 
            self.IMAGE_DIR, self.LABELS)
        
        (self.x_set_test, self.y_set_test) = self.load_data(
            self.ANNOTATION_DIR, self.ANNOTATION_TEST_FILE, 
            self.IMAGE_DIR, self.LABELS)
        
        self.TRAIN_LENGTH = len(self.x_set_train)
        self.VAL_LENGTH = len(self.x_set_val)
        
        self.TRAIN_STEPS = self.TRAIN_LENGTH // self.BATCH_SIZE
        self.VAL_STEPS = self.VAL_LENGTH // self.BATCH_SIZE
        
        print("**Dataset loaded**")
        
    def build_generators(self):
        
        # Building training, validation and test generators 
        self.generator_train = ChexpertSequence(self.x_set_train, self.y_set_train, self.BATCH_SIZE, augmenter)
        self.generator_val = ChexpertSequence(self.x_set_val, self.y_set_val, self.BATCH_SIZE, augmenter)
        self.generator_test = ChexpertSequence(self.x_set_test, self.y_set_test, self.BATCH_SIZE, augmenter)    
        
        print("**Generators successfully built**")
        
    def load_data(self, annotation_path, annotation_file, img_dir, labels):
        """
        Attrs: 
           annotation_path: annotation file path, 'a/b/c'
           annotation_file: annotation file name, 'annotation.csv'
           img_dir: image directory, 'a/b/c'
           labels: an array of labels, ["A", "B", "C"]
        This function returns:
            X: an array of image paths, ['x/y/z.jpg', ...]
            y: an array of sets of labels, [[0,0,0,0,0], ...]
        """
        csv = pd.read_csv(os.path.join(annotation_path,annotation_file))
        x_set = np.array([os.path.join(img_dir,img) for img in csv.iloc[:,0]])
        y_set = np.array(csv[labels])
        print(annotation_file, csv.shape)
        
        return x_set, y_set