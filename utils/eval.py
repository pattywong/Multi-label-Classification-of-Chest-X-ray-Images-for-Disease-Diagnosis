import numpy as np
import os, sys, math
from sklearn.metrics import *
from termcolor import colored, cprint
from keras.utils.np_utils import to_categorical
        
def calculate_auc(y_test_pred_dict, y_test_true_dict, n_classes):
    auc_roc_arr = []
    for index in range(n_classes):
        y_scores = y_test_pred_dict[f'{index}']
        y_true = y_test_true_dict[f'{index}']
        auc_roc = roc_auc_score(y_true, y_scores, average="macro")
        cprint(f'  {index}:' +str(auc_roc), "green" )
        auc_roc_arr.append(auc_roc) 
    cprint(f"average AUC: {np.mean(auc_roc_arr)}", "green")    
    return auc_roc_arr

def calculate_auc_bl(y_test_pred, y_test_true, n_classes):

    # This function returns AUC scores of each class label (Baseline method)
    y_test_pred_dict = {}
    for index_col in range(n_classes):
        y_test_pred_dict[f'{index_col}'] = np.array(y_test_pred)[:,index_col]

    y_test_true_dict = {}
    for index_col in range(n_classes):
        y_test_true_dict[f'{index_col}'] = np.array(y_test_true)[:,index_col]    

    auc_roc_arr = []
    for index in range(n_classes):
        y_scores = y_test_pred_dict[f'{index}']
        y_true = y_test_true_dict[f'{index}']
        auc_roc = roc_auc_score(y_true, y_scores, average="macro")
        cprint(f'  {index}:' +str(auc_roc), "green" )
        auc_roc_arr.append(auc_roc) 
    cprint(f"average AUC: {np.mean(auc_roc_arr)}", "green")    
    return auc_roc_arr


def calculate_auc_label(y_test_pred, y_test_true, n_classes, n_labels):
    '''
    This function returns AUC scores of each class label
    converted from 32-class scores (labelpowerset)
    '''
    y_test_pred = np.array([to_label_scores(i) for i in y_test_pred])
    y_test_true = np.array([np.argmax(i) for i in y_test_true])
    y_test_true = np.array(get_label_set(np.array(y_test_true)))
    label_5_auc = calculate_auc_bl(y_test_pred, y_test_true, n_labels)
    return label_5_auc
    
def calculate_auc_lp(y_test_pred, y_test_true, n_classes, n_labels):
    class_32_auc = calculate_auc_bl(y_test_pred, y_test_true, n_classes)
    y_test_pred = np.array([to_label_scores(i) for i in y_test_pred])
    y_test_true = np.array([np.argmax(i) for i in y_test_true])
    y_test_true = np.array(get_label_set(np.array(y_test_true)))
    label_5_auc = calculate_auc_bl(y_test_pred, y_test_true, n_labels)
    return class_32_auc, label_5_auc
            
def to_label_scores(class_scores):
    # This function converts m-class scores to n-label scores
    n_labels = int(math.log(len(class_scores),2))
    label_scores = [0] * n_labels
    for index in range(len(class_scores)):
        label_set = (convert_to_bin(index,n_labels))
        pos_label_index_arr = np.where(label_set == 1)
        if (len(pos_label_index_arr[0]) != 0):
            for pos_label_index in pos_label_index_arr[0]:
                label_scores[pos_label_index] += class_scores[index]
    return label_scores
    
def get_label_set(class_label_batch):
    '''
    class_label_batch: model prediction in a batch array
    return an array of sets of labels 
    '''
    results = []
    for class_label in class_label_batch:
        label_set = bin(class_label)[2:]
        label_set = [int(i) for i in label_set]
        result = ([0] * (5 - len(label_set))) + label_set
        results.append(result)
    return results

def convert_to_bin(number, width):
    # Convert decimal number to binary number
    str_label_set = format(number, '{fill}{width}b'.format(width=width, fill=0))
    labelset = [int(i) for i in str_label_set]
    return np.array(labelset)