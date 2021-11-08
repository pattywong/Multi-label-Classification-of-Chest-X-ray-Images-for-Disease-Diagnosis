
import os, math
import numpy as np
import pandas as pd
from sklearn.metrics import *
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.utils import class_weight


def count_samples_ct(df, cols, values):
    
    '''
    This function counts number of samples given specific values of labels.
    Attrs:  df: DataFrame
            cols: array of column names
            values: array of values for column names respectively
    Ex: (n, _) = count_samples(df_test, ["Edema","Atelectasis"],[1,0])
    Return: a number of samples (int64), an array of sample ids
    '''
    
    if len(cols) != len(values):
        raise Exception("the lengths of 'cols' and 'values' arrays must be equal.")
    
    for index in range(len(cols)):
        df = df.loc[df[cols[index]]==values[index]]
    
    sample_ids = df.index
    n_samples = len(sample_ids)

    return n_samples, sample_ids

def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list

def cal_corr_rmse(corr_gt_arr, corr_pred_arr):
    gt = []
    pred = []
    for i in range(len(corr_gt_arr)):
        for j in range(i+1,len(corr_gt_arr)):
            gt.append(np.array(corr_gt_arr)[i,j])
            pred.append(np.array(corr_pred_arr)[i,j])
    rmse = math.sqrt(mean_squared_error(gt, pred))
    return gt, pred, rmse

def to_csv(array, filename, index=None, columns=None):
    df = pd.DataFrame(array)
    if index != None: df.index = index
    if columns != None: df.columns = columns
    df.to_csv(f"results/csv/{filename}.csv")

    
def to_ml_cat(indexes, n_classes=5):
    arr = [0] * n_classes
    for index in indexes:
        arr[index] = 1
    return arr


def convert_to_bin(number, width):
    """
    Covert decimal number to binary number
    """
    str_label_set = format(number, '{fill}{width}b'.format(width=width, fill=0))
    labelset = [int(i) for i in str_label_set]
    return np.array(labelset)
    

def get_class_label(binary_list):
    '''
    function description:
        binary_list: an array of labels
        return: a class label
    ex: get_class_label([1.,1.,1.]) return 7
    '''
    class_label = 0 
    binary_list = [str(int(integer)) for integer in binary_list]
    binary = str("".join(binary_list))
    for digit in binary: 
        class_label = class_label*2 + int(digit) 
    return class_label


def class_label_generator(df):
    '''
    function description:
    df_y: dataframe y
    create "Classes" column and generate class label of each set.
    '''
    classes_array = []
    for index in range(len(df)):
        label_set = df.iloc[index,:].values
        class_label = get_class_label(label_set)
        classes_array.append(class_label)
    df['Classes'] = classes_array
    return df


def get_label_set(class_label_batch, n_classes):
    '''
    class_label_batch: model prediction in a batch array
    return an array of sets of labels 
    '''
    
    results = []
    for class_label in class_label_batch:
        label_set = bin(class_label)[2:]
        label_set = [int(i) for i in label_set]
        result = ([0] * (n_classes - len(label_set))) + label_set
        results.append(result)
    return results


# Class weights calculation functions

def count_samples(annotation_path, annotation_file, class_names):
    annotation_file_path = os.path.join(annotation_path, annotation_file)
    df = pd.read_csv(annotation_file_path)
    labels = df[class_names].values
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return len(df), class_positive_counts

def get_class_weights(annotation_path, annotation_file, class_names, y_set_train, mode):
    
    if mode == 'default':
        train_counts, train_pos_counts = count_samples(annotation_path, annotation_file, class_names)
        class_weights = calculate_class_weights(train_counts, train_pos_counts)
    elif mode == 'labelpowerset':
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_set_train), y_set_train)
    
    return class_weights

def calculate_class_weights(total_counts, class_positive_counts, multiply=1):
    """
    Calculate class_weight used in training

    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean

    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights



def count_samples_pn(df, col_name):
    '''
    Ex: count_samples_pn(df_test, label)
    Return: numbers of samples with class '0' and class '1'
    '''
    
    counts = df[f'{col_name}'].value_counts()
    return np.array(counts)

def count_samples_ctm(df, cols, values):
    
    '''
    This function counts number of samples given specific values of labels.
    Attrs:  df: DataFrame
            cols: array of column names
            values: array of values for column names respectively
    Ex: (n, _) = count_samples(df_test, ["Edema","Atelectasis"],[1,0])
    Return: a number of samples (int64), an array of sample ids
    '''
    
    if len(cols) != len(values):
        raise Exception("the lengths of 'cols' and 'values' arrays must be equal.")
    
    for index in range(len(cols)):
        df = df.loc[df[cols[index]]==values[index]]
    
    sample_ids = df.index
    n_samples = len(sample_ids)

    return n_samples, sample_ids

def to_dict_format(y_test_pred, y_test_true, n_classes):

    # extract each disease col
    y_test_pred_dict = {}
    for index_col in range(n_classes):
        y_test_pred_dict[f'{index_col}'] = y_test_pred[:,index_col]

    y_test_true_dict = {}
    for index_col in range(n_classes):
        y_test_true_dict[f'{index_col}'] = y_test_true[:,index_col]

    return y_test_pred_dict, y_test_true_dict

if __name__ == "__main__":
    pass