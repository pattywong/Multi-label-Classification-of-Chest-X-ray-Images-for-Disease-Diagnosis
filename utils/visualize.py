import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from itertools import cycle
import seaborn as sns
from termcolor import colored, cprint
import pandas as pd
from sklearn.metrics import mean_squared_error
import math

def confusion_matrix(df, pair, labels):
    index_dict = {'C': 0, 'L':1, 'E': 2, 'A':3, 'P':4}
    label_indexes = (index_dict[pair[0]] , index_dict[pair[1]])
    label_pair = labels[label_indexes[0]], labels[label_indexes[1]]
    cases = [[0,0], [0,1], [1,0], [1,1]]
    matrix = []
    norm_matrix = []
    for i in range(len(cases)):
        row = []
        for j in range(len(cases)):
            pair_case = (cases[i], cases[j])
            a_true = df[f'{label_pair[0]}-act']
            b_true = df[f'{label_pair[1]}-act']
            a_pred = df[f'{label_pair[0]}-pred']
            b_pred = df[f'{label_pair[1]}-pred']

            count = len(df.loc[(a_true == pair_case[0][0]) & (b_true == pair_case[0][1]) & (a_pred == pair_case[1][0]) & (b_pred == pair_case[1][1])])   
            row.append(count)
        row = np.array(row)
        sum_n = np.sum(row)
        matrix.append(row)
        norm_matrix.append(row / sum_n)
    return np.array(matrix).transpose(), np.array(norm_matrix).transpose()
    
def case_a_to_f_df(csv):
    labels = ["Cardiomegaly", "Lung Opacity", "Edema", "Atelectasis", "Pleural Effusion"]
    methods = csv.Method.unique()
    rows = []
    for method in methods:
        for disease1 in labels:
            labels2 = ["Cardiomegaly", "Lung Opacity", "Edema", "Atelectasis", "Pleural Effusion"]
            labels2.remove(disease1)
            for disease2 in labels2:
                csv_i = csv.loc[(csv.Method==method) & (csv.Disease1==disease1) & (csv.Disease2==disease2)]
                N_total = 13132
                # P(Act-A | Act-B) - checked
                Support_11 = csv_i.iloc[12,7]
                Support_01_11 = csv_i.iloc[4,7] + csv_i.iloc[12,7]
                P_ActA_g_ActB = Support_11/Support_01_11

                # P(Act-A | Pred-B)
                N_ActA_g_PredB = csv_i.iloc[9,5] + csv_i.iloc[11,5] + csv_i.iloc[13,5]+ csv_i.iloc[15,5]  
                N_PredB = csv_i.iloc[1,5] + csv_i.iloc[3,5] + csv_i.iloc[5,5] + csv_i.iloc[7,5] + csv_i.iloc[9,5] + csv_i.iloc[11,5] + csv_i.iloc[13,5]+ csv_i.iloc[15,5]  
                P_ActA_g_PredB = N_ActA_g_PredB/N_PredB

                # P(Act-A | Pred-A)
                N_ActA_g_PredA = csv_i.iloc[10,5] + csv_i.iloc[11,5] + csv_i.iloc[14,5]+ csv_i.iloc[15,5]  
                N_PredA = csv_i.iloc[2,5] + csv_i.iloc[3,5] + csv_i.iloc[6,5] + csv_i.iloc[7,5] + csv_i.iloc[10,5] + csv_i.iloc[11,5] + csv_i.iloc[14,5]+ csv_i.iloc[15,5]  
                P_ActA_g_PredA = N_ActA_g_PredA/N_PredA

                diff = abs(P_ActA_g_ActB - P_ActA_g_PredB)

                row = [disease1, disease2, method] 
                row = row + [P_ActA_g_ActB, P_ActA_g_PredB, P_ActA_g_PredA] + [diff]
                rows.append(row)
                
    df = pd.DataFrame(rows)
    df.columns = ["Disease1", "Disease2", "Method", "P_ActA_g_ActB", "P_ActA_g_PredB", "P_ActA_g_PredA", "diff"]
    return df
    
def gen_raw_cm(df, mode):
    labels = ['Cardiomegaly', 'Lung Opacity', 'Edema', 'Atelectasis', 'Pleural Effusion']
    dict_label = {'C':0, 'L':1, 'E':2, 'A':3, 'P':4}
    dict_type = {0: '00', 1: '01', 2:'10', 3: '11'}
    pairs = ['CL', 'CE', 'CA', 'CP', 'LC', 'LE', 'LA', 'LP','EC', 'EL', 'EA', 'EP', 'AC','AL','AE','AP','PC','PL','PE','PA']
    rows = []
    for pair in pairs:    
        for type1 in range(4):
            for type2 in range(4):
                row=[labels[dict_label[pair[0]]]]
                for l in pair:
                    label_index = dict_label[l]
                    row.append(labels[label_index])
                row.append(dict_type[type1])
                row.append(dict_type[type2])
                cm1, cm2 = confusion_matrix(df, pair, labels)
                row.append( cm1[type2,type1])
                row.append( cm2[type2,type1])
                row.append(np.sum(cm1[:,type1]))
                row.append(mode)
                rows.append(row)
    csv = pd.DataFrame(rows)
    csv.columns = ['Disease', 'Disease1', 'Disease2', 'Act', 'Pred', 'N', 'Norm_N', "Support", "Method" ]
    csv.to_csv(f"{mode}_pol_cm.csv")

def display_roc_graph(labels, y_true, y_scores):
    
    aucs = []
    fpr = dict()
    tpr = dict()
    thres = dict()
    roc_auc = dict()
    
    # Calculate AUC of rach label
    for i in range(len(labels)):
        fpr[i], tpr[i], thres[i] = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple' ])
    for i, color in zip(range(len(labels)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of {0} (area = {1:0.3f})'.format(labels[i], roc_auc[i]))
        aucs.append(roc_auc[i])
    
    print("32 AUC Scores:")
    for i in aucs:
        print(i)
    print("avg. AUC Scores:")
    print(np.mean(aucs))
 
    # Plot
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-label data')
    plt.legend(loc="lower right")
    plt.show()    
        
def cal_auc(labels, y_true, y_scores):
    
    aucs = []
    fpr = dict()
    tpr = dict()
    thres = dict()
    roc_auc = dict()

    # Calculate AUC of rach label
    for i in range(len(labels)):
        fpr[i], tpr[i], thres[i] = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple' ])
    
    for i, color in zip(range(len(labels)), colors):
        aucs.append(roc_auc[i])

    return aucs + [np.mean(aucs)] 

def get_thresholds(y_true, y_scores, labels):
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    thresholds = []
    fpr = dict()
    tpr = dict()
    thres = dict()

    for i in range(len(labels)):
        fpr[i], tpr[i], thres[i] = roc_curve(y_true[:, i], y_scores[:, i])

    # find optimal thresholds
    for i in range(len(labels)):
        optimal_idx = np.argmax(tpr[i] - fpr[i])
        optimal_threshold = thres[i][optimal_idx]
        thresholds.append(optimal_threshold)
    return thresholds
    
def round_scores_df(y_pred_dict, thresholds, labels):
    
    for label in range(len(labels)):
        temp = y_pred_dict[str(label)]
        temp[temp>=thresholds[label]] = 1
        temp[temp<thresholds[label]] = 0
        
    df_pred = pd.DataFrame(y_pred_dict)
    round_y_pred = []
    for sample in range(len(df_pred)):
        round_y_pred.append(np.array(df_pred.loc[sample,:]))
    
    round_y_pred= pd.DataFrame(round_y_pred)
    
    return round_y_pred

def round_scores(y_test_pred, thresholds, labels, n_classes):
    
    y_test_pred_dict = {}
    for index_col in range(n_classes):
        y_test_pred_dict[f'{index_col}'] = np.array(y_test_pred)[:,index_col]
    
    for label in range(n_classes):
        temp = y_test_pred_dict[str(label)]
        temp[temp>=thresholds[label]] = 1
        temp[temp<thresholds[label]] = 0
        
    df_pred = pd.DataFrame(y_test_pred_dict)
    round_y_pred = []
    for sample in range(len(df_pred)):
        round_y_pred.append(np.array(df_pred.loc[sample,:]))
    
    round_y_pred= pd.DataFrame(round_y_pred)
    
    return round_y_pred

def display_corr_matrix(df_y):
    # Only select the requested columns
    df_corr_matrix = df_y
    df_corr_matrix.columns = ['Cardiomegaly', 'Lung Opacity', 'Edema', 'Atelectasis', 'Pleural Effusion']

    # This computes the Pearson coefficient for all couples
    corr = df_corr_matrix.corr().fillna(0)
    corr_array = corr
    
    # Start drawing
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    size = max(6, len(corr.columns)/2.)
    f, ax = plt.subplots(figsize=(size, size))
    ax.set_title('Correlation matrix')

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": 0.5}, ax=ax, annot = True, fmt='.2g',vmin = 0, vmax= 0.7, cmap="YlGnBu" )

def get_corr_values(df_y):
    # Only select the requested columns
    df_corr_matrix = df_y
    # This computes the Pearson coefficient for all couples
    corr = df_corr_matrix.corr().fillna(0)
    return corr

def fit_line(corr_gt_array, corr_pred_array, labels, file_name):
    corr_gt_array = corr_gt_array.values.flatten()
    corr_pred_array = corr_pred_array.values.flatten()
    
    prep_df = []
    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            corr_index = col_index + (row_index* len(labels))
            array = [labels[row_index], labels[col_index], corr_gt_array[corr_index], corr_pred_array[corr_index]]
            prep_df.append(array)

    df = pd.DataFrame(prep_df)
    df.to_excel(f"{file_name}.xlsx")
    
    annotation = []
    for i in range(len(df[0])):
        if (df[0][i][0] == df[1][i][0]):
            annotation.append("ALL")
        else:
            annotation.append(df[0][i][0] + df[1][i][0])
    df['4'] = annotation
    n = df['4']
    to_reverse = [5, 10, 11, 15, 16, 17, 20, 21, 22, 23]
    for i in to_reverse:
        df['4'][i] = df['4'][i][::-1]
    n = df['4']
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # gt - green
    x = np.array(df[2])
    y_act = np.array(df[2])
    m, b = np.polyfit(x, y_act, 1)
    ax.scatter(x,y_act)

    plt.plot(x, m*x + b, 'g', label="ground truth")
    # # pred
    y_pred = np.array(df[3])
    ax.scatter(x,y_pred)
    m, b = np.polyfit(x, y_pred, 1)
    plt.plot(x, m*x + b, 'r', label="prediction")
    plt.legend(loc="upper left")

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y_act[i]), horizontalalignment='right')
        ax.annotate(txt, (x[i], y_pred[i]), horizontalalignment='right')
        
    print(math.sqrt(mean_squared_error(np.array(df[2]), np.array(df[3]))))
    
def to_ml_cat(indexes, n_classes=5):
    arr = [0] * n_classes
    for index in indexes:
        arr[index] = 1
    return arr
    
def get_df_error(y_set_test, round_y_pred, labels):
    err_sample_ids = []
    err_label_indexes = []
    for sample_index in range(len(y_set_test)):
        pred = np.array(round_y_pred.iloc[sample_index])
        gt = np.array(y_set_test.iloc[sample_index])
        comparison = pred == gt
        equal_arrays = comparison.all()
        if equal_arrays == False:
            err_sample_ids.append(sample_index)
            err_label_indexes.append(to_ml_cat(np.where(pred!=gt)[0]))

    df_err_cases = pd.DataFrame()
    df_err_cases['Sample id'] = err_sample_ids
    for i in range(len(labels)):
        df_err_cases[f'{labels[i]}'] = np.array(err_label_indexes).transpose()[i]
    return df_err_cases


def count_err(df_err_cases, labels):
    err_arr = []
    for label in labels:
        err_arr.append(df_err_cases[f'{label}'].value_counts()[1])
    err_arr = np.array(err_arr)
    total_n_err = err_arr.sum()
    return total_n_err, err_arr

def display_matrix(matrix):
    
    matrix = pd.DataFrame(matrix)
    matrix.columns = ["00", "01", "10", "11"]
    matrix.index = ["00", "01", "10", "11"]
    
    fig, ax = plt.subplots(figsize=(6,6))         # Sample figsize in inches
    sns.heatmap(matrix, annot=True, linewidths=.5, ax=ax, vmax=7000, vmin= 0 ,fmt="d", cmap="RdYlGn")
    plt.xlabel("Actual")
    plt.ylabel("Prediction")