import copy

import seaborn as sns
from mmdet.apis import init_detector, inference_detector
import os
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
from tabulate import tabulate
from texttable import Texttable
import latextable
import seaborn as sns
import numpy as np
import h5py
import pandas as pd
import cv2
import copy
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
import sklearn.metrics as metrics
from skimage.transform import resize
from sklearn.metrics import precision_score, recall_score
from scipy.ndimage.filters import gaussian_filter
from shapely.geometry import Polygon
import torch
import torchvision.ops.boxes as bops
import slideio
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)

# load the pd dataframe, set dirs and constants
DATAFRAME_DIR = '/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_TMA.csv'
RESULTS_SAVE_DIR = '/data/public/HULA/WSIs_renal_compartment_segmentations_new3/'
CLASSES = ["Glomerulus", "Arteriole", "Artery"]
LETTERS = [['A', 'B', 'C'], ['D', 'E', 'F']]
# set the val and test ids, strings for slide_id
VAL_ID = 'WCM_'
TEST_ID = 'Turin-'

# load the dataframe
results_dataframe = pd.read_csv(DATAFRAME_DIR)

# get the sub dfs
val_id = 'WCM_'
test_id = 'Turin-'
val_df = results_dataframe.loc[results_dataframe['slide_id'].str.contains(VAL_ID) & results_dataframe['status'].str.contains('processed')]
test_df = results_dataframe.loc[results_dataframe['slide_id'].str.contains(TEST_ID) & results_dataframe['status'].str.contains('processed')]
val_df = val_df.replace('nan', np.NaN)
test_df = test_df.replace('nan', np.NaN)
df_list = [val_df, test_df]

# generate the seaborn figure for our PR curves
# raw
sns.set_style('white', {'font.family': 'serif'})
sns.set_context("paper", font_scale=0.8)
fig, ax = plt.subplots(len(CLASSES), 2, constrained_layout=True, figsize=(8, 6))
global_precision = [{key: np.empty(shape=(0, 1)) for key in CLASSES} for _ in df_list]
global_recall = [{key: np.empty(shape=(0, 1)) for key in CLASSES} for _ in df_list]
passer = [{key: 0 for key in CLASSES} for _ in df_list]
for eval_int, df in enumerate(df_list):
    for slide_id in df['slide_id']:
        for class_id, class_name in enumerate(CLASSES):
            slide_info = df.loc[df['slide_id'].str.contains(slide_id)]
            if pd.isnull(slide_info[class_name + '_pr_precision']).iloc[0] or pd.isnull(slide_info[class_name + '_pr_recall']).iloc[0]:
                continue
            if passer[eval_int][class_name] == 0:
                pr_precision = np.concatenate([np.array([0.]), np.array(list(slide_info[class_name + '_pr_precision'])[0].split(' ')).astype('float'), np.array([1.])])
                global_precision[eval_int][class_name] = np.append(global_precision[eval_int][class_name], np.concatenate([np.array([[0.]]), resize(pr_precision, (500,))[:, np.newaxis], np.array([[1.]])]), axis=0)
                pr_recall = np.concatenate([np.array([1.]), np.array(list(slide_info[class_name + '_pr_recall'])[0].split(' ')).astype('float'), np.array([0.])])
                global_recall[eval_int][class_name] = np.append(global_recall[eval_int][class_name], np.concatenate([np.array([[1.]]), resize(pr_recall, (500,))[:, np.newaxis], np.array([[0.]])]), axis=0)
                passer[eval_int][class_name] += 1
            else:
                pr_precision = np.concatenate([np.array([0.]), np.array(list(slide_info[class_name + '_pr_precision'])[0].split(' ')).astype('float'), np.array([1.])])
                global_precision[eval_int][class_name] = np.append(global_precision[eval_int][class_name], np.concatenate([np.array([[0.]]), resize(pr_precision, (500,))[:, np.newaxis], np.array([[1.]])]), axis=1)
                pr_recall = np.concatenate([np.array([1.]), np.array(list(slide_info[class_name + '_pr_recall'])[0].split(' ')).astype('float'), np.array([0.])])
                global_recall[eval_int][class_name] = np.append(global_recall[eval_int][class_name], np.concatenate([np.array([[1.]]), resize(pr_recall, (500,))[:, np.newaxis], np.array([[0.]])]), axis=1)
            sns.lineplot(x=pr_recall, y=pr_precision, ax=ax[class_id, eval_int], ci=None, alpha=0.3)
    for class_id, class_name in enumerate(CLASSES):
        # get the global average PR curves
        precision = np.mean(global_precision[eval_int][class_name], axis=1)
        recall = np.mean(global_recall[eval_int][class_name], axis=1)
        sns.lineplot(x=recall, y=precision, ax=ax[class_id, eval_int], ci=None, color='red', linewidth=1.8, linestyle="dashed")
        ax[class_id, eval_int].set_xlim([0, 1])
        ax[class_id, eval_int].set_ylim([0, 1])
        ax[class_id, eval_int].text(0, 1.1, LETTERS[eval_int][class_id], fontsize=9, weight='bold')
        if (eval_int == 0) and (class_id == 2):
            print('Not hiding axes for bottom left...')
        elif (eval_int == 0):
            print('Removing x axis from middle area, left...')
            ax[class_id, eval_int].get_xaxis().set_visible(False)
        elif (eval_int == 1) and (class_id == 2):
            print('Removing y axis from bottom right...')
            ax[class_id, eval_int].get_yaxis().set_visible(False)
        else:
            print('Removing x and y axis from middle area, right...')
            ax[class_id, eval_int].get_xaxis().set_visible(False)
            ax[class_id, eval_int].get_yaxis().set_visible(False)
        mean_auc = df[class_name + '_pr_auc'].dropna(how='any').mean()
        std_auc = df[class_name + '_pr_auc'].dropna(how='any').std()
        if eval_int == 0:
            ax[class_id, eval_int].set_title(class_name + ' (Validation, Average AUC: {} \u00B1 {})'.format(mean_auc.round(3),
                                                                                                 std_auc.round(3)))
        else:
            ax[class_id, eval_int].set_title(class_name + ' (Test, Average AUC: {} \u00B1 {})'.format(mean_auc.round(3),
                                                                                                 std_auc.round(3)))
# fig.suptitle('Validation Set Pixel-Wise Precision-Recall Curves', weight='bold', x=0.53)
fig.supxlabel('Recall', weight='bold', x=0.53)
fig.supylabel('Precision', weight='bold', y=0.53)
plt.savefig(RESULTS_SAVE_DIR + 'nonroi_PR_curves.png', dpi=1000)
plt.savefig(RESULTS_SAVE_DIR + 'nonroi_PR_curves.svg', dpi=1000)
plt.show()

# roi
sns.set_style('white', {'font.family': 'serif'})
sns.set_context("paper", font_scale=0.8)
fig, ax = plt.subplots(len(CLASSES), 2, constrained_layout=True, figsize=(8, 6))
global_precision = [{key: np.empty(shape=(0, 1)) for key in CLASSES} for _ in df_list]
global_recall = [{key: np.empty(shape=(0, 1)) for key in CLASSES} for _ in df_list]
passer = [{key: 0 for key in CLASSES} for _ in df_list]
for eval_int, df in enumerate(df_list):
    for slide_id in df['slide_id']:
        for class_id, class_name in enumerate(CLASSES):
            slide_info = df.loc[df['slide_id'].str.contains(slide_id)]
            if pd.isnull(slide_info[class_name + '_pr_precision_rois']).iloc[0] or pd.isnull(slide_info[class_name + '_pr_recall_rois']).iloc[0]:
                continue
            if passer[eval_int][class_name] == 0:
                pr_precision = np.concatenate([np.array([0.]), np.array(list(slide_info[class_name + '_pr_precision_rois'])[0].split(' ')).astype('float'), np.array([1.])])
                global_precision[eval_int][class_name] = np.append(global_precision[eval_int][class_name], np.concatenate([np.array([[0.]]), resize(pr_precision, (500,))[:, np.newaxis], np.array([[1.]])]), axis=0)
                pr_recall = np.concatenate([np.array([1.]), np.array(list(slide_info[class_name + '_pr_recall_rois'])[0].split(' ')).astype('float'), np.array([0.])])
                global_recall[eval_int][class_name] = np.append(global_recall[eval_int][class_name], np.concatenate([np.array([[1.]]), resize(pr_recall, (500,))[:, np.newaxis], np.array([[0.]])]), axis=0)
                passer[eval_int][class_name] += 1
            else:
                pr_precision = np.concatenate([np.array([0.]), np.array(list(slide_info[class_name + '_pr_precision_rois'])[0].split(' ')).astype('float'), np.array([1.])])
                global_precision[eval_int][class_name] = np.append(global_precision[eval_int][class_name], np.concatenate([np.array([[0.]]), resize(pr_precision, (500,))[:, np.newaxis], np.array([[1.]])]), axis=1)
                pr_recall = np.concatenate([np.array([1.]), np.array(list(slide_info[class_name + '_pr_recall_rois'])[0].split(' ')).astype('float'), np.array([0.])])
                global_recall[eval_int][class_name] = np.append(global_recall[eval_int][class_name], np.concatenate([np.array([[1.]]), resize(pr_recall, (500,))[:, np.newaxis], np.array([[0.]])]), axis=1)
            sns.lineplot(x=pr_recall, y=pr_precision, ax=ax[class_id, eval_int], ci=None, alpha=0.3)
    for class_id, class_name in enumerate(CLASSES):
        # get the global average PR curves
        precision = np.mean(global_precision[eval_int][class_name], axis=1)
        recall = np.mean(global_recall[eval_int][class_name], axis=1)
        sns.lineplot(x=recall, y=precision, ax=ax[class_id, eval_int], ci=None, color='red', linewidth=1.8, linestyle="dashed")
        ax[class_id, eval_int].set_xlim([0, 1])
        ax[class_id, eval_int].set_ylim([0, 1])
        ax[class_id, eval_int].text(0, 1.1, LETTERS[eval_int][class_id], fontsize=9, weight='bold')
        if (eval_int == 0) and (class_id == 2):
            print('Not hiding axes for bottom left...')
        elif (eval_int == 0):
            print('Removing x axis from middle area, left...')
            ax[class_id, eval_int].get_xaxis().set_visible(False)
        elif (eval_int == 1) and (class_id == 2):
            print('Removing y axis from bottom right...')
            ax[class_id, eval_int].get_yaxis().set_visible(False)
        else:
            print('Removing x and y axis from middle area, right...')
            ax[class_id, eval_int].get_xaxis().set_visible(False)
            ax[class_id, eval_int].get_yaxis().set_visible(False)
        mean_auc = df[class_name + '_pr_auc_rois'].dropna(how='any').mean()
        std_auc = df[class_name + '_pr_auc_rois'].dropna(how='any').std()
        if eval_int == 0:
            ax[class_id, eval_int].set_title(class_name + ' (Validation, Average AUC: {} \u00B1 {})'.format(mean_auc.round(3),
                                                                                                 std_auc.round(3)))
        else:
            ax[class_id, eval_int].set_title(class_name + ' (Test, Average AUC: {} \u00B1 {})'.format(mean_auc.round(3),
                                                                                                 std_auc.round(3)))
# fig.suptitle('Validation Set Pixel-Wise Precision-Recall Curves', weight='bold', x=0.53)
fig.supxlabel('Recall', weight='bold', x=0.53)
fig.supylabel('Precision', weight='bold', y=0.53)
plt.savefig(RESULTS_SAVE_DIR + 'roi_PR_curves.png', dpi=1000)
plt.savefig(RESULTS_SAVE_DIR + 'roi_PR_curves.svg', dpi=1000)
plt.show()

# generate threshold comparison tables
# raw
rows = [['Confidence Thresholds', 'Glomerulus', 'Arteriole', 'Artery', 'Glomerulus', 'Arteriole', 'Artery'],
        ['0.3'] + [val_df[class_name + '_f1_30'].dropna(how='any').mean() for class_name in CLASSES] + [test_df[class_name + '_f1_30'].dropna(how='any').mean() for class_name in CLASSES],
        ['0.5'] + [val_df[class_name + '_f1_50'].dropna(how='any').mean() for class_name in CLASSES] + [test_df[class_name + '_f1_50'].dropna(how='any').mean() for class_name in CLASSES],
        ['0.7'] + [val_df[class_name + '_f1_70'].dropna(how='any').mean() for class_name in CLASSES] + [test_df[class_name + '_f1_70'].dropna(how='any').mean() for class_name in CLASSES],
        ['0.9'] + [val_df[class_name + '_f1_90'].dropna(how='any').mean() for class_name in CLASSES] + [test_df[class_name + '_f1_90'].dropna(how='any').mean() for class_name in CLASSES],
        ['Dynamic'] + [val_df[class_name + '_f1_auto'].mean() for class_name in CLASSES] + [test_df[class_name + '_f1_auto'].dropna(how='any').mean() for class_name in CLASSES],
        ['Optimistic'] + [val_df[class_name + '_f1_max'].mean() for class_name in CLASSES] + [test_df[class_name + '_f1_max'].dropna(how='any').mean() for class_name in CLASSES]]


table = Texttable()
table.set_cols_align(["c"] * 7)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTexttable Table:')
print(table.draw())

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))

print('\nTexttable Latex:')
print(latextable.draw_latex(table, caption="A comparison of different confidence thresholds on F1 score performance per class."))

# roi
rows = [['Confidence Thresholds', 'Glomerulus', 'Arteriole', 'Artery', 'Glomerulus', 'Arteriole', 'Artery'],
        ['0.3'] + [val_df[class_name + '_f1_30_rois'].dropna(how='any').mean() for class_name in CLASSES] + [test_df[class_name + '_f1_30_rois'].dropna(how='any').mean() for class_name in CLASSES],
        ['0.5'] + [val_df[class_name + '_f1_50_rois'].dropna(how='any').mean() for class_name in CLASSES] + [test_df[class_name + '_f1_50_rois'].dropna(how='any').mean() for class_name in CLASSES],
        ['0.7'] + [val_df[class_name + '_f1_70_rois'].dropna(how='any').mean() for class_name in CLASSES] + [test_df[class_name + '_f1_70_rois'].dropna(how='any').mean() for class_name in CLASSES],
        ['0.9'] + [val_df[class_name + '_f1_90_rois'].dropna(how='any').mean() for class_name in CLASSES] + [test_df[class_name + '_f1_90_rois'].dropna(how='any').mean() for class_name in CLASSES],
        ['Dynamic'] + [val_df[class_name + '_f1_auto_rois'].dropna(how='any').mean() for class_name in CLASSES] + [test_df[class_name + '_f1_auto_rois'].dropna(how='any').mean() for class_name in CLASSES],
        ['Optimistic'] + [val_df[class_name + '_f1_max_rois'].mean() for class_name in CLASSES] + [test_df[class_name + '_f1_max_rois'].dropna(how='any').mean() for class_name in CLASSES]]

table = Texttable()
table.set_cols_align(["c"] * 7)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTexttable Table:')
print(table.draw())

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))

print('\nTexttable Latex:')
print(latextable.draw_latex(table, caption="A comparison of different confidence thresholds on F1 score performance per class."))

# generate other metrics tables
# raw
class_row = [[class_name,
              val_df[class_name + '_iou'].dropna(how='any').mean(),
              val_df[class_name + '_AP'].dropna(how='any').mean(),
              val_df[class_name + '_AR'].dropna(how='any').mean(),
              val_df[class_name + '_F1'].dropna(how='any').mean(),
              val_df[class_name + '_AS'].dropna(how='any').mean(),
              test_df[class_name + '_iou'].dropna(how='any').mean(),
              test_df[class_name + '_AP'].dropna(how='any').mean(),
              test_df[class_name + '_AR'].dropna(how='any').mean(),
              test_df[class_name + '_F1'].dropna(how='any').mean(),
              test_df[class_name + '_AS'].dropna(how='any').mean(),
              ] for class_name in CLASSES]

rows = [['Classes', 'mIOU', 'mAP', 'mAR', 'mF1', 'mAS', 'mIOU', 'mAP', 'mAR', 'mF1', 'mAS']] + class_row

table = Texttable()
table.set_cols_align(["c"] * 11)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTexttable Table:')
print(table.draw())

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))

print('\nTexttable Latex:')
print(latextable.draw_latex(table, caption="Post-cleaning performance metrics."))

# roi
class_row = [[class_name,
              val_df[class_name + '_iou_rois'].dropna(how='any').mean(),
              val_df[class_name + '_AP_rois'].dropna(how='any').mean(),
              val_df[class_name + '_AR_rois'].dropna(how='any').mean(),
              val_df[class_name + '_F1_rois'].dropna(how='any').mean(),
              val_df[class_name + '_AS_rois'].dropna(how='any').mean(),
              test_df[class_name + '_iou_rois'].dropna(how='any').mean(),
              test_df[class_name + '_AP_rois'].dropna(how='any').mean(),
              test_df[class_name + '_AR_rois'].dropna(how='any').mean(),
              test_df[class_name + '_F1_rois'].dropna(how='any').mean(),
              test_df[class_name + '_AS_rois'].dropna(how='any').mean(),
              ] for class_name in CLASSES]

rows = [['Classes', 'mIOU', 'mAP', 'mAR', 'mF1', 'mAS', 'mIOU', 'mAP', 'mAR', 'mF1', 'mAS']] + class_row

table = Texttable()
table.set_cols_align(["c"] * 11)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTexttable Table:')
print(table.draw())

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))

print('\nTexttable Latex:')
print(latextable.draw_latex(table, caption="Post-cleaning performance metrics."))