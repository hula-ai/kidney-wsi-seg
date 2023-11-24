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
from matplotlib.pyplot import cm
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

# load the pd dataframe, set dirs and constants
DATAFRAME_DIR = ['/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Cologne_3.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Szeged_6.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Bari_1.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Cologne_7.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Cologne_5.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Cologne_4.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Szeged_4.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Lille_7.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Bari_7.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Lille_0.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Bari_3.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Cologne_0.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Bari_5.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Szeged_1.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Cologne_2.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Szeged_2.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Cologne_1.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Lille_5.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Lille_4.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Bari_6.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Szeged_5.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Lille_6.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Szeged_7.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Bari_0.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Bari_2.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Lille_1.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Szeged_3.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Cologne_6.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Bari_4.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Szeged_0.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Lille_3.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Lille_2.csv']
# DATAFRAME_DIR = '/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_TMA.csv'
# RESULTS_SAVE_DIR = '/data/pcicales/seg_performance/pangn/'
RESULTS_SAVE_DIR = './pangn_seg_results/2/'
CLASSES = ["Glomerulus"]

# get the classes for classwise results
DISEASE_CLASSES = ["ABMGN", "ANCA", "C3-GN", "CryoglobulinemicGN", "DDD", "Fibrillary", "IAGN",
                   "IgAGN", "Membranous", "MPGN", "PGNMID", "SLEGN-IV"]
class_info = pd.read_csv('/data/public/HULA/Pan_GN/Pan_GN_GT.csv')

# load the dataframe
results_dataframe = pd.read_csv(DATAFRAME_DIR[0])
for frame in DATAFRAME_DIR[1:]:
    temp_res = pd.read_csv(frame)
    results_dataframe = pd.concat([results_dataframe, temp_res], ignore_index=True)

# get the sub dfs
# val_id = 'WCM_'
# test_id = 'Turin-'
val_df = results_dataframe.loc[results_dataframe['status'].str.contains('processed')]
# test_df = results_dataframe.loc[results_dataframe['slide_id'].str.contains(TEST_ID) & results_dataframe['status'].str.contains('processed')]
val_df = results_dataframe.replace('nan', np.NaN)
# test_df = test_df.replace('nan', np.NaN)
df_list = []
# append the dfs
for class_name in DISEASE_CLASSES:
    class_denotation = class_info.loc[class_info['Primary Classification'].str.contains(class_name)]
    bool = []
    for slide_ids in val_df['slide_id']:
        if any(case_ids in slide_ids for case_ids in class_denotation['Biopsy Number']):
            bool.append(True)
        else:
            bool.append(False)
    class_df = val_df.loc[bool]
    df_list.append(class_df)

df_list.append(val_df)

# denoter = DISEASE_CLASSES + ['ALL']
denoter = ["ABMGN", "ANCA", "C3-GN", "CryoGN", "DDD", "Fibrillary", "IAGN",
                   "IgAGN", "Membranous", "MPGN", "PGNMID", "SLEGN-IV", "ALL"]

# generate the seaborn figure for our PR curves
roistrs = ['', '_rois']
# generate global fig
sns.set_style('white', {'font.family': 'serif'})
sns.set_context("paper", font_scale=0.8)
for roistr in roistrs:
    precision_averages = []
    recall_averages = []
    colors = cm.rainbow(np.linspace(0, 1, len(denoter) - 1))
    global_precision = [{key: np.empty(shape=(0, 1)) for key in CLASSES} for _ in df_list]
    global_recall = [{key: np.empty(shape=(0, 1)) for key in CLASSES} for _ in df_list]
    passer = [{key: 0 for key in CLASSES} for _ in df_list]
    fig_glob, ax_glob = plt.subplots(len(CLASSES), 1, constrained_layout=True, figsize=(6, 3))
    # generate local fig
    fig, ax = plt.subplots(3, 4, constrained_layout=True, figsize=(20, 12))
    rowval = 0
    colval = 0
    for eval_int, df in enumerate(df_list):
        if eval_int % 4 == 0 and eval_int > 3:
            rowval += 1
            colval = 0
        for loca_val, slide_id in zip(df.index.values, df['slide_id']):
            for class_id, class_name in enumerate(CLASSES):
                slide_info = df.loc[df['slide_id'].str.contains(slide_id)]
                try:
                    if pd.isnull(slide_info[class_name + '_pr_precision{}'.format(roistr)])[loca_val] or pd.isnull(slide_info[class_name + '_pr_recall{}'.format(roistr)])[loca_val]:
                        continue
                    if passer[eval_int][class_name] == 0:
                        pr_precision = np.concatenate([np.array([0.]), np.array(list(slide_info[class_name + '_pr_precision{}'.format(roistr)])[0].split(' ')).astype('float'), np.array([1.])])
                        global_precision[eval_int][class_name] = np.append(global_precision[eval_int][class_name], np.concatenate([np.array([[0.]]), resize(pr_precision, (500,))[:, np.newaxis], np.array([[1.]])]), axis=0)
                        pr_recall = np.concatenate([np.array([1.]), np.array(list(slide_info[class_name + '_pr_recall{}'.format(roistr)])[0].split(' ')).astype('float'), np.array([0.])])
                        global_recall[eval_int][class_name] = np.append(global_recall[eval_int][class_name], np.concatenate([np.array([[1.]]), resize(pr_recall, (500,))[:, np.newaxis], np.array([[0.]])]), axis=0)
                        passer[eval_int][class_name] += 1
                    else:
                        pr_precision = np.concatenate([np.array([0.]), np.array(list(slide_info[class_name + '_pr_precision{}'.format(roistr)])[0].split(' ')).astype('float'), np.array([1.])])
                        global_precision[eval_int][class_name] = np.append(global_precision[eval_int][class_name], np.concatenate([np.array([[0.]]), resize(pr_precision, (500,))[:, np.newaxis], np.array([[1.]])]), axis=1)
                        pr_recall = np.concatenate([np.array([1.]), np.array(list(slide_info[class_name + '_pr_recall{}'.format(roistr)])[0].split(' ')).astype('float'), np.array([0.])])
                        global_recall[eval_int][class_name] = np.append(global_recall[eval_int][class_name], np.concatenate([np.array([[1.]]), resize(pr_recall, (500,))[:, np.newaxis], np.array([[0.]])]), axis=1)
                    if denoter[eval_int] != 'ALL':
                        ax[rowval, colval].plot(pr_recall, pr_precision,
                                label=denoter[eval_int], alpha=0.15)
                except:
                    continue

        for class_id, class_name in enumerate(CLASSES):
            # get the global average PR curves
            precision = np.mean(global_precision[eval_int][class_name], axis=1)
            recall = np.mean(global_recall[eval_int][class_name], axis=1)
            mean_auc = df[class_name + '_pr_auc{}'.format(roistr)].dropna(how='any').mean()
            std_auc = df[class_name + '_pr_auc{}'.format(roistr)].dropna(how='any').std()
            if denoter[eval_int] != 'ALL':
                ax_glob.plot(recall, precision, color=colors[eval_int], linewidth=2,
                             label='{0} (mAUC = {1:0.2f})'.format(denoter[eval_int], mean_auc))
                ax[rowval, colval].plot(recall, precision, '--', color='black', linewidth=2,
                             label=denoter[eval_int])
                ax[rowval, colval].set_xlim([0, 1.01])
                ax[rowval, colval].set_ylim([0, 1.01])
            else:
                ax_glob.plot(recall, precision, '--', color='black', linewidth=2,
                             label='All Classes (mAUC = {0:0.2f})'.format(mean_auc))
            if denoter[eval_int] != 'ALL':
                ax[rowval, colval].set_title('{} mAUC: {} \u00B1 {}'.format(denoter[eval_int], mean_auc.round(3), std_auc.round(3)))
        colval += 1
        # if denoter[eval_int] != 'ALL':
        #     fig.supxlabel('Recall', x=0.53)
        #     fig.supylabel('Precision', y=0.53)
        #     fig.savefig(RESULTS_SAVE_DIR + '{}_PR_curves{}.png'.format(denoter[eval_int], roistr), dpi=1000)
        #     fig.savefig(RESULTS_SAVE_DIR + '{}_PR_curves{}.svg'.format(denoter[eval_int], roistr), dpi=1000)
        #     fig.show()
        #     plt.close(fig)

    # generate the global classwise figure
    # sns.set_context("paper", font_scale=1.5)
    fig.supxlabel('Recall')
    fig.supylabel('Precision')
    fig.savefig(RESULTS_SAVE_DIR + 'CLASSWISE_PR_curves{}.png'.format(roistr), dpi=1000)
    fig.savefig(RESULTS_SAVE_DIR + 'CLASSWISE_PR_curves{}.svg'.format(roistr), dpi=1000)
    fig.show()
    plt.close(fig)

    # generate the global figure
    # sns.set_context("paper", font_scale=0.8)
    ax_glob.set_xlim([0, 1.01])
    ax_glob.set_ylim([0, 1.01])
    handles, labels = ax_glob.get_legend_handles_labels()
    lgd = ax_glob.legend(handles, labels, bbox_to_anchor=(1.1, 1))
    fig_glob.supxlabel('Recall', x=0.36) # x=0.53)
    fig_glob.supylabel('Precision', y=0.53)
    fig_glob.savefig(RESULTS_SAVE_DIR + 'ALL_PR_curves{}.png'.format(roistr), dpi=1000, bbox_inches='tight')
    fig_glob.savefig(RESULTS_SAVE_DIR + 'ALL_PR_curves{}.svg'.format(roistr), dpi=1000, bbox_inches='tight')
    plt.close(fig_glob)

    for print_val, df in zip(denoter, df_list):
        print('##' * 40)
        print(print_val)
        print('##' * 40)
        class_row = [[class_name,
                      df[class_name + '_iou{}'.format(roistr)].dropna(how='any').mean(),
                      df[class_name + '_AP{}'.format(roistr)].dropna(how='any').mean(),
                      df[class_name + '_AR{}'.format(roistr)].dropna(how='any').mean(),
                      df[class_name + '_F1{}'.format(roistr)].dropna(how='any').mean(),
                      df[class_name + '_AS{}'.format(roistr)].dropna(how='any').mean(),
                      # test_df[class_name + '_iou_rois'].dropna(how='any').mean(),
                      # test_df[class_name + '_AP_rois'].dropna(how='any').mean(),
                      # test_df[class_name + '_AR_rois'].dropna(how='any').mean(),
                      # test_df[class_name + '_F1_rois'].dropna(how='any').mean(),
                      # test_df[class_name + '_AS_rois'].dropna(how='any').mean(),
                      ] for class_name in CLASSES]

        rows = [['Classes', 'mIOU', 'mAP', 'mAR', 'mF1', 'mAS']] + class_row

        table = Texttable()
        table.set_cols_align(["c"] * 6)
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
#
# ##########################
# ####### roi ##############
# ##########################
# global_precision = [{key: np.empty(shape=(0, 1)) for key in CLASSES} for _ in df_list]
# global_recall = [{key: np.empty(shape=(0, 1)) for key in CLASSES} for _ in df_list]
# passer = [{key: 0 for key in CLASSES} for _ in df_list]
# for eval_int, df in enumerate(df_list):
#     sns.set_style('white', {'font.family': 'serif'})
#     sns.set_context("paper", font_scale=0.8)
#     fig, ax = plt.subplots(len(CLASSES), 1, constrained_layout=True, figsize=(4, 3))
#     for loca_val, slide_id in zip(df.index.values, df['slide_id']):
#         for class_id, class_name in enumerate(CLASSES):
#             slide_info = df.loc[df['slide_id'].str.contains(slide_id)]
#             try:
#                 if pd.isnull(slide_info[class_name + '_pr_precision_rois'])[loca_val] or pd.isnull(slide_info[class_name + '_pr_recall_rois'])[loca_val]:
#                     continue
#                 if passer[eval_int][class_name] == 0:
#                     pr_precision = np.concatenate([np.array([0.]), np.array(list(slide_info[class_name + '_pr_precision_rois'])[0].split(' ')).astype('float'), np.array([1.])])
#                     global_precision[eval_int][class_name] = np.append(global_precision[eval_int][class_name], np.concatenate([np.array([[0.]]), resize(pr_precision, (500,))[:, np.newaxis], np.array([[1.]])]), axis=0)
#                     pr_recall = np.concatenate([np.array([1.]), np.array(list(slide_info[class_name + '_pr_recall_rois'])[0].split(' ')).astype('float'), np.array([0.])])
#                     global_recall[eval_int][class_name] = np.append(global_recall[eval_int][class_name], np.concatenate([np.array([[1.]]), resize(pr_recall, (500,))[:, np.newaxis], np.array([[0.]])]), axis=0)
#                     passer[eval_int][class_name] += 1
#                 else:
#                     pr_precision = np.concatenate([np.array([0.]), np.array(list(slide_info[class_name + '_pr_precision_rois'])[0].split(' ')).astype('float'), np.array([1.])])
#                     global_precision[eval_int][class_name] = np.append(global_precision[eval_int][class_name], np.concatenate([np.array([[0.]]), resize(pr_precision, (500,))[:, np.newaxis], np.array([[1.]])]), axis=1)
#                     pr_recall = np.concatenate([np.array([1.]), np.array(list(slide_info[class_name + '_pr_recall_rois'])[0].split(' ')).astype('float'), np.array([0.])])
#                     global_recall[eval_int][class_name] = np.append(global_recall[eval_int][class_name], np.concatenate([np.array([[1.]]), resize(pr_recall, (500,))[:, np.newaxis], np.array([[0.]])]), axis=1)
#                 sns.lineplot(x=pr_recall, y=pr_precision, ax=ax, ci=None, alpha=0.15)
#             except:
#                 continue
#     for class_id, class_name in enumerate(CLASSES):
#         # get the global average PR curves
#         precision = np.mean(global_precision[eval_int][class_name], axis=1)
#         recall = np.mean(global_recall[eval_int][class_name], axis=1)
#         sns.lineplot(x=recall, y=precision, ax=ax, ci=None, color='black', linewidth=1.8, linestyle="dashed")
#         ax.set_xlim([0, 1])
#         ax.set_ylim([0, 1])
#         # ax[class_id, eval_int].text(0, 1.1, LETTERS[eval_int][class_id], fontsize=9, weight='bold')
#         # if (eval_int == 0) and (class_id == 2):
#         #     print('Not hiding axes for bottom left...')
#         # elif (eval_int == 0):
#         #     print('Removing x axis from middle area, left...')
#         #     ax[class_id, eval_int].get_xaxis().set_visible(False)
#         # elif (eval_int == 1) and (class_id == 2):
#         #     print('Removing y axis from bottom right...')
#         #     ax[class_id, eval_int].get_yaxis().set_visible(False)
#         # else:
#         #     print('Removing x and y axis from middle area, right...')
#         #     ax[class_id, eval_int].get_xaxis().set_visible(False)
#         #     ax[class_id, eval_int].get_yaxis().set_visible(False)
#         mean_auc = df[class_name + '_pr_auc_rois'].dropna(how='any').mean()
#         std_auc = df[class_name + '_pr_auc_rois'].dropna(how='any').std()
#         if eval_int == 0:
#             ax.set_title('{} (Average AUC: {} \u00B1 {})'.format(denoter[eval_int], mean_auc.round(3),
#                                                                                                  std_auc.round(3)))
#         else:
#             ax.set_title('{} (Average AUC: {} \u00B1 {})'.format(denoter[eval_int], mean_auc.round(3),
#                                                                                                  std_auc.round(3)))
#     # fig.suptitle('Validation Set Pixel-Wise Precision-Recall Curves', weight='bold', x=0.53)
#     fig.supxlabel('Recall', weight='bold', x=0.53)
#     fig.supylabel('Precision', weight='bold', y=0.53)
#     plt.savefig(RESULTS_SAVE_DIR + '{}_roi_PR_curves.png'.format(denoter[eval_int]), dpi=1000)
#     plt.savefig(RESULTS_SAVE_DIR + '{}_roi_PR_curves.svg'.format(denoter[eval_int]), dpi=1000)
#     plt.show()
#     plt.close(fig)
#
# # generate threshold comparison tables
# # raw
# # rows = [['Confidence Thresholds', 'Glomerulus'],
# #         ['0.3'] + [val_df[class_name + '_f1_30'].dropna(how='any').mean() for class_name in CLASSES],
# #         ['0.5'] + [val_df[class_name + '_f1_50'].dropna(how='any').mean() for class_name in CLASSES],
# #         ['0.7'] + [val_df[class_name + '_f1_70'].dropna(how='any').mean() for class_name in CLASSES],
# #         ['0.9'] + [val_df[class_name + '_f1_90'].dropna(how='any').mean() for class_name in CLASSES],
# #         ['Dynamic'] + [val_df[class_name + '_f1_auto'].mean() for class_name in CLASSES],
# #         ['Optimistic'] + [val_df[class_name + '_f1_max'].mean() for class_name in CLASSES]]
# #
# #
# # table = Texttable()
# # table.set_cols_align(["c"] * 2)
# # table.set_deco(Texttable.HEADER | Texttable.VLINES)
# # table.add_rows(rows)
# #
# # print('Tabulate Table:')
# # print(tabulate(rows, headers='firstrow'))
# #
# # print('\nTexttable Table:')
# # print(table.draw())
# #
# # print('\nTabulate Latex:')
# # print(tabulate(rows, headers='firstrow', tablefmt='latex'))
# #
# # print('\nTexttable Latex:')
# # print(latextable.draw_latex(table, caption="A comparison of different confidence thresholds on F1 score performance per class."))
#
# # roi
# rows = [['Confidence Thresholds', 'Glomerulus'],
#         ['0.3'] + [val_df[class_name + '_f1_30_rois'].dropna(how='any').mean() for class_name in CLASSES],
#         ['0.5'] + [val_df[class_name + '_f1_50_rois'].dropna(how='any').mean() for class_name in CLASSES],
#         ['0.7'] + [val_df[class_name + '_f1_70_rois'].dropna(how='any').mean() for class_name in CLASSES],
#         ['0.9'] + [val_df[class_name + '_f1_90_rois'].dropna(how='any').mean() for class_name in CLASSES],
#         ['Dynamic'] + [val_df[class_name + '_f1_auto_rois'].dropna(how='any').mean() for class_name in CLASSES],
#         ['Optimistic'] + [val_df[class_name + '_f1_max_rois'].mean() for class_name in CLASSES]]
#
# table = Texttable()
# table.set_cols_align(["c"] * 2)
# table.set_deco(Texttable.HEADER | Texttable.VLINES)
# table.add_rows(rows)
#
# print('Tabulate Table:')
# print(tabulate(rows, headers='firstrow'))
#
# print('\nTexttable Table:')
# print(table.draw())
#
# print('\nTabulate Latex:')
# print(tabulate(rows, headers='firstrow', tablefmt='latex'))
#
# print('\nTexttable Latex:')
# print(latextable.draw_latex(table, caption="A comparison of different confidence thresholds on F1 score performance per class."))
#
# # generate other metrics tables
# # raw
# # class_row = [[class_name,
# #               val_df[class_name + '_iou'].dropna(how='any').mean(),
# #               val_df[class_name + '_AP'].dropna(how='any').mean(),
# #               val_df[class_name + '_AR'].dropna(how='any').mean(),
# #               val_df[class_name + '_F1'].dropna(how='any').mean(),
# #               val_df[class_name + '_AS'].dropna(how='any').mean(),
# #               # test_df[class_name + '_iou'].dropna(how='any').mean(),
# #               # test_df[class_name + '_AP'].dropna(how='any').mean(),
# #               # test_df[class_name + '_AR'].dropna(how='any').mean(),
# #               # test_df[class_name + '_F1'].dropna(how='any').mean(),
# #               # test_df[class_name + '_AS'].dropna(how='any').mean(),
# #               ] for class_name in CLASSES]
# #
# # rows = [['Classes', 'mIOU', 'mAP', 'mAR', 'mF1', 'mAS']] + class_row
# #
# # table = Texttable()
# # table.set_cols_align(["c"] * 6)
# # table.set_deco(Texttable.HEADER | Texttable.VLINES)
# # table.add_rows(rows)
# #
# # print('Tabulate Table:')
# # print(tabulate(rows, headers='firstrow'))
# #
# # print('\nTexttable Table:')
# # print(table.draw())
# #
# # print('\nTabulate Latex:')
# # print(tabulate(rows, headers='firstrow', tablefmt='latex'))
# #
# # print('\nTexttable Latex:')
# # print(latextable.draw_latex(table, caption="Post-cleaning performance metrics."))
#
# # roi
# for print_val, df in zip(denoter, df_list):
#     print('##' * 40)
#     print(print_val)
#     print('##' * 40)
#     class_row = [[class_name,
#                   df[class_name + '_iou_rois'].dropna(how='any').mean(),
#                   df[class_name + '_AP_rois'].dropna(how='any').mean(),
#                   df[class_name + '_AR_rois'].dropna(how='any').mean(),
#                   df[class_name + '_F1_rois'].dropna(how='any').mean(),
#                   df[class_name + '_AS_rois'].dropna(how='any').mean(),
#                   # test_df[class_name + '_iou_rois'].dropna(how='any').mean(),
#                   # test_df[class_name + '_AP_rois'].dropna(how='any').mean(),
#                   # test_df[class_name + '_AR_rois'].dropna(how='any').mean(),
#                   # test_df[class_name + '_F1_rois'].dropna(how='any').mean(),
#                   # test_df[class_name + '_AS_rois'].dropna(how='any').mean(),
#                   ] for class_name in CLASSES]
#
#     rows = [['Classes', 'mIOU', 'mAP', 'mAR', 'mF1', 'mAS']] + class_row
#
#     table = Texttable()
#     table.set_cols_align(["c"] * 6)
#     table.set_deco(Texttable.HEADER | Texttable.VLINES)
#     table.add_rows(rows)
#
#     print('Tabulate Table:')
#     print(tabulate(rows, headers='firstrow'))
#
#     print('\nTexttable Table:')
#     print(table.draw())
#
#     print('\nTabulate Latex:')
#     print(tabulate(rows, headers='firstrow', tablefmt='latex'))
#
#     print('\nTexttable Latex:')
#     print(latextable.draw_latex(table, caption="Post-cleaning performance metrics."))