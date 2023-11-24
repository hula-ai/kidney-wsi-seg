import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# use this only when you are sure code is working as expected, ignore numpy warnings from mmdetection
# import warnings
# warnings.filterwarnings("ignore")

# ToDo: Find a way to log and add the recall metrics
# path to results directory
# /home/cougarnet.uh.edu/srizvi7/anaconda3/envs/openmmlab_03292022_5/bin/python visualize_log_file.py
RESULTS_DIR = '/data/syed/mmdet/run20_swin_gan_img_train_iter2_autoaug/'  # Need end '/'

# name of the relevant config file, to be used for model results
# CONFIG = [pos_py for pos_py in os.listdir(RESULTS_DIR) if pos_py.endswith('.py')][0]

# if you want to generate the pseudo labels, set to True
SUPER = False

# if you want to evaluate one model
EPOCH = 12

# input list of json files given possible training ints
JSON_FILES = [pos_json for pos_json in os.listdir(RESULTS_DIR) if pos_json.endswith('.json')]

# generate blank dataframes for AP results
bbox_val_AP = pd.DataFrame(columns=['epoch', 'bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75',
                                        'bbox_mAP_s', 'bbox_mAP_m', 'bbox_mAP_l'])

class_bbox_val_AP = pd.DataFrame(columns=['epoch', "bbox_AP_Glomerulus", "bbox_AP_Arteriole", "bbox_AP_Artery"])

seg_val_AP = pd.DataFrame(columns=['epoch', 'segm_mAP', 'segm_mAP_50', 'segm_mAP_75',
                                   'segm_mAP_s', 'segm_mAP_m', 'segm_mAP_l'])

class_segm_val_AP = pd.DataFrame(columns=['epoch', "segm_AP_Glomerulus", "segm_AP_Arteriole", "segm_AP_Artery"])

# extract key data
for file in JSON_FILES:
    data = [json.loads(line) for line in open(RESULTS_DIR + file, 'r')]
    for log_line in data:
        if 'bbox_mAP' in log_line.keys():
            bbox_val_AP = bbox_val_AP.append(
                pd.DataFrame([[log_line[x] for x in bbox_val_AP.columns.values]],
                             columns=bbox_val_AP.columns.values))
            class_bbox_val_AP = class_bbox_val_AP.append(
                pd.DataFrame([[log_line[x] for x in class_bbox_val_AP.columns.values]],
                             columns=class_bbox_val_AP.columns.values))
            seg_val_AP = seg_val_AP.append(
                pd.DataFrame([[log_line[x] for x in seg_val_AP.columns.values]],
                             columns=seg_val_AP.columns.values))
            class_segm_val_AP = class_segm_val_AP.append(
                pd.DataFrame([[log_line[x] for x in class_segm_val_AP.columns.values]],
                             columns=class_segm_val_AP.columns.values))

# sort the dataframe by epoch
bbox_val_AP = bbox_val_AP.sort_values(by='epoch')
class_bbox_val_AP = class_bbox_val_AP.sort_values(by='epoch')
seg_val_AP = seg_val_AP.sort_values(by='epoch')
class_segm_val_AP = class_segm_val_AP.sort_values(by='epoch')

# log the winning val rounds
bbox_val_AP_winners = []
bbox_val_AP_legend = []
class_bbox_val_AP_winners = []
class_bbox_val_AP_legend = []
seg_val_AP_winners = []
seg_val_AP_legend = []
class_segm_val_AP_winners = []
class_segm_val_AP_legend = []

for metrics in bbox_val_AP.columns.values[1:]:
    winner = bbox_val_AP[metrics].argmax()
    winner_val = bbox_val_AP[metrics].max()
    bbox_val_AP_legend.append(metrics + ' (Best: {}, Epoch: {})'.format(winner_val, winner + 1))
    if winner in bbox_val_AP_winners:
        continue
    else:
        bbox_val_AP_winners.append(winner)

for metrics in class_bbox_val_AP.columns.values[1:]:
    winner = class_bbox_val_AP[metrics].argmax()
    winner_val = class_bbox_val_AP[metrics].max()
    class_bbox_val_AP_legend.append(metrics + ' (Best: {}, Epoch: {})'.format(winner_val, winner + 1))
    if winner in class_bbox_val_AP_winners:
        continue
    else:
        class_bbox_val_AP_winners.append(winner)

for metrics in seg_val_AP.columns.values[1:]:
    winner = seg_val_AP[metrics].argmax()
    winner_val = seg_val_AP[metrics].max()
    seg_val_AP_legend.append(metrics + ' (Best: {}, Epoch: {})'.format(winner_val, winner + 1))
    if winner in seg_val_AP_winners:
        continue
    else:
        seg_val_AP_winners.append(winner)

for metrics in class_segm_val_AP.columns.values[1:]:
    winner = class_segm_val_AP[metrics].argmax()
    winner_val = class_segm_val_AP[metrics].max()
    class_segm_val_AP_legend.append(metrics + ' (Best: {}, Epoch: {})'.format(winner_val, winner + 1))
    if winner in class_segm_val_AP_winners:
        continue
    else:
        class_segm_val_AP_winners.append(winner)

# generating net winners list
in_bbox = set(bbox_val_AP_winners)
class_in_bbox = set(class_bbox_val_AP_winners)
in_seg = set(seg_val_AP_winners)
class_in_seg = set(class_segm_val_AP_winners)
unique_seg = in_seg - in_bbox
class_unique_seg = class_in_seg - class_in_bbox
net_val_AP_winners = bbox_val_AP_winners + list(unique_seg)
class_net_val_AP_winners = class_bbox_val_AP_winners + list(class_unique_seg)

# creating line plots for bbox and segm AP results
# sns.set(font_scale=2)
sns.set_style("whitegrid")
bbox_val_AP = bbox_val_AP.melt('epoch', var_name='cols', value_name='vals')
g = sns.catplot(x="epoch", y="vals", hue='cols', data=bbox_val_AP, kind='point', facet_kws={'legend_out': True})
for winner in bbox_val_AP_winners:
    plt.plot([winner, winner], [0, max(bbox_val_AP['vals'])], color='red', linestyle='dashed')
g.set(ylim=(0, None))
g.set(xlim=(0, None))
g.set(title='Bounding Box Validation Performance (Precision)')
g.set(ylabel='Score')
g.set(xlabel='Validation Round')
g.legend.set_title('Metrics')
for t, l in zip(g.legend.texts, bbox_val_AP_legend):
    t.set_text(l)
    t.set_fontsize('8')
g.legend.set_bbox_to_anchor((1.15, 0.5))
plt.savefig(RESULTS_DIR + "bbox_val_AP_line.png", bbox_inches='tight', pad_inches=0.2)

sns.set_style("whitegrid")
seg_val_AP = seg_val_AP.melt('epoch', var_name='cols',  value_name='vals')
g = sns.catplot(x="epoch", y="vals", hue='cols', data=seg_val_AP, kind='point', facet_kws={'legend_out': True})
for winner in seg_val_AP_winners:
    plt.plot([winner, winner], [0, max(seg_val_AP['vals'])], color='red', linestyle='dashed')
g.set(ylim=(0, None))
g.set(xlim=(0, None))
g.set(title='Segmentation Validation Performance (Precision)')
g.set(ylabel='Score')
g.set(xlabel='Validation Round')
g.legend.set_title('Metrics')
for t, l in zip(g.legend.texts, seg_val_AP_legend):
    t.set_text(l)
    t.set_fontsize('8')
g.legend.set_bbox_to_anchor((1.15, 0.5))
g.savefig(RESULTS_DIR + "seg_val_AP_line.png", bbox_inches='tight', pad_inches=0.2)

# creating line plots for classwise bbox and segm AP results
sns.set_style("whitegrid")
class_bbox_val_AP = class_bbox_val_AP.melt('epoch', var_name='cols', value_name='vals')
g = sns.catplot(x="epoch", y="vals", hue='cols', data=class_bbox_val_AP, kind='point', facet_kws={'legend_out': True})
for winner in class_bbox_val_AP_winners:
    plt.plot([winner, winner], [0, max(class_bbox_val_AP['vals'])], color='red', linestyle='dashed')
g.set(ylim=(0, None))
g.set(xlim=(0, None))
g.set(title='Bounding Box Validation Performance (Precision)')
g.set(ylabel='Score')
g.set(xlabel='Validation Round')
g.legend.set_title('Metrics')
for t, l in zip(g.legend.texts, class_bbox_val_AP_legend):
    t.set_text(l)
    t.set_fontsize('8')
g.legend.set_bbox_to_anchor((1.15, 0.5))
g.savefig(RESULTS_DIR + "class_bbox_val_AP_line.png", bbox_inches='tight', pad_inches=0.3)

sns.set_style("whitegrid")
class_segm_val_AP = class_segm_val_AP.melt('epoch', var_name='cols',  value_name='vals')
g = sns.catplot(x="epoch", y="vals", hue='cols', data=class_segm_val_AP, kind='point', facet_kws={'legend_out': True})
for winner in class_segm_val_AP_winners:
    plt.plot([winner, winner], [0, max(class_segm_val_AP['vals'])], color='red', linestyle='dashed')
g.set(ylim=(0, None))
g.set(xlim=(0, None))
g.set(title='Segmentation Validation Performance (Precision)')
g.set(ylabel='Score')
g.set(xlabel='Validation Round')
g.legend.set_title('Metrics')
for t, l in zip(g.legend.texts, class_segm_val_AP_legend):
    t.set_text(l)
    t.set_fontsize('8')
g.legend.set_bbox_to_anchor((1.15, 0.5))
g.savefig(RESULTS_DIR + "class_seg_val_AP_line.png", bbox_inches='tight', pad_inches=0.3)
