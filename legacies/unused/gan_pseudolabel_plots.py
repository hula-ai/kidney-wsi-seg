import os
import json
import statistics

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycocotools.mask as mask
from tqdm import tqdm


"""
This script is an adaptation of the script Pietro provided for creating the violin plots of pseudo annotatino
sizes and counts. Useful for visualizing the balance between pseudo-annotation classes.
"""


def polygonFromMask(maskedArr):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:  # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        raise ValueError('No polygons.')
        # return [[]]
    return segmentation


# Define parts of COCO dataset dictionary
categories = [
    {
        "id": 0,
        "name": "Background",
        "supercategory": "Background"
    },
    {
        "id": 1,
        "name": "Glomerulus",
        "supercategory": "Compartment"
    },
    {
        "id": 2,
        "name": "Arteriole",
        "supercategory": "Compartment"
    },
    {
        "id": 3,
        "name": "Artery",
        "supercategory": "Compartment"
    },
]

MODEL_DIR = "/data/syed/mmdet/results/run19_ep2_gan25k/"  # ToDo: Specify directory, plots will be saved to here
mmdet_segm_test_path = "/data/syed/mmdet/results/run19_ep2_25k_json_results.segm.json"  # ToDo: Specify path to pseudo annotation json file

# Filtering cutoffs obtained from confidence clustering script
FILTERING_CUTOFFS = {'Glomerulus': 0.5723221302032471, 'Arteriole': 0.3031373500823975, 'Artery': 0.5274305939674377}
FILTERING_CUTOFF_LIST = [0, 0.5723221302032471, 0.3031373500823975, 0.5274305939674377]

# Load mmdetection test pseudoannotations
with open(mmdet_segm_test_path) as f:
    segm_data = json.load(f)
print("Annotation length before filtering:", len(segm_data))

# Filter
mmdet_annot_filtered = []
for annot in segm_data:
    if annot["score"] > FILTERING_CUTOFF_LIST[annot["category_id"]]:
        mmdet_annot_filtered.append(annot)
segm_data = mmdet_annot_filtered  # Reset variable name
print("Annotation length after filtering:", len(segm_data))

# Need key "area" for each annotation, compute here
segm_data_with_area = []
for annot in tqdm(segm_data):
    try:
        maskedArr = mask.decode(annot["segmentation"])
        segm_polygon = polygonFromMask(maskedArr)
        area = float((maskedArr > 0.0).sum())
        annot["segmentation"] = segm_polygon
        annot["area"] = area
        segm_data_with_area.append(annot)
    except ValueError as ve:
        # print("No polygons for annotation")
        pass
segm_data = segm_data_with_area
print("List length after calculating area for each annotation:", len(segm_data))


segm_annot_count = {mykey['id']: 0 for mykey in categories}
segm_annot_area = {mykey['id']: [] for mykey in categories}
segm_score_count = {mykey['id']: [] for mykey in categories}
for annot in segm_data:
    segm_annot_count[annot['category_id']] += 1
    segm_score_count[annot['category_id']].append(annot['score'])
    segm_annot_area[annot['category_id']].append(annot['area'])

for key in range(0, len(segm_annot_count.keys())):
    if segm_annot_area[key] == []:
        del segm_annot_area[key]
        del segm_annot_count[key]
        categories = [ele for ele in categories if ele['id'] != key]
print(categories)
print(len(segm_annot_count.keys()))

segm_annot_mean = {mykey['id']: statistics.mean(segm_score_count[mykey['id']]) for mykey in categories}
segm_area_mean = {mykey['id']: statistics.mean(segm_annot_area[mykey['id']]) for mykey in categories}
segm_annot_std = {mykey['id']: statistics.stdev(segm_score_count[mykey['id']]) for mykey in categories}
segm_annot_max = {mykey['id']: max(segm_score_count[mykey['id']]) for mykey in categories}
segm_annot_min = {mykey['id']: min(segm_score_count[mykey['id']]) for mykey in categories}

print('Predicted annotations class counts: ' + str(segm_annot_count))
print('Predicted annotations mask area averages: ' + str(segm_area_mean))
print('Predicted annotations score averages: ' + str(segm_annot_mean))
print('Predicted annotations score std: ' + str(segm_annot_std))
print('Predicted annotations score max: ' + str(segm_annot_max))
print('Predicted annotations score min: ' + str(segm_annot_min))
segm_annot_count = {mykey['name']: segm_annot_count[mykey['id']] for mykey in categories}

for i, key in enumerate(list(segm_annot_area.keys())):
    segm_annot_area[list(segm_annot_count.keys())[i] + ': {}'.format(segm_annot_count[list(segm_annot_count.keys())[i]])] = segm_annot_area.pop(key)
    segm_score_count[list(segm_annot_count.keys())[i] + ': {}'.format( segm_annot_count[list(segm_annot_count.keys())[i]])] = segm_score_count.pop(key)

# generating violin plots describing predicted annotation counts, size per class
plt.figure()
segm_annot_area = pd.DataFrame.from_dict(segm_annot_area, orient='index')
segm_annot_area = segm_annot_area.T
segm_annot_area = segm_annot_area.melt()
segm_annot_area = segm_annot_area.dropna()
p = sns.boxplot(x='variable', y='value', data=segm_annot_area)
p.get_figure().set_figwidth(0.7 * len(segm_annot_mean))
p.set(ylim=(0, 100000))
p.set(title='Pred. Area/Counts Per Class')
p.set(ylabel='Area (Seg. Mask Pixels)')
p.set(xlabel='Class: Counts')
p.set_xticklabels(p.get_xticklabels(), rotation=45)
plt.tight_layout()
p.get_figure().savefig(MODEL_DIR + "gan_25k_FILT_pred_annotations_sizes_counts.png")

# generating violin plots describing predicted annotation counts, scores per class
plt.figure()
segm_score_count = pd.DataFrame.from_dict(segm_score_count, orient='index')
segm_score_count = segm_score_count.T
segm_score_count = segm_score_count.melt()
segm_score_count = segm_score_count.dropna()
ps = sns.violinplot(x='variable', y='value', data=segm_score_count)
ps.get_figure().set_figwidth(0.7 * len(segm_annot_mean))
ps.set(ylim=(0, None))
ps.set(title='Pred. Scores/Counts Per Class')
ps.set(ylabel='Score')
ps.set(xlabel='Class: Counts')
ps.set_xticklabels(ps.get_xticklabels(), rotation=45)
plt.tight_layout()
ps.get_figure().savefig(MODEL_DIR + "gan_25k_FILT_pred_annotations_scores_counts.png")

# delete 'score', adjust ids based on the maximum GT id value to avoid repeat ids, set 'iscrowd' based on seg mask format (0 = polygon)
# id_add = max(data['annotations'], key=lambda ev: ev['id'])['id']
# for i in range(len(segm_data)):
#     segm_data[i]['image_id'] = segm_data[i]['image_id'] + len(data['images'])
#     segm_data[i]['iscrowd'] = 0
#     segm_data[i]['id'] = id_add + i + 1
#     del segm_data[i]['score']

"""
Outputs for GAN 25k images run 11 epoch 4 autoaug:
{'Glomerulus': 0.5822800993919373, 'Arteriole': 0.24685976803302764, 'Artery': 0.5061547160148621}
[{'id': 1, 'name': 'Glomerulus', 'supercategory': 'Compartment'}, {'id': 2, 'name': 'Arteriole', 'supercategory': 'Compartment'}, {'id': 3, 'name': 'Artery', 'supercategory': 'Compartment'}]
3
Predicted annotations class counts: {1: 61863, 2: 24849, 3: 14868}
Predicted annotations mask area averages: {1: 408503.7184423646, 2: 69716.16109300173, 3: 849027.1762173796}
Predicted annotations score averages: {1: 0.9577606243831547, 2: 0.46846704883922574, 3: 0.8421612111052295}
Predicted annotations score std: {1: 0.0824473120893427, 2: 0.18075713444196073, 3: 0.14273590035292677}
Predicted annotations score max: {1: 0.9989338517189026, 2: 0.9861884713172913, 3: 0.9974004030227661}
Predicted annotations score min: {1: 0.5823073387145996, 2: 0.2468603253364563, 3: 0.5062583684921265}


Outputs for GAN 25k images run 19 epoch 2 autoaug
FILTERING_CUTOFFS = {'Glomerulus': 0.5723221302032471, 'Arteriole': 0.3031373500823975, 'Artery': 0.5274305939674377}

List length after calculating area for each annotation: 125494
[{'id': 1, 'name': 'Glomerulus', 'supercategory': 'Compartment'}, {'id': 2, 'name': 'Arteriole', 'supercategory': 'Compartment'}, {'id': 3, 'name': 'Artery', 'supercategory': 'Compartment'}]
3
Predicted annotations class counts: {1: 64112, 2: 43109, 3: 18273}
Predicted annotations mask area averages: {1: 380439.99043860746, 2: 63480.563362638895, 3: 813934.4956493187}
Predicted annotations score averages: {1: 0.9535911689885007, 2: 0.5850455903271942, 3: 0.8706843966086797}
Predicted annotations score std: {1: 0.09312808722203361, 2: 0.19559299130850324, 3: 0.14327451709321617}
Predicted annotations score max: {1: 0.9995847344398499, 2: 0.993980884552002, 3: 0.9993686079978943}
Predicted annotations score min: {1: 0.5723705887794495, 2: 0.303139328956604, 3: 0.5274444222450256}

"""

