import json
import cv2
import statistics
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# This is the original file sent by Pietro containing some code for handling test output json file. Don't use this file, the code has been adapted into other files

MODEL_DIR = ""
# '/lhome1/c-pcicalese/results/nuimage/ins_seg/SuperNet/CascadeRCNN/7_13_2021_resnext101_baseline/epoch_23/'
SEGM = 'run11_ep4_json_results.segm.json'  # 'segm_init_clean.json'
TEST_DUMMY = '/lhome1/c-pcicalese/datasets/nuimage/annotations/instances_nuimage_dummy_coco_rm_no_mask_mmdetection.json'
TRAIN = '/lhome1/c-pcicalese/datasets/nuimage/annotations/instances_nuimage_train_coco_rm_no_mask_mmdetection.json'
DATA = '/lhome1/c-pcicalese/datasets/nuimage/'

###########################################
######## AUXILLARY FUNCTIONS ##############
###########################################
# function to convert RLE to polygon format, mmdetection outputs are encoded as RLE for some reason
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
    return segmentation

###########################################
############## GT DATA LOAD ###############
###########################################
# load the GT annotation file
with open(TRAIN) as data_file:
    data = json.load(data_file)


###########################################
######## PRED DATA CHECK ##################
###########################################
# load the inference dummy .json file, add the length of the GT annotation 'images' to the 'id', avoid repeat id values
with open(TEST_DUMMY) as data_file:
    dummy_data = json.load(data_file)
    for i in range(len(dummy_data['images'])):
        dummy_data['images'][i]['id'] = dummy_data['images'][i]['id'] + len(data['images'])

# load the mmdetection segmentation output json file, filter annotations, adjust 'id', remove 'score' once filtered
# convert annotations to polygons (or just match to GT), compute area of masks
with open(MODEL_DIR + SEGM) as data_file:
    segm_data = json.load(data_file)

# create counts of the predicted annotations, evaluate average score per class, generate stats, generate stat plots
segm_annot_count = {mykey['id']: 0 for mykey in data['categories']}
segm_annot_area = {mykey['id']: [] for mykey in data['categories']}
segm_score_count = {mykey['id']: [] for mykey in data['categories']}
for annot in segm_data:
    segm_annot_count[annot['category_id']] += 1
    segm_score_count[annot['category_id']].append(annot['score'])
    segm_annot_area[annot['category_id']].append(annot['area'])

for key in range(1, len(segm_annot_count.keys())):
    if segm_annot_area[key] == []:
        del segm_annot_area[key]
        del segm_annot_count[key]
    data['categories'] = [ele for ele in data['categories'] if ele['id'] != key]

segm_annot_mean = {mykey['id']: statistics.mean(segm_score_count[mykey['id']]) for mykey in data['categories']}
segm_area_mean = {mykey['id']: statistics.mean(segm_annot_area[mykey['id']]) for mykey in data['categories']}
segm_annot_std = {mykey['id']: statistics.stdev(segm_score_count[mykey['id']]) for mykey in data['categories']}
segm_annot_max = {mykey['id']: max(segm_score_count[mykey['id']]) for mykey in data['categories']}
segm_annot_min = {mykey['id']: min(segm_score_count[mykey['id']]) for mykey in data['categories']}

print('Predicted annotations class counts: ' + str(segm_annot_count))
print('Predicted annotations mask area averages: ' + str(segm_area_mean))
print('Predicted annotations score averages: ' + str(segm_annot_mean))
print('Predicted annotations score std: ' + str(segm_annot_std))
print('Predicted annotations score max: ' + str(segm_annot_max))
print('Predicted annotations score min: ' + str(segm_annot_min))
segm_annot_count = {mykey['name']: segm_annot_count[mykey['id']] for mykey in data['categories']}

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
    p.get_figure().savefig(MODEL_DIR + "ins_seg_FILT_pred_annotations_sizes_counts.png")

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
    ps.get_figure().savefig(MODEL_DIR + "ins_seg_FILT_pred_annotations_scores_counts.png")
    # delete 'score', adjust ids based on the maximum GT id value to avoid repeat ids, set 'iscrowd' based on seg mask format (0 = polygon)
    id_add = max(data['annotations'], key=lambda ev: ev['id'])['id']
    for i in range(len(segm_data)):
        segm_data[i]['image_id'] = segm_data[i]['image_id'] + len(data['images'])
        segm_data[i]['iscrowd'] = 0
        segm_data[i]['id'] = id_add + i + 1
        del segm_data[i]['score']

###########################################
### FINAL KD ANNOT. FILE GEN ##############
###########################################
# load the GT annotation file
with open(TRAIN) as data_file:
    data = json.load(data_file)

# saving the coarse only annotations file
coarse_data = dummy_data.copy()
coarse_data['annotations'] = segm_data
with open(MODEL_DIR + 'KD_160k_train_pseudolabel_on_unlabeled_and_test_coarse.json', 'w') as outfile:
    json.dump(coarse_data, outfile)

# append the pred image names to the data file
for i in range(len(dummy_data['images'])):
    data['images'].append(dummy_data['images'][i])
# append the pred annotations to the data file
for i in range(len(segm_data)):
    data['annotations'].append(segm_data[i])
# write the final KD json file
with open(MODEL_DIR + 'KD_160k_train_pseudolabel_on_unlabeled_and_test_with_train.json', 'w') as outfile:
    json.dump(data, outfile)
