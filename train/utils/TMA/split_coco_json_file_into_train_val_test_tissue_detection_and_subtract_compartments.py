import os
import json
import mmcv
import numpy as np
import pandas as pd


# normal
# all_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4.json')
# train_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_train_701515.json')
# val_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_val_701515.json')
# test_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_test_701515.json')
# tissue detection and subtract smaller compartments
all_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_tissue_detection_and_subtract_smaller_compartments.json')
train_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_tissue_detection_and_subtract_smaller_compartments_train_701515_reduced.json')
val_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_tissue_detection_and_subtract_smaller_compartments_val_701515_reduced.json')
test_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_tissue_detection_and_subtract_smaller_compartments_test_701515_reduced.json')

# test_path = os.path.join(data_path, 'coco_tma_tile_validation_fold0_fixed.json')

# f = open(train_path)
with open(all_path) as f:
  data = json.load(f)
for key, value in data.items():
  print('[*] key: ', key, ', type: ', type(value), ', len(value): ', len(value))

train_images, train_annotations = [], []
val_images, val_annotations = [], []
test_images, test_annotations = [], []
n_slide_ids = 551 # 60/20/20 = 331/110/110
                  # 70/15/15 = 386/82/83
# img_pnt = 0
anno_len = len(data['annotations'])
cur_slide_id = None
count_slide_id = 0
count_crop_id = 0
dict_of_mapping_from_cropid_to_split = {}
for image in data['images'][::50]:
  slide_id = image['file_name'].split('/')[-2]
  crop_id = image['id']
  # print('[*] crop_id: ', crop_id)
  count_crop_id += 1
  if cur_slide_id is None or slide_id != cur_slide_id:
    print('[*] slide_id', slide_id)
    count_slide_id += 1
    cur_slide_id = slide_id
  if count_slide_id <= 386:
    train_images.append(image)
    dict_of_mapping_from_cropid_to_split[crop_id] = 'train'
  elif count_slide_id > 386 and count_slide_id <= 468:
    val_images.append(image)
    dict_of_mapping_from_cropid_to_split[crop_id] = 'val'
  else:
    test_images.append(image)
    dict_of_mapping_from_cropid_to_split[crop_id] = 'test'

images_have_mask = []
list_of_annotations = data['annotations']
sorted_list_of_annotations = sorted(list_of_annotations, key=lambda x: x['image_id']) # sort because the if .. else .. (line 383 - 400) in `create_coco_jon_file_2_multiproc.py`
dict_of_counting_bboxes_of_each_crop = {}
for anno_idx in range(anno_len):
  print('[*] anno_idx: {}/{}'.format(anno_idx, anno_len))
  corr_crop_id = sorted_list_of_annotations[anno_idx]['image_id']
  if corr_crop_id not in dict_of_mapping_from_cropid_to_split: # currently bring this if before the following if
    continue # only need this for a smaller version of json files (line 38 [::100]), mmdet cannot run on large json files
  if len(images_have_mask) == 0 or corr_crop_id != images_have_mask[-1]:
    images_have_mask.append(corr_crop_id)
  if corr_crop_id not in dict_of_counting_bboxes_of_each_crop:
    dict_of_counting_bboxes_of_each_crop[corr_crop_id] = 0
  dict_of_counting_bboxes_of_each_crop[corr_crop_id] += 1
  # handle the problem of some crops that do not have any gt mask (maybe some white crops originally denoted as cortex now do not have any mask)
  # there are cases that corr_crop_id decreases -> more generalized implementation
  if dict_of_mapping_from_cropid_to_split[corr_crop_id] == 'train':
    train_annotations.append(sorted_list_of_annotations[anno_idx])
  elif dict_of_mapping_from_cropid_to_split[corr_crop_id] == 'val':
    val_annotations.append(sorted_list_of_annotations[anno_idx])
  elif dict_of_mapping_from_cropid_to_split[corr_crop_id] == 'test':
    test_annotations.append(sorted_list_of_annotations[anno_idx])
print('[!] Complete annotation loop')

# TODO: remove images that do not have any mask (white region)
for idx, image_have_mask in enumerate(images_have_mask):
  if idx == len(images_have_mask) - 1:
    break
  if image_have_mask > images_have_mask[idx+1]:
    print('=> List images_have_mask does not increase')
    print(image_have_mask, ' ', images_have_mask[idx+1])
    break

for idx, train_image in enumerate(train_images):
  if idx == len(train_images) - 1:
    break
  if train_image['id'] > train_images[idx+1]['id']:
    print('=> Train list does not increase')
    break
for idx, val_image in enumerate(val_images):
  if idx == len(val_images) - 1:
    break
  if val_image['id'] > val_images[idx+1]['id']:
    print('=> Val list does not increase')
    break
for idx, test_image in enumerate(test_images):
  if idx == len(test_images) - 1:
    break
  if test_image['id'] > test_images[idx+1]['id']:
    print('=> Test list does not increase')
    break
print('==> All tests complete')

pnt = 0
new_train_images , new_val_images, new_test_images = [], [], []
for idx, train_image in enumerate(train_images):
  print('In train loop {}'.format(train_image['id']))
  print('images_have_mask[pnt]: ', images_have_mask[pnt])
  if train_image['id'] == images_have_mask[pnt]:
    new_train_images.append(train_image)
    # images_have_mask.remove(train_image['id'])
    pnt += 1
for idx, val_image in enumerate(val_images):
  print('In val loop {}'.format(val_image['id']))
  print('images_have_mask[pnt]: ', images_have_mask[pnt])
  if val_image['id'] == images_have_mask[pnt]:
    new_val_images.append(val_image)
    # images_have_mask.remove(val_image['id'])
    pnt += 1
for idx, test_image in enumerate(test_images):
  if pnt >= len(images_have_mask):
    break
  print('In test loop {}'.format(test_image['id']))
  print('images_have_mask[pnt]: ', images_have_mask[pnt])
  if test_image['id'] == images_have_mask[pnt]:
    new_test_images.append(test_image)
    # images_have_mask.remove(test_image['id'])
    pnt += 1
train_images = new_train_images
val_images = new_val_images
test_images = new_test_images

# REMOVE LATER - CREATE A SMALL SUBSET
# # --- train ---
# train_images = train_images[:10] 
# new_train_annotations = []
# crop_ids = []
# for train_image in train_images:
#   crop_ids.append(train_image['id'])
# for train_annotation in train_annotations:
#   if train_annotation['image_id'] in crop_ids:
#     new_train_annotations.append(train_annotation)
# train_annotations = new_train_annotations
# # --- val ---
# val_images = val_images[:10] 
# new_val_annotations = []
# crop_ids = []
# for val_image in val_images:
#   crop_ids.append(val_image['id'])
# for val_annotation in val_annotations:
#   if val_annotation['image_id'] in crop_ids:
#     new_val_annotations.append(val_annotation)
# val_annotations = new_val_annotations
# # --- test ---
# test_images = test_images[:10] 
# new_test_annotations = []
# crop_ids = []
# for test_image in test_images:
#   crop_ids.append(test_image['id'])
# for test_annotation in test_annotations:
#   if test_annotation['image_id'] in crop_ids:
#     new_test_annotations.append(test_annotation)
# test_annotations = new_test_annotations

print('[*] len(train_images): ', len(train_images))
print('[*] len(val_images): ', len(val_images))
print('[*] len(test_images): ', len(test_images))
print('[*] len(train_annotations): ', len(train_annotations))
print('[*] len(val_annotations): ', len(val_annotations))
print('[*] len(test_annotations): ', len(test_annotations))
print('-> Total slide ids: ', count_slide_id)
print('-> Total crop ids: ', count_crop_id)
print('-> Total images in all splits: ', len(train_images)+len(val_images)+len(test_images))
print('-> Total annotations in all splits: ', len(train_annotations)+len(val_annotations)+len(test_annotations))
for key, val in dict_of_counting_bboxes_of_each_crop.items():
  print('Crop id {} has {} bboxes.'.format(key, val))
# === Delete crops that have more than 50 annotations ===
# train
refined_train_images = []
for train_image in train_images:
  if dict_of_counting_bboxes_of_each_crop[train_image['id']] <= 50:
    refined_train_images.append(train_image)
refined_train_annotations = []
for train_annotation in train_annotations:
  if dict_of_counting_bboxes_of_each_crop[train_annotation['image_id']] <= 50:
    refined_train_annotations.append(train_annotation)
# val
refined_val_images = []
for val_image in val_images:
  if dict_of_counting_bboxes_of_each_crop[val_image['id']] <= 50:
    refined_val_images.append(val_image)
refined_val_annotations = []
for val_annotation in val_annotations:
  if dict_of_counting_bboxes_of_each_crop[val_annotation['image_id']] <= 50:
    refined_val_annotations.append(val_annotation)
# test 
refined_test_images = []
for test_image in test_images:
  if dict_of_counting_bboxes_of_each_crop[test_image['id']] <= 50:
    refined_test_images.append(test_image)
refined_test_annotations = []
for test_annotation in test_annotations:
  if dict_of_counting_bboxes_of_each_crop[test_annotation['image_id']] <= 50:
    refined_test_annotations.append(test_annotation)
print('[*] len(refined_train_images): ', len(refined_train_images))
print('[*] len(refined_val_images): ', len(refined_val_images))
print('[*] len(refined_test_images): ', len(refined_test_images))
print('[*] len(refined_train_annotations): ', len(refined_train_annotations))
print('[*] len(refined_val_annotations): ', len(refined_val_annotations))
print('[*] len(refined_test_annotations): ', len(refined_test_annotations))
print('-> Total images in all splits: ', len(refined_train_images)+len(refined_val_images)+len(refined_test_images))
print('-> Total annotations in all splits: ', len(refined_train_annotations)+len(refined_val_annotations)+len(refined_test_annotations))
# guarantee 6 classes exist in each class

train_coco_format_json = dict(
    info={
      'description': 'HULA TMA 4096 Tile Dataset Segmentation 6 classes - train',
      'version': '1.0',
      'year': 2023,
      'contributor': 'hqvo3',
      'date created': '2023/02/18'
    },
    images=refined_train_images,
    annotations=refined_train_annotations,
    categories=[{'id':0, 'name': 'Background', 'supercategory': 'Background'},
                {'id':1, 'name': 'Cortex', 'supercategory': 'Compartment'},
                {'id':2, 'name': 'Glomerulus', 'supercategory': 'Compartment'},
                {'id':3, 'name': 'Arteriole', 'supercategory': 'Compartment'},
                {'id':4, 'name': 'Artery', 'supercategory': 'Compartment'},
                {'id':5, 'name': 'Medulla', 'supercategory': 'Compartment'},
                {'id':6, 'name': 'CapsuleOther', 'supercategory': 'Compartment'}])
val_coco_format_json = dict(
    info={
      'description': 'HULA TMA 4096 Tile Dataset Segmentation 6 classes - val',
      'version': '1.0',
      'year': 2023,
      'contributor': 'hqvo3',
      'date created': '2023/02/18'
    },
    images=refined_val_images,
    annotations=refined_val_annotations,
    categories=[{'id':0, 'name': 'Background', 'supercategory': 'Background'},
                {'id':1, 'name': 'Cortex', 'supercategory': 'Compartment'},
                {'id':2, 'name': 'Glomerulus', 'supercategory': 'Compartment'},
                {'id':3, 'name': 'Arteriole', 'supercategory': 'Compartment'},
                {'id':4, 'name': 'Artery', 'supercategory': 'Compartment'},
                {'id':5, 'name': 'Medulla', 'supercategory': 'Compartment'},
                {'id':6, 'name': 'CapsuleOther', 'supercategory': 'Compartment'}])
test_coco_format_json = dict(
    info={
      'description': 'HULA TMA 4096 Tile Dataset Segmentation 6 classes - test',
      'version': '1.0',
      'year': 2023,
      'contributor': 'hqvo3',
      'date created': '2023/02/18'
    },
    images=refined_test_images,
    annotations=refined_test_annotations,
    categories=[{'id':0, 'name': 'Background', 'supercategory': 'Background'},
                {'id':1, 'name': 'Cortex', 'supercategory': 'Compartment'},
                {'id':2, 'name': 'Glomerulus', 'supercategory': 'Compartment'},
                {'id':3, 'name': 'Arteriole', 'supercategory': 'Compartment'},
                {'id':4, 'name': 'Artery', 'supercategory': 'Compartment'},
                {'id':5, 'name': 'Medulla', 'supercategory': 'Compartment'},
                {'id':6, 'name': 'CapsuleOther', 'supercategory': 'Compartment'}])

mmcv.dump(train_coco_format_json, train_path)
mmcv.dump(val_coco_format_json, val_path)
mmcv.dump(test_coco_format_json, test_path)