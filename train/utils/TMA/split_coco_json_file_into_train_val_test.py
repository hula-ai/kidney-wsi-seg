import os
import json
import mmcv
import numpy as np
import pandas as pd


# normal
all_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4.json')
train_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_train_701515.json')
val_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_val_701515.json')
test_path = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_test_701515.json')

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
anno_pnt = 0
anno_len = len(data['annotations'])
cur_slide_id = None
count_slide_id = 0
for image in data['images']:
  slide_id = image['file_name'].split('/')[-2]
  print('[*] slide_id', slide_id)
  print('[*] count_slide_id', count_slide_id)
  crop_id = image['id']
  if cur_slide_id is None or slide_id != cur_slide_id:
    count_slide_id += 1
    cur_slide_id = slide_id
  if count_slide_id <= 386:
    train_images.append(image)
  elif count_slide_id > 386 and count_slide_id <= 468:
    val_images.append(image)
  else:
    test_images.append(image)
  while anno_pnt < anno_len:
    print('[*] anno_pnt: ', anno_pnt)
    corr_crop_id = data['annotations'][anno_pnt]['image_id']
    if corr_crop_id == crop_id:
      if count_slide_id <= 386:
        train_annotations.append(data['annotations'][anno_pnt])
      elif count_slide_id > 386 and count_slide_id <= 468:
        val_annotations.append(data['annotations'][anno_pnt])
      else:
        test_annotations.append(data['annotations'][anno_pnt])
      anno_pnt += 1
    else:
      break

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
# guarantee 6 classes exist in each class

train_coco_format_json = dict(
    info={
      'description': 'HULA TMA 4096 Tile Dataset Segmentation 6 classes - train',
      'version': '1.0',
      'year': 2023,
      'contributor': 'hqvo3',
      'date created': '2023/02/18'
    },
    images=train_images,
    annotations=train_annotations,
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
    images=val_images,
    annotations=val_annotations,
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
    images=test_images,
    annotations=test_annotations,
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