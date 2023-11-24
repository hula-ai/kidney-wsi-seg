import os
import json

import numpy as np
from mmdet.apis import init_detector, inference_detector
import mmcv


INPUT_PATH = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_512/coco_tma_tile_rm4_train_701515.json'
# Opening JSON file
f = open(INPUT_PATH)
  
# returns JSON object as 
# a dictionary
data = json.load(f)

print(type(data))
print('=== KEYS ===')
for key, val in data.items():
  print(key)

print()
print('--- IMAGES ---')
# print(data['images'])
print(type(data['images']), ' ', len(data['images']))
print("- first element - data['images'][0]: ", data['images'][0])
print("- second element - data['images'][1]: ", data['images'][1])
print("- third element - data['images'][2]: ", data['images'][2])
print("- keys of data['images'][0]: ", data['images'][0].keys())

print()
print('--- ANNOTATIONS ---')
# print(data['annotations'])
print(type(data['annotations']), ' ', len(data['annotations']))
print("- first element - data['annotations][0]: ", data['annotations'][0])
print("- second element - data['annotations][1]: ", data['annotations'][1])
print("- third element - data['annotations][2]: ", data['annotations'][2])
print("- keys of data['annotations'][0]: ", data['annotations'][0].keys())

print()
print('--- CATEGORIES ---')
for catego in data['categories']:
  print(catego)
print(type(data['categories']), ' ', len(data['categories']))

# each image crop may have more than one category
dict_of_mapping_from_imageid_to_categoryids = {}
# each image crop may have more than one contour, but only one category
dict_of_mapping_from_imageid_to_segmentations = {}
for anno in data['annotations']:
  imageid = anno['image_id']
  categoryid = anno['category_id']
  annoid = anno['id']
  if imageid not in dict_of_mapping_from_imageid_to_categoryids: 
    dict_of_mapping_from_imageid_to_categoryids[imageid] = []
  dict_of_mapping_from_imageid_to_categoryids[imageid].append(categoryid)
  if imageid not in dict_of_mapping_from_imageid_to_segmentations: 
    dict_of_mapping_from_imageid_to_segmentations[imageid] = []
  dict_of_mapping_from_imageid_to_segmentations[imageid].append(annoid)

# count
uniq_list_of_num_categoryids = []
for key, val in dict_of_mapping_from_imageid_to_categoryids.items():
  uniq_list_of_num_categoryids.append(len(set(val))) # try remove set() and 2 printings below show [1..41] without 40 (1)
uniq_list_of_num_categoryids = set(uniq_list_of_num_categoryids)
uniq_list_of_num_segmentations = []
for key, val in dict_of_mapping_from_imageid_to_segmentations.items():
  uniq_list_of_num_segmentations.append(len(set(val))) # try remove set() and 2 printings below show [1..41] without 40 (2)
uniq_list_of_num_segmentations = set(uniq_list_of_num_segmentations)
# => From (1)(2), it means each crop may have more than one annotations, and these annotations may belong to different category ids
# => When using set(), (2) still shows [1..41] without 40 because each anno id is unique, while (1) only shows [1, 2, 3, 4, 5, 6] because we have 6 compartment classes
print()
print('--> uniq_list_of_num_categoryids: ', uniq_list_of_num_categoryids)
print('--> uniq_list_of_num_segmentations: ', uniq_list_of_num_segmentations)

# --- create dict of mapping from category ids to image_id (crop id) ---
dict_of_mapping_from_categoryids_to_imageids = {}
uniq_list_of_sorted_categoryids = []
for key, val in dict_of_mapping_from_imageid_to_categoryids.items():
  sorted_categoryids = list(set(val))
  sorted_categoryids.sort()
  for idx, catego in enumerate(sorted_categoryids):
    sorted_categoryids[idx] = str(catego)
  str_sorted_categoryids = '_'.join(sorted_categoryids)

  if str_sorted_categoryids not in dict_of_mapping_from_categoryids_to_imageids:
    dict_of_mapping_from_categoryids_to_imageids[str_sorted_categoryids] = []
  dict_of_mapping_from_categoryids_to_imageids[str_sorted_categoryids].append(key)
print()
print('--> all str_sorted_categoryids: ', dict_of_mapping_from_categoryids_to_imageids.keys())
# print("--> dict_of_mapping_from_categoryids_to_imageids['1']: ", dict_of_mapping_from_categoryids_to_imageids['1'])
# --- create dict of mapping from image_id to file_name (file path) ---
dict_of_mapping_from_imageid_to_filename = {}
for crop_info in data['images']:
  dict_of_mapping_from_imageid_to_filename[crop_info['id']] = crop_info['file_name']
print()
# for imageid in dict_of_mapping_from_categoryids_to_imageids['1'][:5]:
#   print('[*] imageid: ', imageid)
#   print('- corr file name: ', dict_of_mapping_from_imageid_to_filename[imageid])


print('[*] #crop ids have single class 1: ', len(dict_of_mapping_from_categoryids_to_imageids['1'])) # 1 means crops that have only one category 1
print('[*] #crop ids have single class 2: ', len(dict_of_mapping_from_categoryids_to_imageids['2'])) # 2 means crops that have only one category 2
print('[*] #crop ids have single class 3: ', len(dict_of_mapping_from_categoryids_to_imageids['3'])) # 3 means crops that have only one category 3
print('[*] #crop ids have single class 4: ', len(dict_of_mapping_from_categoryids_to_imageids['4'])) # 4 means crops that have only one category 4
print('[*] #crop ids have single class 5: ', len(dict_of_mapping_from_categoryids_to_imageids['5'])) # 5 means crops that have only one category 5
print('[*] #crop ids have single class 6: ', len(dict_of_mapping_from_categoryids_to_imageids['6'])) # 6 means crops that have only one category 6

# crop_id_list_1 = [0, 4500, 9000, 13000, 17500]
# crop_id_list_2 = [0, 30, 60, 100, 139]
# crop_id_list_3 = [0, 45, 85, 125, 175]
# crop_id_list_4 = [0, 20, 45, 55, 70]
# crop_id_list_5 = [0, 1000, 2000, 3000, 4000]
# crop_id_list_6 = [0, 1500, 2500, 3500, 4500]
crop_id_list_1 = range(0, 55000, 100)
crop_id_list_2 = range(0, 1000,10)
crop_id_list_3 = range(0, 720, 20)
crop_id_list_4 = range(0, 270, 2)
crop_id_list_5 = range(0, 44500, 200)
crop_id_list_6 = range(0, 30280, 150)

n_crops = len(crop_id_list_1)
imageids_dict = {}
imageids_dict['1'] = []
imageids_dict['2'] = []
imageids_dict['3'] = []
imageids_dict['4'] = []
imageids_dict['5'] = []
imageids_dict['6'] = []

for idx in range(len(crop_id_list_1)):
  imageids_dict['1'].append(dict_of_mapping_from_categoryids_to_imageids['1'][crop_id_list_1[idx]])

for idx in range(len(crop_id_list_2)):
  imageids_dict['2'].append(dict_of_mapping_from_categoryids_to_imageids['2'][crop_id_list_2[idx]])

for idx in range(len(crop_id_list_3)):
  imageids_dict['3'].append(dict_of_mapping_from_categoryids_to_imageids['3'][crop_id_list_3[idx]])

for idx in range(len(crop_id_list_4)):
  imageids_dict['4'].append(dict_of_mapping_from_categoryids_to_imageids['4'][crop_id_list_4[idx]])

for idx in range(len(crop_id_list_5)):
  imageids_dict['5'].append(dict_of_mapping_from_categoryids_to_imageids['5'][crop_id_list_5[idx]])
  
for idx in range(len(crop_id_list_6)):
  imageids_dict['6'].append(dict_of_mapping_from_categoryids_to_imageids['6'][crop_id_list_6[idx]])

# --- INIT SEGMENTATION ---
# Specify the path to model config and checkpoint file
config_file = './configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py'
# checkpoint_file = '/data/hqvo3/mmdet/run2/latest.pth' # MOANA
checkpoint_file = '/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/run_6_classes_ps512/latest.pth' # MAUI
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# for idx in range(n_crops):
for catego in imageids_dict.keys():
  imageids = imageids_dict[catego]
  for imageid in imageids:
    corr_filename = dict_of_mapping_from_imageid_to_filename[imageid] # corr file path
    img_crop_name = corr_filename.split('/')[-1]
    
    # --- START SEGMENTATION ---
    
    # INPUT_PATH = '/data/public/HULA/WSIs_renal_compartment_segmentations_tma_3classes_new_wsis/Autoseg_Glomerulus_img'
    # crop_list = os.listdir(INPUT_PATH)
    # Read json test file

    # out_folder = '/data/hqvo3/final_results/digital_pathology/MILxseg/inf_tma_model_on_tma_glom_patches/2' # MOANA
    out_folder = '/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/inf_tma_model_on_tma_glom_patches_test_split_json/10'
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, catego), exist_ok=True)

    # test a single image and show the results
    img_crop_path = corr_filename
    # print('- img crop path: ', img_crop_path)
    # img = os.path.join('/data/public/HULA/', img_crop_path)  # or img = mmcv.imread(img), which will only load it once
    try:
      result = inference_detector(model, img_crop_path)
      catego_int = int(catego)
      # print('--> result[0]: ', result[0])
      # print('--> result[1]: ', result[1])
      # print('--> type(result[0]): ', type(result[0]))
      # print('--> type(result[1]): ', type(result[1]))
      # print('--> len(result[0]): ', len(result[0]))
      # print('--> len(result[1]): ', len(result[1]))
      # print('--> type(result[0][catego_int][0]): ', type(result[0][catego_int][0]))
      # print('--> type(result[1][catego_int][0]): ', type(result[1][catego_int][0]))
      # print('--> result[0][catego_int][0].shape: ', result[0][catego_int][0].shape)
      # print('--> result[1][catego_int][0].shape: ', result[1][catego_int][0].shape)
      # print('--> np.unique(result[0][catego_int][0]: ', np.unique(result[0][catego_int][0]))
      # print('--> np.unique(result[1][catego_int][0]: ', np.unique(result[1][catego_int][0]))
      # print('--> type(result): ', type(result))
    except:
      continue
    # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files
    # crop_name = img_crop_path.rsplit('/', 1)[-1]
    model.show_result(img_crop_path, result, out_file=os.path.join(out_folder, catego, img_crop_name))