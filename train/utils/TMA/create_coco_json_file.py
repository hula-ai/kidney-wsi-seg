###########################################################
# WITH TISSUE DETECTION AND REMOVE SMALLER COMPARTMENTS   #
# INSIDE BIG COMPARTMENTS (CORTEX, MEDULLA, CAPSULEOTHER) #
###########################################################

'''
for each crop that has cortex, medulla, and capsule other
  if that crop has 3 old types of compartments (glomerulus, arteriole, artery):
    it will be split into 2 similar images but with different annotations:
      1) with only glomerulus, arteriole, artery annotations
      2) with only cortex, medulla, and capsule others
    these 2 new similar images are considered as 2 different training samples

    Note: there is no need for double images in the drive, just need double images in the json files
'''

'''
  6-class instance segmentation
'''
import os

import cv2
import json
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import tifffile
import torch
import torchvision

from tissue_seg.tissue_seg_unet import UNet
from tissue_seg.tissue_seg_predict import predict_img


INPUT_PATH = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048' # on Maui
OUTPUT_PATH = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/coco_tma_tile_rm4_tissue_detection_and_subtract_smaller_compartments.json' # on Maui # remove polygon list having less than or equal to 4 elements

# data_infos = mmcv.load(ann_file) # TODO: We have a json file for each original slide

list_of_slide_ids = os.listdir(INPUT_PATH)
list_of_slide_ids.sort()
dict_of_mapping_slide_id_to_crop_imgs = {}
dict_of_mapping_slide_id_to_crop_masks = {}
for slide_id in list_of_slide_ids:
  slide_json_name = ''.join([slide_id, '-tiles.json'])
  # Add crop list for each slide id
  json_path = os.path.join(INPUT_PATH, slide_id, slide_json_name)
  if os.path.isfile(json_path):
    dict_of_mapping_slide_id_to_crop_imgs[slide_id] = []
    dict_of_mapping_slide_id_to_crop_masks[slide_id] = []
    with open(json_path) as f:
      json_data = json.load(f)
    for tile_dict in json_data['tiles']:
      dict_of_mapping_slide_id_to_crop_imgs[slide_id].append(tile_dict['image']) # just a image file name, not a path
      dict_of_mapping_slide_id_to_crop_masks[slide_id].append(tile_dict['labels']) # just a mask file name, not a path

annotations = []
images = []
obj_count = 0
crop_img_idx = -1
slide_count = 0

# --- INIT MODEL FOR TISSUE DETECTION ---
tissue_net = UNet(n_channels=3, n_classes=4, bilinear=False)
tissue_device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
tissue_net.to(device=tissue_device)
# tissue_net.load_state_dict(torch.load('/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_checkpoints/best_model.pth', map_location=tissue_device)) # on Moana
# tissue_type_norm = np.load('/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_checkpoints/train_norm.npy') # on Moana
tissue_net.load_state_dict(torch.load('/project/hnguyen2/hqvo3/Intermediary_results/digital_pathology/UNet_tissue_seg_checkpoints/best_model.pth', map_location=tissue_device)) # on Maui
tissue_type_norm = np.load('/project/hnguyen2/hqvo3/Intermediary_results/digital_pathology/UNet_tissue_seg_checkpoints/train_norm.npy') # on Maui
# for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())): # Iterate over crop images

# --- Iterate over slide ids ---
for slide_id, list_of_crop_img_names in dict_of_mapping_slide_id_to_crop_imgs.items():
  print('[*] slide_id: ', slide_id)
  print('[*] crop_img_idx: ', crop_img_idx)
  slide_count += 1
  print('[*] slide_count: ', slide_count)

  list_of_crop_mask_names = dict_of_mapping_slide_id_to_crop_masks[slide_id]
  print('[**] len(list_crop_masks)', len(list_of_crop_mask_names))
  for crop_id, crop_img_name in enumerate(list_of_crop_img_names):
    # print('[**] crop_id: ', crop_id)
    # if crop_id != 382:
    #   continue
    # idx
    crop_img_idx += 1

    filename = crop_img_name
    img_path = os.path.join(INPUT_PATH, slide_id, crop_img_name)
    # height, width = mmcv.imread(img_path).shape[:2] # TODO: do not need to read image for height, width because we know them

    images.append(dict(
        id=crop_img_idx,
        file_name=img_path,
        height=2048,
        width=2048))

    # Do we need to add channel axis
    ''' !! IMPORTANT DEFINITION !!:
    The old json file of Pietro has all `iscrowd` being equal to 0

    - `iscrowd` = 0: a mask has only a single object, but maybe with multiple polygons due to occluded
    - `iscrowd` = 1: a mask has a collection of objects

    Read more about `iscrowd` at cocodataset.org/#format-data
    '''

    crop_mask_name = list_of_crop_mask_names[crop_id]
    mask_path = os.path.join(INPUT_PATH, slide_id, crop_mask_name)
    # for _, obj in v['regions'].items(): # TODO: Iterate over 6 masks (numpy arrays), do we really need the background mask? no
    ids_to_classes = {1: "Cortex", 2: "Glomerulus", 3: "Arteriole", 4: "Artery", 5: "Medulla", 6: "CapsuleOther"}
    
    # ===== TODO: tissue detection =====
    # The initial code is before the loop
    # load the thumbnail
    ori_wsi_crop = cv2.imread(img_path, cv2.IMREAD_COLOR)  # load a 2048x2048 crop
    wsi_crop = cv2.cvtColor(ori_wsi_crop, cv2.COLOR_BGR2RGB)
    # normalize it after converting to torch and float
    wsi_crop = torch.from_numpy(wsi_crop).permute(2, 0, 1)
    wsi_crop = wsi_crop / 255.
    wsi_crop = torchvision.transforms.Normalize(list(tissue_type_norm[0]), list(tissue_type_norm[1]))(wsi_crop)
    # pass it through our model
    wsi_crop = wsi_crop.unsqueeze(0)
    wsi_crop = wsi_crop.to(device=tissue_device, dtype=torch.float32)
    prediction_tissue_mask = predict_img(net=tissue_net, img=wsi_crop) # -> this step produces a one-hot mask of 4 classes with shape (#channels or #classes, height, width)
    # get the binary tissue mask
    downscaled_mask = np.argmax(prediction_tissue_mask, axis=0) # a mask have one channel, each pixel value is either (0, 1, 2, 3). 0 for background, 1 2 3 are for 3 old types of compartments
    downscaled_mask[downscaled_mask > 0] = 255 # all 3 tissue types are seen as tissue regions and annotated as white color
    downscaled_mask = downscaled_mask.astype('uint8')
    # upscale the mask
    full_tissue_masking_canvas = cv2.resize(downscaled_mask, (2048, 2048), interpolation=cv2.INTER_NEAREST)

    # ===== TODO: subtract smaller compartments =====
    # subtract directly in the mask image, not the contour
    # TODO: only subtract overlapped regions
    cortex_mask = tifffile.imread(mask_path, key=1)
    _, cortex_thresh = cv2.threshold(cortex_mask, 127, 255, 0)
    medulla_mask = tifffile.imread(mask_path, key=5)
    _, medulla_thresh = cv2.threshold(medulla_mask, 127, 255, 0)
    capsuleother_mask = tifffile.imread(mask_path, key=6)
    _, capsuleother_thresh = cv2.threshold(capsuleother_mask, 127, 255, 0)
    glomerulus_mask = tifffile.imread(mask_path, key=2)
    _, glomerulus_thresh = cv2.threshold(glomerulus_mask, 127, 255, 0)
    arteriole_mask = tifffile.imread(mask_path, key=3)
    _, arteriole_thresh = cv2.threshold(arteriole_mask, 127, 255, 0)
    artery_mask = tifffile.imread(mask_path, key=4)
    _, artery_thresh = cv2.threshold(artery_mask, 127, 255, 0)
    # --- remove white rims of cortex, medulla, capsuleother ---
    # convert to bool type
    bool_cortex_thresh = cortex_thresh != 0
    bool_medulla_thresh = medulla_thresh != 0
    bool_capsuleother_thresh = capsuleother_thresh != 0
    bool_glomerulus_thresh = glomerulus_thresh != 0
    bool_arteriole_thresh = arteriole_thresh != 0
    bool_artery_thresh = artery_thresh != 0
    bool_tissue_mask = full_tissue_masking_canvas != 0
    tissue_cortex_thresh = np.logical_and(bool_tissue_mask, bool_cortex_thresh)
    tissue_medulla_thresh = np.logical_and(bool_tissue_mask, bool_medulla_thresh)
    tissue_capsuleother_thresh = np.logical_and(bool_tissue_mask, bool_capsuleother_thresh)

    # print unique pixel values
    unique_tissue_cortex_thresh = np.unique(tissue_cortex_thresh)
    unique_tissue_medulla_thresh = np.unique(tissue_medulla_thresh)
    unique_tissue_capsuleother_thresh = np.unique(tissue_capsuleother_thresh)
    unique_glomerulus_thresh = np.unique(bool_glomerulus_thresh)
    unique_arteriole_thresh = np.unique(bool_arteriole_thresh)
    unique_artery_thresh = np.unique(bool_artery_thresh)

    # cnt1 = 0
    # cnt2 = 0
    # if len(unique_tissue_cortex_thresh) == 2:
    #   cnt1 += 1
    # if len(unique_tissue_medulla_thresh) == 2:
    #   cnt1 += 1
    # if len(unique_tissue_capsuleother_thresh) == 2:
    #   cnt1 += 1
    # if len(unique_glomerulus_thresh) == 2:
    #   cnt2 += 1
    # if len(unique_arteriole_thresh) == 2:
    #   cnt2 += 1
    # if len(unique_artery_thresh) == 2:
    #   cnt2 += 1
    # if cnt1 != 2 or cnt2 != 2:
    #   continue

    # print('- np.unique(tissue_cortex_thresh): ', unique_tissue_cortex_thresh)
    # print('- np.unique(tissue_medulla_thresh): ', unique_tissue_medulla_thresh)
    # print('- np.unique(tissue_capsuleother_thresh): ', unique_tissue_capsuleother_thresh)
    # print('- np.unique(glomerulus_thresh): ', unique_glomerulus_thresh)
    # print('- np.unique(arteriole_thresh): ', unique_arteriole_thresh)
    # print('- np.unique(artery_thresh): ', unique_artery_thresh)

    # convert to bool type
    # bool_cortex_thresh = cortex_thresh != 0
    # bool_medulla_thresh = medulla_thresh != 0
    # bool_capsuleother_thresh = capsuleother_thresh != 0

    # --- Refine cortex ---
    # INTERSECT
    its1 = np.logical_and(tissue_cortex_thresh, bool_glomerulus_thresh)
    its2 = np.logical_and(tissue_cortex_thresh, bool_arteriole_thresh)
    its3 = np.logical_and(tissue_cortex_thresh, bool_artery_thresh)
    # UNION
    uni1 = np.logical_or(its1, its2)
    uni2 = np.logical_or(uni1, its3)
    # convert bool array back to (0, 255) array
    union_mat1 = uni2*255
    # print('- np.unique(union_mat): ', np.unique(union_mat1))
    # SUBTRACT
    tissue_cortex_thresh = tissue_cortex_thresh*255
    refined_cortex_thresh = tissue_cortex_thresh - union_mat1

    # --- Refine medulla ---
    # INTERSECT
    its1 = np.logical_and(tissue_medulla_thresh, bool_glomerulus_thresh)
    its2 = np.logical_and(tissue_medulla_thresh, bool_arteriole_thresh)
    its3 = np.logical_and(tissue_medulla_thresh, bool_artery_thresh)
    # UNION
    uni1 = np.logical_or(its1, its2)
    uni2 = np.logical_or(uni1, its3)
    # convert bool array back to (0, 255) array
    union_mat2 = uni2*255
    # print('- np.unique(union_mat): ', np.unique(union_mat2))
    # SUBTRACT
    tissue_medulla_thresh = tissue_medulla_thresh*255
    refined_medulla_thresh = tissue_medulla_thresh - union_mat2

    # --- Refine capsuleother --- It is very important that we follow the steps below for the case where a glomerulus is just partly inside the cortex, doing normal minus would cause negative pixel values
    # INTERSECT
    its1 = np.logical_and(tissue_capsuleother_thresh, bool_glomerulus_thresh)
    its2 = np.logical_and(tissue_capsuleother_thresh, bool_arteriole_thresh)
    its3 = np.logical_and(tissue_capsuleother_thresh, bool_artery_thresh)
    # UNION
    uni1 = np.logical_or(its1, its2)
    uni2 = np.logical_or(uni1, its3)
    # convert bool array back to (0, 255) array
    union_mat3 = uni2*255
    # print('- np.unique(union_mat): ', np.unique(union_mat3))
    # SUBTRACT
    tissue_capsuleother_thresh = tissue_capsuleother_thresh*255
    refined_capsuleother_thresh = tissue_capsuleother_thresh - union_mat3

    # --- APPEND ONE MORE IMAGE IF THERE ARE SMALLER COMPARTMENTS INSIDE 3 NEW BIG COMPARTMENTS (if there are holes in the mask annotations) ---
    # check if there is a smaller compartment inside
    # print('----- create a copy image for 3 new compartments -----')
    there_are_smaller_compartments = False
    if len(np.unique(union_mat1)) == 2 or len(np.unique(union_mat2)) == 2 or len(np.unique(union_mat3)) == 2:
      there_are_smaller_compartments = True
      # create the same image crop, but with black color for small compartments
      # print('- img_path: ', img_path)
      cur_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      # print('- cur_img.shape: ', cur_img.shape)
      # set black color for smaller compartments
      union_of_small_compartment_masks1 = np.logical_or(bool_glomerulus_thresh, bool_arteriole_thresh)
      union_of_small_compartment_masks = np.logical_or(union_of_small_compartment_masks1, bool_artery_thresh)
      invert_union_of_small_compartment_masks = np.bitwise_not(union_of_small_compartment_masks)
      binary_invert_union_of_small_compartment_masks = invert_union_of_small_compartment_masks*1
      binary_invert_union_of_small_compartment_masks = binary_invert_union_of_small_compartment_masks.astype(np.uint8)
      # tiling into 3 channels
      # print('- binary_invert_union_of_small_compartment_masks.shape: ', binary_invert_union_of_small_compartment_masks.shape)
      binary_invert_union_of_small_compartment_masks = binary_invert_union_of_small_compartment_masks[..., np.newaxis]
      three_channels_binary_invert_union_of_small_compartment_masks = np.repeat(binary_invert_union_of_small_compartment_masks, 3, 2)
      # print('- three_channels_binary_invert_union_of_small_compartment_masks.shape: ', three_channels_binary_invert_union_of_small_compartment_masks.shape)
      # multiply to get only rois
      new_img = cur_img * three_channels_binary_invert_union_of_small_compartment_masks
      # save new image to the same folder of img_path
      old_img_path = img_path.rsplit('/', 1)[0]
      old_img_name = img_path.rsplit('/', 1)[-1]
      old_img_name_head = old_img_name.rsplit('.', 1)[0]
      old_img_name_tail = old_img_name.rsplit('.', 1)[-1]
      new_img_name = ''.join([old_img_name_head, '-new.', old_img_name_tail])
      new_img_path = os.path.join(old_img_path, new_img_name)
      # print('--> new_img_path: ', new_img_path)
      cv2.imwrite(new_img_path, new_img)

      images.append(dict(
          id=crop_img_idx+1, # crop_img_idx must be updated with +2, not +1
          file_name=new_img_path,
          height=2048,
          width=2048))



    # !!! Doing like below will cause the problem of negative pixel values: 0 - 255
    # refined_cortex_thresh = cortex_thresh - glomerulus_thresh - arteriole_thresh - artery_thresh
    # refined_medulla_thresh = medulla_thresh - glomerulus_thresh - arteriole_thresh - artery_thresh
    # refined_capsuleother_thresh = capsuleother_thresh - glomerulus_thresh - arteriole_thresh - artery_thresh

    # print('- np.unique(cortex_thresh): ', np.unique(cortex_thresh))

    # TODO: how to represent a groundtruth mask that has hole
    # --- SAVE FIGURE ---
    # ori_crop = cv2.imread(img_path)
    # f, ax = plt.subplots(6, 6, figsize=(20, 20))
    # ax[0,0].imshow(ori_crop)
    # ax[0,0].set_title("Ori crop")
    # if there_are_smaller_compartments:
    #   ax[0,1].imshow(new_img)
    #   ax[0,1].set_title("New ori crop")
    # ax[1,0].imshow(cortex_thresh)
    # ax[1,0].set_title("Ori cortex mask")
    # ax[1,1].imshow(medulla_thresh)
    # ax[1,1].set_title("Ori medulla mask")
    # ax[1,2].imshow(capsuleother_thresh)
    # ax[1,2].set_title("Ori capsuleother mask")
    # ax[1,3].imshow(glomerulus_thresh)
    # ax[1,3].set_title("Ori glomerulus mask")
    # ax[1,4].imshow(arteriole_thresh)
    # ax[1,4].set_title("Ori arteriole mask")
    # ax[1,5].imshow(artery_thresh)
    # ax[1,5].set_title("Ori artery mask")
    # ax[2,0].imshow(bool_tissue_mask)
    # ax[2,0].set_title("Bool tissue mask")
    # ax[3,0].imshow(tissue_cortex_thresh)
    # ax[3,0].set_title("Tissue cortex thresh")
    # ax[3,1].imshow(tissue_medulla_thresh)
    # ax[3,1].set_title("Medulla cortex thresh")
    # ax[3,2].imshow(tissue_capsuleother_thresh)
    # ax[3,2].set_title("Capsuleother cortex thresh")
    # ax[4,0].imshow(refined_cortex_thresh)
    # ax[4,0].set_title("Refined cortex")
    # ax[4,1].imshow(refined_medulla_thresh)
    # ax[4,1].set_title("Refined medulla")
    # ax[4,2].imshow(refined_capsuleother_thresh)
    # ax[4,2].set_title("Refined capsuleother")
    # plt.subplots_adjust(top=0.9, wspace=0.3, hspace=0.3)

    # plt.savefig('./output/tma_seg_pre_{}_{}.png'.format(slide_count, crop_id))
    # plt.close()

    # Glomerulus, Arteriole, Artery
    for compartment_id in range(1, 7):
      if compartment_id == 1:
        thresh = refined_cortex_thresh
      elif compartment_id == 2:
        thresh = glomerulus_thresh
      elif compartment_id == 3:
        thresh = arteriole_thresh
      elif compartment_id == 4:
        thresh = artery_thresh
      elif compartment_id == 5:
        thresh = refined_medulla_thresh
      elif compartment_id == 6:
        thresh = refined_capsuleother_thresh

      thresh = thresh.astype(np.uint8)
      contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

      for cnt_ind, contour in enumerate(contours):
        # --- DO NOT GET SUB CONTOURS IF THEY ARE INSIDE BIGGER CONTOURS ---
        if hierarchy[0, cnt_ind, 3] != -1: # Read more about this at https://stackoverflow.com/questions/37479338/how-to-remove-a-contour-inside-contour-in-python-opencv
          continue
        contour = np.squeeze(contour, axis=1)
        px = []
        py = []
        for xy in contour:
          px.append(xy[0])
          py.append(xy[1])
        # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [(x, y) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x] # list of single elements - x then y
        if len(poly) <= 4: # https://github.com/cocodataset/cocoapi/issues/139#issuecomment-708327483
          continue

        x_min, y_min, x_max, y_max = (
            min(px), min(py), max(px), max(py))
        
        if there_are_smaller_compartments and compartment_id in [1, 5, 6]:
          data_anno = dict(
              image_id=crop_img_idx+1,
              id=obj_count,
              category_id=compartment_id, # TODO: category_id should be valid, all values in category_id should belong to id in categories
              bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
              area=(x_max - x_min) * (y_max - y_min),
              segmentation=[poly],
              iscrowd=0)
        else:
          data_anno = dict(
              image_id=crop_img_idx,
              id=obj_count,
              category_id=compartment_id, # TODO: category_id should be valid, all values in category_id should belong to id in categories
              bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
              area=(x_max - x_min) * (y_max - y_min),
              segmentation=[poly],
              iscrowd=0)
        annotations.append(data_anno)
        obj_count += 1
    if there_are_smaller_compartments:
      crop_img_idx += 1

coco_format_json = dict(
    info={
      'description': 'HULA TMA 4096 Tile Dataset Segmentation 6 classes',
      'version': '1.0',
      'year': 2023,
      'contributor': 'hqvo3',
      'date created': '2023/02/18'
    },
    images=images,
    annotations=annotations,
    categories=[{'id':0, 'name': 'Background', 'supercategory': 'Background'},
                {'id':1, 'name': 'Cortex', 'supercategory': 'Compartment'},
                {'id':2, 'name': 'Glomerulus', 'supercategory': 'Compartment'},
                {'id':3, 'name': 'Arteriole', 'supercategory': 'Compartment'},
                {'id':4, 'name': 'Artery', 'supercategory': 'Compartment'},
                {'id':5, 'name': 'Medulla', 'supercategory': 'Compartment'},
                {'id':6, 'name': 'CapsuleOther', 'supercategory': 'Compartment'}])

mmcv.dump(coco_format_json, OUTPUT_PATH)