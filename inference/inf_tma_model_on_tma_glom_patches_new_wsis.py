import os
import numpy as np

from mmdet.apis import init_detector, inference_detector
import mmcv


# Specify the path to model config and checkpoint file
config_file = './configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py'
checkpoint_file = '/data/hqvo3/mmdet/run2/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# INPUT_PATH = '/data/public/HULA/WSIs_renal_compartment_segmentations_tma_3classes_new_wsis/Autoseg_Glomerulus_img'
# crop_list = os.listdir(INPUT_PATH)
# Read json test file

out_folder = '/data/hqvo3/final_results/digital_pathology/MILxseg/inf_tma_model_on_tma_glom_patches/2'
os.makedirs(out_folder, exist_ok=True)
for img_crop_name in crop_list:
  # test a single image and show the results
  img_crop_path = os.path.join(INPUT_PATH, img_crop_name)
  print('- img crop path: ', img_crop_path)
  # img = os.path.join('/data/public/HULA/', img_crop_path)  # or img = mmcv.imread(img), which will only load it once
  try:
    result = inference_detector(model, img_crop_path)
  except:
    continue
  # visualize the results in a new window
  # model.show_result(img, result)
  # or save the visualization results to image files
  # crop_name = img_crop_path.rsplit('/', 1)[-1]
  model.show_result(img_crop_path, result, out_file=os.path.join(out_folder, img_crop_name))