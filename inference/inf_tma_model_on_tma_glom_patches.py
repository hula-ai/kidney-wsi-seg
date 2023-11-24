import os
import numpy as np

from mmdet.apis import init_detector, inference_detector
import mmcv


# Specify the path to model config and checkpoint file
config_file = './configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py'
# checkpoint_file = '/data/hqvo3/mmdet/run2/latest.pth' # --- moana ---
# checkpoint_file = '/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/run_3_classes_ps4096_from_moana/latest.pth' # --- maui 3 classes---
checkpoint_file = '/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/run_6_classes_ps2048/epoch_1.pth' # --- maui 6 classes---

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# INPUT_PATH = '/data/public/HULA/TMA/folds/TMA_GT_FOLD_0.npz' # --- moana ---
INPUT_PATH = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/folds/TMA_GT_FOLD_0.npz' # --- maui ---
fold_info = np.load(INPUT_PATH, allow_pickle=True)
print('--- ID ---')
print(fold_info['ID'])
print(len(fold_info['ID']))
print('--- CL ---')
print(fold_info['CL'])
print(len(fold_info['CL']))
print('--- GLOM_PAS ---')
nbr_of_case_ids= len(fold_info['CL'])
  
# out_folder = '/data/hqvo3/final_results/digital_pathology/MILxseg/inf_tma_model_on_tma_glom_patches/1' # --- moana ---
out_folder = '/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/inf_tma_model_on_tma_glom_patches_fold_npz_3classes/3' # --- maui ---
os.makedirs(out_folder, exist_ok=True)

for idx in range(nbr_of_case_ids):
  print('[*] Case id: ', idx)
  cur_case_id = fold_info['ID'][idx]
  cur_label = fold_info['CL'][idx]
  cur_crop_list = fold_info['GLOM_PAS'][idx]
  case_id_folder = os.path.join(out_folder, ''.join([cur_case_id, ' - ', cur_label]))
  os.makedirs(case_id_folder, exist_ok=True)

  if len(cur_crop_list) > 10:
    cur_crop_list = cur_crop_list[:10]
  for multi_crop_path in cur_crop_list:
    # test a single image and show the results
    img_crop_path = multi_crop_path.replace('multi', 'img')
    img_crop_path = img_crop_path.replace('-img', '')
    print('- img crop path: ', img_crop_path)
    img = os.path.join('/project/hnguyen2/hqvo3/Datasets/digital_pathology', img_crop_path)  # or img = mmcv.imread(img), which will only load it once
    try:
      result = inference_detector(model, img)
      print('Succeed!!')
    except:
      continue
    # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files
    crop_name = img_crop_path.rsplit('/', 1)[-1]
    model.show_result(img, result, out_file=os.path.join(case_id_folder, crop_name))

# img = '/data/public/HULA/Pan_GN/Bari_GN/img/Bari 10026 PAS - 2022-05-26 01.23.54.ndpi_Glomerulus_(1.00,100162,3727,481,629).png'  # or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# # visualize the results in a new window
# model.show_result(img, result)
# # or save the visualization results to image files
# out_path = '/data/hqvo3/final_results/digital_pathology/MILxseg/inf_tma_model_on_pangn_glom_patches/result.jpg'
# model.show_result(img, result, out_file=out_path)

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)