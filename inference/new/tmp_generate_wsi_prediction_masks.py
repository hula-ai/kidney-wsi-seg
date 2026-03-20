import copy
import gc
from multiprocessing import (
  set_start_method,
  Manager,
  shared_memory
)
import multiprocessing as mp
import time

import torchvision.transforms

from mmdet.apis import init_detector, inference_detector
import os
import shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tifffile
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
import pandas as pd
import math
import cv2
import copy
from paquo.projects import QuPathProject
import multiprocessing as mp
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
from torch import nn
import torch.backends.cudnn as cudnn
import slideio
from aicsimageio import AICSImage
from openslide import OpenSlide
import openslide
from tissue_seg.tissue_seg_dataloader import BasicDataset
from tissue_seg.tissue_seg_unet import UNet
from tissue_seg.tissue_seg_plot_masks import plot_img_and_mask
from tissue_seg.tissue_seg_predict import predict_img
import argparse
matplotlib.use('Agg')

from function_definition import (
  process_contours, \
  get_args, \
  sub_df_uniques
)
from function_new_sharedmemory import (
  loop1, loop2, loop3, loop4, \
  loop5, loop6, loop7, \
  loop8, loop9, loop10, \
  loop11, loop12, loop13
)
from class_definition import (
  MLP
)

if __name__ == '__main__':
  # try:
  #   set_start_method('spawn')
  # except RuntimeError:
  #   pass
  # set_start_method('spawn')
  set_start_method('forkserver')

  # construct the tissue segmentation model
  args = get_args()
  
  tissue_net = UNet(n_channels=3, n_classes=4, bilinear=args.bilinear)
  
  tissue_device = torch.device('cuda:{}'.format(args.cuda_tiseg) if torch.cuda.is_available() else 'cpu')
  
  tissue_net.to(device=tissue_device)
  tissue_net.load_state_dict(torch.load(args.tissue_model, map_location=tissue_device))
  
  tissue_type_norm = np.load(args.train_norm_npy)
  
  print('Tissue seg. Model loaded!')
  
  # construct the model that we are going to use to process data
  
  # Specify the path to model config and checkpoint file
  config_file = args.swin_config
  checkpoint_file = args.tissue_type_model
  # config_file = '/home/cougarnet.uh.edu/hqvo3/Projects/digital_pathology/codes/MILxseg/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py'
  # checkpoint_file = '/data/hqvo3/mmdet/run_6_classes/latest.pth'
  
  # previously, each DF was instantiated for the entire set; because multi-gpu is tricky in mmdetection, we now break up
  # each DF into 8 sub DFs
  SUB_DF_COUNT = 1
  # set sub df ID, pythonic
  SUB_DF_ID = args.sub_df_id
  
  # build the model from a config file and a checkpoint file
  device = 'cuda:{}'.format(args.cuda_compseg)
  model = init_detector(config_file, checkpoint_file, device=device)
  
  # pick the project we are processing, can be any of the qupath_proj_files keys
  PROJECT = args.quproj
  
  # build the conf cutoff model
  conf_mlp = torch.load(args.val_to_test)
  conf_mlp = conf_mlp.float()
  conf_mlp.eval()
  conf_mlp_norm_val_to_test = np.load(args.val_to_test_norm)
  conf_mlp_norm_test_to_val = np.load(args.test_to_val_norm)
  
  # set the GPU batch size
  GPU_BATCH_SIZE = args.gpu_batch_size 
  
  # first we need to get an array of ground truth masks that we will use to evaluate performance
  # set the classes
  print('Getting our ground truths for accuracy/mask updating when possible...')
  CLASSES = ["Glomerulus", "Arteriole", "Artery"]
  # CLASSES = ['Cortex', 'Glomerulus', 'Arteriole', 'Artery', 'Medulla', 'CapsuleOther']
  
  SPECIAL_CLASS_CASES = {'Glomerulus': ['Globally sclerosed', 'non-globally sclerosed'],
                         'Arteriole': [],
                         'Artery': []}
  
  GT_MASKS = {key: [] for key in CLASSES}
  GT_MASKS_RAW_NAMES = {key: [] for key in CLASSES}
  GT_PATHS = ['/data/public/HULA/AMR/mask_val1/',
              '/data/public/HULA/AMR/mask_val2/',
              '/data/public/HULA/IgA/masks/',
              '/data/public/HULA/Pan_GN/Cologne_GN/mask/',
              '/data/public/HULA/Pan_GN/Szeged_GN/mask/',
              '/data/public/HULA/SLE/mask/',
              '/data/public/HULA/Pan_GN/Lille_GN/mask',
              '/data/public/HULA/TMA/mask/',
              '/data/public/HULA/Pan_GN/Bari_GN/mask/']
  
  # get the raw file paths
  for sup_dirs in GT_PATHS:
    for root, dirs, files in os.walk(os.path.abspath(sup_dirs)):
      for file in files:
        for class_name in CLASSES:
          if class_name in file:
            GT_MASKS_RAW_NAMES[class_name].append(file)
            GT_MASKS[class_name].append((os.path.join(root, file)))
          elif any(special_class in file for special_class in SPECIAL_CLASS_CASES[class_name]):
            GT_MASKS_RAW_NAMES[class_name].append(file)
            GT_MASKS[class_name].append((os.path.join(root, file)))
  
  # get the ids for indexing
  print('Get the WSI names for where we have gts per class...')
  GT_MASKS_WSI_IDS = {key: [] for key in CLASSES}
  for class_name in CLASSES:
    for file in GT_MASKS_RAW_NAMES[class_name]:
      if class_name in file:
        wsi_name = file.split('_' + class_name + '_')[0]
        GT_MASKS_WSI_IDS[class_name].append(wsi_name)
      elif any(special_class in file for special_class in SPECIAL_CLASS_CASES[class_name]):
        for special_class in SPECIAL_CLASS_CASES[class_name]:
          if special_class in file:
            wsi_name = file.split('_' + special_class + '_')[0]
            GT_MASKS_WSI_IDS[class_name].append(wsi_name)
  
  for class_name in CLASSES:
    GT_MASKS_WSI_IDS[class_name] = np.unique(np.array(GT_MASKS_WSI_IDS[class_name]))
  
  # get the various qupath project meta data
  TMA_qupath_proj_imgs = [im.image_name for im in QuPathProject(os.path.join(args.qupath_proj_path, 'TMA/TMASegmentationR14/project.qpproj')).images]
  AMR_qupath_proj_imgs = [im.image_name for im in QuPathProject(os.path.join(args.qupath_proj_path, 'AMR/AMRValidation2/project.qpproj')).images]
  Pan_GN_Cologne_qupath_proj_imgs = [im.image_name for im in QuPathProject(os.path.join(args.qupath_proj_path, 'Pan_GN/MultiGNSegmentationCologneR3/project.qpproj')).images]
  Pan_GN_Szeged_qupath_proj_imgs = [im.image_name for im in QuPathProject(os.path.join(args.qupath_proj_path, 'Pan_GN/PanGNSzegedSegmentationR3/project.qpproj')).images]
  Pan_GN_Lille_qupath_proj_imgs = [im.image_name for im in QuPathProject(os.path.join(args.qupath_proj_path, 'Pan_GN/PanGNLilleSegmentationR4/project.qpproj')).images]
  Pan_GN_Bari_qupath_proj_imgs = [im.image_name for im in QuPathProject(os.path.join(args.qupath_proj_path, 'Pan_GN/PanGNBariSegmentationR2/project.qpproj')).images]
  SLE_qupath_proj_imgs = [im.image_name for im in QuPathProject(os.path.join(args.qupath_proj_path, 'SLE/SLE-SegmentationV4/project.qpproj')).images]
  Podonet_qupath_proj_imgs = [im.image_name for im in QuPathProject(os.path.join(args.qupath_proj_path, 'Podonet/PodonetParaffinSegmentationR1/project.qpproj')).images]
  TMA_qupath_proj_imgs.sort()
  AMR_qupath_proj_imgs.sort()
  Pan_GN_Cologne_qupath_proj_imgs.sort()
  Pan_GN_Szeged_qupath_proj_imgs.sort()
  Pan_GN_Lille_qupath_proj_imgs.sort()
  Pan_GN_Bari_qupath_proj_imgs.sort()
  SLE_qupath_proj_imgs.sort()
  Podonet_qupath_proj_imgs.sort()
  
  
  qupath_proj_files = {'TMA': TMA_qupath_proj_imgs,
                       'AMR': AMR_qupath_proj_imgs,
                       'Pan_GN_Cologne': Pan_GN_Cologne_qupath_proj_imgs,
                       'Pan_GN_Szeged': Pan_GN_Szeged_qupath_proj_imgs,
                       'Pan_GN_Lille': Pan_GN_Lille_qupath_proj_imgs,
                       'Pan_GN_Bari': Pan_GN_Bari_qupath_proj_imgs,
                       'SLE': SLE_qupath_proj_imgs,
                       'Podonet': Podonet_qupath_proj_imgs
                      }
  
  # qupath_proj_files = {'TMA': [im.image_name for im in QuPathProject('/data/public/HULA/TMA/QuPath_Projects/TMASegmentationR14/project.qpproj').images],
  #                      'AMR': [im.image_name for im in QuPathProject('/data/public/HULA/AMR/QuPath_Projects/AMRValidation2/project.qpproj').images],
  #                      'Pan_GN_Cologne': [im.image_name for im in QuPathProject('/data/public/HULA/Pan_GN/QuPath_Projects/MultiGNSegmentationCologneR3/project.qpproj').images],
  #                      'Pan_GN_Szeged': [im.image_name for im in QuPathProject('/data/public/HULA/Pan_GN/QuPath_Projects/PanGNSzegedSegmentationR3/project.qpproj').images],
  #                      'Pan_GN_Lille': [im.image_name for im in QuPathProject('/data/public/HULA/Pan_GN/QuPath_Projects/PanGNLilleSegmentationR4/project.qpproj').images],
  #                      'Pan_GN_Bari': [im.image_name for im in QuPathProject('/data/public/HULA/Pan_GN/QuPath_Projects/PanGNBariSegmentationR2/project.qpproj').images],
  #                      'SLE': [im.image_name for im in QuPathProject('/data/public/HULA/SLE/QuPath_Projects/SLE-SegmentationV4/project.qpproj').images],
  #                      'Podonet': [im.image_name for im in QuPathProject('/data/public/HULA/Podonet/qupath_projects/PodonetParaffinSegmentationR1/project.qpproj').images]
  #                      }
  
  # set in/out dims, only supports square predictions for now
  IN_DIM = args.in_dim
  OUT_DIM = args.out_dim
  OUT_RATIO = IN_DIM/OUT_DIM
  
  # set the figure scales, destination to plot pngs, binary masks, colors for figs
  FIG_SCALE = 16
  COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0)]
  
  # set dir names
  OUTPUT_DIR = args.output_dir
  # if os.path.exists(OUTPUT_DIR):
  #   shutil.rmtree(OUTPUT_DIR)
  PLOT_DIR = OUTPUT_DIR + 'Global_masks_plots_autoseg/'
  MASK_DIR = {class_name: OUTPUT_DIR + 'Autoseg_' + class_name + '_mask/' for class_name in CLASSES}
  MULTI_DIR = {class_name: OUTPUT_DIR + 'Autoseg_' + class_name + '_multi/' for class_name in CLASSES}
  IMG_DIR = {class_name: OUTPUT_DIR + 'Autoseg_' + class_name + '_img/' for class_name in CLASSES}
  TI_DIR = OUTPUT_DIR + 'Autoseg_TI_tiles/'
  # PLOT_DIR = '/data/public/HULA/WSIs_renal_compartment_segmentations/Global_masks_plots_autoseg/'
  # MASK_DIR = {class_name: '/data/public/HULA/WSIs_renal_compartment_segmentations/Autoseg_' + class_name + '_mask/' for class_name in CLASSES}
  # MULTI_DIR = {class_name: '/data/public/HULA/WSIs_renal_compartment_segmentations/Autoseg_' + class_name + '_multi/' for class_name in CLASSES}
  # IMG_DIR = {class_name: '/data/public/HULA/WSIs_renal_compartment_segmentations/Autoseg_' + class_name + '_img/' for class_name in CLASSES}
  # TI_DIR = '/data/public/HULA/WSIs_renal_compartment_segmentations/Autoseg_TI_tiles/'
  
  # make the dirs if needed
  if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR, exist_ok=True)
  for mask_dir in MASK_DIR.values():
    if not os.path.exists(mask_dir):
      os.makedirs(mask_dir, exist_ok=True)
  for multi_dir in MULTI_DIR.values():
    if not os.path.exists(multi_dir):
      os.makedirs(multi_dir, exist_ok=True)
  for img_dir in IMG_DIR.values():
    if not os.path.exists(img_dir):
      os.makedirs(img_dir, exist_ok=True)
  if not os.path.exists(TI_DIR):
    os.makedirs(TI_DIR, exist_ok=True)
  
  # get the data and pass it through the model
  
  # first we get the csv containing the tissue mask processing information, which we will use to parse samples
  PROCESS_DF = pd.read_csv(args.clam_process_df)
  
  # set the directory for the slides and the respective tissue mask data
  WSI_PATH = args.wsi_path
  # WSI_MASK_PATH = '/data/public/HULA/WSIs_renal_tissue_masks_CLAM/'
  CSV_PATH = args.csv_path
  
  # get the unique directory extension for the .mrxs file formats
  mrxs_special_dir = {}
  # mrxs_dirs = next(os.walk(WSI_PATH))[1]
  mrxs_dirs = None
  for root, dirs, files in os.walk(WSI_PATH):
    mrxs_dirs = dirs
    break
  if mrxs_dirs is None:
    print('!! No mrxs dirs')
  else:
    print('[*] mrxs_dirs: ', mrxs_dirs)
    for mrxs_dir in mrxs_dirs:
      print('[*] mrxs_dir: ', mrxs_dir)
      mrxs_slides = os.listdir(WSI_PATH + mrxs_dir)
      for mrxs_slide in mrxs_slides:
        if '.mrxs' in mrxs_slide:
          mrxs_special_dir[mrxs_slide] = mrxs_dir
  
  
  # get the raw lists/dicts for appending to our pd frame
  # lists
  class_conflist = [key + '_conf_cutoff' for key in CLASSES]
  class_roctprlist = [key + '_pr_precision' for key in CLASSES]
  class_rocfprlist = [key + '_pr_recall' for key in CLASSES]
  class_rocthreshlist = [key + '_pr_thresh' for key in CLASSES]
  class_rocauclist = [key + '_pr_auc' for key in CLASSES]
  class_roctprcleanlist = [key + '_pr_precision_rois' for key in CLASSES]
  class_rocfprcleanlist = [key + '_pr_recall_rois' for key in CLASSES]
  class_rocthreshcleanlist = [key + '_pr_thresh_rois' for key in CLASSES]
  class_rocauccleanlist = [key + '_pr_auc_rois' for key in CLASSES]
  class_idealconflist = [key + '_conf_delta' for key in CLASSES]
  class_idealconfcleanlist = [key + '_conf_delta_rois' for key in CLASSES]
  class_f1_30 = [key + '_f1_30' for key in CLASSES]
  class_f1_50 = [key + '_f1_50' for key in CLASSES]
  class_f1_70 = [key + '_f1_70' for key in CLASSES]
  class_f1_90 = [key + '_f1_90' for key in CLASSES]
  class_f1_auto = [key + '_f1_auto' for key in CLASSES]
  class_f1_clean30 = [key + '_f1_30_rois' for key in CLASSES]
  class_f1_clean50 = [key + '_f1_50_rois' for key in CLASSES]
  class_f1_clean70 = [key + '_f1_70_rois' for key in CLASSES]
  class_f1_clean90 = [key + '_f1_90_rois' for key in CLASSES]
  class_f1_cleanauto = [key + '_f1_auto_rois' for key in CLASSES]
  class_f1_maxcut = [key + '_f1_maxcut' for key in CLASSES]
  class_f1_cleanmaxcut = [key + '_f1_maxcut_rois' for key in CLASSES]
  class_f1_max = [key + '_f1_max' for key in CLASSES]
  class_f1_cleanmax = [key + '_f1_max_rois' for key in CLASSES]
  class_f1_rawcount = [key + '_raw_count_histo' for key in CLASSES]
  class_f1_freqcount = [key + '_freq_count_histo' for key in CLASSES]
  class_f1_cleanrawcount = [key + '_raw_count_histo_rois' for key in CLASSES]
  class_f1_cleanfreqcount = [key + '_freq_count_histo_rois' for key in CLASSES]
  class_iouslist = [key + '_iou' for key in CLASSES]
  class_aplist = [key + '_AP' for key in CLASSES]
  class_arlist = [key + '_AR' for key in CLASSES]
  class_f1list = [key + '_F1' for key in CLASSES]
  class_aslist = [key + '_AS' for key in CLASSES]
  class_iouscleanlist = [key + '_iou_rois' for key in CLASSES]
  class_apcleanlist = [key + '_AP_rois' for key in CLASSES]
  class_arcleanlist = [key + '_AR_rois' for key in CLASSES]
  class_f1cleanlist = [key + '_F1_rois' for key in CLASSES]
  class_ascleanlist = [key + '_AS_rois' for key in CLASSES]
  class_countslist = [key + '_counts' for key in CLASSES]
  
  # combine lists
  full_meta_list = class_conflist + class_roctprlist + class_rocfprlist + class_rocthreshlist + \
  class_rocauclist + class_roctprcleanlist + class_rocfprcleanlist + class_rocthreshcleanlist + class_rocauccleanlist + \
  class_idealconflist + class_idealconfcleanlist + class_f1_30 + class_f1_50 + class_f1_70 + class_f1_90 + class_f1_auto + \
  class_f1_clean30 + class_f1_clean50 + class_f1_clean70 + class_f1_clean90 + class_f1_cleanauto + \
  class_f1_maxcut + class_f1_cleanmaxcut + class_f1_max + class_f1_cleanmax + class_f1_rawcount + class_f1_freqcount + \
  class_f1_cleanrawcount + class_f1_cleanfreqcount + \
  class_iouslist + class_aplist + class_arlist + class_f1list + class_aslist + class_iouscleanlist + class_apcleanlist + \
  class_arcleanlist + class_f1cleanlist + class_ascleanlist + class_countslist
  
  # dicts
  class_confdict = {key: 'nan' for key in class_conflist}
  class_tprdict = {key: 'nan' for key in class_roctprlist}
  class_fprdict = {key: 'nan' for key in class_rocfprlist}
  class_threshdict = {key: 'nan' for key in class_rocthreshlist}
  class_aucdict = {key: 'nan' for key in class_rocauclist}
  class_tprcleandict = {key: 'nan' for key in class_roctprcleanlist}
  class_fprcleandict = {key: 'nan' for key in class_rocfprcleanlist}
  class_threshcleandict = {key: 'nan' for key in class_rocthreshcleanlist}
  class_auccleandict = {key: 'nan' for key in class_rocauccleanlist}
  class_idealconfdict = {key: 'nan' for key in class_idealconflist}
  class_idealconfcleandict = {key: 'nan' for key in class_idealconfcleanlist}
  class_f1_30_dict = {key: 'nan' for key in class_f1_30}
  class_f1_50_dict = {key: 'nan' for key in class_f1_50}
  class_f1_70_dict = {key: 'nan' for key in class_f1_70}
  class_f1_90_dict = {key: 'nan' for key in class_f1_90}
  class_f1_auto_dict = {key: 'nan' for key in class_f1_auto}
  class_f1_clean30_dict = {key: 'nan' for key in class_f1_clean30}
  class_f1_clean50_dict = {key: 'nan' for key in class_f1_clean50}
  class_f1_clean70_dict = {key: 'nan' for key in class_f1_clean70}
  class_f1_clean90_dict = {key: 'nan' for key in class_f1_clean90}
  class_f1_cleanauto_dict = {key: 'nan' for key in class_f1_cleanauto}
  class_f1_maxcut_dict = {key: 'nan' for key in class_f1_maxcut}
  class_f1_cleanmaxcut_dict = {key: 'nan' for key in class_f1_cleanmaxcut}
  class_f1_max_dict = {key: 'nan' for key in class_f1_max}
  class_f1_cleanmax_dict = {key: 'nan' for key in class_f1_cleanmax}
  class_f1_rawcount_dict = {key: 'nan' for key in class_f1_rawcount}
  class_f1_freqcount_dict = {key: 'nan' for key in class_f1_freqcount}
  class_f1_cleanrawcount_dict = {key: 'nan' for key in class_f1_rawcount}
  class_f1_cleanfreqcount_dict = {key: 'nan' for key in class_f1_freqcount}
  class_iousdict = {key: 'nan' for key in class_iouslist}
  class_apdict = {key: 'nan' for key in class_aplist}
  class_ardict = {key: 'nan' for key in class_arlist}
  class_f1dict = {key: 'nan' for key in class_f1list}
  class_asdict = {key: 'nan' for key in class_aslist}
  class_iouscleandict = {key: 'nan' for key in class_iouscleanlist}
  class_apcleandict = {key: 'nan' for key in class_apcleanlist}
  class_arcleandict = {key: 'nan' for key in class_arcleanlist}
  class_f1cleandict = {key: 'nan' for key in class_f1cleanlist}
  class_ascleandict = {key: 'nan' for key in class_ascleanlist}
  class_countsdict = {key: 'nan' for key in class_countslist}
  
  # full dict
  full_meta_dict = {**class_confdict,
                    **class_tprdict, **class_fprdict, **class_threshdict, **class_aucdict,
                    **class_tprcleandict, **class_fprcleandict, **class_threshcleandict, **class_auccleandict, **class_idealconfdict,
                    **class_idealconfcleandict, **class_f1_30_dict, **class_f1_50_dict, **class_f1_70_dict, **class_f1_90_dict,
                    **class_f1_auto_dict, **class_f1_clean30_dict, **class_f1_clean50_dict, **class_f1_clean70_dict,
                    **class_f1_rawcount_dict, **class_f1_freqcount_dict, **class_f1_cleanrawcount_dict, **class_f1_cleanfreqcount_dict,
                    **class_f1_clean90_dict, **class_f1_cleanauto_dict,
                    **class_f1_maxcut_dict, **class_f1_cleanmaxcut_dict, **class_f1_max_dict, **class_f1_cleanmax_dict,
                    **class_iousdict, **class_apdict, **class_ardict, **class_f1dict, **class_asdict, **class_iouscleandict,
                    **class_apcleandict, **class_arcleandict, **class_f1cleandict, **class_ascleandict, **class_countsdict}
  
  
  # generate a processing file if there isnt one already
  print('[*] csv_path: ', os.path.join(CSV_PATH, 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID)))
  if os.path.isfile(os.path.join(CSV_PATH, 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID))):
    # get all of the file names, this assumes you have generated all SUB_DF_COUNT dfs already
    print('Checking for new tissue annotations...')
    original_slideids = []
    original_slideids += pd.read_csv(CSV_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID))['slide_id'].to_list()

    list_slide_ids = os.listdir(WSI_PATH)
    for idx, slide_id in enumerate(list_slide_ids):
      last_dot_index = slide_id.rfind('.')
      slide_id_out_dir_name = ''.join([slide_id[0:last_dot_index], '_dot_', slide_id[last_dot_index+1:]])
      for tissue_class, tissue_class_dir in IMG_DIR.items():
        os.makedirs(os.path.join(tissue_class_dir, slide_id_out_dir_name), exist_ok=True)
        os.makedirs(os.path.join(MASK_DIR[tissue_class], slide_id_out_dir_name), exist_ok=True)
        os.makedirs(os.path.join(MULTI_DIR[tissue_class], slide_id_out_dir_name), exist_ok=True)

    ori_list_slide_ids = os.listdir(WSI_PATH)
    list_slide_ids = []
    for ori_slide_id in ori_list_slide_ids:
      if '.' in ori_slide_id:
        list_slide_ids.append(ori_slide_id)

    # Remove scene and series posfix
    new_original_slideids = []
    for original_slideid in original_slideids:
      if '.scn' in original_slideid:
        new_original_slideids.append(original_slideid.split(' - Series ')[0])
      elif '.czi' in original_slideid:
        new_original_slideids.append(original_slideid.split(' - Scene ')[0])
      else:
        new_original_slideids.append(original_slideid)

    original_slideids = set(new_original_slideids)
    uniques = list(set(list_slide_ids) - original_slideids)
    # uniques = []
    print('--> uniques: ', uniques)

    if len(uniques) > 0:
      print('{} new slides found! Adding them for processing...'.format(len(uniques)))

      for idx, slide_id in enumerate(uniques):
        last_dot_index = slide_id.rfind('.')
        slide_id_out_dir_name = ''.join([slide_id[0:last_dot_index], '_dot_', slide_id[last_dot_index+1:]])
        for tissue_class, tissue_class_dir in IMG_DIR.items():
          os.makedirs(os.path.join(tissue_class_dir, slide_id_out_dir_name), exist_ok=True)
          os.makedirs(os.path.join(MASK_DIR[tissue_class], slide_id_out_dir_name), exist_ok=True)
          print('glom path dir: ', os.path.join(MULTI_DIR[tissue_class], slide_id_out_dir_name))
          os.makedirs(os.path.join(MULTI_DIR[tissue_class], slide_id_out_dir_name), exist_ok=True)

      sub_process_df = pd.read_csv(CSV_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID))
      # sub_uniques = sub_df_uniques(uniques, SUB_DF_COUNT)[sub_id] # Why the i-th element is of the corresponding sub_id

      for slide_id in uniques:
        print('----- Add {} to process csv -----'.format(slide_id))
        last_dot_index = slide_id.rfind('.')
        slide_id_out_dir_name = ''.join([slide_id[0:last_dot_index], '_dot_', slide_id[last_dot_index+1:]])

        if slide_id.rsplit('.', 1)[1] == 'scn':
          slide = slideio.open_slide(os.path.join(WSI_PATH, slide_id), "SCN")
          num_scenes = slide.num_scenes
          print('-> num_scenes in scn: ', num_scenes)
          for index in range(0, num_scenes):
            new_slide_id = ''.join([slide_id, ' - Series ', str(index+1)])
            print('Name with Series: ', slide.get_scene(index).name)
            base_dict = {'slide_id': new_slide_id,
                        'status': 'tbp'}
            rep_dict = {'reprocess': 0, 'sub_out_dir': slide_id_out_dir_name}
            full_dict = {**base_dict, **full_meta_dict.copy(), **rep_dict}
            # process_df = process_df.append(full_dict, ignore_index=True)
            sub_process_df = pd.concat([sub_process_df, pd.DataFrame.from_records([full_dict])], ignore_index=True)
        elif slide_id.rsplit('.', 1)[1] == 'czi':
          slide = slideio.open_slide(os.path.join(WSI_PATH, slide_id), "CZI")
          num_scenes = slide.num_scenes
          print('-> num_scenes in czi: ', num_scenes)
          for index in range(0, num_scenes):
            new_slide_id = ''.join([slide_id, ' - Scene ', str(index+1)])
            print('Name with Series: ', slide.get_scene(index).name)
            base_dict = {'slide_id': new_slide_id,
                        'status': 'tbp'}
            rep_dict = {'reprocess': 0, 'sub_out_dir': slide_id_out_dir_name}
            full_dict = {**base_dict, **full_meta_dict.copy(), **rep_dict}
            # process_df = process_df.append(full_dict, ignore_index=True)
            sub_process_df = pd.concat([sub_process_df, pd.DataFrame.from_records([full_dict])], ignore_index=True)
        else:
          # print('- QuPath proj, slide id: ', idx, slide_id)
          base_dict = {'slide_id': slide_id,
                        'status': 'tbp'}
          rep_dict = {'reprocess': 0, 'sub_out_dir': slide_id_out_dir_name}
          full_dict = {**base_dict, **full_meta_dict.copy(), **rep_dict}
          # process_df = process_df.append(full_dict, ignore_index=True)
          sub_process_df = pd.concat([sub_process_df, pd.DataFrame.from_records([full_dict])], ignore_index=True)

        # base_dict = {'slide_id': slide_id,
        #               'status': 'tbp'}
        # rep_dict = {'reprocess': 0, 'sub_out_dir': slide_id_out_dir_name}
        # full_dict = {**base_dict, **full_meta_dict.copy(), **rep_dict}
        # sub_process_df = pd.concat([sub_process_df, pd.DataFrame.from_records([full_dict])], ignore_index=True)
      print('--> Count #rows of new csv: ', len(sub_process_df.index))
      sub_process_df.to_csv(CSV_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID), index=False)
    else:
      print('No new slides found.')
    print('Loading our processing df...')
    process_df = pd.read_csv(CSV_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID))
  else:
    print('Generating our processing df...')
    ori_list_slide_ids = os.listdir(WSI_PATH)
    list_slide_ids = []
    for ori_slide_id in ori_list_slide_ids:
      if '.' in ori_slide_id:
        list_slide_ids.append(ori_slide_id)
    # list_slide_ids = [f for f in os.listdir(WSI_PATH) if os.path.isfile(f)]
    print('[check] list_slide_ids: ', list_slide_ids)
    process_df = pd.DataFrame(columns=['slide_id', 'status'] + full_meta_list.copy() + ['reprocess'])
    for idx, slide_id in enumerate(list_slide_ids):
      print('----- Add {} to process csv -----'.format(slide_id))
      last_dot_index = slide_id.rfind('.')
      slide_id_out_dir_name = ''.join([slide_id[0:last_dot_index], '_dot_', slide_id[last_dot_index+1:]])
      for tissue_class, tissue_class_dir in IMG_DIR.items():
        os.makedirs(os.path.join(tissue_class_dir, slide_id_out_dir_name), exist_ok=True)
        os.makedirs(os.path.join(MASK_DIR[tissue_class], slide_id_out_dir_name), exist_ok=True)
        print('glom path dir: ', os.path.join(MULTI_DIR[tissue_class], slide_id_out_dir_name))
        os.makedirs(os.path.join(MULTI_DIR[tissue_class], slide_id_out_dir_name), exist_ok=True)
      if slide_id.rsplit('.', 1)[1] == 'scn':
        slide = slideio.open_slide(os.path.join(WSI_PATH, slide_id), "SCN")
        num_scenes = slide.num_scenes
        print('-> num_scenes in scn: ', num_scenes)
        for index in range(0, num_scenes):
          new_slide_id = ''.join([slide_id, ' - Series ', str(index+1)])
          print('Name with Series: ', slide.get_scene(index).name)
          base_dict = {'slide_id': new_slide_id,
                      'status': 'tbp'}
          rep_dict = {'reprocess': 0, 'sub_out_dir': slide_id_out_dir_name}
          full_dict = {**base_dict, **full_meta_dict.copy(), **rep_dict}
          # process_df = process_df.append(full_dict, ignore_index=True)
          process_df = pd.concat([process_df, pd.DataFrame.from_records([full_dict])], ignore_index=True)
      elif slide_id.rsplit('.', 1)[1] == 'czi':
        slide = slideio.open_slide(os.path.join(WSI_PATH, slide_id), "CZI")
        num_scenes = slide.num_scenes
        print('-> num_scenes in czi: ', num_scenes)
        for index in range(0, num_scenes):
          new_slide_id = ''.join([slide_id, ' - Scene ', str(index+1)])
          print('Name with Series: ', slide.get_scene(index).name)
          base_dict = {'slide_id': new_slide_id,
                      'status': 'tbp'}
          rep_dict = {'reprocess': 0, 'sub_out_dir': slide_id_out_dir_name}
          full_dict = {**base_dict, **full_meta_dict.copy(), **rep_dict}
          # process_df = process_df.append(full_dict, ignore_index=True)
          process_df = pd.concat([process_df, pd.DataFrame.from_records([full_dict])], ignore_index=True)
      else:
        # print('- QuPath proj, slide id: ', idx, slide_id)
        base_dict = {'slide_id': slide_id,
                      'status': 'tbp'}
        rep_dict = {'reprocess': 0, 'sub_out_dir': slide_id_out_dir_name}
        full_dict = {**base_dict, **full_meta_dict.copy(), **rep_dict}
        # process_df = process_df.append(full_dict, ignore_index=True)
        process_df = pd.concat([process_df, pd.DataFrame.from_records([full_dict])], ignore_index=True)
    process_df.to_csv(os.path.join(CSV_PATH, 'process_list_{}_{}.csv'.format((PROJECT), SUB_DF_ID)), index=False)
  
  # we iterate through the df to generate our predictions
  for index, row in process_df.iterrows():
    print('===== [INDEX AND SLIDE ID] {}, {} ====='.format(index, row['slide_id']))
    if index >= args.from_index and index <= args.to_index:
      pass
    else:
      continue
    if row['status'] == 'processed':
      # check if the reprocess flag has been raised
      if row['reprocess']:
        if args.do_not_print_much is False:
          print('{} has already been processed, but the reprocess flag is set to 1. Reprocessing...'.format(row['slide_id']))
      else:
        if args.do_not_print_much is False:
          print('{} has already been processed! Skipping. '
                'If you want to reprocess, set the reprocess flag to 1.'.format(row['slide_id']))
        continue
      if args.do_not_print_much is False:
        print('Processing {}...'.format(row['slide_id']))
    elif row['status'] != 'tbp' and args.continue_from_previous_run is True:
      continue
  
    if args.do_not_print_much is False:
      print('Deleting old prediction masks/multi/tiles for {} if we have them...'.format(row['slide_id']))
    # make the supercategory list
    sup_cats = []
    # append the classes list
    sup_cats.append(CLASSES)
    # get the mixed cats
    for sup_class in range(0, len(CLASSES)):
      if sup_class == (len(CLASSES) - 1):
        continue
      for sub_class in range(sup_class + 1, len(CLASSES)):
        sup_cats.append(CLASSES[sup_class] + 'OR' + CLASSES[sub_class])
    # delete the old masks, multis
    for mask_dir, multi_dir in zip(MASK_DIR.values(), MULTI_DIR.values()):
      for old_pred_mask in os.listdir(mask_dir):
        splits = [old_pred_mask.split('_{}_'.format(class_name)) for class_name in sup_cats]
        for split in splits:
          if len(split) > 1:
            if (row['slide_id'] in split) or ((row['slide_id'].split(' - Series ')[0] in split) and ('.tif' in row['slide_id'])):
              if args.do_not_print_much is False:
                print('Deleting {}...'.format(old_pred_mask))
              os.remove(mask_dir + old_pred_mask)
              os.remove(multi_dir + old_pred_mask.split('-mask.png')[0] + '-multi.png')
    # delete the old TIs
    for old_ti in os.listdir(TI_DIR):
      splits = old_ti.split('_xcoord_')
      if (row['slide_id'] in splits[0]) or ((row['slide_id'].split(' - Series ')[0] in splits[0]) and ('.tif' in row['slide_id'])):
        if args.do_not_print_much is False:
          print('Deleting {}...'.format(old_ti))
        os.remove(TI_DIR + old_ti)
  
    # load the WSI
    print('loading the wsi ...')
    if args.do_not_print_much is False:
      print('Loading {}...'.format(row['slide_id']))
    if '- macro' in row['slide_id']:
      if args.do_not_print_much is False:
        print('Thumbnail found. Skipping...')
      process_df.loc[index, 'status'] = 'thumbnail'
      continue
    elif ('.scn' in row['slide_id']) or ('.sc_' in row['slide_id']):
      if args.do_not_print_much is False:
        print('Leica scan found, using SlideIO to load {}'.format(row['slide_id']))
      scene_id = int(row['slide_id'].split(' - Series ')[1][0])
      full_path = WSI_PATH + row['slide_id'].split(' - Series ')[0]
      slideio_wsi = slideio.open_slide(full_path, driver="SCN")
      scene = slideio_wsi.get_scene(scene_id - 1)
      try:
        wsi = scene.read_block()
      except:
        if args.do_not_print_much is False:
          print('Scene loading failed! Likely an issue with scan file. Skipping...')
        process_df.loc[index, 'status'] = 'failed_to_load'
        continue
    elif ('.svs' in row['slide_id']):
      if args.do_not_print_much is False:
        print('.svs scan found, using SlideIO to load {}'.format(row['slide_id']))
      full_path = WSI_PATH + row['slide_id']
      slideio_wsi = slideio.open_slide(full_path, driver="SVS")
      scene = slideio_wsi.get_scene(0)
      try:
        wsi = scene.read_block()
      except:
        if args.do_not_print_much is False:
          print('Scene loading failed! Likely an issue with scan file. Skipping...')
        process_df.loc[index, 'status'] = 'failed_to_load'
        continue
    elif ('.czi' in row['slide_id']) and (' - Scene ' in row['slide_id']):
      print('Loading czi with scene !!')
      if args.do_not_print_much is False:
        print('.czi scan found, using AICSImage to load {}'.format(row['slide_id']))
      scene_id = int(row['slide_id'].split(' - Scene ')[1])
      full_path = os.path.join(WSI_PATH, row['slide_id'].split(' - Scene ')[0])
      try:
        full_scan = AICSImage(full_path)
        full_scan.set_scene(scene_id - 1)
        wsi = full_scan.get_image_data("YXS", T=0)
        wsi = cv2.cvtColor(wsi, cv2.COLOR_BGR2RGB)
      except:
        if args.do_not_print_much is False:
          print('Scene loading failed! Likely an issue with scan file. Skipping...')
        process_df.loc[index, 'status'] = 'failed_to_load'
        continue
    elif ('.czi' in row['slide_id']) and (' - Scene ' not in row['slide_id']):
      if args.do_not_print_much is False:
        print('.czi scan found, using AICSImage to load {}'.format(row['slide_id']))
      scene_id = 0
      full_path = os.path.join(WSI_PATH, row['slide_id'])
      try:
        full_scan = AICSImage(full_path)
        full_scan.set_scene(scene_id)
        wsi = full_scan.get_image_data("YXS", T=0)
        wsi = cv2.cvtColor(wsi, cv2.COLOR_BGR2RGB)
      except:
        if args.do_not_print_much is False:
          print('Scene loading failed! Likely an issue with scan file. Skipping...')
        process_df.loc[index, 'status'] = 'failed_to_load'
        continue
    elif ('.mrxs' in row['slide_id']):
      if args.do_not_print_much is False:
        print('.mrxs scan found, using openslide to load {}'.format(row['slide_id']))
        print('mrxs_special_dir: ', mrxs_special_dir)
        print(type(mrxs_special_dir))
      for key, val in mrxs_special_dir.items():
        if args.do_not_print_much is False:
          print(key)
      # full_path = os.path.join(WSI_PATH + mrxs_special_dir[row['slide_id']], row['slide_id'])
      # full_path = os.path.join(WSI_PATH, row['slide_id'])
      # slide = OpenSlide(full_path)
      try:
        # full_path = os.path.join(WSI_PATH + mrxs_special_dir[row['slide_id']], row['slide_id'])
        full_path = os.path.join(WSI_PATH, row['slide_id'])
        slide = OpenSlide(full_path)
        # slide = openslide.open_slide(full_path)
        try:
          if ((int(slide.properties['openslide.bounds-width']) * int(slide.properties['openslide.bounds-height'])) > (200000 * 100000)):
            if args.do_not_print_much is False:
              print('Scene loading failed! The MIRAX ROI is too big. Skipping...')
            process_df.loc[index, 'status'] = 'failed_to_load'
            continue
          else:
            mirax_x = int(slide.properties['openslide.bounds-x'])
            mirax_y = int(slide.properties['openslide.bounds-y'])
            mirax_width = int(slide.properties['openslide.bounds-width'])
            mirax_height = int(slide.properties['openslide.bounds-height'])
            # wsi = np.array(slide.read_region((mirax_x, mirax_y), 0, (mirax_width, mirax_height)).convert('RGB'),
            #                  dtype='uint8')
            wsi = np.zeros((mirax_height, mirax_width, 3), dtype='uint8')
            for widths in range(0, mirax_width, mirax_width // 3):
              for heights in range(0, mirax_height, mirax_height // 3):
                if (widths + mirax_width // 3) > mirax_width:
                  end_width = mirax_width
                else:
                  end_width = widths + mirax_width // 3
                if (heights + mirax_height // 3) > mirax_height:
                  end_height = mirax_height
                else:
                  end_height = heights + mirax_height // 3
                wsi[heights:end_height, widths:end_width, :] = \
                  np.array(slide.read_region((widths + mirax_x, heights + mirax_y), 0,
                                         (end_width - widths, end_height - heights)).convert('RGB'),
                                          dtype='uint8')
                if args.do_not_print_much is False:
                  print('MIRAX patch [{} - {}, {} - {}] added to our WSI image!'.format(widths + mirax_x,
                                                                                      end_width + mirax_x,
                                                                                      heights + mirax_y,
                                                                                      end_height + mirax_y))
        except:
          if ((int(slide.properties['openslide.bounds-width']) * int(slide.properties['openslide.bounds-height'])) > (200000 * 100000)):
            if args.do_not_print_much is False:
              print('Scene loading failed! The MIRAX ROI is too big. Skipping...')
            process_df.loc[index, 'status'] = 'failed_to_load'
            continue
          else:
            wsi = np.zeros((slide.level_dimensions[0][1], slide.level_dimensions[0][0], 3), dtype='uint8')
            # load the mrxs files in pieces as they are very large
            for widths in range(0, slide.level_dimensions[0][0], slide.level_dimensions[0][0]//5):
              for heights in range(0, slide.level_dimensions[0][1], slide.level_dimensions[0][1]//5):
                if (widths + slide.level_dimensions[0][0]//5) > slide.level_dimensions[0][0]:
                  end_width = slide.level_dimensions[0][0]
                else:
                  end_width = widths + slide.level_dimensions[0][0]//5
                if (heights + slide.level_dimensions[0][1]//5) > slide.level_dimensions[0][1]:
                  end_height = slide.level_dimensions[0][1]
                else:
                  end_height = heights + slide.level_dimensions[0][1]//5
                wsi[heights:end_height,widths:end_width,:] = \
                  np.array(slide.read_region((widths, heights), 0, (end_width-widths, end_height-heights)).convert('RGB'),
                           dtype='uint8')
                if args.do_not_print_much is False:
                  print('MIRAX patch [{} - {}, {} - {}] added to our WSI image!'.format(widths,
                                                                                        end_width,
                                                                                        heights,
                                                                                        end_height))
      except:
        if args.do_not_print_much is False:
          print('Scene loading failed! Likely an issue with scan file. Skipping...')
        process_df.loc[index, 'status'] = 'failed_to_load'
        continue
    elif ('.tif' in row['slide_id']):
      if args.do_not_print_much is False:
        print('.tif scan found, using tifffile/slideio to load {}'.format(row['slide_id']))
      if ' - Series ' in row['slide_id']:
        scene_id = int(row['slide_id'].split(' - Series ')[1][0])
        full_path = os.path.join(WSI_PATH, row['slide_id'].split(' - Series ')[0])
      else:
        scene_id = 0
        full_path = os.path.join(WSI_PATH, row['slide_id'])
      if scene_id > 0:
        if args.do_not_print_much is False:
          print('WARNING: scene id for a .tif image is higher than 0, there is likely an error...')
        process_df.loc[index, 'status'] = 'failed_to_load'
        continue
      else:
        try:
          wsi = tifffile.imread(full_path)
        except:
          slideio_wsi = slideio.open_slide(full_path, driver="GDAL")
          scene = slideio_wsi.get_scene(scene_id)
          try:
            wsi = scene.read_block()
          except:
            if args.do_not_print_much is False:
              print('Scene loading failed! Likely an issue with scan file. Skipping...')
            process_df.loc[index, 'status'] = 'failed_to_load'
            continue
    else:
      print('**** USING THIS ONE TO READ WSI ****')
      full_path = os.path.join(WSI_PATH, row['slide_id'])
      if args.do_not_print_much is False:
        print('Using tifffile/slideio to load {}'.format(row['slide_id']))
      try:
        wsi = tifffile.imread(full_path)
      except:
        print('--- slideio ---')
        slideio_wsi = slideio.open_slide(full_path, driver="GDAL")
        scene = slideio_wsi.get_scene(0)
        try:
          wsi = scene.read_block()
        except:
          if args.do_not_print_much is False:
            print('Scene loading failed! Likely an issue with scan file. Skipping...')
          process_df.loc[index, 'status'] = 'last_else_failed_to_load'
          process_df.to_csv(CSV_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID), index=False)
          continue
          
  
    if wsi.shape[0] == 0:
      if args.do_not_print_much is False:
        print('{} was not properly loaded by tifffile. The tiff may be corrupt! Skipping..')
      process_df.loc[index, 'status'] = 'failed_to_load'
      continue
  
    if args.do_not_print_much is False:
      print('{} loaded!'.format(row['slide_id']))
    # instantiate a WSI mask np array for both the confidence mapped pixel values and the counts
    if args.do_not_print_much is False:
      print('Instantiating our pixel conf/counts np arrays...')
    out_shape = (int(wsi.shape[0]//OUT_RATIO), int(wsi.shape[1]//OUT_RATIO), 3)

    try:
      wsi_mask = np.zeros(out_shape, dtype='float64')
    except:
      process_df.loc[index, 'status'] = 'unable_to_alloc_mem_for_wsi_mask'
      process_df.to_csv(CSV_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID), index=False)
      del wsi
      gc.collect()
      continue

    if args.do_not_print_much is False:
      print('Pixel conf/counts np arrays instantiated! Generating GT masks if we have them...')
  
    #################################
    ########### GT MAPPING ##########
    #################################
    # get the tissue mask
    # load the thumbnail
    wsi_thumb = cv2.resize(wsi, (int(args.scale*4096), int(args.scale*4096)), interpolation=cv2.INTER_CUBIC)
    # normalize it after converting to torch and float
    wsi_thumb = torch.from_numpy(wsi_thumb).permute(2, 0, 1)
    wsi_thumb = wsi_thumb / 255.
    wsi_thumb = torchvision.transforms.Normalize(list(tissue_type_norm[0]), list(tissue_type_norm[1]))(wsi_thumb)
    # pass it through our model
    wsi_thumb = wsi_thumb.unsqueeze(0)
    wsi_thumb = wsi_thumb.to(device=tissue_device, dtype=torch.float32)
    downscaled_mask = predict_img(net=tissue_net, img=wsi_thumb)
    # get the binary tissue mask
    downscaled_mask = np.argmax(downscaled_mask, axis=0)
    downscaled_mask[downscaled_mask > 0] = 255
    downscaled_mask = downscaled_mask.astype('uint8')
    # upscale the mask
    full_tissue_masking_canvas = cv2.resize(downscaled_mask, (wsi.shape[1], wsi.shape[0]), interpolation=cv2.INTER_NEAREST)
    # get the contours and fill them on the full scaled mask
    roi_coords, _ = cv2.findContours(full_tissue_masking_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # remove contours with an area less than 2000
    roi_coords = [contour for contour in roi_coords if cv2.contourArea(contour) > 2000]
    cv2.drawContours(full_tissue_masking_canvas, roi_coords, -1, 255, thickness=cv2.FILLED)
    # scale the canvas back down for future use
    full_tissue_masking_canvas = cv2.resize(full_tissue_masking_canvas, (int(wsi.shape[1]//OUT_RATIO), int(wsi.shape[0]//OUT_RATIO)),
                                            interpolation=cv2.INTER_NEAREST)
    # now convert masks to bool for later
    full_tissue_masking_canvas[full_tissue_masking_canvas > 0] = 1
    full_tissue_masking_canvas = full_tissue_masking_canvas.astype('bool')
  
  
    # get the gt masks if there are any
    class_gt_tissue_masks = {key: [] for key in CLASSES}
    class_gt_masks = {key: [] for key in CLASSES}
    class_roi_gt_masks = {key: [] for key in CLASSES}
    # make the special case for the tif
    # === LOOP 1 ===
    start_time = time.time()
    class_gt_tissue_masks, class_roi_gt_masks, class_gt_masks = loop1(args, CLASSES, row, GT_MASKS_WSI_IDS, GT_MASKS, wsi_mask, SPECIAL_CLASS_CASES, OUT_RATIO, \
                                class_gt_masks, roi_coords, OUT_DIM, IN_DIM, class_gt_tissue_masks, class_roi_gt_masks)
    print('===> loop1 takes: {} seconds'.format(time.time() - start_time))
  
    #############################
    ######## PRED MAPPING #######
    #############################
  
    # get the patch coordinates
    tot_coords = process_contours(args,
                                 roi_coords,
                                 patch_size=4096,
                                 step_size=1024,
                                 use_padding=True,
                                 top_left=None,
                                 bot_right=None,
                                 img_h=wsi.shape[0],
                                 img_w=wsi.shape[1])
    if len(tot_coords) < 100:
      process_df.loc[index, 'status'] = 'too_few_patches_returned_by_tot_coords'
      process_df.to_csv(CSV_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID), index=False)
      err_flag = True
      del wsi
      del wsi_mask
      del wsi_thumb
      del downscaled_mask
      del tot_coords
      gc.collect()
      continue
  
  
    # extract the patches from the WSI, append them to a single torch object after resizing
    # get a list of np array patches
    if args.do_not_print_much is False:
      print('Extracting patches for WSI {}...'.format(row['slide_id']))
    
    ### A WRAPPER LOOP FOR LOOP 2 AND LOOP 3 TO DEAL WITH OVERFLOW MEMORY ###
    start_idx = 0
    err_flag = False
    manager = Manager()
    shared_class_prediction_confs = manager.dict()
    # class_prediction_confs['Glomerulus'] = [] # Wrong, should also be shared list
    # class_prediction_confs['Arteriole'] = []
    # class_prediction_confs['Artery'] = []
    shared_class_prediction_confs['Glomerulus'] = manager.list()
    shared_class_prediction_confs['Arteriole'] = manager.list()
    shared_class_prediction_confs['Artery'] = manager.list()
    while True:
      # ================================================ LOOP 2 ===========================================================
      start_time = time.time()
      patches, next_idx = loop2(args, tot_coords, wsi, IN_DIM, OUT_DIM, start_idx=start_idx)
      print('[main] next_idx: ', next_idx)
      print('===> loop2 takes: {} seconds'.format(time.time() - start_time))
  
      # ================================================ LOOP 3 ====================================================
      # create a band which we use to remove annotations that are fully in them
      kernel = np.ones((OUT_DIM, OUT_DIM), dtype='bool')
      kernel[OUT_DIM//10:-OUT_DIM//10, OUT_DIM//10:-OUT_DIM//10] = False
  
      if args.do_not_print_much is False:
        print('{} patches extracted! Passing patches through our model...'.format(len(patches)))
      # iterate through the list according to batch size, generate prediction confidence dict so we can get cutoffs
      class_prediction_confs = {key: [] for key in CLASSES}
      start_time = time.time()

      return_mask = True if next_idx == -1 else False
      print('[main] return_mask: ', return_mask)
      try:
        ret_wsi_mask = loop3(args, patches, GPU_BATCH_SIZE, inference_detector, model, tot_coords, kernel, \
                        wsi_mask, CLASSES, OUT_RATIO, OUT_DIM, args.cuda_compseg, start_idx, shared_class_prediction_confs, return_mask=return_mask)
      except:
        process_df.loc[index, 'status'] = 'prob_at_loop3'
        process_df.to_csv(CSV_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID), index=False)

        shr_name = 'global_wsi_mask{}'.format(args.cuda_compseg)
        try:
          shm = shared_memory.SharedMemory(create=False, size=wsi_mask.nbytes, name=shr_name)
          shm.close()
          shm.unlink()
        except:
          print('No shared memory to release')

        err_flag = True
        del patches
        del wsi
        del wsi_thumb
        del downscaled_mask
        del tot_coords
        del wsi_mask
        gc.collect()
        break
      print('===> loop3 takes: {} seconds'.format(time.time() - start_time))
      # patches.clear()
      # condition for break
      if next_idx == -1:
        wsi_mask = ret_wsi_mask  
        break
      else:
        start_idx = next_idx

    if err_flag is True:
      continue

    if args.do_not_print_much is False:
      print('Patch-wise prediction for {} complete!'.format(row['slide_id']))
    wsi_mask_cleaned = wsi_mask
    # print('[main] np.unique(wsi_mask): ', np.unique(wsi_mask))
    # print('[main] np.unique(wsi_mask_cleaned): ', np.unique(wsi_mask_cleaned))
    # plot heatmaps?
  
    if args.do_not_print_much is False:
      print('Generating our cutoff thresholds...')
    slide_wise_cutoffs = {key: 1 for key in CLASSES}
    # === LOOP 4 ===
    start_time = time.time()
    loop4(args, CLASSES, GT_MASKS_WSI_IDS, row, wsi_mask_cleaned, class_gt_tissue_masks, process_df, \
          index, conf_mlp_norm_test_to_val, conf_mlp_norm_val_to_test, slide_wise_cutoffs)
    print('===> loop4 takes: {} seconds'.format(time.time() - start_time))
  
    if args.do_not_print_much is False:
      print('Cutoff thresholds generated! Class thresholds: {}, Generating our global binary masks...'.format(slide_wise_cutoffs))
    # blur the prediction confidence values first to avoid empty areas between the same annotations
    # === LOOP 5 ===
    start_time = time.time()
    wsi_mask_cleaned = loop5(args, CLASSES, row, GT_MASKS_WSI_IDS, full_tissue_masking_canvas, wsi_mask_cleaned, class_gt_masks, \
          slide_wise_cutoffs, process_df, index, class_gt_tissue_masks, class_roi_gt_masks)
    print('===> loop5 takes: {} seconds'.format(time.time() - start_time))
  
    # change the dtype
    wsi_mask_cleaned = wsi_mask_cleaned.astype('uint8')
    if args.do_not_print_much is False:
      print('Global binary masks generated! Generating our classwise contours...')
    all_contours = {key: [] for key in CLASSES}
    aux_contours = {key: [] for key in CLASSES}
    # === LOOP 6 ==
    start_time = time.time()
    all_contours, aux_contours = loop6(args, CLASSES, row, GT_MASKS_WSI_IDS, wsi_mask_cleaned, \
          class_gt_tissue_masks, aux_contours, all_contours)
    print('===> loop6 takes: {} seconds'.format(time.time() - start_time))
  
    if args.do_not_print_much is False:
      print('Contours extracted. Extracting TI tiles...')
    x_coords = np.array([])
    y_coords = np.array([])
    sum_percents = np.array([])
    wsi_tile_mask = np.sum(wsi_mask_cleaned, axis=2).astype('bool')
    xor_tile_mask = np.bitwise_xor(full_tissue_masking_canvas, wsi_tile_mask)
    # === LOOP 7 ===
    start_time = time.time()
    x_coords, y_coords = loop7(args, xor_tile_mask, x_coords, y_coords, sum_percents)
    print('===> loop7 takes: {} seconds'.format(time.time() - start_time))
  
    # extract the tiles
    if args.do_not_print_much is False:
      print('{} total TI tiles extracted! Saving...'.format(len(x_coords)))
    # === LOOP 8 ===
    start_time = time.time()
    try:
      loop8(args, x_coords, y_coords, xor_tile_mask, wsi, row, TI_DIR)
    except:
      process_df.loc[index, 'status'] = 'prob_at_loop8_no_tile_name_to_be_saved'
      process_df.to_csv(CSV_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID), index=False)
      continue

    print('===> loop8 takes: {} seconds'.format(time.time() - start_time))
  
    if args.do_not_print_much is False:
      print('TI tiles saved. Generating WSI thumbnail image for visualization...')
    # plot the contours and save the binary masks
    # get the scaled image and contours
    wsi_mini = cv2.resize(wsi, dsize=(wsi.shape[1] // FIG_SCALE, wsi.shape[0] // FIG_SCALE))
  
    coef_y = wsi_mini.shape[0] / wsi_mask_cleaned.shape[0]
    coef_x = wsi_mini.shape[1] / wsi_mask_cleaned.shape[1]
  
    mini_all_contours = {key: copy.deepcopy(all_contours[key]) for key in CLASSES}
    # === LOOP 9 ===
    start_time = time.time()
    loop9(args, CLASSES, mini_all_contours, coef_x, coef_y)
    print('===> loop9 takes: {} seconds'.format(time.time() - start_time))
  
    # get contour counts, rounded contour cutoffs
    contour_counts = {key: len(mini_all_contours[key]) for key in CLASSES}
    slide_wise_cutoffs = {key: round(slide_wise_cutoffs[key], 3) for key in CLASSES}
  
    # plotting the figure
    fig, axs = plt.subplots(1)
    # first draw the contours on the image (currently only supports 8 classes)
    # === LOOP 10 ===
    start_time = time.time()
    loop10(args, CLASSES, wsi_mini, mini_all_contours, COLORS, axs, full_tissue_masking_canvas, coef_x, coef_y)
    print('===> loop10 takes: {} seconds'.format(time.time() - start_time))
  
    axs.imshow(wsi_mini)
    axs.legend(title='Classes', title_fontsize=4, fontsize=3, loc='center left', bbox_to_anchor=(1, 0.5))
    axs.get_xaxis().set_visible(False)
    axs.get_yaxis().set_visible(False)
    axs.set_title('Classwise contours, image name: {} \n'
                  'Class confidence thresholds: {} \n'
                  'Class annotation counts: {}'.format(row['slide_id'], slide_wise_cutoffs, contour_counts), fontsize=4,
                  y=1.001, fontweight='bold')
    fig.tight_layout()
    plt.savefig(PLOT_DIR + row['slide_id'] + '.png', dpi=1000, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
  
    if args.do_not_print_much is False:
      print('Plot generated. Updating our processing DF, and checking for GT...')
    # === LOOP 11 ===
    start_time = time.time()
    all_contours = loop11(args, CLASSES, process_df, index, contour_counts, slide_wise_cutoffs, row, GT_MASKS_WSI_IDS, \
           wsi_mask_cleaned, full_tissue_masking_canvas, class_gt_tissue_masks, class_gt_masks, all_contours, aux_contours)
    print('===> loop11 takes: {} seconds'.format(time.time() - start_time))
  
    # we will need to check if any contours intersect, in which cas we will save a new contour with a combined id
    # make a storage variable which we will use to store contours that have no intersection
    if args.do_not_print_much is False:
      print('Identifying uncertain contours, saving them...')
    delete_ids = {keys: [] for keys in CLASSES}
    # === LOOP 12 ===
    start_time = time.time()
    delete_ids = loop12(args, CLASSES, all_contours, OUT_RATIO, row, wsi, \
           IMG_DIR, MASK_DIR, MULTI_DIR, delete_ids)
    print('===> loop12 takes: {} seconds'.format(time.time() - start_time))
  
    if args.do_not_print_much is False:
      print('Saving the remaining contours...')
    # save the remaining masks
    # === LOOP 13 ===
    start_time = time.time()
    loop13(args, CLASSES, all_contours, delete_ids, OUT_RATIO, \
           row, wsi, IMG_DIR, MASK_DIR, MULTI_DIR)
    print('===> loop13 takes: {} seconds'.format(time.time() - start_time))
  
    if args.do_not_print_much is False:
      print('All contours saved. The slide has been processed.')
    process_df.loc[index, 'status'] = 'processed'
    process_df.loc[index, 'reprocess'] = 0
    process_df.to_csv(CSV_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID), index=False)

    del patches
    del wsi_mask
    del wsi_mini
    del coef_y
    del coef_x
    del mini_all_contours
    del contour_counts
    del slide_wise_cutoffs
    del x_coords
    del y_coords
    del sum_percents
    del wsi_tile_mask
    del xor_tile_mask
    del wsi_mask_cleaned
    del all_contours
    del aux_contours
    del class_prediction_confs
    del kernel
    del tot_coords
    del wsi
    del out_shape
    del wsi_thumb
    del downscaled_mask
    del full_tissue_masking_canvas
    del roi_coords
    del class_gt_tissue_masks
    del class_gt_masks
    del class_roi_gt_masks
    del delete_ids
    gc.collect()