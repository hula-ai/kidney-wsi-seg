import copy

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


def polygonFromMask(maskedArr):
  # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
  contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  segmentation = []
  valid_poly = 0
  for contour in contours:
  # Valid polygons have >= 6 coordinates (3 points)
     if contour.size >= 6:
        segmentation.append(contour.astype(float).flatten().tolist())
        valid_poly += 1
  if valid_poly == 0:
     raise ValueError
  return segmentation

def binary_metrics(pred_mask, gt_mask):

    negative = 0
    positive = 1

    pred_mask_area = np.count_nonzero(pred_mask == positive)
    gt_mask_area = np.count_nonzero(gt_mask == positive)
    print('Computing intersection...')
    intersection = np.count_nonzero(np.logical_and(pred_mask, gt_mask))
    print('Computing IoU...')
    if (pred_mask_area+gt_mask_area-intersection) == 0:
        iou = 1
    elif intersection == 0:
        iou = 0
    else:
        iou = intersection/(pred_mask_area+gt_mask_area-intersection)
    print('Computing Precision/Recall metrics...')
    print('Computing TP...')
    tp = np.sum(np.logical_and(pred_mask == positive, gt_mask == positive))
    print('Computing TN...')
    tn = np.sum(np.logical_and(pred_mask == negative, gt_mask == negative))
    print('Computing FP...')
    fp = np.sum(np.logical_and(pred_mask == positive, gt_mask == negative))
    print('Computing FN...')
    fn = np.sum(np.logical_and(pred_mask == negative, gt_mask == positive))

    if tp == 0:
        filtered_precision = 0
        filtered_recall = 0
        filtered_f1score = 0
    else:
        filtered_precision = tp / (tp + fp)
        filtered_recall = tp / (tp + fn)
        filtered_f1score = 2 * ((filtered_precision * filtered_recall) / (filtered_precision + filtered_recall))

    if tn == 0:
        filtered_specificity = 0
    else:
        filtered_specificity = tn / (tn + fp)

    return iou, filtered_precision, filtered_recall, filtered_f1score, filtered_specificity

def contour_intersect(cnt_ref, cnt_query):
    ## Contours are both an np array of points
    ## Check for bbox intersection, then check pixel intersection if bboxes intersect

    # first check if it is possible that any of the contours intersect
    x1, y1, w1, h1 = cv2.boundingRect(cnt_ref)
    x2, y2, w2, h2 = cv2.boundingRect(cnt_query)
    # get contour areas
    area_ref = cv2.contourArea(cnt_ref)
    area_query = cv2.contourArea(cnt_query)
    # get coordinates as tensors
    box1 = torch.tensor([[x1, y1, x1 + w1, y1 + h1]], dtype=torch.float)
    box2 = torch.tensor([[x2, y2, x2 + w2, y2 + h2]], dtype=torch.float)
    # get bbox iou
    iou = bops.box_iou(box1, box2)

    if iou == 0:
        # bboxes dont intersect, so contours dont either
        return False
    else:
        # bboxes intersect, now check pixels
        # get the height, width, x, and y of the smaller contour
        if area_ref >= area_query:
            h = h2
            w = w2
            x = x2
            y = y2
        else:
            h = h1
            w = w1
            x = x1
            y = y1

        # get a canvas to draw the small contour and subspace of the large contour
        contour_canvas_ref = np.zeros((h, w), dtype='uint8')
        contour_canvas_query = np.zeros((h, w), dtype='uint8')
        # draw the pixels areas, filled (can also be outline)
        cv2.drawContours(contour_canvas_ref, [cnt_ref], -1, 255, thickness=cv2.FILLED,
                         offset=(-x, -y))
        cv2.drawContours(contour_canvas_query, [cnt_query], -1, 255, thickness=cv2.FILLED,
                         offset=(-x, -y))

        # check for any pixel overlap
        return np.any(np.bitwise_and(contour_canvas_ref, contour_canvas_query))

def contour_iou(cnt_ref, cnt_query):
    ## Contour is a list of points

    polygon1 = Polygon(cnt_ref)
    polygon2 = Polygon(cnt_query)
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    return intersect / union

class Contour_Checking_fn(object):
	# Defining __call__ method
	def __call__(self, pt):
		raise NotImplementedError

# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(Contour_Checking_fn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [(center[0] - self.shift, center[1] - self.shift),
                          (center[0] + self.shift, center[1] + self.shift),
                          (center[0] + self.shift, center[1] - self.shift),
                          (center[0] - self.shift, center[1] + self.shift)
                          ]
        else:
            all_points = [center]

        for points in all_points:
            if cv2.pointPolygonTest(self.cont, (int(points[0]), int(points[1])), False) >= 0:
                return 1
        return 0

def process_contours(contours,
                     patch_size=4096,
                     step_size=1024,
                     use_padding=True,
                     top_left=None,
                     bot_right=None,
                     img_h=0,
                     img_w=0):

    n_contours = len(contours)
    fp_chunk_size = math.ceil(n_contours * 0.05)
    tot_coords = np.array([])
    for idx, cont in enumerate(contours):
        if (idx + 1) % fp_chunk_size == fp_chunk_size:
            print('Processing contour {}/{}'.format(idx, n_contours))

        asset_dict, attr_dict = process_contour(cont, patch_size, step_size,
                                                use_padding, top_left, bot_right, img_h, img_w)

        if len(asset_dict) > 0:
            if len(tot_coords) == 0:
                tot_coords = asset_dict['coords']
            else:
                tot_coords = np.append(tot_coords, asset_dict['coords'], axis=0)

    return tot_coords

def process_contour(cont,
                    patch_size=4096,
                    step_size=1024,
                    use_padding=True,
                    top_left=None,
                    bot_right=None,
                    img_h=0,
                    img_w=0):

    start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (
    0, 0, patch_size, patch_size)

    ref_patch_size = (patch_size, patch_size)

    if use_padding:
        stop_y = start_y + h
        stop_x = start_x + w
    else:
        stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
        stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)

    print("Bounding Box:", start_x, start_y, w, h)
    print("Contour Area:", cv2.contourArea(cont))

    if bot_right is not None:
        stop_y = min(bot_right[1], stop_y)
        stop_x = min(bot_right[0], stop_x)
    if top_left is not None:
        start_y = max(top_left[1], start_y)
        start_x = max(top_left[0], start_x)

    if bot_right is not None or top_left is not None:
        w, h = stop_x - start_x, stop_y - start_y
        if w <= 0 or h <= 0:
            print("Contour is not in specified ROI, skip")
            return {}, {}
        else:
            print("Adjusted Bounding Box:", start_x, start_y, w, h)

    cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)

    step_size_x = step_size
    step_size_y = step_size

    x_range = np.arange(start_x, stop_x, step=step_size_x)
    y_range = np.arange(start_y, stop_y, step=step_size_y)
    x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
    coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

    results = [process_coord_candidate(coord, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
    results = np.array([result for result in results if result is not None])

    print('Extracted {} coordinates'.format(len(results)))

    if len(results) > 1:
        asset_dict = {'coords': results}

        attr = {'patch_size': patch_size}

        attr_dict = {'coords': attr}
        return asset_dict, attr_dict

    else:
        return {}, {}

def process_coord_candidate(coord, ref_patch_size, cont_check_fn):
    if isInContours(cont_check_fn, ref_patch_size, coord):
        return coord
    else:
        return None

def isInContours(cont_check_fn, ref_patch_size, pt):
    if cont_check_fn(pt):
        return 1
    return 0

class MLP(nn.Module):
    '''
      Multilayer Perceptron for regression.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(401, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)

def get_args():
    # unet tissue segmentation arguments
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_checkpoints/best_model.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--viz', '-v', default=False, action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', default=True, action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--cuda_tiseg', '-ct', default='cuda:4', help='Specify gpu to be used for tissue unet')
    parser.add_argument('--cuda_compseg', '-cc', default='cuda:4', help='Specify gpu to be used for compartment seg')
    parser.add_argument('--sub_df_id', '-sdf', default=6, type=int, help='Specify the sub df id, must be between 0 and 7')
    parser.add_argument('--quproj', '-proj', default='SLE', help='Specify the qupath project id')
    parser.add_argument('--from_index', type=int, required=True, help='Processing from index')
    parser.add_argument('--to_index', type=int, required=True, help='Processing to index')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir')
    parser.add_argument('--wsi_seg_path', type=str, required=True, help='WSI segmentation path')

    return parser.parse_args()

def sub_df_uniques(lst, n):
    return [lst[i::n] for i in range(n)]

# construct the tissue segmentation model
args = get_args()

tissue_net = UNet(n_channels=3, n_classes=4, bilinear=args.bilinear)

tissue_device = torch.device(args.cuda_tiseg if torch.cuda.is_available() else 'cpu')

tissue_net.to(device=tissue_device)
tissue_net.load_state_dict(torch.load(args.model, map_location=tissue_device))

tissue_type_norm = np.load('/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_checkpoints/train_norm.npy')

print('Tissue seg. Model loaded!')

# construct the model that we are going to use to process data

# Specify the path to model config and checkpoint file
config_file = '/home/cougarnet.uh.edu/pcicales/PycharmProjects/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py'
checkpoint_file = '/home/cougarnet.uh.edu/pcicales/PycharmProjects/mmdetection/models/epoch_4.pth'
# config_file = '/home/cougarnet.uh.edu/hqvo3/Projects/digital_pathology/codes/MILxseg/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py'
# checkpoint_file = '/data/hqvo3/mmdet/run_6_classes/latest.pth'

# previously, each DF was instantiated for the entire set; because multi-gpu is tricky in mmdetection, we now break up
# each DF into 8 sub DFs
SUB_DF_COUNT = 1
# set sub df ID, pythonic
SUB_DF_ID = args.sub_df_id

# build the model from a config file and a checkpoint file
device = args.cuda_compseg
model = init_detector(config_file, checkpoint_file, device=device)

# pick the project we are processing, can be any of the qupath_proj_files keys
PROJECT = args.quproj

# build the conf cutoff model
conf_mlp = torch.load('/data/public/HULA/WSIs_renal_compartment_segmentations/val_to_test.pth')
conf_mlp = conf_mlp.float()
conf_mlp.eval()
conf_mlp_norm_val_to_test = np.load('/data/public/HULA/WSIs_renal_compartment_segmentations/val_to_test_norm.npy')
conf_mlp_norm_test_to_val = np.load('/data/public/HULA/WSIs_renal_compartment_segmentations/test_to_val_norm.npy')

# set the GPU batch size
GPU_BATCH_SIZE = 22

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
TMA_qupath_proj_imgs = [im.image_name for im in QuPathProject('/data/public/HULA/TMA/QuPath_Projects/TMASegmentationR14/project.qpproj').images]
AMR_qupath_proj_imgs = [im.image_name for im in QuPathProject('/data/public/HULA/AMR/QuPath_Projects/AMRValidation2/project.qpproj').images]
Pan_GN_Cologne_qupath_proj_imgs = [im.image_name for im in QuPathProject('/data/public/HULA/Pan_GN/QuPath_Projects/MultiGNSegmentationCologneR3/project.qpproj').images]
Pan_GN_Szeged_qupath_proj_imgs = [im.image_name for im in QuPathProject('/data/public/HULA/Pan_GN/QuPath_Projects/PanGNSzegedSegmentationR3/project.qpproj').images]
Pan_GN_Lille_qupath_proj_imgs = [im.image_name for im in QuPathProject('/data/public/HULA/Pan_GN/QuPath_Projects/PanGNLilleSegmentationR4/project.qpproj').images]
Pan_GN_Bari_qupath_proj_imgs = [im.image_name for im in QuPathProject('/data/public/HULA/Pan_GN/QuPath_Projects/PanGNBariSegmentationR2/project.qpproj').images]
SLE_qupath_proj_imgs = [im.image_name for im in QuPathProject('/data/public/HULA/SLE/QuPath_Projects/SLE-SegmentationV4/project.qpproj').images]
Podonet_qupath_proj_imgs = [im.image_name for im in QuPathProject('/data/public/HULA/Podonet/qupath_projects/PodonetParaffinSegmentationR1/project.qpproj').images]
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
IN_DIM = 4096
OUT_DIM = 2048
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
PROCESS_DF = pd.read_csv('/data/public/HULA/WSIs_renal_tissue_masks_CLAM/process_list_autogen.csv')

# set the directory for the slides and the respective tissue mask data
WSI_PATH = '/data/public/HULA/Podonet/wsi/'
WSI_MASK_PATH = '/data/public/HULA/WSIs_renal_tissue_masks_CLAM/'
WSI_SEG_PATH = args.wsi_seg_path

# get the unique directory extension for the .mrxs file formats
mrxs_special_dir = {}
mrxs_dirs = next(os.walk(WSI_PATH))[1]
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
if os.path.isfile(os.path.join(WSI_SEG_PATH, 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID))):
    print('[**] path to process_list_csv: ', WSI_SEG_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID))
    # get all of the file names, this assumes you have generated all SUB_DF_COUNT dfs already
    print('Checking for new tissue annotations...')
    original_slideids = []
    for sub_id in range(SUB_DF_COUNT):
        original_slideids += pd.read_csv(WSI_SEG_PATH + 'process_list_{}_{}.csv'.format(PROJECT, sub_id))['slide_id'].to_list()
    uniques = list(set(qupath_proj_files[PROJECT]) - set(original_slideids))
    print('--> uniques: ', uniques)
    if len(uniques) > 0:
        print('{} new slides found! Adding them for processing...'.format(len(uniques)))
        for sub_id in range(SUB_DF_COUNT):
            sub_process_df = pd.read_csv(WSI_SEG_PATH + 'process_list_{}_{}.csv'.format(PROJECT, sub_id))
            sub_uniques = sub_df_uniques(uniques, SUB_DF_COUNT)[sub_id] # Why the i-th element is of the corresponding sub_id
            for new_slides in sub_uniques:
                print('Adding... {}'.format(new_slides))
                base_dict = {'slide_id': new_slides,
                              'status': 'tbp'}
                rep_dict = {'reprocess': 0}
                full_dict = {**base_dict, **full_meta_dict.copy(), **rep_dict}
                sub_process_df = sub_process_df.append(full_dict, ignore_index=True)
            print('--> Count #rows of new csv: ', len(sub_process_df.index))
            sub_process_df.to_csv(WSI_SEG_PATH + 'process_list_{}_{}.csv'.format(PROJECT, sub_id), index=False)
    else:
        print('No new slides found.')
    print('Loading our processing df...')
    process_df = pd.read_csv(WSI_SEG_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID))
else:
    print('Generating our processing df...')
    process_df = pd.DataFrame(columns=['slide_id', 'status'] + full_meta_list.copy() + ['reprocess'])

    qupath_proj_imgs = list(set(qupath_proj_files[PROJECT]))
    qupath_proj_imgs.sort()
    sub_uniques = sub_df_uniques(qupath_proj_imgs, SUB_DF_COUNT)
    # print('[**] sub_uniques: ', sub_uniques)
    # for tmp_idx in range(SUB_DF_COUNT):
    #   print('- tmp_idx: {}, count: {}'.format(tmp_idx, len(sub_uniques[tmp_idx])))
    #   print(sub_uniques[tmp_idx][0])
    #   if 'KAMBHAM_8_HE_2.svs' in sub_uniques[tmp_idx]:
    #     print('KAMBHAM_8_HE_2.svs in sub df id {}'.format(tmp_idx))
    for idx, slide_id in enumerate(sub_uniques[SUB_DF_ID]):
        # print('- QuPath proj, slide id: ', idx, slide_id)
        base_dict = {'slide_id': slide_id,
                      'status': 'tbp'}
        rep_dict = {'reprocess': 0}
        full_dict = {**base_dict, **full_meta_dict.copy(), **rep_dict}
        # process_df = process_df.append(full_dict, ignore_index=True)
        process_df = pd.concat([process_df, pd.DataFrame.from_records([full_dict])], ignore_index=True)
    process_df.to_csv(WSI_SEG_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID), index=False)

# we iterate through the df to generate our predictions
for index, row in process_df.iterrows():
    print('[INDEX AND SLIDE ID]', index, ' ', row['slide_id'])
    if index >= args.from_index and index <= args.to_index:
      pass
    else:
      continue
    if row['status'] == 'processed':
        # check if the reprocess flag has been raised
        if row['reprocess']:
            print('{} has already been processed, but the reprocess flag is set to 1. Reprocessing...'.format(row['slide_id']))
        else:
            print('{} has already been processed! Skipping. '
                  'If you want to reprocess, set the reprocess flag to 1.'.format(row['slide_id']))
            continue
        print('Processing {}...'.format(row['slide_id']))

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
                        print('Deleting {}...'.format(old_pred_mask))
                        os.remove(mask_dir + old_pred_mask)
                        os.remove(multi_dir + old_pred_mask.split('-mask.png')[0] + '-multi.png')
    # delete the old TIs
    for old_ti in os.listdir(TI_DIR):
        splits = old_ti.split('_xcoord_')
        if (row['slide_id'] in splits[0]) or ((row['slide_id'].split(' - Series ')[0] in splits[0]) and ('.tif' in row['slide_id'])):
            print('Deleting {}...'.format(old_ti))
            os.remove(TI_DIR + old_ti)

    # load the WSI
    print('Loading {}...'.format(row['slide_id']))
    if '- macro' in row['slide_id']:
        print('Thumbnail found. Skipping...')
        process_df.loc[index, 'status'] = 'thumbnail'
        continue
    elif ('.scn' in row['slide_id']) or ('.sc_' in row['slide_id']):
        print('Leica scan found, using SlideIO to load {}'.format(row['slide_id']))
        scene_id = int(row['slide_id'].split(' - Series ')[1][0])
        full_path = WSI_PATH + row['slide_id'].split(' - Series ')[0]
        slideio_wsi = slideio.open_slide(full_path, driver="SCN")
        scene = slideio_wsi.get_scene(scene_id - 1)
        try:
            wsi = scene.read_block()
        except:
            print('Scene loading failed! Likely an issue with scan file. Skipping...')
            process_df.loc[index, 'status'] = 'failed_to_load'
            continue
    elif ('.svs' in row['slide_id']):
        print('.svs scan found, using SlideIO to load {}'.format(row['slide_id']))
        full_path = WSI_PATH + row['slide_id']
        slideio_wsi = slideio.open_slide(full_path, driver="SVS")
        scene = slideio_wsi.get_scene(0)
        try:
            wsi = scene.read_block()
        except:
            print('Scene loading failed! Likely an issue with scan file. Skipping...')
            process_df.loc[index, 'status'] = 'failed_to_load'
            continue
    elif ('.czi' in row['slide_id']) and (' - Scene #' in row['slide_id']):
        print('.czi scan found, using AICSImage to load {}'.format(row['slide_id']))
        scene_id = int(row['slide_id'].split(' - Scene #')[1])
        full_path = os.path.join(WSI_PATH, row['slide_id'].split(' - Scene #')[0])
        try:
            full_scan = AICSImage(full_path)
            full_scan.set_scene(scene_id - 1)
            wsi = full_scan.get_image_data("YXS", T=0)
            wsi = cv2.cvtColor(wsi, cv2.COLOR_BGR2RGB)
        except:
            print('Scene loading failed! Likely an issue with scan file. Skipping...')
            process_df.loc[index, 'status'] = 'failed_to_load'
            continue
    elif ('.czi' in row['slide_id']) and (' - Scene #' not in row['slide_id']):
        print('.czi scan found, using AICSImage to load {}'.format(row['slide_id']))
        scene_id = 0
        full_path = os.path.join(WSI_PATH, row['slide_id'])
        try:
            full_scan = AICSImage(full_path)
            full_scan.set_scene(scene_id)
            wsi = full_scan.get_image_data("YXS", T=0)
            wsi = cv2.cvtColor(wsi, cv2.COLOR_BGR2RGB)
        except:
            print('Scene loading failed! Likely an issue with scan file. Skipping...')
            process_df.loc[index, 'status'] = 'failed_to_load'
            continue
    elif ('.mrxs' in row['slide_id']):
        print('.mrxs scan found, using openslide to load {}'.format(row['slide_id']))
        print('mrxs_special_dir: ', mrxs_special_dir)
        print(type(mrxs_special_dir))
        for key, val in mrxs_special_dir.items():
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
                            print('MIRAX patch [{} - {}, {} - {}] added to our WSI image!'.format(widths + mirax_x,
                                                                                                  end_width + mirax_x,
                                                                                                  heights + mirax_y,
                                                                                                  end_height + mirax_y))
            except:
                if ((int(slide.properties['openslide.bounds-width']) * int(slide.properties['openslide.bounds-height'])) > (200000 * 100000)):
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
                            print('MIRAX patch [{} - {}, {} - {}] added to our WSI image!'.format(widths,
                                                                                                  end_width,
                                                                                                  heights,
                                                                                                  end_height))
        except:
            print('Scene loading failed! Likely an issue with scan file. Skipping...')
            process_df.loc[index, 'status'] = 'failed_to_load'
            continue
    elif ('.tif' in row['slide_id']):
        print('.tif scan found, using tifffile/slideio to load {}'.format(row['slide_id']))
        if ' - Series ' in row['slide_id']:
            scene_id = int(row['slide_id'].split(' - Series ')[1][0])
            full_path = os.path.join(WSI_PATH, row['slide_id'].split(' - Series ')[0])
        else:
            scene_id = 0
            full_path = os.path.join(WSI_PATH, row['slide_id'])
        if scene_id > 0:
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
                    print('Scene loading failed! Likely an issue with scan file. Skipping...')
                    process_df.loc[index, 'status'] = 'failed_to_load'
                    continue
    else:
        full_path = os.path.join(WSI_PATH, row['slide_id'])
        print('Using tifffile/slideio to load {}'.format(row['slide_id']))
        try:
            wsi = tifffile.imread(full_path)
        except:
            slideio_wsi = slideio.open_slide(full_path, driver="GDAL")
            scene = slideio_wsi.get_scene(0)
            try:
                wsi = scene.read_block()
            except:
                print('Scene loading failed! Likely an issue with scan file. Skipping...')
                process_df.loc[index, 'status'] = 'failed_to_load'
                continue

    if wsi.shape[0] == 0:
        print('{} was not properly loaded by tifffile. The tiff may be corrupt! Skipping..')
        process_df.loc[index, 'status'] = 'failed_to_load'
        continue

    print('{} loaded!'.format(row['slide_id']))
    # instantiate a WSI mask np array for both the confidence mapped pixel values and the counts
    print('Instantiating our pixel conf/counts np arrays...')
    out_shape = (int(wsi.shape[0]//OUT_RATIO), int(wsi.shape[1]//OUT_RATIO), 3)
    wsi_mask = np.zeros(out_shape, dtype='float64')
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
    for class_name in CLASSES:
        if '.tif' in row['slide_id']:
            gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name]) or (
                        row['slide_id'].split(' - Series ')[0] in GT_MASKS_WSI_IDS[class_name])
        else:
            gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name])

        if gt_bool:
            print('{} ground truths found for {}! Loading masks...'.format(class_name, row['slide_id']))

            # special case for tif
            if '.tif' in row['slide_id']:
                gt_class_masks = [file for file in GT_MASKS[class_name] if
                                  ((row['slide_id'] in file) or (row['slide_id'].split(' - Series ')[0] in file))]
            else:
                gt_class_masks = [file for file in GT_MASKS[class_name] if row['slide_id'] in file]

            gt_class_masks_np = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in gt_class_masks]
            print('Generating GT global binary mask...')
            class_gt_wsi = np.zeros((wsi_mask.shape[0], wsi_mask.shape[1]), dtype='uint8')

            for mask, mask_file in zip(gt_class_masks_np, gt_class_masks):
                # check if the class exception is in the file name
                if any(special_case in mask_file for special_case in SPECIAL_CLASS_CASES[class_name]):
                    for special_case in SPECIAL_CLASS_CASES[class_name]:
                        if special_case in mask_file:
                            class_splitter = special_case
                else:
                    class_splitter = class_name
                # get the coordinates, h, w from the image
                x, y, width, height = list(
                    map(int, mask_file.split('_' + class_splitter + '_')[-1].split(')')[0].split(',')[1:]))
                # convert coordinate, mask to our out size
                x, y, width, height = int(x // OUT_RATIO), int(y // OUT_RATIO), int(width // OUT_RATIO), int(
                    height // OUT_RATIO)
                # map the mask to the class_gt_wsi
                if (width < 2) or (height < 2):
                    continue
                else:
                    mask = cv2.resize(mask, dsize=(width, height))
                # checking for masks that are over the edge on the left and top of the wsi
                if (x < 0) and (y < 0):
                    x_wsi = x + abs(x)
                    width_wsi = width - abs(x)
                    y_wsi = y + abs(y)
                    height_wsi = height - abs(y)
                    class_gt_wsi[y_wsi:y_wsi + height_wsi, x_wsi:x_wsi + width_wsi] += mask[abs(y):, abs(x):]
                # checking for masks that are over the right and top of the wsi
                elif ((x + width) > class_gt_wsi.shape[1]) and (y < 0):
                    y_wsi = y + abs(y)
                    height_wsi = height - abs(y)
                    x_wsi = width - ((x + width) - class_gt_wsi.shape[1])
                    class_gt_wsi[y_wsi:y_wsi + height_wsi, x:x + width] += mask[abs(y):, :x_wsi]
                # checking for masks that are over the right and bottom of the wsi
                elif ((x + width) > class_gt_wsi.shape[1]) and ((y + height) > class_gt_wsi.shape[0]):
                    y_wsi = height - ((y + height) - class_gt_wsi.shape[0])
                    x_wsi = width - ((x + width) - class_gt_wsi.shape[1])
                    class_gt_wsi[y:y + height, x:x + width] += mask[:y_wsi, :x_wsi]
                # checking for masks that are over the left and bottom of the wsi
                elif (x < 0) and ((y + height) > class_gt_wsi.shape[0]):
                    x_wsi = x + abs(x)
                    width_wsi = width - abs(x)
                    y_wsi = height - ((y + height) - class_gt_wsi.shape[0])
                    class_gt_wsi[y:y + height, x_wsi:x_wsi + width_wsi] += mask[:y_wsi, abs(x):]
                # checking for masks that are over the top on the wsi
                elif y < 0:
                    y_wsi = y + abs(y)
                    height_wsi = height - abs(y)
                    class_gt_wsi[y_wsi:y_wsi + height_wsi, x:x + width] += mask[abs(y):, :]
                # checking for masks that are over the left on the wsi
                elif x < 0:
                    x_wsi = x + abs(x)
                    width_wsi = width - abs(x)
                    class_gt_wsi[y:y + height, x_wsi:x_wsi + width_wsi] += mask[:, abs(x):]
                # checking for masks that are over the right on the wsi
                elif ((x + width) > class_gt_wsi.shape[1]):
                    x_wsi = width - ((x + width) - class_gt_wsi.shape[1])
                    class_gt_wsi[y:y + height, x:x + width] += mask[:, :x_wsi]
                # checking for masks that are over the bottom on the wsi
                elif ((y + height) > class_gt_wsi.shape[0]):
                    y_wsi = height - ((y + height) - class_gt_wsi.shape[0])
                    class_gt_wsi[y:y + height, x:x + width] += mask[:y_wsi, :]
                # the rest are fully in range
                else:
                    class_gt_wsi[y:y + height, x:x + width] += mask

            # correct the mask so that all values are binary
            class_gt_wsi[class_gt_wsi > 0] = 255
            class_gt_masks[class_name] = class_gt_wsi
            print('Masks {} binary mapped! Getting our ROI masks...'.format(class_name))

            # get the variables for computing rois
            gt_mask = class_gt_wsi.copy()
            gt_mask[gt_mask == 255] = 1
            gt_mask_roi = gt_mask.copy()

            # get the canvas
            tissue_masking_canvas = np.zeros(class_gt_wsi.shape)
            # get the roi version of our masks
            # patch the coords together, scale them down
            for tissue_contours in roi_coords:
                if len(tissue_contours) < 3:
                    continue
                else:
                    # make a temporary canvas
                    temp_tissue_masking_canvas = np.zeros(gt_mask.shape, dtype='uint8')
                    # rescale the contours
                    roi_contour = np.array(tissue_contours)
                    for roi_k, roi_contours in enumerate([roi_contour]):
                        roi_contours[:, :, 0] = roi_contours[:, :, 0] * (OUT_DIM / IN_DIM)
                        roi_contours[:, :, 1] = roi_contours[:, :, 1] * (OUT_DIM / IN_DIM)

                    # draw onto the canvas
                    cv2.drawContours(temp_tissue_masking_canvas, [roi_contour], -1, 1, thickness=cv2.FILLED)

                    # check if any GTs lie in the tissue mask
                    if np.any(gt_mask_roi[np.where(temp_tissue_masking_canvas > 0)]):
                        tissue_masking_canvas += temp_tissue_masking_canvas

            # now append the ROI masks
            tissue_masking_canvas[tissue_masking_canvas > 0] = 1
            gt_mask_roi[np.where(tissue_masking_canvas < 1)] = 0
            class_gt_tissue_masks[class_name] = tissue_masking_canvas.astype('bool')
            class_roi_gt_masks[class_name] = gt_mask_roi

        else:
            print('No GTs found for {}.'.format(class_name))

    #############################
    ######## PRED MAPPING #######
    #############################

    # get the patch coordinates
    tot_coords = process_contours(roi_coords,
                                 patch_size=4096,
                                 step_size=1024,
                                 use_padding=True,
                                 top_left=None,
                                 bot_right=None,
                                 img_h=wsi.shape[0],
                                 img_w=wsi.shape[1])


    # extract the patches from the WSI, append them to a single torch object after resizing
    # get a list of np array patches
    patches = []
    print('Extracting patches for WSI {}...'.format(row['slide_id']))
    for coords in tot_coords:
        patch = wsi[coords[1]:coords[1] + IN_DIM, coords[0]:coords[0] + IN_DIM, :]
        # check to make sure that the patch is the correct size, if it isnt, pad it according to where we are going over
        if patch.shape != (IN_DIM, IN_DIM, 3):
            if ((coords[1] + IN_DIM) > wsi.shape[0]) and ((coords[0] + IN_DIM) > wsi.shape[1]):
                # bottom and right padding
                patch = np.stack(
                    [np.pad(patch[:, :, c], ((0, (coords[1] + IN_DIM) - wsi.shape[0]), (0, (coords[0] + IN_DIM) - wsi.shape[1])),
                            mode='constant', constant_values=0) for c in range(3)], axis=2)
            elif (coords[1] + IN_DIM) > wsi.shape[0]:
                # bottom padding
                patch = np.stack(
                    [np.pad(patch[:, :, c], ((0, (coords[1] + IN_DIM) - wsi.shape[0]), (0, 0)),
                            mode='constant', constant_values=0) for c in range(3)], axis=2)
            elif (coords[0] + IN_DIM) > wsi.shape[1]:
                # right padding
                patch = np.stack(
                    [np.pad(patch[:, :, c], ((0, 0), (0, (coords[0] + IN_DIM) - wsi.shape[1])),
                            mode='constant', constant_values=0) for c in range(3)], axis=2)
        patch = cv2.resize(patch, dsize=(OUT_DIM, OUT_DIM))
        # convert to bgr to make compatible with the mmdetection pipeline
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        patches.append(patch)

    # create a band which we use to remove annotations that are fully in them
    kernel = np.ones((OUT_DIM, OUT_DIM), dtype='bool')
    kernel[OUT_DIM//10:-OUT_DIM//10, OUT_DIM//10:-OUT_DIM//10] = False

    print('{} patches extracted! Passing patches through our model...'.format(len(patches)))
    # iterate through the list according to batch size, generate prediction confidence dict so we can get cutoffs
    img_batches = []
    class_prediction_confs = {key: [] for key in CLASSES}
    for i in range(0, len(patches), GPU_BATCH_SIZE):
        # get the batch and pass it through our model
        # check for exceeding dimensions
        if (i + GPU_BATCH_SIZE) > (len(patches) - 1):
            print('Passing patch {} to {}...'.format(i+1, len(patches)))
            patches_batch = patches[i:]
            patch_ids = list(range(i, len(patches)))
        else:
            print('Passing patch {} to {}...'.format(i+1, i + GPU_BATCH_SIZE))
            patches_batch = patches[i:i + GPU_BATCH_SIZE]
            patch_ids = list(range(i, i + GPU_BATCH_SIZE))
        # run inference on the batch
        batch_result = inference_detector(model, patches_batch)

        # parse the bboxes and masks, load values
        for patch_id, prediction in enumerate(batch_result):

            # zip the bboxes and masks by class id
            for class_id, (bbox, segm) in enumerate(zip(prediction[0], prediction[1])):

                # continue if we dont have any predictions for a class
                if len(segm) == 0:
                    continue

                # update the values using our patch location coords
                patch_coordinates = tot_coords[patch_ids[patch_id]]

                # update our pixel counts
                # get our counts kernel array
                counts = kernel.copy()[0:wsi_mask.shape[0] - int(patch_coordinates[1] // OUT_RATIO),
                                       0:wsi_mask.shape[1] - int(patch_coordinates[0] // OUT_RATIO)]

                # zip the bboxes and masks by their relative prediction
                for class_bbox, class_segm in zip(bbox, segm):

                    # check to see if any pixels are annotated, skip the annotation if not
                    if np.any(class_segm):

                        # check to make sure that the patch is over the edge, if it is, cut the extra
                        class_segm = class_segm[
                                      0:wsi_mask.shape[0] - int(patch_coordinates[1] // OUT_RATIO),
                                      0:wsi_mask.shape[1] - int(patch_coordinates[0] // OUT_RATIO)]

                        # check if the annotation is fully within the edge of the image, skip if it is
                        edge_canvas_pixels = np.concatenate([class_segm[0, :-1], class_segm[:-1, -1],
                                                             class_segm[-1, ::-1], class_segm[-2:0:-1, 0]])
                        # get the edge pixels
                        edge_mask_pixels = np.sum(cv2.Canny(class_segm.copy().astype('uint8'), 0, 1) / 255)

                        if (np.sum(class_segm[counts]) >= (0.9 * np.sum(class_segm))) and np.sum(edge_canvas_pixels) >= (0.2 * edge_mask_pixels):
                            print('Small edge instance found, skipping...')
                            continue
                        elif np.sum(edge_canvas_pixels) >= (0.4 * edge_mask_pixels):
                            print('Large edge instance found, skipping...')
                            continue

                        # get the confidence output of the prediction, append them to the dict of conf values
                        conf_class_pred = class_bbox[-1]
                        # set the confidence pixel values
                        class_segm_conf = class_segm * conf_class_pred
                        # add the confidence value
                        class_prediction_confs[CLASSES[class_id]].append(conf_class_pred)
                        # get the mask areas where we have lower values
                        index_append = np.where(wsi_mask[int(patch_coordinates[1]//OUT_RATIO):int(patch_coordinates[1]//OUT_RATIO) + OUT_DIM,
                                                        int(patch_coordinates[0]//OUT_RATIO):int(patch_coordinates[0]//OUT_RATIO) + OUT_DIM,
                                                        class_id] < class_segm_conf)
                        # add the conf values
                        wsi_mask[int(patch_coordinates[1]//OUT_RATIO):int(patch_coordinates[1]//OUT_RATIO) + OUT_DIM,
                        int(patch_coordinates[0]//OUT_RATIO):int(patch_coordinates[0]//OUT_RATIO) + OUT_DIM,
                        class_id][index_append] = class_segm_conf[index_append].astype('float64') # note

                    else:
                        print('Bounding box with no segmented pixels detected, skipping...')

    print('Patch-wise prediction for {} complete!'.format(row['slide_id']))
    wsi_mask_cleaned = wsi_mask
    # plot heatmaps?

    print('Generating our cutoff thresholds...')
    slide_wise_cutoffs = {key: 1 for key in CLASSES}
    for class_index, class_name in enumerate(CLASSES):
        # use rois if they are available
        if '.tif' in row['slide_id']:
            gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name]) or (
                        row['slide_id'].split(' - Series ')[0] in GT_MASKS_WSI_IDS[class_name])
        else:
            gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name])

        if gt_bool:
            class_cutoff_np_roi, class_cutoff_counts_roi = np.unique(
                wsi_mask_cleaned[:, :, class_index][class_gt_tissue_masks[class_name]][np.nonzero(wsi_mask_cleaned[:, :, class_index][class_gt_tissue_masks[class_name]])],
                return_counts=True)
            bins_roi = np.linspace(0, 1, 200)
            raw_count_histo_roi = np.histogram(class_cutoff_np_roi, bins_roi, weights=class_cutoff_np_roi)[0]
            raw_count_histo_cuts_roi = np.histogram(class_cutoff_np_roi, bins_roi, weights=class_cutoff_np_roi)[1]
            freq_count_histo_roi = np.zeros_like(raw_count_histo_roi)
            for thresh_id, thresh in enumerate(raw_count_histo_cuts_roi):
                if thresh_id == (len(raw_count_histo_cuts_roi) - 1):
                    continue
                for conf_id, conf_value in enumerate(class_cutoff_np_roi):
                    if (conf_value >= thresh) and (conf_value < raw_count_histo_cuts_roi[thresh_id + 1]):
                        freq_count_histo_roi[thresh_id] += class_cutoff_counts_roi[conf_id]
            # min max scale and append, log scale counts first to avoid dying neurons
            freq_count_histo_roi = np.where(freq_count_histo_roi != 0, np.log(freq_count_histo_roi), 0)
            raw_count_histo_roi = (raw_count_histo_roi - np.min(raw_count_histo_roi)) / np.max(raw_count_histo_roi)
            freq_count_histo_roi = (freq_count_histo_roi - np.min(freq_count_histo_roi)) / np.max(freq_count_histo_roi)
            # append the values
            process_df.loc[index, class_name + '_raw_count_histo_rois'] = ' '.join(raw_count_histo_roi.astype('str'))
            process_df.loc[index, class_name + '_freq_count_histo_rois'] = ' '.join(freq_count_histo_roi.astype('str'))

        # generate non roi to log, and for cases where we dont have GT/Rois
        class_cutoff_np, class_cutoff_counts = np.unique(
            wsi_mask_cleaned[:, :, class_index][np.nonzero(wsi_mask_cleaned[:, :, class_index])],
            return_counts=True)
        bins = np.linspace(0, 1, 200)
        raw_count_histo = np.histogram(class_cutoff_np, bins, weights=class_cutoff_np)[0]
        raw_count_histo_cuts = np.histogram(class_cutoff_np, bins, weights=class_cutoff_np)[1]
        freq_count_histo = np.zeros_like(raw_count_histo)
        for thresh_id, thresh in enumerate(raw_count_histo_cuts):
            if thresh_id == (len(raw_count_histo_cuts) - 1):
                continue
            for conf_id, conf_value in enumerate(class_cutoff_np):
                if (conf_value >= thresh) and (conf_value < raw_count_histo_cuts[thresh_id + 1]):
                    freq_count_histo[thresh_id] += class_cutoff_counts[conf_id]
        # min max scale and append, log scale counts first to avoid dying neurons
        freq_count_histo = np.where(freq_count_histo != 0, np.log(freq_count_histo), 0)
        raw_count_histo = (raw_count_histo - np.min(raw_count_histo)) / np.max(raw_count_histo)
        freq_count_histo = (freq_count_histo - np.min(freq_count_histo)) / np.max(freq_count_histo)
        # append the values
        process_df.loc[index, class_name + '_raw_count_histo'] = ' '.join(raw_count_histo.astype('str'))
        process_df.loc[index, class_name + '_freq_count_histo'] = ' '.join(freq_count_histo.astype('str'))

        # eval with the network, get the confidence threshold
        # generate the tensor, use roi if we have it
        if gt_bool:
            count_tensor = torch.from_numpy(np.concatenate((raw_count_histo_roi, freq_count_histo_roi)))
        else:
            count_tensor = torch.from_numpy(np.concatenate((raw_count_histo, freq_count_histo)))
        if 'WCM_' in row['slide_id']:
            conf_mlp = torch.load('/data/public/HULA/WSIs_renal_compartment_segmentations/test_to_val.pth')
            conf_mlp = conf_mlp.float()
            conf_mlp.eval()
            # normalize
            count_tensor = (count_tensor - torch.tensor(conf_mlp_norm_test_to_val[:, 0][:-len(CLASSES)])) / torch.tensor(conf_mlp_norm_test_to_val[:, 1][:-len(CLASSES)])
            count_tensor = torch.nan_to_num(count_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            conf_mlp = torch.load('/data/public/HULA/WSIs_renal_compartment_segmentations/val_to_test.pth')
            conf_mlp = conf_mlp.float()
            conf_mlp.eval()
            # normalize
            print('[!!] count_tensor.shape: ', count_tensor.shape)
            print('[!!] torch.tensor(conf_mlp_norm_val_to_test[:, 0][:-len(CLASSES)]).shape: ', torch.tensor(conf_mlp_norm_val_to_test[:, 0][:-len(CLASSES)]).shape)
            count_tensor = (count_tensor - torch.tensor(conf_mlp_norm_val_to_test[:, 0][:-len(CLASSES)])) / torch.tensor(conf_mlp_norm_val_to_test[:, 1][:-len(CLASSES)])
            count_tensor = torch.nan_to_num(count_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        # append the ohe label
        class_label = np.array(class_index)
        one_hot = np.zeros(len(CLASSES))
        one_hot[class_label] = 1
        count_tensor = torch.cat((count_tensor, torch.from_numpy(one_hot)))
        count_tensor = count_tensor.float()
        with torch.no_grad():
            thresh_tensor = conf_mlp(count_tensor)
        # finally get the class cutoff value
        if (class_name == 'Artery') and (thresh_tensor.item() < 0.65):
            thresh = 0.65
        elif (class_name == 'Arteriole') and (thresh_tensor.item() < 0.45):
            thresh = 0.45
        elif (class_name == 'Glomerulus') and (thresh_tensor.item() < 0.85):
            thresh = 0.85
        else:
            thresh = thresh_tensor.item()
        slide_wise_cutoffs[class_name] = thresh
        print('{} class cutoff generated for {} (threshold: {})'.format(class_name,
                                                                             row['slide_id'], thresh))

    print('Cutoff thresholds generated! Class thresholds: {}, Generating our global binary masks...'.format(slide_wise_cutoffs))
    # blur the prediction confidence values first to avoid empty areas between the same annotations
    for class_index, class_name in enumerate(CLASSES):
        # generate classwise AUC curves/ROC values if we have the GTs
        if '.tif' in row['slide_id']:
            gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name]) or (
                        row['slide_id'].split(' - Series ')[0] in GT_MASKS_WSI_IDS[class_name])
        else:
            gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name])

        if gt_bool:
            print('Generating ROC/AUC metrics for raw and masked {} predictions...'.format(class_name))
            # prediction metrics, need to get the areas where prediction/gt values are non zero, or we will be
            # biasing performance to background
            pred_non_zero_mask = full_tissue_masking_canvas
            raw_class_conf = wsi_mask_cleaned[:, :, class_index][pred_non_zero_mask]
            raw_gt = class_gt_masks[class_name][pred_non_zero_mask]
            raw_gt[raw_gt == 255] = 1
            if (np.max(raw_gt) == 0) or (np.max(raw_class_conf) == 0):
                print('No {} overlap for ROC/AUC...'.format(class_name))

            else:
                raw_precision, raw_recall, raw_threshold = metrics.precision_recall_curve(raw_gt, raw_class_conf)
                raw_pr_auc = metrics.auc(raw_recall, raw_precision)
                raw_pr_f1 = (2 * raw_precision * raw_recall) / (raw_precision + raw_recall)
                raw_pr_f1 = np.nan_to_num(raw_pr_f1)
                raw_pr_f1_max = raw_threshold[np.argmax(raw_pr_f1)]
                raw_f1_30 = raw_pr_f1[np.argmin(np.absolute(raw_threshold - 0.3))]
                raw_f1_50 = raw_pr_f1[np.argmin(np.absolute(raw_threshold - 0.5))]
                raw_f1_70 = raw_pr_f1[np.argmin(np.absolute(raw_threshold - 0.7))]
                raw_f1_90 = raw_pr_f1[np.argmin(np.absolute(raw_threshold - 0.9))]
                raw_f1_auto = raw_pr_f1[np.argmin(np.absolute(raw_threshold - slide_wise_cutoffs[class_name]))]

                # append values to our pd
                process_df.loc[index, class_name + '_pr_precision'] = ' '.join(raw_precision.round(4).astype('str'))
                process_df.loc[index, class_name + '_pr_recall'] = ' '.join(raw_recall.round(4).astype('str'))
                process_df.loc[index, class_name + '_pr_thresh'] = ' '.join(np.concatenate((raw_threshold.round(4), np.array([1]))).astype('str'))
                process_df.loc[index, class_name + '_pr_auc'] = raw_pr_auc
                process_df.loc[index, class_name + '_conf_delta'] = slide_wise_cutoffs[class_name] - raw_pr_f1_max
                process_df.loc[index, class_name + '_f1_30'] = raw_f1_30
                process_df.loc[index, class_name + '_f1_50'] = raw_f1_50
                process_df.loc[index, class_name + '_f1_70'] = raw_f1_70
                process_df.loc[index, class_name + '_f1_90'] = raw_f1_90
                process_df.loc[index, class_name + '_f1_auto'] = raw_f1_auto
                process_df.loc[index, class_name + '_f1_maxcut'] = raw_pr_f1_max
                process_df.loc[index, class_name + '_f1_max'] = np.max(raw_pr_f1)
                print('{} ideal conf. threshold: {}'.format(class_name, raw_pr_f1_max))

            # masked prediction metrics
            pred_non_zero_mask = class_gt_tissue_masks[class_name]
            roi_class_conf = wsi_mask_cleaned[:, :, class_index][pred_non_zero_mask]
            roi_gt = class_roi_gt_masks[class_name][pred_non_zero_mask]
            if (len(roi_gt) == 0) or (np.max(roi_gt) == 0) or (np.max(roi_class_conf) == 0):
                print('No {} overlap for ROC/AUC rois...'.format(class_name))

            else:
                roi_precision, roi_recall, roi_threshold = metrics.precision_recall_curve(roi_gt, roi_class_conf)
                roi_pr_auc = metrics.auc(roi_recall, roi_precision)
                roi_pr_f1 = (2 * roi_precision * roi_recall) / (roi_precision + roi_recall)
                roi_pr_f1 = np.nan_to_num(roi_pr_f1)
                roi_pr_f1_max = roi_threshold[np.argmax(roi_pr_f1)]
                roi_f1_30 = roi_pr_f1[np.argmin(np.absolute(roi_threshold - 0.3))]
                roi_f1_50 = roi_pr_f1[np.argmin(np.absolute(roi_threshold - 0.5))]
                roi_f1_70 = roi_pr_f1[np.argmin(np.absolute(roi_threshold - 0.7))]
                roi_f1_90 = roi_pr_f1[np.argmin(np.absolute(roi_threshold - 0.9))]
                roi_f1_auto = roi_pr_f1[np.argmin(np.absolute(roi_threshold - slide_wise_cutoffs[class_name]))]

                # append values to our pd
                process_df.loc[index, class_name + '_pr_precision_rois'] = ' '.join(roi_precision.round(4).astype('str'))
                process_df.loc[index, class_name + '_pr_recall_rois'] = ' '.join(roi_recall.round(4).astype('str'))
                process_df.loc[index, class_name + '_pr_thresh_rois'] = ' '.join(np.concatenate((roi_threshold.round(4), np.array([1]))).astype('str'))
                process_df.loc[index, class_name + '_pr_auc_rois'] = roi_pr_auc
                process_df.loc[index, class_name + '_conf_delta_rois'] = slide_wise_cutoffs[class_name] - roi_pr_f1_max
                process_df.loc[index, class_name + '_f1_30_rois'] = roi_f1_30
                process_df.loc[index, class_name + '_f1_50_rois'] = roi_f1_50
                process_df.loc[index, class_name + '_f1_70_rois'] = roi_f1_70
                process_df.loc[index, class_name + '_f1_90_rois'] = roi_f1_90
                process_df.loc[index, class_name + '_f1_auto_rois'] = roi_f1_auto
                process_df.loc[index, class_name + '_f1_maxcut_rois'] = roi_pr_f1_max
                process_df.loc[index, class_name + '_f1_max_rois'] = np.max(roi_pr_f1)
                print('{} ideal conf. threshold (ROI only): {}'.format(class_name, roi_pr_f1_max))
        else:
            print('No {} predictions for ROC/AUC...'.format(class_name))

        # cleaning masks
        wsi_mask_cleaned[:, :, class_index][
            wsi_mask_cleaned[:, :, class_index] >= slide_wise_cutoffs[class_name]] = 255
        wsi_mask_cleaned[:, :, class_index][
            wsi_mask_cleaned[:, :, class_index] < slide_wise_cutoffs[class_name]] = 0

    # change the dtype
    wsi_mask_cleaned = wsi_mask_cleaned.astype('uint8')
    print('Global binary masks generated! Generating our classwise contours...')
    all_contours = {key: [] for key in CLASSES}
    aux_contours = {key: [] for key in CLASSES}
    for class_index, class_name in enumerate(CLASSES):
        # get the contours
        if '.tif' in row['slide_id']:
            gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name]) or (
                        row['slide_id'].split(' - Series ')[0] in GT_MASKS_WSI_IDS[class_name])
        else:
            gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name])
        if gt_bool:
            if np.max(wsi_mask_cleaned[:, :, class_index][np.invert(class_gt_tissue_masks[class_name])]) > 0:
                print('Non ROI {} prediction pixels detected!'.format(class_name))
                # check to make sure that the pred masks are not present in the GT tissue masks, we dont want to save
                # spillover annotations.
                # get all the contours
                class_contours, class_hierarchys = cv2.findContours(wsi_mask_cleaned[:, :, class_index],
                                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                gt_tissue_contours, gt_tissue_hierarchys = cv2.findContours(class_gt_tissue_masks[class_name].astype('uint8'),
                                                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # check if they overlap with the GT tissue masks, save the ones that dont as aux contours
                aux_class_contours = []
                for contour in class_contours:
                    contour_bools = []
                    for gt_tissue_contour in gt_tissue_contours:
                        aux_bool = contour_intersect(gt_tissue_contour, contour)
                        contour_bools.append(aux_bool)
                    if not np.any(np.array(contour_bools)):
                        print('{} raw auxiliary contour found. Appending...'.format(class_name))
                        aux_class_contours.append(contour)

                if len(aux_class_contours) > 0:
                    aux_class_contours = [contour for contour in aux_class_contours if cv2.contourArea(contour) > 25]
                    print('{} total auxiliary {} class masks found!'.format(len(aux_class_contours), class_name))
                    aux_contours[class_name] = aux_class_contours
                else:
                    print('WARNING: No {} auxiliary class masks found! Likely only detected spillover.'.format(
                        class_name))

        # get the contours
        class_contours, class_hierarchys = cv2.findContours(wsi_mask_cleaned[:, :, class_index],
                                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # append the contours
        if len(class_contours) > 0:
            class_contours = [contour for contour in class_contours if cv2.contourArea(contour) > 25]
            print('{} total {} class masks found!'.format(len(class_contours), class_name))
            all_contours[class_name] = class_contours
        else:
            print('WARNING: No {} class masks found! Check thresholds/slide content/slide quality'.format(class_name))

    print('Contours extracted. Extracting TI tiles...')
    x_coords = np.array([])
    y_coords = np.array([])
    sum_percents = np.array([])
    wsi_tile_mask = np.sum(wsi_mask_cleaned, axis=2).astype('bool')
    xor_tile_mask = np.bitwise_xor(full_tissue_masking_canvas, wsi_tile_mask)
    for y_coord in range(0, xor_tile_mask.shape[0]-2048, 1024):
        for x_coord in range(0, xor_tile_mask.shape[1]-2048, 1024):
            sum_area_percent = np.sum(xor_tile_mask[y_coord:y_coord + 2048, x_coord:x_coord + 2048]) / (2048 * 2048)
            if sum_area_percent > 0.4:
                x_coords = np.append(x_coords, [x_coord])
                y_coords = np.append(y_coords, [y_coord])
                sum_percents = np.append(sum_percents, [sum_area_percent])

    # extract the tiles
    print('{} total TI tiles extracted! Saving...'.format(len(x_coords)))
    for x, y in zip(x_coords, y_coords):
        x_full = int(x * 2)
        y_full = int(y * 2)
        mask = xor_tile_mask[int(y):int(y) + 2048, int(x):int(x) + 2048].astype('uint8')
        mask_full = cv2.resize(mask, (0,0), fx=2, fy=2)
        tile = wsi[y_full:y_full + 4096, x_full:x_full + 4096, :]
        tile = tile.copy()
        tile[np.invert(mask_full.astype('bool'))] = 255
        tile_name = row['slide_id'] + '_xcoord_{}_ycoord_{}.png'.format(x_full, y_full)
        cv2.imwrite(TI_DIR + tile_name, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

    print('TI tiles saved. Generating WSI thumbnail image for visualization...')
    # plot the contours and save the binary masks
    # get the scaled image and contours
    wsi_mini = cv2.resize(wsi, dsize=(wsi.shape[1] // FIG_SCALE, wsi.shape[0] // FIG_SCALE))

    coef_y = wsi_mini.shape[0] / wsi_mask_cleaned.shape[0]
    coef_x = wsi_mini.shape[1] / wsi_mask_cleaned.shape[1]

    mini_all_contours = {key: copy.deepcopy(all_contours[key]) for key in CLASSES}
    for class_index in CLASSES:
        if len(mini_all_contours[class_index]) == 0:
            continue
        for k, contours in enumerate(mini_all_contours[class_index]):
            contours[:, :, 0] = contours[:, :, 0] * coef_x
            contours[:, :, 1] = contours[:, :, 1] * coef_y

    # get contour counts, rounded contour cutoffs
    contour_counts = {key: len(mini_all_contours[key]) for key in CLASSES}
    slide_wise_cutoffs = {key: round(slide_wise_cutoffs[key], 3) for key in CLASSES}

    # plotting the figure
    fig, axs = plt.subplots(1)
    # first draw the contours on the image (currently only supports 8 classes)
    for k, class_name in enumerate(CLASSES):
        # draw red filled contour on image background
        back = wsi_mini.copy()
        cv2.drawContours(back, mini_all_contours[class_name], -1, COLORS[k], thickness=cv2.FILLED)

        # blend with original image
        alpha = 0.35
        wsi_mini = cv2.addWeighted(wsi_mini, 1 - alpha, back, alpha, 0)
        x, y = 0, 0
        axs.scatter(x, y, c=np.array(list(COLORS[k]))/255.0, s=1, label=class_name)
        if k == (len(CLASSES) - 1):
            back = wsi_mini.copy()
            tissue_contours_plot, tissue_hierarchys_plot = cv2.findContours(full_tissue_masking_canvas.astype('uint8'),
                                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for k, contours in enumerate(tissue_contours_plot):
                contours[:, :, 0] = contours[:, :, 0] * coef_x
                contours[:, :, 1] = contours[:, :, 1] * coef_y

            cv2.drawContours(back, tissue_contours_plot, -1, COLORS[-1], thickness=2)

            # blend with original image
            alpha = 0.4
            wsi_mini = cv2.addWeighted(wsi_mini, 1 - alpha, back, alpha, 0)
            x, y = 0, 0
            axs.scatter(x, y, c=np.array(list(COLORS[-1])) / 255.0, s=1, label='Tissue Contours')

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

    print('Plot generated. Updating our processing DF, and checking for GT...')
    for class_index, class_name in enumerate(CLASSES):
        process_df.loc[index, class_name + '_counts'] = contour_counts[class_name]
        process_df.loc[index, class_name + '_conf_cutoff'] = slide_wise_cutoffs[class_name]

        # make the special case for the tif
        if '.tif' in row['slide_id']:
            gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name]) or (row['slide_id'].split(' - Series ')[0] in GT_MASKS_WSI_IDS[class_name])
        else:
            gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name])

        if gt_bool:
            print('Computing cleaned mask performance for {}...'.format(class_name))
            # get the variables for computing performance
            pred_mask = wsi_mask_cleaned[:, :, class_index][full_tissue_masking_canvas]
            pred_mask[pred_mask == 255] = 1
            pred_mask_roi = wsi_mask_cleaned[:, :, class_index][class_gt_tissue_masks[class_name]]
            pred_mask_roi[pred_mask_roi == 255] = 1
            gt_mask = class_gt_masks[class_name][full_tissue_masking_canvas]
            gt_mask[gt_mask == 255] = 1
            gt_mask_roi = class_gt_masks[class_name][class_gt_tissue_masks[class_name]]
            gt_mask_roi[gt_mask_roi == 255] = 1

            # compute metrics
            iou, filtered_precision, filtered_recall, filtered_f1score, filtered_specificity = binary_metrics(pred_mask, gt_mask)
            roi_iou, roi_filtered_precision, roi_filtered_recall, roi_filtered_f1score, roi_filtered_specificity = binary_metrics(pred_mask_roi, gt_mask_roi)

            print('Saving performance metrics...')
            process_df.loc[index, class_name + '_iou'] = iou
            process_df.loc[index, class_name + '_AP'] = filtered_precision
            process_df.loc[index, class_name + '_AR'] = filtered_recall
            process_df.loc[index, class_name + '_F1'] = filtered_f1score
            process_df.loc[index, class_name + '_AS'] = filtered_specificity
            process_df.loc[index, class_name + '_iou_rois'] = roi_iou
            process_df.loc[index, class_name + '_AP_rois'] = roi_filtered_precision
            process_df.loc[index, class_name + '_AR_rois'] = roi_filtered_recall
            process_df.loc[index, class_name + '_F1_rois'] = roi_filtered_f1score
            process_df.loc[index, class_name + '_AS_rois'] = roi_filtered_specificity

            print('Storing the GTs and auxiliary predictions...')
            class_contours, class_hierarchys = cv2.findContours(class_gt_masks[class_name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            all_contours[class_name] = list(class_contours)
            if len(aux_contours[class_name]) > 0:
                all_contours[class_name] = all_contours[class_name] + aux_contours[class_name]

        else:
            print('No {} ground truths found for {}.'.format(class_name, row['slide_id']))

    # we will need to check if any contours intersect, in which cas we will save a new contour with a combined id
    # make a storage variable which we will use to store contours that have no intersection
    print('Identifying uncertain contours, saving them...')
    delete_ids = {keys: [] for keys in CLASSES}
    for contour_index_id in range(len(CLASSES)):
        if contour_index_id == (len(CLASSES) - 1):
            continue
        for saved_ids, saved_contours in enumerate(all_contours[CLASSES[contour_index_id]]):
            for sub_contour_index_id in range(contour_index_id+1, len(CLASSES)):
                for saved_sub_ids, saved_sub_contours in enumerate(all_contours[CLASSES[sub_contour_index_id]]):
                    # compute the iou of the contour bboxes, find the ones with too much intersection and log them
                    x1, y1, w1, h1 = cv2.boundingRect(saved_contours)
                    x2, y2, w2, h2 = cv2.boundingRect(saved_sub_contours)
                    # get tensors
                    box1 = torch.tensor([[x1, y1, x1 + w1, y1 + h1]], dtype=torch.float)
                    box2 = torch.tensor([[x2, y2, x2 + w2, y2 + h2]], dtype=torch.float)
                    # get iou
                    iou = bops.box_iou(box1, box2)

                    if iou.item() > 0.0:
                        print('Overlap detected! IoU: {}'.format(iou.item()))

                        # check if there is class uncertainty
                        if iou.item() > 0.5:
                            print('Uncertain prediction class detected! IoU: {}, '
                                  'Saving larger mask as mixed class {} and {}...'.format(iou.item(),
                                                                                          CLASSES[contour_index_id],
                                                                                          CLASSES[sub_contour_index_id]))
                            area_1 = cv2.contourArea(saved_contours)
                            area_2 = cv2.contourArea(saved_sub_contours)
                            if area_1 >= area_2:
                                saved_contours_scaled = saved_contours.copy()
                            else:
                                saved_contours_scaled = saved_sub_contours.copy()
                            # get the scaled contour
                            saved_contours_scaled[:, :, 0] = saved_contours_scaled[:, :, 0] * OUT_RATIO
                            saved_contours_scaled[:, :, 1] = saved_contours_scaled[:, :, 1] * OUT_RATIO
                            # get the edges of the contour
                            x, y, w, h = cv2.boundingRect(saved_contours_scaled)
                            # convert the contour to a self standing frame, instantiate a canvas
                            contour_canvas = np.zeros((h, w), dtype='uint8')
                            # draw the contour
                            cv2.drawContours(contour_canvas, [saved_contours_scaled], -1, 255, thickness=cv2.FILLED, offset=(-x, -y))
                            # save the binary mask and the multi image
                            file_name = row['slide_id'] + '_' + \
                                        CLASSES[contour_index_id] + 'OR' + CLASSES[sub_contour_index_id] + '_' + \
                                        '({},{},{},{},{})-mask.png'.format(1.00, x, y, w, h)
                            multi_file_name = row['slide_id'] + '_' + \
                                        CLASSES[contour_index_id] + 'OR' + CLASSES[sub_contour_index_id] + '_' + \
                                        '({},{},{},{},{})-multi.png'.format(1.00, x, y, w, h)
                            img_file_name = row['slide_id'] + '_' + \
                                        CLASSES[contour_index_id] + 'OR' + CLASSES[sub_contour_index_id] + '_' + \
                                        '({},{},{},{},{}).png'.format(1.00, x, y, w, h)
                            img_x = x - 300
                            img_y = y - 300
                            img_x_end = x + w + 300
                            img_y_end = y + h + 300
                            if img_x < 0:
                                img_x_end += img_x
                                img_x = 0
                            if img_y < 0:
                                img_y_end += img_y
                                img_y = 0
                            if img_x_end > wsi.shape[1]:
                                img_x += (img_x_end - wsi.shape[1])
                                img_x_end = wsi.shape[1]
                            if img_y_end > wsi.shape[0]:
                                img_y += (img_y_end - wsi.shape[0])
                                img_y_end = wsi.shape[0]
                            compartment = wsi[y:y + h, x:x + w, :].copy()
                            compartment_img = wsi[img_y:img_y_end, img_x:img_x_end, :].copy()
                            compartment[np.invert(contour_canvas.astype('bool'))] = 0
                            cv2.imwrite(MULTI_DIR[CLASSES[contour_index_id]] + multi_file_name,
                                        cv2.cvtColor(compartment, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(IMG_DIR[CLASSES[contour_index_id]] + img_file_name,
                                        cv2.cvtColor(compartment_img, cv2.COLOR_RGB2BGR))
                            if '.tif' in row['slide_id']:
                                file_name_2 = row['slide_id'].split(' - Series ')[0] + '_' + \
                                            CLASSES[contour_index_id] + 'OR' + CLASSES[sub_contour_index_id] + '_' + \
                                            '({},{},{},{},{})-mask.png'.format(1.00, x, y, w, h)
                                cv2.imwrite(MASK_DIR[CLASSES[contour_index_id]] + file_name, contour_canvas)
                                cv2.imwrite(MASK_DIR[CLASSES[contour_index_id]] + file_name_2, contour_canvas)
                            else:
                                cv2.imwrite(MASK_DIR[CLASSES[contour_index_id]] + file_name, contour_canvas)

                            delete_ids[CLASSES[sub_contour_index_id]].append(saved_sub_ids)
                            delete_ids[CLASSES[contour_index_id]].append(saved_ids)

                        # # check if one annotation is completely within another, in which case we save the larger annotation
                        # elif (x1 < x2) and (y1 < y2):
                        #     if ((x2 + w2) < (x1 + w1)) and ((y2 + h2) < (y1 + h1)):
                        #         print('Small sub annotation detected! Saving the larger annotation only...')
                        #         # append the delete id
                        #         delete_ids[CLASSES[sub_contour_index_id]].append(saved_sub_ids)
                        #         # save the contour mask
                        #         saved_contours_scaled = saved_contours.copy()
                        #         # get the scaled contour
                        #         saved_contours_scaled[:, :, 0] = saved_contours_scaled[:, :, 0] * OUT_RATIO
                        #         saved_contours_scaled[:, :, 1] = saved_contours_scaled[:, :, 1] * OUT_RATIO
                        #         # get the edges of the contour
                        #         x, y, w, h = cv2.boundingRect(saved_contours_scaled)
                        #         # convert the contour to a self standing frame, instantiate a canvas
                        #         contour_canvas = np.zeros((h, w), dtype='uint8')
                        #         # draw the contour
                        #         cv2.drawContours(contour_canvas, [saved_contours_scaled], -1, 255,
                        #                          thickness=cv2.FILLED,
                        #                          offset=(-x, -y))
                        #         # save the binary mask
                        #         file_name = row['slide_id'] + '_' + \
                        #                     CLASSES[contour_index_id] + '_' + \
                        #                     '({},{},{},{},{})-mask.png'.format(1.00, x, y, w, h)
                        #         cv2.imwrite(MASK_DIR[CLASSES[contour_index_id]] + file_name, contour_canvas)
                        #
                        # # check the other one as well
                        # elif (x2 < x1) and (y2 < y1):
                        #     if ((x1 + w1) < (x2 + w2)) and ((y1 + h1) < (y2 + h2)):
                        #         print('Small sub annotation detected! Saving the larger annotation only...')
                        #         # append the delete id
                        #         delete_ids[CLASSES[contour_index_id]].append(saved_ids)
                        #         # save the contour mask
                        #         saved_contours_scaled = saved_sub_contours.copy()
                        #         # get the scaled contour
                        #         saved_contours_scaled[:, :, 0] = saved_contours_scaled[:, :, 0] * OUT_RATIO
                        #         saved_contours_scaled[:, :, 1] = saved_contours_scaled[:, :, 1] * OUT_RATIO
                        #         # get the edges of the contour
                        #         x, y, w, h = cv2.boundingRect(saved_contours_scaled)
                        #         # convert the contour to a self standing frame, instantiate a canvas
                        #         contour_canvas = np.zeros((h, w), dtype='uint8')
                        #         # draw the contour
                        #         cv2.drawContours(contour_canvas, [saved_contours_scaled], -1, 255,
                        #                          thickness=cv2.FILLED,
                        #                          offset=(-x, -y))
                        #         # save the binary mask
                        #         file_name = row['slide_id'] + '_' + \
                        #                     CLASSES[sub_contour_index_id] + '_' + \
                        #                     '({},{},{},{},{})-mask.png'.format(1.00, x, y, w, h)
                        #         cv2.imwrite(MASK_DIR[CLASSES[contour_index_id]] + file_name, contour_canvas)

    print('Saving the remaining contours...')
    # save the remaining masks
    for class_names in CLASSES:
        for cont_id, contours in enumerate(all_contours[class_names]):
            if cont_id in delete_ids[class_names]:
                continue
            # save the contour mask
            saved_contours_scaled = contours.copy()
            # get the scaled contour
            saved_contours_scaled[:, :, 0] = saved_contours_scaled[:, :, 0] * OUT_RATIO
            saved_contours_scaled[:, :, 1] = saved_contours_scaled[:, :, 1] * OUT_RATIO
            # get the edges of the contour
            x, y, w, h = cv2.boundingRect(saved_contours_scaled)
            # convert the contour to a self standing frame, instantiate a canvas
            contour_canvas = np.zeros((h, w), dtype='uint8')
            # draw the contour
            cv2.drawContours(contour_canvas, [saved_contours_scaled], -1, 255, thickness=cv2.FILLED,
                             offset=(-x, -y))
            # save the binary mask and the multi image
            file_name = row['slide_id'] + '_' + \
                        class_names + '_' + \
                        '({},{},{},{},{})-mask.png'.format(1.00, x, y, w, h)
            multi_file_name = row['slide_id'] + '_' + \
                        class_names + '_' + \
                        '({},{},{},{},{})-multi.png'.format(1.00, x, y, w, h)
            img_file_name = row['slide_id'] + '_' + \
                            class_names + '_' + \
                            '({},{},{},{},{}).png'.format(1.00, x, y, w, h)
            img_x = x - 300
            img_y = y - 300
            img_x_end = x + w + 300
            img_y_end = y + h + 300
            if img_x < 0:
                img_x_end += img_x
                img_x = 0
            if img_y < 0:
                img_y_end += img_y
                img_y = 0
            if img_x_end > wsi.shape[1]:
                img_x += (img_x_end - wsi.shape[1])
                img_x_end = wsi.shape[1]
            if img_y_end > wsi.shape[0]:
                img_y += (img_y_end - wsi.shape[0])
                img_y_end = wsi.shape[0]
            compartment = wsi[y:y + h, x:x + w, :].copy()
            compartment_img = wsi[img_y:img_y_end, img_x:img_x_end, :].copy()
            compartment[np.invert(contour_canvas.astype('bool'))] = 0
            cv2.imwrite(MULTI_DIR[class_names] + multi_file_name, cv2.cvtColor(compartment, cv2.COLOR_RGB2BGR))
            cv2.imwrite(IMG_DIR[class_names] + img_file_name,
                        cv2.cvtColor(compartment_img, cv2.COLOR_RGB2BGR))
            if '.tif' in row['slide_id']:
                file_name_2 = row['slide_id'].split(' - Series ')[0] + '_' + \
                              class_names + '_' + \
                              '({},{},{},{},{})-mask.png'.format(1.00, x, y, w, h)
                cv2.imwrite(MASK_DIR[class_names] + file_name, contour_canvas)
                cv2.imwrite(MASK_DIR[class_names] + file_name_2, contour_canvas)
            else:
                cv2.imwrite(MASK_DIR[class_names] + file_name, contour_canvas)

    print('All contours saved. The slide has been processed.')
    process_df.loc[index, 'status'] = 'processed'
    process_df.loc[index, 'reprocess'] = 0
    process_df.to_csv(WSI_SEG_PATH + 'process_list_{}_{}.csv'.format(PROJECT, SUB_DF_ID), index=False)
