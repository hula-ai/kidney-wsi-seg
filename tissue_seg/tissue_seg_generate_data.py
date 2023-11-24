import copy

from mmdet.apis import init_detector, inference_detector
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
import pandas as pd
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import copy
from paquo.projects import QuPathProject
from scipy import ndimage
from skimage import io
from skimage import color
from skimage import segmentation
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
# from czifile import CziFile
from aicsimageio import AICSImage

# first we need to get an array of ground truth masks that we will use to evaluate performance
# set the classes
print('Getting our ground truths for accuracy/mask updating when possible...')
CLASSES = ["Cortex", "Medulla", "CapsuleOther"]

GT_MASKS = {key: [] for key in CLASSES}
GT_MASKS_RAW_NAMES = {key: [] for key in CLASSES}
GT_PATHS = ['/data/public/HULA/AMR/mask_val1/', '/data/public/HULA/AMR/mask_val2/',
            '/data/public/HULA/IgA/masks/', '/data/public/HULA/Pan_GN/Cologne_GN/mask/',
            '/data/public/HULA/Pan_GN/Szeged_GN/mask/', '/data/public/HULA/Pan_GN/Lille_GN/mask/',
            '/data/public/HULA/SLE/mask/', '/data/public/HULA/TMA/mask/']

# get the raw file paths
for sup_dirs in GT_PATHS:
    for root, dirs, files in os.walk(os.path.abspath(sup_dirs)):
        for file in files:
            for class_name in CLASSES:
                if class_name in file:
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

for class_name in CLASSES:
    GT_MASKS_WSI_IDS[class_name] = np.unique(np.array(GT_MASKS_WSI_IDS[class_name]))

# save the WSI dir, img and mask dirs, scale of the output
WSI_PATH = '/data/public/HULA/WSIs_renal/'
THUMBNAIL_PATH = '/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_imgs/'
MASK_PATH = '/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_masks/'
PLOT_PATH = '/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_plots/'
SCALE = 4096

# set the resolution for the tissue seg crops

# get all of the unique WSI names
ALL_SLIDES = np.array([])
for key in GT_MASKS_WSI_IDS.keys():
    set_diff = np.setdiff1d(np.array(GT_MASKS_WSI_IDS[key]), ALL_SLIDES)
    ALL_SLIDES = np.append(ALL_SLIDES, set_diff)

# get all of the thumbnail names so that we can check if we already have masks
CURRENT_DATA = os.listdir('/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_imgs')

# Set whether or not to regenerate data
REGEN_MODE = False

# extract the wsi, get the scaled down version and original dimensions
for slide in ALL_SLIDES:
    if not REGEN_MODE:
        if any(slide in current_data for current_data in CURRENT_DATA):
            print('Already have tissue type masks for {}'.format(slide))
            continue

    # we load the wsi here and pass it, this is so we can test for failure cases
    if ('.scn' in slide) or ('.sc_' in slide):
        print('Leica scan found, using SlideIO to load {}'.format(slide))
        scene_id = int(slide.split(' - Series ')[1][0])
        full_path = os.path.join(WSI_PATH, slide.split(' - Series ')[0])
        slideio_wsi = slideio.open_slide(full_path, driver="SCN")
        scene = slideio_wsi.get_scene(scene_id - 1)
        try:
            wsi_img = scene.read_block()
        except:
            print('Scene loading failed! Likely an issue with scan file. Skipping...')
            continue
    elif ('.svs' in slide):
        print('.svs scan found, using SlideIO to load {}'.format(slide))
        full_path = os.path.join(WSI_PATH, slide)
        slideio_wsi = slideio.open_slide(full_path, driver="SVS")
        scene = slideio_wsi.get_scene(0)
        try:
            wsi_img = scene.read_block()
        except:
            print('Scene loading failed! Likely an issue with scan file. Skipping...')
            continue
    elif ('.czi' in slide):
        print('.czi scan found, using AICSImage to load {}'.format(slide))
        scene_id = int(slide.split(' - Scene #')[1][0])
        full_path = os.path.join(WSI_PATH, slide.split(' - Scene #')[0])
        try:
            full_scan = AICSImage(full_path)
            full_scan.set_scene(scene_id - 1)
            wsi_img = full_scan.get_image_data("YXS", T=0)
            wsi_img = cv2.cvtColor(wsi_img, cv2.COLOR_BGR2RGB)
        except:
            print('Scene loading failed! Likely an issue with scan file. Skipping...')
            continue
    elif ('.tif' in slide):
        print('.tif scan found, using tifffile to load {}'.format(slide))
        if ' - Series ' in slide:
            scene_id = int(slide.split(' - Series ')[1][0])
            full_path = os.path.join(WSI_PATH, slide.split(' - Series ')[0])
        else:
            scene_id = 0
            full_path = os.path.join(WSI_PATH, slide)
        if scene_id > 0:
            print('WARNING: scene id for a .tif image is higher than 0, there is likely an error...')
            continue
        else:
            try:
                wsi_img = tifffile.imread(full_path)
            except:
                slideio_wsi = slideio.open_slide(full_path, driver="GDAL")
                scene = slideio_wsi.get_scene(0)
                try:
                    wsi_img = scene.read_block()
                except:
                    print('Scene loading failed! Likely an issue with scan file. Skipping...')
                    continue
    else:
        full_path = os.path.join(WSI_PATH, slide)
        print('Using tifffile to load {}'.format(slide))
        try:
            wsi_img = tifffile.imread(full_path)
        except:
            slideio_wsi = slideio.open_slide(full_path, driver="GDAL")
            scene = slideio_wsi.get_scene(0)
            try:
                wsi_img = scene.read_block()
            except:
                print('Scene loading failed! Likely an issue with scan file. Skipping...')
                continue

    if wsi_img.shape[0] == 0:
        print('{} was not properly loaded by tifffile. The tiff may be corrupt! Skipping...')
        continue

    # scale down the image after storing x and y, save the thumbnail
    wsi_x = wsi_img.shape[0]
    wsi_y = wsi_img.shape[1]
    wsi_thumb = cv2.resize(wsi_img, dsize=(SCALE, SCALE), interpolation=cv2.INTER_CUBIC)


    # get the msk ratio values
    # mask_x_ratio = wsi_x / SCALE
    # mask_y_ratio = wsi_y / SCALE
    mask_x_ratio = 1
    mask_y_ratio = 1

    # map the masks to our mask array
    wsi_mask = np.zeros(shape=(wsi_img.shape[0], wsi_img.shape[1]))
    # load the masks, map them
    for class_id, class_name in enumerate(CLASSES):
        for mask_id in GT_MASKS[class_name]:
            if slide in mask_id:
                try:
                    mask_raw = cv2.imread(mask_id, cv2.IMREAD_GRAYSCALE)
                except:
                    mask_raw = Image.open(mask_id)
                    mask_raw = np.asarray(mask_raw)
                mask_raw = mask_raw / 255
                mask_raw = mask_raw * (class_id + 1)
                # get the coordinates for scaling
                class_splitter = class_name
                # get the coordinates, h, w from the image
                x, y, width, height = list(
                    map(int, mask_id.split('_' + class_splitter + '_')[-1].split(')')[0].split(',')[1:]))
                # convert coordinate, mask to our out size
                x, y, width, height = int(x // mask_x_ratio), int(y // mask_y_ratio), int(width // mask_x_ratio), int(
                    height // mask_y_ratio)
                # remove padding and adjust coordinates, assumes equal padding
                edge_canvas_pixels = np.concatenate([mask_raw[0, :-1], mask_raw[:-1, -1],
                                                     mask_raw[-1, ::-1], mask_raw[-2:0:-1, 0]])
                mask = mask_raw.copy()
                if np.sum(edge_canvas_pixels) == 0:
                    p = np.where(mask != 0)
                    mask = mask[min(p[0]): max(p[0]) + 1, min(p[1]): max(p[1]) + 1]
                    x = x + (mask_raw.shape[1] - mask.shape[1]) // 2
                    y = y + (mask_raw.shape[0] - mask.shape[0]) // 2
                    width -= mask_raw.shape[1] - mask.shape[1]
                    height -= mask_raw.shape[0] - mask.shape[0]

                # map the mask to the class_gt_wsi
                mask = cv2.resize(mask, dsize=(width, height))
                # checking for masks that are over the edge on the left and top of the wsi
                if (x < 0) and (y < 0):
                    x_wsi = x + abs(x)
                    width_wsi = width - abs(x)
                    y_wsi = y + abs(y)
                    height_wsi = height - abs(y)
                    wsi_mask[y_wsi:y_wsi + height_wsi, x_wsi:x_wsi + width_wsi] += mask[abs(y):, abs(x):]
                # checking for masks that are over the right and top of the wsi
                elif ((x + width) > wsi_mask.shape[1]) and (y < 0):
                    y_wsi = y + abs(y)
                    height_wsi = height - abs(y)
                    x_wsi = width - ((x + width) - wsi_mask.shape[1])
                    wsi_mask[y_wsi:y_wsi + height_wsi, x:x + width] += mask[abs(y):, :x_wsi]
                # checking for masks that are over the right and bottom of the wsi
                elif ((x + width) > wsi_mask.shape[1]) and ((y + height) > wsi_mask.shape[0]):
                    y_wsi = height - ((y + height) - wsi_mask.shape[0])
                    x_wsi = width - ((x + width) - wsi_mask.shape[1])
                    wsi_mask[y:y + height, x:x + width] += mask[:y_wsi, :x_wsi]
                # checking for masks that are over the left and bottom of the wsi
                elif (x < 0) and ((y + height) > wsi_mask.shape[0]):
                    x_wsi = x + abs(x)
                    width_wsi = width - abs(x)
                    y_wsi = height - ((y + height) - wsi_mask.shape[0])
                    wsi_mask[y:y + height, x_wsi:x_wsi + width_wsi] += mask[:y_wsi, abs(x):]
                # checking for masks that are over the right on the wsi
                elif ((x + width) > wsi_mask.shape[1]):
                    x_wsi = width - ((x + width) - wsi_mask.shape[1])
                    wsi_mask[y:y + height, x:x + width] += mask[:, :x_wsi]
                # checking for masks that are over the bottom on the wsi
                elif ((y + height) > wsi_mask.shape[0]):
                    y_wsi = height - ((y + height) - wsi_mask.shape[0])
                    wsi_mask[y:y + height, x:x + width] += mask[:y_wsi, :]
                # checking for masks that are over the top on the wsi
                elif y < 0:
                    y_wsi = y + abs(y)
                    height_wsi = height - abs(y)
                    wsi_mask[y_wsi:y_wsi + height_wsi, x:x + width] += mask[abs(y):, :]
                # checking for masks that are over the left on the wsi
                elif x < 0:
                    x_wsi = x + abs(x)
                    width_wsi = width - abs(x)
                    wsi_mask[y:y + height, x_wsi:x_wsi + width_wsi] += mask[:, abs(x):]
                # the rest are fully in range
                else:
                    wsi_mask[y:y + height, x:x + width] += mask

    # save the mask
    wsi_mask = cv2.resize(wsi_mask, dsize=(SCALE, SCALE), interpolation=cv2.INTER_NEAREST).astype('uint8')
    wsi_mask[wsi_mask > len(CLASSES)] = 0
    fig = io.imshow(color.label2rgb(wsi_mask, wsi_thumb))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(PLOT_PATH + slide + '_tissue_thumbnail.png', dpi=1000, bbox_inches='tight')
    plt.close()
    cv2.imwrite(THUMBNAIL_PATH + slide + '_tissue_thumbnail.png', cv2.cvtColor(wsi_thumb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(MASK_PATH + slide + '_tissue_thumbnail.png', wsi_mask)
    print('{} processed.'.format(slide))







