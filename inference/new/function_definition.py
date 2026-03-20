import argparse
import math

import cv2
import numpy as np

from shapely.geometry import Polygon
import torch
import torchvision.ops.boxes as bops

from class_definition import isInContourV3_Easy


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

def binary_metrics(args, pred_mask, gt_mask):
  negative = 0
  positive = 1

  pred_mask_area = np.count_nonzero(pred_mask == positive)
  gt_mask_area = np.count_nonzero(gt_mask == positive)
  if args.do_not_print_much is False:
    print('Computing intersection...')
  intersection = np.count_nonzero(np.logical_and(pred_mask, gt_mask))
  if args.do_not_print_much is False:
    print('Computing IoU...')
  if (pred_mask_area+gt_mask_area-intersection) == 0:
    iou = 1
  elif intersection == 0:
    iou = 0
  else:
    iou = intersection/(pred_mask_area+gt_mask_area-intersection)
  if args.do_not_print_much is False:
    print('Computing Precision/Recall metrics...')
    print('Computing TP...')
  tp = np.sum(np.logical_and(pred_mask == positive, gt_mask == positive))
  if args.do_not_print_much is False:
    print('Computing TN...')
  tn = np.sum(np.logical_and(pred_mask == negative, gt_mask == negative))
  if args.do_not_print_much is False:
    print('Computing FP...')
  fp = np.sum(np.logical_and(pred_mask == positive, gt_mask == negative))
  if args.do_not_print_much is False:
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

def process_contours(args,
                     contours,
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
      if args.do_not_print_much is False:
        print('Processing contour {}/{}'.format(idx, n_contours))

    asset_dict, attr_dict = process_contour(args, cont, patch_size, step_size,
                                          use_padding, top_left, bot_right, img_h, img_w)

    if len(asset_dict) > 0:
      if len(tot_coords) == 0:
        tot_coords = asset_dict['coords']
      else:
        tot_coords = np.append(tot_coords, asset_dict['coords'], axis=0)

  return tot_coords

def process_contour(args,
                    cont,
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

  if args.do_not_print_much is False:
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
      if args.do_not_print_much is False:
        print("Contour is not in specified ROI, skip")
      return {}, {}
    else:
      if args.do_not_print_much is False:
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

  if args.do_not_print_much is False:
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

def get_args():
  # unet tissue segmentation arguments
  parser = argparse.ArgumentParser(description='Predict masks from input images')
  # --- checkpoints, config, and norm file ---
  parser.add_argument('--tissue_model', '-tm', default='./checkpoints/best_model.pth', metavar='FILE',
                      help='Specify the tissue seg checkpoint file')
  parser.add_argument('--train_norm_npy', '-tnn', default='./checkpoints/train_norm.npy', metavar='FILE',
                      help='Specify the train_norm.npy file')
  parser.add_argument('--swin_config', '-sc', default='./checkpoints/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py', metavar='FILE',
                      help='Specify the mmdet swin config file')
  parser.add_argument('--tissue_type_model', '-ttm', default='./checkpoints/epoch_4.pth', metavar='FILE',
                      help='Specify the tissue type instance segmentation swin file')
  parser.add_argument('--val_to_test', '-vtt', default='./checkpoints/val_to_test.pth', metavar='FILE',
                      help='Specify the val_to_test pth file')
  parser.add_argument('--test_to_val', '-ttv', default='./checkpoints/test_to_val.pth', metavar='FILE',
                      help='Specify the test_to_val pth file')
  parser.add_argument('--val_to_test_norm', '-vttn', default='./checkpoints/val_to_test_norm.npy', metavar='FILE',
                      help='Specify the val_to_test_norm file')
  parser.add_argument('--test_to_val_norm', '-ttvn', default='./checkpoints/test_to_val_norm.npy', metavar='FILE',
                      help='Specify the test_to_val_norm file')
  parser.add_argument('--clam_process_df', '-cpd', default='./checkpoints/process_list_autogen_CLAM.csv', metavar='FILE',
                      help='Specify the clam process df file')
  parser.add_argument('--qupath_proj_path', '-qpp', default='./checkpoints/qupath_projects', metavar='FILE',
                      help='Specify the path to Qupath projects')
  # ------------------------------------------
  # --- my addition ---
  parser.add_argument('--wsi_path', type=str, required=True, help='path to input WSIs')
  parser.add_argument('--do_not_print_much', '-dnpm', default=False, action='store_true', help='Do not print many outputs')
  parser.add_argument('--edge_detector', type=str, default='laplacian', help='laplacian or canny')
  parser.add_argument('--in_dim', type=int, default=4096, help='Original crop dimension from WSI')
  parser.add_argument('--out_dim', type=int, default=2048, help='Downscaled crop dimension to feed into Swin-MaskRCNN')
  parser.add_argument('--n_cml_patches', type=int, default=10, help='The number of epochs to feed into multi processing')
  parser.add_argument('--n_mp_cml_eps', type=int, default=50, help='The number of epochs to feed into multi processing')
  parser.add_argument('--n_mp_processes', type=int, default=30, help='The number of multiprocessing processes')
  parser.add_argument('--n_gpus', type=int, default=4, help='The number of gpus used (multiprocessing)')
  parser.add_argument('--gpu_batch_size', type=int, default=4, help='The batch size on each gpu')
  # parser.add_argument('--new_run', '-nr', default=False, action='store_true', help='Start a new run or still continue from previous run. If continues, then the code will read the updated csv file, or else it will create a new csv file and run everything again')
  parser.add_argument('--continue_from_previous_run', '-cfpr', default=False, action='store_true', help='Continue right on the slide that fails in the previous run, ignore slides having previous error run')
  # ------------------------------------------

  parser.add_argument('--viz', '-v', default=False, action='store_true',
                      help='Visualize the images as they are processed')
  parser.add_argument('--no-save', '-n', default=True, action='store_true', help='Do not save the output masks')
  parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                      help='Minimum probability value to consider a mask pixel white')
  parser.add_argument('--scale', '-s', type=float, default=0.5,
                      help='Scale factor for the input images')
  parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
  parser.add_argument('--cuda_tiseg', '-ct', type=int, default=4, help='Specify gpu to be used for tissue unet')
  parser.add_argument('--cuda_compseg', '-cc', type=int, default=4, help='Specify gpu to be used for compartment seg')
  parser.add_argument('--sub_df_id', '-sdf', default=6, type=int, help='Specify the sub df id, must be between 0 and 7')
  parser.add_argument('--quproj', '-proj', default='SLE', help='Specify the qupath project id')
  parser.add_argument('--from_index', type=int, required=True, help='Processing from index')
  parser.add_argument('--to_index', type=int, required=True, help='Processing to index')
  parser.add_argument('--output_dir', type=str, required=True, help='Output dir')
  parser.add_argument('--csv_path', type=str, required=True, help='pre csv path')

  return parser.parse_args()

def sub_df_uniques(lst, n):
  return [lst[i::n] for i in range(n)]