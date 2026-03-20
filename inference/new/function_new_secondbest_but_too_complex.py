# Second best - the sharing of global variables for multi processes use a complex technique like a trick
import ctypes
import inspect # https://www.quora.com/In-Python-how-do-I-get-a-function-name-as-a-string-without-calling-the-function
               # https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function-without-using-traceback
from contextlib import closing
import time
import random
import multiprocessing
from multiprocessing import (
  set_start_method,
  Value,
  Lock,
  Manager,
  Pool
)
from sys import getsizeof

import cv2
import numpy as np
import sklearn.metrics as metrics
import torch
import torchvision.ops.boxes as bops

from function_definition import (
  binary_metrics, \
  contour_intersect
)

from scalene import scalene_profiler
                               

def post_processing_for_each_batch_result(class_prediction_confs, one_sample_result, wsi_h, wsi_w, wsi_c, patch_coordinates, kernel, classes, out_ratio, out_dim):
  global mng_wsi_mask
  value = random.random() * 5
  time.sleep(value)

  with mng_wsi_mask.get_lock():
    print('[multiproc] Start')
    # global mng_class_prediction_confs
    wsi_mask = np.frombuffer(mng_wsi_mask.get_obj()).reshape((wsi_h, wsi_w, wsi_c))

  # update our pixel counts
  # get our counts kernel array
  counts = kernel.copy()[0:wsi_mask.shape[0] - int(patch_coordinates[1] // out_ratio),
                         0:wsi_mask.shape[1] - int(patch_coordinates[0] // out_ratio)]

  # zip the bboxes and masks by class id
  for class_id, (bbox, segm) in enumerate(zip(one_sample_result[0], one_sample_result[1])):
    print('[multiproc] class_id: ', class_id)
    # continue if we dont have any predictions for a class
    if len(segm) == 0:
      continue

    # zip the bboxes and masks by their relative prediction
    for class_bbox, class_segm in zip(bbox, segm):
      # check to see if any pixels are annotated, skip the annotation if not
      if np.any(class_segm):
        # check to make sure that the patch is over the edge, if it is, cut the extra
        class_segm = class_segm[
                      0:wsi_mask.shape[0] - int(patch_coordinates[1] // out_ratio),
                      0:wsi_mask.shape[1] - int(patch_coordinates[0] // out_ratio)]

        # check if the annotation is fully within the edge of the image, skip if it is
        edge_canvas_pixels = np.concatenate([class_segm[0, :-1], class_segm[:-1, -1],
                                             class_segm[-1, ::-1], class_segm[-2:0:-1, 0]])
        # get the edge pixels
        # if args.edge_detector == 'canny':
        #   edge_mask_pixels = np.sum(cv2.Canny(class_segm.copy().astype('uint8'), 0, 1) / 255)
        # elif args.edge_detector == 'laplacian':
        #   edge_mask_pixels = np.sum(cv2.Laplacian(class_segm.copy().astype('uint8'), cv2.CV_64F) / 255)

        # if (np.sum(class_segm[counts]) >= (0.9 * np.sum(class_segm))) and np.sum(edge_canvas_pixels) >= (0.2 * edge_mask_pixels):
        #   if args.do_not_print_much is False:
        #     print('[In {}] Small edge instance found, skipping...'.format(inspect.stack()[0][3]))
        #   continue
        # elif np.sum(edge_canvas_pixels) >= (0.4 * edge_mask_pixels):
        #   if args.do_not_print_much is False:
        #     print('[In {}] Large edge instance found, skipping...'.format(inspect.stack()[0][3]))
        #   continueclass_prediction_confs

        # get the confidence output of the prediction, append them to the dict of conf values
        conf_class_pred = class_bbox[-1]
        # set the confidence pixel values
        class_segm_conf = class_segm * conf_class_pred
        # add the confidence value
        start_time = time.time()
        class_prediction_confs[classes[class_id]].append(conf_class_pred)
        print('=> update mng dict time: ', time.time() - start_time)
        # get the mask areas where we have lower values
        index_append = np.where(wsi_mask[int(patch_coordinates[1]//out_ratio):int(patch_coordinates[1]//out_ratio) + out_dim,
                                        int(patch_coordinates[0]//out_ratio):int(patch_coordinates[0]//out_ratio) + out_dim,
                                        class_id] < class_segm_conf)
        # add the conf values
        with mng_wsi_mask.get_lock():
          wsi_mask[int(patch_coordinates[1]//out_ratio):int(patch_coordinates[1]//out_ratio) + out_dim,
          int(patch_coordinates[0]//out_ratio):int(patch_coordinates[0]//out_ratio) + out_dim,
          class_id][index_append] = class_segm_conf[index_append].astype('float64') # note
  
      else:
        # if args.do_not_print_much is False:
        print('[In {}] Bounding box with no segmented pixels detected, skipping...'.format(inspect.stack()[0][3]))
  print('!! A process completes !!')

def loop1(args, CLASSES, row, GT_MASKS_WSI_IDS, GT_MASKS, wsi_mask, SPECIAL_CLASS_CASES, OUT_RATIO,
          class_gt_masks, roi_coords, OUT_DIM, IN_DIM, class_gt_tissue_masks, class_roi_gt_masks):
  for class_name in CLASSES:
    if '.tif' in row['slide_id']:
      gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name]) or (
                  row['slide_id'].split(' - Series ')[0] in GT_MASKS_WSI_IDS[class_name])
    else:
      gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name])

    if gt_bool:
      if args.do_not_print_much is False:
        print('[In {}] {} ground truths found for {}! Loading masks...'.format(inspect.stack()[0][3], class_name, row['slide_id']))

      # special case for tif
      if '.tif' in row['slide_id']:
        gt_class_masks = [file for file in GT_MASKS[class_name] if
                          ((row['slide_id'] in file) or (row['slide_id'].split(' - Series ')[0] in file))]
      else:
        gt_class_masks = [file for file in GT_MASKS[class_name] if row['slide_id'] in file]

      gt_class_masks_np = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in gt_class_masks]
      if args.do_not_print_much is False:
        print('[In {}] Generating GT global binary mask...'.format(inspect.stack()[0][3]))
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
      if args.do_not_print_much is False:
        print('[In {}] Masks {} binary mapped! Getting our ROI masks...'.format(inspect.stack()[0][3], class_name))

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
      if args.do_not_print_much is False:
        print('[In {}] No GTs found for {}.'.format(inspect.stack()[0][3], class_name))

def loop2(args, tot_coords, wsi, IN_DIM, OUT_DIM, patches):
  print('[In loop2] Start')
  for coords in tot_coords:
    print('[In loop2] new coords')
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

    print('[In loop2] End')

def init_worker(shared_wsi_mask):
  global mng_wsi_mask
  mng_wsi_mask = shared_wsi_mask
  # global mng_class_prediction_confs
  # mng_class_prediction_confs = shared_class_prediction_confs

def loop3(args, patches, GPU_BATCH_SIZE, inference_detector, model, tot_coords, kernel, wsi_mask, class_prediction_confs, CLASSES,
          OUT_RATIO, OUT_DIM):
  print('[**] start of loop 3')

  # scalene_profiler.start()
  total_inference_time = 0
  total_loop_time = 0

  all_batch_results = []
  for i in range(0, len(patches), GPU_BATCH_SIZE):
    # get the batch and pass it through our model
    # check for exceeding dimensions
    print('[**] loop ith: ', i)
    if (i + GPU_BATCH_SIZE) > (len(patches) - 1):
      if args.do_not_print_much is False:
        print('[In {}] Passing patch {} to {}...'.format(inspect.stack()[0][3], i+1, len(patches)))
      patches_batch = patches[i:]
      patch_ids = list(range(i, len(patches)))
    else:
      if args.do_not_print_much is False:
        print('[In {}] Passing patch {} to {}...'.format(inspect.stack()[0][3], i+1, i + GPU_BATCH_SIZE))
      patches_batch = patches[i:i + GPU_BATCH_SIZE]
      patch_ids = list(range(i, i + GPU_BATCH_SIZE))
    # run inference on the batch
    start_time = time.time()
    batch_result = inference_detector(model, patches_batch)
    total_inference_time += time.time() - start_time

    for one_sample_result in batch_result:
      all_batch_results.append(one_sample_result)

    # parse the bboxes and masks, load values
    # print('[loop] len(batch_result): ', len(batch_result))
    # print('[loop] len(batch_result[0][0]): ', len(batch_result[0][0]))
    # print('[loop] len(batch_result[0][0][0]): ', len(batch_result[0][0][0]))

  start_time = time.time()
  patch_ids = list(range(0, len(patches)))

  # for patch_id, prediction in enumerate(batch_result):
  wsi_h, wsi_w, wsi_c = wsi_mask.shape
  print('[**] start of multi processing ')


  print('[**] multi processing, init confs')
  # seqs = ['Glomerulus', 'Arteriole', 'Artery']
  manager = Manager()
  # ori_class_prediction_confs = class_prediction_confs.copy()
  shared_class_prediction_confs = manager.dict()
  # class_prediction_confs['Glomerulus'] = [] # Wrong, should also be shared list
  # class_prediction_confs['Arteriole'] = []
  # class_prediction_confs['Artery'] = []
  shared_class_prediction_confs['Glomerulus'] = manager.list()
  shared_class_prediction_confs['Arteriole'] = manager.list()
  shared_class_prediction_confs['Artery'] = manager.list()

  print('[**] multi processing, init wsi_mask')
  shared_wsi_mask = multiprocessing.Array(ctypes.c_double, wsi_h*wsi_w*wsi_c)
  ori_wsi_mask = np.frombuffer(shared_wsi_mask.get_obj()).copy()
  # update the values using our patch location coords
  # patch_coordinates = tot_coords.copy()[patch_ids[patch_id]]
  with closing(Pool(processes=25, initializer=init_worker, initargs=(shared_wsi_mask,))) as pool:
    funclist = []
    mp_results = []

    for patch_id, one_sample_result in enumerate(all_batch_results):
      a_func = pool.apply_async(post_processing_for_each_batch_result, args=(shared_class_prediction_confs, one_sample_result, wsi_h, wsi_w, wsi_c, tot_coords[patch_ids[patch_id]], kernel, CLASSES, OUT_RATIO, OUT_DIM))
      funclist.append(a_func)

    for func in funclist:
      an_mp_result = func.get(timeout=50)
      mp_results.append(an_mp_result)
    # pool.map_async(post_processing_for_each_batch_result, [(shared_class_prediction_confs, one_sample_result, wsi_h, wsi_w, wsi_c, tot_coords[patch_ids[patch_id]], kernel, CLASSES, OUT_RATIO, OUT_DIM) \
    #                                                        for patch_id, one_sample_result in enumerate(all_batch_results)])

    print('[**] multi processing, convert mng confs to pure confs')
    # class_prediction_confs = shared_class_prediction_confs.get()
    class_prediction_confs['Glomerulus'] = list(shared_class_prediction_confs['Glomerulus'])
    class_prediction_confs['Arteriole'] = list(shared_class_prediction_confs['Arteriole'])
    class_prediction_confs['Artery'] = list(shared_class_prediction_confs['Artery'])
    print('[In loop3] updated class_prediction_confs: ', class_prediction_confs)
    print('[**] multi processing, convert mng wsi_mask to pure wsi_mask')

    # pool.close()
    # pool.terminate()
  pool.join()
  wsi_mask = np.frombuffer(shared_wsi_mask.get_obj())
  # assert (wsi_mask==ori_wsi_mask).all()
  print('=> np.unique(wsi_mask): ', np.unique(wsi_mask))
  print('=> np.unique(ori_wsi_mask): ', np.unique(ori_wsi_mask))
  wsi_mask = wsi_mask.reshape(wsi_h, wsi_w, wsi_c)
  saved_wsi_mask = (wsi_mask*255).astype(np.uint8)
  resized_wsi_mask = cv2.resize(saved_wsi_mask, (2048, 2048))
  cv2.imwrite('./newest_global_mask.png', resized_wsi_mask)

  print('[In loop3] updated class_prediction_confs: ', class_prediction_confs)

  print('[**] end of multi processing')

  total_loop_time += time.time() - start_time
  print('~~~~~~~> Only inference time: {}'.format(total_inference_time))
  print('~~~~~~~> Only loop time: {}'.format(total_loop_time))
  # scalene_profiler.stop()
  return wsi_mask

def loop4(args, CLASSES, GT_MASKS_WSI_IDS, row, wsi_mask_cleaned, class_gt_tissue_masks,
          process_df, index, conf_mlp_norm_test_to_val, conf_mlp_norm_val_to_test, slide_wise_cutoffs):
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
      conf_mlp = torch.load(args.test_to_val)
      conf_mlp = conf_mlp.float()
      conf_mlp.eval()
      # normalize
      count_tensor = (count_tensor - torch.tensor(conf_mlp_norm_test_to_val[:, 0][:-len(CLASSES)])) / torch.tensor(conf_mlp_norm_test_to_val[:, 1][:-len(CLASSES)])
      count_tensor = torch.nan_to_num(count_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    else:
      conf_mlp = torch.load(args.val_to_test)
      conf_mlp = conf_mlp.float()
      conf_mlp.eval()
      # normalize
      if args.do_not_print_much is False:
        print('[In {}] count_tensor.shape: {}'.format(inspect.stack()[0][3], count_tensor.shape))
        print('[In {}] torch.tensor(conf_mlp_norm_val_to_test[:, 0][:-len(CLASSES)]).shape: {}'.format(inspect.stack()[0][3], torch.tensor(conf_mlp_norm_val_to_test[:, 0][:-len(CLASSES)]).shape))
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
    if args.do_not_print_much is False:
      print('[In {}] {} class cutoff generated for {} (threshold: {})'.format(inspect.stack()[0][3], class_name,
                                                                         row['slide_id'], thresh))
  
  resized_wsi_mask = cv2.resize((wsi_mask_cleaned*255).astype(np.uint8), (2048, 2048))
  cv2.imwrite('./newest_global_mask_cleaned_loop4.png', resized_wsi_mask)

def loop5(args, CLASSES, row, GT_MASKS_WSI_IDS, full_tissue_masking_canvas, wsi_mask_cleaned, class_gt_masks,
          slide_wise_cutoffs, process_df, index, class_gt_tissue_masks, class_roi_gt_masks):
  for class_index, class_name in enumerate(CLASSES):
    # generate classwise AUC curves/ROC values if we have the GTs
    if '.tif' in row['slide_id']:
      gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name]) or (
                  row['slide_id'].split(' - Series ')[0] in GT_MASKS_WSI_IDS[class_name])
    else:
      gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name])

    if gt_bool:
      if args.do_not_print_much is False:
        print('[In {}] Generating ROC/AUC metrics for raw and masked {} predictions...'.format(inspect.stack()[0][3], class_name))
      # prediction metrics, need to get the areas where prediction/gt values are non zero, or we will be
      # biasing performance to background
      pred_non_zero_mask = full_tissue_masking_canvas
      raw_class_conf = wsi_mask_cleaned[:, :, class_index][pred_non_zero_mask]
      raw_gt = class_gt_masks[class_name][pred_non_zero_mask]
      raw_gt[raw_gt == 255] = 1
      if (np.max(raw_gt) == 0) or (np.max(raw_class_conf) == 0):
        if args.do_not_print_much is False:
          print('[In {}] No {} overlap for ROC/AUC...'.format(inspect.stack()[0][3], class_name))

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
        if args.do_not_print_much is False:
          print('[In {}] {} ideal conf. threshold: {}'.format(inspect.stack()[0][3], class_name, raw_pr_f1_max))

      # masked prediction metrics
      pred_non_zero_mask = class_gt_tissue_masks[class_name]
      roi_class_conf = wsi_mask_cleaned[:, :, class_index][pred_non_zero_mask]
      roi_gt = class_roi_gt_masks[class_name][pred_non_zero_mask]
      if (len(roi_gt) == 0) or (np.max(roi_gt) == 0) or (np.max(roi_class_conf) == 0):
        if args.do_not_print_much is False:
          print('[In {}] No {} overlap for ROC/AUC rois...'.format(inspect.stack()[0][3], class_name))

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
        if args.do_not_print_much is False:
          print('[In {}] {} ideal conf. threshold (ROI only): {}'.format(inspect.stack()[0][3], class_name, roi_pr_f1_max))
    else:
      if args.do_not_print_much is False:
        print('[In {}] No {} predictions for ROC/AUC...'.format(inspect.stack()[0][3], class_name))

    # cleaning masks
    wsi_mask_cleaned[:, :, class_index][
        wsi_mask_cleaned[:, :, class_index] >= slide_wise_cutoffs[class_name]] = 255
    wsi_mask_cleaned[:, :, class_index][
        wsi_mask_cleaned[:, :, class_index] < slide_wise_cutoffs[class_name]] = 0
  
  resized_wsi_mask = cv2.resize((wsi_mask_cleaned*255).astype(np.uint8), (2048, 2048))
  cv2.imwrite('./newest_global_mask_cleaned_loop5.png', resized_wsi_mask)

def loop6(args, CLASSES, row, GT_MASKS_WSI_IDS, wsi_mask_cleaned, class_gt_tissue_masks,
          aux_contours, all_contours):
  for class_index, class_name in enumerate(CLASSES):
    # get the contours
    if '.tif' in row['slide_id']:
      gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name]) or (
                  row['slide_id'].split(' - Series ')[0] in GT_MASKS_WSI_IDS[class_name])
    else:
      gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name])
    if gt_bool:
      if np.max(wsi_mask_cleaned[:, :, class_index][np.invert(class_gt_tissue_masks[class_name])]) > 0:
        if args.do_not_print_much is False:
          print('[In {}] Non ROI {} prediction pixels detected!'.format(inspect.stack()[0][3], class_name))
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
            if args.do_not_print_much is False:
              print('[In {}] {} raw auxiliary contour found. Appending...'.format(inspect.stack()[0][3], class_name))
            aux_class_contours.append(contour)

        if len(aux_class_contours) > 0:
          aux_class_contours = [contour for contour in aux_class_contours if cv2.contourArea(contour) > 25]
          if args.do_not_print_much is False:
            print('[In {}] {} total auxiliary {} class masks found!'.format(inspect.stack()[0][3], len(aux_class_contours), class_name))
          aux_contours[class_name] = aux_class_contours
        else:
          if args.do_not_print_much is False:
            print('[In {}] WARNING: No {} auxiliary class masks found! Likely only detected spillover.'.format(inspect.stack()[0][3],
                    class_name))

    # get the contours
    class_contours, class_hierarchys = cv2.findContours(wsi_mask_cleaned[:, :, class_index],
                                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # append the contours
    if len(class_contours) > 0:
      class_contours = [contour for contour in class_contours if cv2.contourArea(contour) > 25]
      if args.do_not_print_much is False:
        print('[In {}] {} total {} class masks found!'.format(inspect.stack()[0][3], len(class_contours), class_name))
      all_contours[class_name] = class_contours
    else:
      # if args.do_not_print_much is False:
      print('[In {}] WARNING: No {} class masks found! Check thresholds/slide content/slide quality'.format(inspect.stack()[0][3], class_name))

def loop7(args, xor_tile_mask, x_coords, y_coords, sum_percents):
  for y_coord in range(0, xor_tile_mask.shape[0]-2048, 1024):
    for x_coord in range(0, xor_tile_mask.shape[1]-2048, 1024):
      print('!! [loop7] still has some loop!!')
      sum_area_percent = np.sum(xor_tile_mask[y_coord:y_coord + 2048, x_coord:x_coord + 2048]) / (2048 * 2048)
      print('!! [loop7] sum_area_percent: !!', sum_area_percent)
      if sum_area_percent > 0.4:
        x_coords = np.append(x_coords, [x_coord])
        y_coords = np.append(y_coords, [y_coord])
        sum_percents = np.append(sum_percents, [sum_area_percent])
  print('!! [loop7] some x_coords!!: ', x_coords)
  print('!! [loop7] some y_coords!!: ', y_coords)

def loop8(args, x_coords, y_coords, xor_tile_mask, wsi, row, TI_DIR):
  for x, y in zip(x_coords, y_coords):
    print('!! [loop8] x_coords, y_coords !!')
    x_full = int(x * 2)
    y_full = int(y * 2)
    mask = xor_tile_mask[int(y):int(y) + 2048, int(x):int(x) + 2048].astype('uint8')
    mask_full = cv2.resize(mask, (0,0), fx=2, fy=2)
    tile = wsi[y_full:y_full + 4096, x_full:x_full + 4096, :]
    tile = tile.copy()
    tile[np.invert(mask_full.astype('bool'))] = 255
    tile_name = row['slide_id'] + '_xcoord_{}_ycoord_{}.png'.format(x_full, y_full)
    cv2.imwrite(TI_DIR + tile_name, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

def loop9(args, CLASSES, mini_all_contours, coef_x, coef_y):
  for class_index in CLASSES:
    if len(mini_all_contours[class_index]) == 0:
      continue
    for k, contours in enumerate(mini_all_contours[class_index]):
      contours[:, :, 0] = contours[:, :, 0] * coef_x
      contours[:, :, 1] = contours[:, :, 1] * coef_y

def loop10(args, CLASSES, wsi_mini, mini_all_contours, COLORS, axs, full_tissue_masking_canvas,
           coef_x, coef_y):
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

def loop11(args, CLASSES, process_df, index, contour_counts, slide_wise_cutoffs, row, GT_MASKS_WSI_IDS,
           wsi_mask_cleaned, full_tissue_masking_canvas, class_gt_tissue_masks, class_gt_masks, all_contours, aux_contours):
  for class_index, class_name in enumerate(CLASSES):
    process_df.loc[index, class_name + '_counts'] = contour_counts[class_name]
    process_df.loc[index, class_name + '_conf_cutoff'] = slide_wise_cutoffs[class_name]

    # make the special case for the tif
    if '.tif' in row['slide_id']:
      gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name]) or (row['slide_id'].split(' - Series ')[0] in GT_MASKS_WSI_IDS[class_name])
    else:
      gt_bool = (row['slide_id'] in GT_MASKS_WSI_IDS[class_name])

    if gt_bool:
      if args.do_not_print_much is False:
        print('[In {}] Computing cleaned mask performance for {}...'.format(inspect.stack()[0][3], class_name))
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
      iou, filtered_precision, filtered_recall, filtered_f1score, filtered_specificity = binary_metrics(args, pred_mask, gt_mask)
      roi_iou, roi_filtered_precision, roi_filtered_recall, roi_filtered_f1score, roi_filtered_specificity = binary_metrics(args, pred_mask_roi, gt_mask_roi)

      if args.do_not_print_much is False:
        print('[In {}] Saving performance metrics...'.format(inspect.stack()[0][3]))
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

      if args.do_not_print_much is False:
        print('[In {}] Storing the GTs and auxiliary predictions...'.format(inspect.stack()[0][3]))
      class_contours, class_hierarchys = cv2.findContours(class_gt_masks[class_name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      print("!! [loop1] class_gt_masks['Glomerulus'] !!", np.unique(class_gt_masks['Glomerulus']))
      print("!! [loop1] class_gt_masks['Arteriole'] !!", np.unique(class_gt_masks['Arteriole']))
      print("!! [loop1] class_gt_masks['Artery'] !!", np.unique(class_gt_masks['Artery']))
      all_contours[class_name] = list(class_contours)
      if len(aux_contours[class_name]) > 0:
        all_contours[class_name] = all_contours[class_name] + aux_contours[class_name]
      if len(all_contours > 0):
        print('!! [loop11] there are contours !!')

    else:
      if args.do_not_print_much is False:
        print('[In {}] No {} ground truths found for {}.'.format(inspect.stack()[0][3], class_name, row['slide_id']))

def loop12(args, CLASSES, all_contours, OUT_RATIO, row, wsi, IMG_DIR, MASK_DIR, MULTI_DIR, delete_ids):
  for contour_index_id in range(len(CLASSES)):
    if contour_index_id == (len(CLASSES) - 1):
      continue
    for saved_ids, saved_contours in enumerate(all_contours[CLASSES[contour_index_id]]):
      for sub_contour_index_id in range(contour_index_id+1, len(CLASSES)):
        for saved_sub_ids, saved_sub_contours in enumerate(all_contours[CLASSES[sub_contour_index_id]]):
          print('!! [loop12] Iterate over contours to save crops !!')
          # compute the iou of the contour bboxes, find the ones with too much intersection and log them
          x1, y1, w1, h1 = cv2.boundingRect(saved_contours)
          x2, y2, w2, h2 = cv2.boundingRect(saved_sub_contours)
          # get tensors
          box1 = torch.tensor([[x1, y1, x1 + w1, y1 + h1]], dtype=torch.float)
          box2 = torch.tensor([[x2, y2, x2 + w2, y2 + h2]], dtype=torch.float)
          # get iou
          iou = bops.box_iou(box1, box2)

          if iou.item() > 0.0:
            if args.do_not_print_much is False:
              print('[In {}] Overlap detected! IoU: {}'.format(inspect.stack()[0][3], iou.item()))

            # check if there is class uncertainty
            if iou.item() > 0.5:
              if args.do_not_print_much is False:
                print('[In {}] Uncertain prediction class detected! IoU: {}, '
                      'Saving larger mask as mixed class {} and {}...'.format(inspect.stack()[0][3], iou.item(),
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

def loop13(args, CLASSES, all_contours, delete_ids, OUT_RATIO, row, wsi, IMG_DIR, MASK_DIR, MULTI_DIR):
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
