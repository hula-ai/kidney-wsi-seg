
## Introduction

This repository is a fork of mmdetection used for running Noisy Student Segmentation experiments

## Important Custom Config Files
* Swin Transformer config files
  * ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py
    * Used for training swin transformers from scratch. This means either a regular experiment training on real labels, or a finetuning experiment where a checkpoint is loaded in that was trained on pseudolabels
  * ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment_gan_img_train.py
    * Used for training on pseudo labels created by a teacher network
  * ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment_gan_inference_2.py
    * Used for inference on GAN-generated images by a teacher network
* Dataset config files
  * ./configs/base/datasets/HULA_compartment_instance.py
    * Used for regular training from scratch
  * ./configs/base/datasets/HULA_compartment_instance_gan_img_train.py
    * Used for training on pseudo labels
  * ./configs/base/datasets/HULA_compartment_instance_gan_inference_2.py
    * Used for inference on GAN-generated images
* Base model config files
  * ./configs/base/models/mask_rcnn_r50_fpn_HULA_compartment.py
    * Custom mask rcnn base for instance segmentation on TMA dataset
* Base schedule config files
  * ./configs/base/schedules/schedule_1x_HULA_swin.py
    * Custom schedule for Swin Transformer, decays learning rate at epoch 13 and 16



## TMA Dataset JSON Files (on NAS)
* /data/syed//coco_tma_generated_25k.json
  * This is a dummy COCO file used during inference on GAN-generated images, it just has 25k GAN image filenames and no annotations 
* /data/syed/coco_tma_tile_train_folds_1-4.json
  * This is the COCO json file for the TMA training dataset
* /data/syed/coco_tma_tile_validation_fold0_fixed.json
  * This is the COCO json file for the TMA validation dataset


## Extra custom files in repository
* ./convert_pseudolabels_to_COCO_json.py
  * This file is used to convert a json file with pseudo annotations outputted by test script (./tools/test.py) into a full COCO json file. The test file output is nearly in COCO format, but a few adjustments are made in this file to it.
* ./gan_pseudolabel_plots.py
  * This file takes the json file with pseudo annotations output by ./tools/test.py, and creates some plots/prints some information on classwise annotation counts and areas.
  * ./json_test_confidence_clustering.py
    * This file does clustering of confidence values of pseudo annotations in test file output json. This helps find cutoff needed to be used in the ./convert_pseudolabels_to_COCO_json.py file 


## Notes
1. To Do inference on GAN-generated images:
   1. Run the test script (./tools/test.py), there is a command in the script for running a teacher model checkpoint on GAN-generated images and outputting pseudo labels to a json file
   2. The outputted json file is incomplete, so run the script at ./convert_pseudolabels_to_COCO_json.py to convert the test output json file into complete COCO format
   3. Now the json file created can be used as pseudo labels to train a student on.
2. For distributed training on pseudo labels, there is a command written in the script ./train_single_gpu.py that is helpful for distributed training


## Experiments already run
* Experiments are on NAS. Directory explanation:
  * /data/syed/ - top level directory, dataset COCO json files will be here
    * /data/syed/mmdet
      * /data/syed/mmdet/analyze_results - on a few early experiments I visualized pseudo labels generated in test script. These are generated using ./tools/analysis_tools/analyze_results.py
      * /data/syed/mmdet/browse_dataset - visualizing real annotations on a few tiles in the TMA dataset
      * /data/syed/mmdet/results - test script (./tools/test.py) output for different training runs
      * /data/syed/mmdet/run______
        * Training runs are kept here
        * run17 is the first noisy student iteration pseudo label training
        * run19 is the first noisy student iteration fine tuning
        * run20 is the second noisy student iteration pseudo label training
