In the `train` subfolder, there are 5 config files that you should look at:

#### 1/ The main file `./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py`.
- The architecture is Swin transformer.
- The data mask annotations are saved in coco format
- Some important variables:
  - `pretrained`: link to official pretrained model file provided by authors
  - `work_dir`: where to save our checkpoint files
  - `gpu_ids`: which gpus are used to train
  - `load_from`: directory path to our previous checkpoint, for finetuning
  - `out_indices`: the number of classes, indice for each class, remember to count the background as 0

#### 2/ '../_base_/models/mask_rcnn_r50_fpn_HULA_compartment.py'
- Base model MaskRCNN
- Some important variables:
  - `num_classes` in `bbox_head`: an integer denoting the number of classes in the dataset
  - `num_classes` in `mask_head`: an integer denoting the number of classes in the dataset
    
#### 3/ '../_base_/datasets/HULA_compartment_instance2048x_autoaug.py'
- Definitions of all things related to dataset from input path, classes, normalization mean and standard deviation, augmentation, image resolution, path to annotation file, samples per gpu.
- Some important variables:
  - `data_root`: path to the folder that contain .json annotation file
  - `CLASSES`: tuple of classes
  - `ann_file`: path to a json file, which contains class label of each tile
    
#### 4/ '../_base_/schedules/schedule_1x_HULA_swin.py'
- Optimizer config
- Some important variables:
  - `type`: for example "AdamW"
  - `lr`: learning rate
  - `weight_decay`
  - `warmup_iters`
  - `warmup_ratio`
  - `max_epochs`

#### 5/ '../_base_/default_runtime.py'

#### 6/ For changing used gpus

- 2 positions in the training command
- `gpu_ids` in `./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py`