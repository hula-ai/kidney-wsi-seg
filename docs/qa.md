# Train 3 classes vs train 6 classes:

Some config files and hyperparameters to be changed:

- `./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py`: the attribute `out_indices` IS NOT ALLOWED TO BE CHANGED
- `./configs/_base_/models/mask_rcnn_r50_fpn_HULA_compartment.py`: 2 positions of the attribute `num_classes`, the attribute `out_indices` IS NOT ALLOWED TO BE CHANGED
- `./configs/_base_/datasets/HULA_compartment_instance2048x_autoaug.py`: the tuple `CLASSES`

### Attributes in the mmdet data pipeline `./configs/_base_/datasets/HULA_compartment_instance2048x_autoaug.py`

To reduce the scale of Cortex, Medulla, CapsuleOther 

In this [link](https://mmdetection.readthedocs.io/en/v2.20.0/_modules/mmdet/datasets/pipelines/transforms.html), find the class "Pad".

**Note:** 
- In maui sbatch and srun, need to set enough ram, cpu for loading annotation files
- They need to be changed in training, evaluation, and inference stage.

# To train on a single class

Follow this [link](https://github.com/open-mmlab/mmdetection/issues/7599#issuecomment-1084497646) to set the `CLASSES` correctly

# Change size/scale to train on 3 new classes (Cortex, Medulla, CapsuleOther)

- `./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py`: the attribute `window_size`
- `./configs/_base_/datasets/HULA_compartment_instance2048x_autoaug.py`: the attributes `crop_size` in `RandomCrop`
- `./configs/_base_/models/mask_rcnn_r50_fpn_HULA_compartment.py`: the attribute `mask_size`

# Config saving iters

- Go to this link `https://mmdetection.readthedocs.io/en/dev/tutorials/config.html`
- Search "checkpoint_config". Attribute "interval" is used to set the save interval.
- This post also leads to the original checkpoint class "CheckpointHook" (https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py). Two attributes we should notice are `interval` and `by_epoch`.
- The `checkpoint_config` can be found at `./configs/_base_/default_runtime.py`

# Resume from vs load from

To continue from previous training

Read this https://mmdetection.readthedocs.io/en/v2.2.1/getting_started.html#train-with-multiple-gpus

# `pretrained` vs `load_from`

The two attributes are in the file `./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py`

Read about `pretrained` vs `load_from` here: https://github.com/open-mmlab/mmdetection/issues/1247

When finetuning an old 3-class model with an increase in the number of classes (6 class), `pretrained` should be used because we only need to load the backbone, not the heads, ...

# The error "server socket has failed to listen"

```sh
RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:6007 (errno: 98 - Address already in use). The server socket has failed to bind to 0.0.0.0:6007 (errno: 98 - Address already in use)
```

Solution: go to `tools/dist_train.sh` and change port

# An error when resuming checkpoint `AssertionError: If capturable=False, state_steps should not be CUDA tensors`

Install pytorch 1.12.1 instead of pytorch 1.12.0

View this link https://github.com/pytorch/pytorch/issues/80809#issuecomment-1175211598

# An error of large json file causes SIGGTERM

Read more about it at this [link](https://github.com/open-mmlab/mmcv/issues/1969#issuecomment-1220664266)

Basically, we should decrease the number of crop samples in each slide id

# Training normally for several epochs then CUDA out of memory

This can be because some crops that have too many bounding boxes. Read more [here](https://github.com/open-mmlab/mmdetection/issues/188)

# Remove smaller contours that are inside big contour

View this [link](https://stackoverflow.com/questions/37479338/how-to-remove-a-contour-inside-contour-in-python-opencv)

# Multi-processing to generate json file - tissue detection and subtract smaller compartments

- [Basic python multi-processing - machine learning mastery](https://machinelearningmastery.com/multiprocessing-in-python/)
- SuperFastPython - same author as Machine learning mastery
  - [Python Multiprocessing: The complete guide](https://superfastpython.com/multiprocessing-in-python/)
  - [Multiprocessing Pool map() Multiple Arguments](https://superfastpython.com/multiprocessing-pool-map-multiple-arguments/)
- [`Pool` or `Process`](https://stackoverflow.com/questions/31711378/python-multiprocessing-how-to-know-to-use-pool-or-process)
- [`Pool.map` with multiple arguments](https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments)
- RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start
method. Solved by this [link](https://github.com/pytorch/pytorch/issues/1494#issuecomment-305993854)
  - another [link](https://github.com/pytorch/pytorch/issues/40403#issuecomment-648515174)
  - [best practices and tips](https://pytorch.org/docs/master/notes/multiprocessing.html)

# Important issues:
- If test set is very large, evaluation will be timed out. Solution is at this [link](https://github.com/open-mmlab/mmdetection/issues/6534#issuecomment-1105357931):