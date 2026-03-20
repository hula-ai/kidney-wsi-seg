from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv import Config
import mmcv

import os.path as osp

"""
Single-GPU training command:
/home/cougarnet.uh.edu/srizvi7/anaconda3/envs/openmmlab_03292022_5/bin/python train_single_gpu.py

Distributed training command:
CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh \
/home/cougarnet.uh.edu/srizvi7/Desktop/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment_gan_img_train.py 4
"""

# Two different config files, specify based on if you want to train on real images and annotations or GAN generated
# images and pseudo annotations.

# cfg = Config.fromfile("/home/cougarnet.uh.edu/srizvi7/Desktop/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_HULA_compartment.py")
cfg = Config.fromfile("/home/cougarnet.uh.edu/srizvi7/Desktop/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment_gan_img_train.py")

datasets = [build_dataset(cfg.data.train)]

model = build_detector(
    cfg.model,
    train_cfg=cfg.get("train_cfg"),
    test_cfg=cfg.get("test_cfg")
)

model.CLASSES = datasets[0].CLASSES

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

train_detector(model, datasets, cfg, distributed=False, validate=True)

