
"""
This swin configuration file is intended for training swin transformers from scratch. This may be a regular
 experiment training a swin transformer on real labels, or it may be a fine tuning experiment where we load
 a checkpoint trained on GAN pseudo labels, and then fine tune on real data.

Use this config if you want to train on real labels from scratch.

Other customized configs needed:
- HULA_compartment_instance2048x_autoaug.py: Autoaugment TMA compartment dataset, real training labels.
- schedule_1x_HULA_swin.py: AdamW optimizer + scheduling for swin transformer
"""


_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_HULA_compartment.py',
    '../_base_/datasets/HULA_compartment_instance2048x_autoaug.py',
    '../_base_/schedules/schedule_1x_HULA_swin.py',
    '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

# work_dir = "/data/hqvo3/mmdet/run2/"  # ToDo: Change experiment name
# work_dir = "/data/hqvo3/mmdet/run_6_classes_small/"  # ToDo: Change experiment name
# work_dir = "/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/run_6_classes_ps2048_loadfrom_3classes"
# work_dir = "/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/run_6_classes_ps2048_loadfrom_3classes_tissue_detection_and_subtract_smaller_compartments"
work_dir = "/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/run_3_new_classes_resize_wsi_2048"
gpu_ids = range(0, 4)
# gpu_ids = 0
seed = 0

# Adding in load_from checkpoint model
# Read about `pretrained` vs `load_from` here: https://github.com/open-mmlab/mmdetection/issues/1247
# ToDo: If finetuning after pseudo-annotation training, change this path to checkpoint
# load_from = '/data/syed/mmdet/run17_swin_gan_img_train_autoaug/epoch_15.pth'
# load_from = '/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/run_6_classes_ps512/epoch_1.pth'
# load_from = '/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/run_3_classes_ps4096_from_moana/latest.pth'
