
"""
This swin configuration file is intended for running inference on GAN images with a trained swin transformer.
A separate inference config file may not be needed, but one was created anyways to be safe. Use this config
if you want to run inference on GAN-generated images.

Other customized configs needed:
- HULA_compartment_instance_gan_inference_2.py: Autoaugment TMA compartment dataset, real training labels.
- schedule_1x_HULA_swin.py: AdamW optimizer + scheduling for swin transformer
"""

_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_HULA_compartment.py',
    '../_base_/datasets/HULA_compartment_instance_gan_inference_2.py',
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

work_dir = "/data/syed/mmdet/run20_swin_multi_gpu_autoaug/"  # ToDo: Change experiment name
gpu_ids = range(5, 6)
seed = 0
