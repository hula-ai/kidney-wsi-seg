
"""
This dataset config file is for training on GAN generated images + pseudo annotations from teacher network.
 The json file from test script output needs to be specified in the training annotation dictionary below, so
 that pseudo annotations are used during training.
"""

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/syed/'  # ToDo: Change path to personal directory structure
img_scale = (2048, 2048)
CLASSES = ("Glomerulus", "Arteriole", "Artery")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', img_scale=(2048, 2048), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(768, 2048), (896, 2048), (1024, 2048), (1152, 2048),
                           (1280, 2048), (1408, 2048), (1536, 2048), (1664, 2048),
                           (1792, 2048), (1920, 2048), (2048, 2048)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(1152, 2048), (1280, 2048), (1408, 2048)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(624, 768),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(768, 2048), (896, 2048), (1024, 2048),
                                     (1152, 2048), (1280, 2048), (1408, 2048),
                                     (1536, 2048), (1664, 2048), (1792, 2048),
                                     (1920, 2048), (2048, 2048)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 2048),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_run19_ep2_25k_pseudolabeled.json',  # ToDo: put in name of json file produced by test script containing pseudoannotations made by teacher network on GAN images
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_tile_validation_fold0_fixed.json',
        img_prefix='',
        pipeline=test_pipeline,
        samples_per_gpu=24,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_tile_validation_fold0_fixed.json',
        img_prefix='',
        pipeline=test_pipeline,
        samples_per_gpu=24,
        classes=CLASSES))
evaluation = dict(metric=['bbox', 'segm'], classwise=True, classwise_log=True)
