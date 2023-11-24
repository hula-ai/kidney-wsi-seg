
"""
This dataset config file is for running inference on GAN generated images. It uses a dummy config file to load 25k
GAN generated images which will be annotated by the teacher network.
"""

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/syed/'
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
    # dict(type='Pad', size_divisor=32),
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
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_tile_train_folds_1-4.json',
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_generated_25k.json',  # coco_tma_generated_25k.json is a dummy config file containing image names of 25k GAN images
        img_prefix='',
        pipeline=test_pipeline,
        samples_per_gpu=24,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_generated_25k.json',
        img_prefix='',
        pipeline=test_pipeline,
        samples_per_gpu=24,
        classes=CLASSES))
evaluation = dict(metric=['bbox', 'segm'])  # , classwise=True, classwise_log=True
