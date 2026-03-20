# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/syed/'
img_scale = (2048, 2048)
CLASSES = ("Glomerulus", "Arteriole", "Artery")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# This training pipeline has the custom augmentation pipeline including flips, shifts, and photometric distortions.
# This caused problems when training a swin transformer and having it run inference on GAN images, the model was
# not predicting segmentations correctly. Don't use this.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=True),
    dict(type='Resize',
         img_scale=img_scale,
         keep_ratio=True),
    dict(type='RandomFlip',
         direction=["horizontal", "vertical", "diagonal"],
         flip_ratio=[0.2, 0.2, 0.2]),
    dict(type='RandomShift',
         shift_ratio=0.5,
         max_shift_px=512,
         filter_thr_px=32),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    # dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', direction=["horizontal", "vertical", "diagonal"], flip_ratio=[0.2, 0.2, 0.2]),
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
evaluation = dict(metric=['bbox', 'segm'], classwise=True, classwise_log=True)  # ToDo: take out classwise_log for analyze_results and test
