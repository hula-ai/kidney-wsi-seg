
"""
This dataset config file is for training a model from scratch on TMA dataset. It loads the training and validation
folds and real annotations for the TMA dataset.
"""

# dataset settings
dataset_type = 'CocoDataset'
# data_root = '/data/syed/'  # ToDo: Change path to personal directory structure
# data_root = '/data/public/HULA/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks/'  # === MOANA === ToDo: Change path to personal directory structure
# data_root = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks_2048/'  # === MAUI === ToDo: Change path to personal directory structure
data_root = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/TMA/QuPath_Projects/TMASegmentationR14/downsampled_wsis/'
# CLASSES = ("Glomerulus", "Arteriole", "Artery") # 3 classes
# CLASSES = ("Cortex", "Glomerulus", "Arteriole", "Artery", "Medulla", "CapsuleOther") # 6 classes
# CLASSES = ("Cortex", "Medulla", "CapsuleOther")
CLASSES = ("Cortex", )

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
            # dict(type='Pad', size_divisor=32),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# data = dict(
#     samples_per_gpu=3,
#     workers_per_gpu=8,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'coco_tma_tile_train_folds_1-4.json',
#         img_prefix='',
#         pipeline=train_pipeline,
#         classes=CLASSES),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'coco_tma_tile_validation_fold0_fixed.json',
#         img_prefix='',
#         pipeline=test_pipeline,
#         samples_per_gpu=12,
#         classes=CLASSES),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'coco_tma_tile_validation_fold0_fixed.json',
#         img_prefix='',
#         pipeline=test_pipeline,
#         samples_per_gpu=12,
#         classes=CLASSES))

# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'coco_tma_tile_rm4_tissue_detection_and_subtract_smaller_compartments_train_701515_reduced.json',
#         img_prefix='',
#         pipeline=train_pipeline,
#         classes=CLASSES),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'coco_tma_tile_rm4_tissue_detection_and_subtract_smaller_compartments_val_701515_reduced.json',
#         img_prefix='',
#         pipeline=test_pipeline,
#         samples_per_gpu=2,
#         classes=CLASSES),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'coco_tma_tile_rm4_tissue_detection_and_subtract_smaller_compartments_test_701515_reduced.json',
#         img_prefix='',
#         pipeline=test_pipeline,
#         samples_per_gpu=2,
#         classes=CLASSES))
# evaluation = dict(metric=['bbox', 'segm'], classwise=True, classwise_log=True)  # ToDo: take out classwise_log for analyze_results and test

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix='',
        pipeline=train_pipeline,
        classes=CLASSES),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix='',
        pipeline=test_pipeline,
        samples_per_gpu=2,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix='',
        pipeline=test_pipeline,
        samples_per_gpu=2,
        classes=CLASSES))
evaluation = dict(metric=['bbox', 'segm'], classwise=True, classwise_log=True)  # ToDo: take out classwise_log for analyze_results and test