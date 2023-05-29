_base_ = '/mnt/sda/2022-0526/home/scc/ytx/Software/mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py'


data_root = '/mnt/sda/2022-0526/home/scc/ytx/ANN/Main_Project/data/StanfordDogs/'
metainfo = {
    'classes': ('dog', )
}

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

default_hooks = dict(visualization=dict(type='DetVisualizationHook', draw=True))
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0),  # Change pad_val if necessary
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(512, 512), pad_val=0),  # Change pad_val if necessary
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'coco_train.json',
        data_prefix=dict(img=data_root + 'Images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'coco_test.json',
        data_prefix=dict(img=data_root + 'Images/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'coco_test.json')
test_evaluator = val_evaluator

work_dir = '/mnt/sda/2022-0526/home/scc/ytx/ANN/Main_Project/work_dirs/bird/'
