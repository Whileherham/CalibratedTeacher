_base_ = "base.py"

model = dict(
    type = 'RetinaNet',
    backbone=dict(
        depth=101,
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="path_to_pretrained_resnet"
        ),
    ),
    bbox_head=dict(
            type='RetinaHead_adaptivenegweight2_focaliou',
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            miniou=1,
            ioufocala=0.25,
            ioufocalgamma=2,
            ioufocalbias=0.5,
            ioufocalk=3,
            ioudetach=False),
)

data_root = 'path_to_coco'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file="path_to_coco",
        img_prefix="data/coco/train2017/",
    ),
)

semi_wrapper = dict(
    type="Cali_read_full_100",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.4,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        mining_th_score=0.6,
        mining_warmup=0,
        mining_min_size=10,
        do_merge=True,
        save_results=True,
        save_dir="${work_dir}/save",
        save_size=1000,
        ori_thr = 0.4,
        strict_thr = 0.7,
        cali_method='logistic',
        cali_input='c'
    ),
    test_cfg=dict(inference_on="student"),
)

mul = 1
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
lr_config = dict(step=[120000 * mul, 160000 * mul])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000)
resume_from = 'path_to_calibratedteacher/iter_152000.pth'
