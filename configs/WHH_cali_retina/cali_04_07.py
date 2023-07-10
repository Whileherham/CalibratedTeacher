_base_ = "base.py"

model = dict(
    backbone=dict(
        depth=101,
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="path_to_pretrained_resnet"
        ),
    )
)

data_root = 'path_to_coco'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file="path_to_annotations(json)",
        img_prefix="data/coco/train2017/",
    ),
)

semi_wrapper = dict(
    type="Cali_read",
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
    ),
    test_cfg=dict(inference_on="student"),
)

mul = 1
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
lr_config = dict(step=[120000 * mul, 160000 * mul])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=152000)
