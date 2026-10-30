"""Bounded two-device throughput check using the sibling model checkout."""
_base_ = [
    '../../../../../../mmdetection3d/configs/pointpillars/'
    'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
]
train_dataloader = dict(batch_size=4, num_workers=10, persistent_workers=True,
                        dataset=dict(indices=640))
train_cfg = dict(max_epochs=1, val_interval=1000)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
default_hooks = dict(logger=dict(interval=1),
                     checkpoint=dict(interval=1000, save_last=False))
