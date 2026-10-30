"""Three-iteration integration smoke test for the sibling official checkout."""
_base_ = [
    '../../../../../../mmdetection3d/configs/pointpillars/'
    'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
]
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=3,
                 val_interval=1000)
train_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
default_hooks = dict(logger=dict(interval=1),
                     checkpoint=dict(interval=1000, by_epoch=False))
