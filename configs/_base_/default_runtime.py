checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/vmip/slurm_nfs/pre_models/SWA-VFNetX-1-18-53.4_VFNetX-R2101-41e-0.01-0.0001-52.2.pth'
resume_from = None
workflow = [('train', 1)]
