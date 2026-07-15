# Copyright (c) OpenMMLab. All rights reserved.
# from mmcv.parallel import is_module_wrapper

from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS

@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        # if is_module_wrapper(model):
        #     model = model.module
        model.module.set_epoch(epoch)