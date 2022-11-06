import os.path as osp
import time
import torch

from mtl.utils.checkpoint_util import save_checkpoint
from mtl.utils.runtime_util import get_host_info
from mtl.utils.misc_util import is_list_of
from mtl.utils.path_util import symlink
from ..core_runner import RUNNERS
from .base_runner import BaseRunner


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.
    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if torch.cuda.is_available():
            model_device = next(self.model.module.parameters()).device
            for key, value in data_batch.items():
                if key == "img_metas" or key == "gt_masks":
                    continue
                if isinstance(value, list):
                    data_batch[key] = [
                        vd.cuda(model_device, non_blocking=True) for vd in value
                    ]
                else:
                    data_batch[key] = value.cuda(model_device, non_blocking=True)
        if train_mode:
            outputs = self.model(**data_batch, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError("train processor must return a dict")
            if "log_vars" in outputs:
                self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
        else:
            outputs = self.model(**data_batch, **kwargs)
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook("before_train_epoch")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook("before_train_iter")
            if self.fp16:
                with torch.cuda.amp.autocast():
                    self.run_iter(
                        data_batch, train_mode=True, epoch=self._epoch, **kwargs
                    )
            else:
                self.run_iter(data_batch, train_mode=True, epoch=self._epoch, **kwargs)
            self.call_hook("after_train_iter")
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook("before_val_iter")
            with torch.no_grad():
                self.run_iter(data_batch, train_mode=False, **kwargs)
            self.call_hook("after_val_iter")

        self.call_hook("after_val_epoch")

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.
        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert is_list_of(workflow, list)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            self._max_epochs = max_epochs

        assert (
            self._max_epochs is not None
        ), "max_epochs must be specified during instantiation"

        for i, flow in enumerate(workflow):
            mode = flow[0]
            epochs = flow[1]
            if mode == "train":
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        if self.rank == 0:
            self.logger.info(
                "Start running, host: %s, work_dir: %s", get_host_info(), work_dir
            )
            self.logger.info("workflow: %s, max: %d epochs", workflow, self._max_epochs)
        self.call_hook("before_run")

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an ' "epoch"
                        )
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        "mode in workflow must be a str, but got {}".format(type(mode))
                    )

                for _ in range(epochs):
                    if mode == "train" and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook("after_run")

    def save_checkpoint(
        self,
        out_dir,
        by_epoch=True,
        filename_tmpl="epoch_{}.pth",
        save_optimizer=True,
        meta=None,
        create_symlink=True,
    ):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(f"meta should be a dict or None, but got {type(meta)}")
        if self.meta is not None:
            meta.update(self.meta)

        if by_epoch:
            filename = filename_tmpl.format(self.epoch + 1)
        else:
            filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, "latest.pth")
            symlink(filename, dst_file)
