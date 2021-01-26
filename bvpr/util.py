import os
import os.path as osp
from copy import deepcopy

import torch
import pytorch_lightning as pl


__all__ = (
    "process_config",
)


def process_config(config, dataset):
    config = deepcopy(config)
    config["text_encoder"]["corpus"] = dataset.corpus
    return config


def make_mask(real, downsized):
    B = len(real)
    dh, dw = downsized
    mask = torch.zeros(B, 1, dh, dw, dtype=torch.uint8)
    for i in range(B):
        this_h, this_w = real[i]
        mask[i, 0, :this_h, :this_w] = 1
    return mask


def create_checkpoint_callback(config, log_dir):
    checkpoints_path = osp.join(log_dir, "checkpoints")
    config["checkpoint"]["filepath"] = osp.join(checkpoints_path, "{epoch:03d}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**config["checkpoint"])
    last_ckpt = osp.join(checkpoints_path, "last.ckpt")
    last_ckpt = last_ckpt if osp.isfile(last_ckpt) else None
    ckpt_path = config["trainer"]["resume_from_checkpoint"]

    if last_ckpt is not None and ckpt_path is not None:
        raise Exception("resume checkpoint passed (last.ckpt exists already)")

    ckpt_path = last_ckpt if ckpt_path is None else ckpt_path
    if ckpt_path is not None and not osp.isfile(ckpt_path):
        raise Exception("ckpt does not exist at {}".format(ckpt_path))

    return checkpoint_callback, ckpt_path
