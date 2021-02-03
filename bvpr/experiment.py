import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule, LightningDataModule

from bvpr.models import *
from bvpr.criterion import *
from bvpr.evaluation import *


class SegmentationExperiment(LightningModule):
    """Lightning Module for Segmentation Experiments"""
    def __init__(self, config):
        super().__init__()
        self.model = eval(config["model"]["architecture"])(config["model"])
        self.criterion = eval(config["criterion"])()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.thresholds = torch.arange(0, 1, step=0.05).tolist()
        self.IoU_thresholds = torch.arange(0.5, 1.0, 0.1).reshape(1, -1)
        self.save_hyperparameters(config)
        self.config = config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs) 

    def training_step(self, batch, batch_index):
        image, text, size, target = batch
        predicted = self(image, text, size)
        loss = self.criterion(predicted, target, size)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("trn_loss", loss)

    def validation_step(self, batch, batch_index):
        image, text, size, target = batch
        predicted = self(image, text, size=size)
        loss = self.criterion(predicted, target, size)

        if isinstance(predicted, tuple) or isinstance(predicted, list):
            predicted = torch.sigmoid(predicted[-1])

        I, U = compute_thresholded(predicted, target, self.thresholds)
        B = image.size(0)
        return {
            "loss": loss,
            "I": I,  # intersection
            "U": U,  # union
            "B": B,  # batch_size
        }

    def validation_epoch_end(self, outputs):
        cum_I = torch.zeros(len(self.thresholds))
        cum_U = cum_I.detach().clone()
        num_instances, total_loss = 0, 0.0
        num_correct = torch.zeros(
            len(self.thresholds),
            self.IoU_thresholds.numel())
        total_IoU = torch.zeros(len(self.thresholds))

        for output in outputs:
            num_instances += output["B"]
            total_loss += output["loss"]
            I, U = output["I"], output["U"]
            this_IoU = I / U
            total_IoU += torch.sum(this_IoU, dim=0)
            cum_I += I.sum(0)
            cum_U += U.sum(0)
            this_IoU = this_IoU.unsqueeze(-1)
            num_correct += torch.sum(this_IoU >= self.IoU_thresholds, dim=0)

        precision = num_correct / num_instances
        cum_IoU = 100*(cum_I / cum_U)
        mIoU = 100*(total_IoU / num_instances)

        IoU = mIoU  # FIXME: add option for this
        threshold_idx = IoU.argmax().item()
        threshold_val = self.thresholds[threshold_idx]
        this_precision = precision[threshold_idx].tolist()

        self.log("val_loss", total_loss / num_instances)
        self.log("threshold", threshold_val)
        self.log("mIoU", mIoU[threshold_idx].item(), prog_bar=True)
        self.log("cum_IoU", cum_IoU[threshold_idx].item())
        for (th, pr) in zip(self.IoU_thresholds.tolist()[0], this_precision):
            self.log("precision@{:.2f}".format(th), pr)

    def test_step(self, batch, batch_index):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = eval("torch.optim.{}".format(
            self.config["optimizer"]["method"]))
        optimizers = [optimizer(
            self.model.parameters(),
            **self.config["optimizer"]["params"])]

        if self.config.get("scheduler") is None:
            return optimizers, []

        scheduler = eval("torch.optim.lr_scheduler.{}".format(
            self.config["scheduler"]["method"]))
        schedulers = [{
            "scheduler": scheduler(
                optimizers[0], **self.config["scheduler"]["params"]),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }]
        return optimizers, schedulers  