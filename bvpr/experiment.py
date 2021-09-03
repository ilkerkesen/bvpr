from bvpr.datamodule import color_collate_fn
import os
import os.path as osp
from functools import reduce
from math import isnan

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from skimage import io
import lpips

from bvpr.models import *
from bvpr.criterion import *
from bvpr.evaluation import *
from bvpr.util import pretty_acc
from bvpr.submodules import BERTEncoder


class BaseExperiment(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = eval(config["model"]["architecture"])(config["model"])
        self.criterion = eval(config["criterion"])()
        self.save_hyperparameters(config)
        self.config = config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs) 

    def configure_optimizers(self):
        optimizer = eval("torch.optim.{}".format(
            self.config["optimizer"]["method"]))

        components = self.model.children()
        components = [c for c in components if not isinstance(c, BERTEncoder)]

        params = []
        for component in self.model.children():
            param = {"params": component.parameters()}
            if issubclass(type(component), BERTEncoder):
                # optimizer = torch.optim.AdamW
                param["lr"] = 5e-5
            params.append(param)

        optimizers = [optimizer(params, **self.config["optimizer"]["params"])]

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


class SegmentationExperiment(BaseExperiment):
    """Lightning Module for Segmentation Experiments"""
    def __init__(self, config):
        super().__init__(config)
        self.thresholds = torch.arange(0, 1, step=0.05).tolist()
        self.IoU_thresholds = torch.arange(0.5, 1.0, 0.1).reshape(1, -1)

    def training_step(self, batch, batch_index):
        image, text, size = batch["input"], batch["text"], batch["size"]
        text_l = batch.get("text_l")
        predicted = self(image, text, size=size, text_l=text_l)
        loss = self.criterion(predicted, batch["target"], size)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("trn_loss", loss)

    def validation_step(self, batch, batch_index):
        image, text, size = batch["input"], batch["text"], batch["size"]
        text_l = batch.get("text_l")
        target = batch["target"]
        predicted = self(image, text, size=size, text_l=text_l)
        loss = self.criterion(predicted, target, size)

        if isinstance(predicted, tuple) or isinstance(predicted, list):
            predicted = predicted[-1]
        predicted = torch.sigmoid(predicted)

        I, U = compute_thresholded(predicted, target, self.thresholds, size)
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

        IoU = cum_IoU  # FIXME: add option for this
        threshold_idx = IoU.argmax().item()
        threshold_val = self.thresholds[threshold_idx]
        this_precision = precision[threshold_idx].tolist()

        self.log("val_loss", total_loss / num_instances)
        self.log("threshold", threshold_val)
        self.log("mIoU", mIoU[threshold_idx].item())
        self.log("cum_IoU", cum_IoU[threshold_idx].item(), prog_bar=True)
        for (th, pr) in zip(self.IoU_thresholds.tolist()[0], this_precision):
            self.log("precision@{:.2f}".format(th), pr)

    def test_step(self, batch, batch_index, dataloader_idx):
        outputs = [] # idx, split, phrase, intersection, union, IoU
        data = self.test_dataloader()[dataloader_idx].dataset
        split = data.split
        index2word = self.model.text_encoder.config["corpus"].dictionary.idx2word
        image, text, size = batch["input"], batch["text"], batch["size"]
        text_l = batch.get("text_l")
        target = batch["target"]
        predicted = self(image, text, size=size, text_l=text_l)

        if isinstance(predicted, tuple) or isinstance(predicted, list):
            predicted = predicted[-1]
        predicted = torch.sigmoid(predicted)
        threshold = self.config["threshold"]
        thresholded = (predicted > threshold).float().data
        intersection, union = compute_iou(thresholded, target, size)
        intersection, union = intersection.tolist(), union.tolist()

        for i in range(len(intersection)):
            word_indices = text[:, i].tolist()
            words = [index2word[index] for index in word_indices if index > 0]
            sentence = " ".join(words)
            I, U = intersection[i], union[i]
            index = batch["index"][i]
            outputs.append((index, split, sentence, I, U, I / U))
        return sorted(outputs, key=lambda x: x[0])

    def test_epoch_end(self, outputs):
        output_file = osp.abspath(osp.expanduser(self.config["output"]))
        with open(output_file, "w") as f:
            for dataset_outputs in outputs:
                for batch_outputs in dataset_outputs:
                    lines = [",".join([str(x) for x in output]) + "\n"
                             for output in batch_outputs]
                    f.writelines(lines)


class ColorizationExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.criterion = None
        if config.get("use_priors", True):
            self.priors = config.get("priors", None)
        else:
            self.priors = None
        if self.priors is not None:
            self.priors = self.priors.to('cuda:0')
        self.loss_fn_alex = lpips.LPIPS(net='alex').to("cuda:0")
        
    def loss_fn(self, scores, targets, soft_targets):
        if soft_targets is None:
            return F.cross_entropy(scores, targets, self.priors)
        else:
            logprobs = F.log_softmax(scores, dim=1)
            weighted = soft_targets * self.priors.view(1, -1, 1, 1)
            output = -logprobs * weighted
            return output.sum() / weighted.sum()

    def training_step(self, batch, batch_index):
        scores = self(batch["images"], batch["captions"], batch["captions_l"])
        loss = self.loss_fn(scores, batch["targets"], batch["soft_targets"])
        if isnan(loss.item()):
            import ipdb; ipdb.set_trace()
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("trn_loss", loss)

    def validation_step(self, batch, batch_index):
        # L, caption, size, ab = batch
        targets = batch["targets"]
        scores = self(batch["images"], batch["captions"], batch["captions_l"])
        loss = self.loss_fn(scores, targets, batch["soft_targets"])
        top1, top5, num_pixels = compute_pixel_acc(scores, targets)
        num_pixels = targets.numel()

        rgbs = torch.round(255 * batch["rgbs"])
        pred = torch.round(255 * self.lab2rgb(batch["Ls"], scores))
        psnr_val = psnr(pred, rgbs)

        rgbs = rgbs / (255. / 2.) - 1.
        pred = pred / (255. / 2.) - 1.
        d = self.loss_fn_alex(pred, rgbs)
        lpips_val = d.mean().item()

        if isnan(loss.item()) or isnan(lpips_val) or isnan(psnr_val.item()):
            import ipdb; ipdb.set_trace()

        return {
            "loss": loss,
            "N": num_pixels,
            "top1": top1,
            "top5": top5,
            "psnr": psnr_val,
            "lpips": lpips_val,
        }

    def validation_epoch_end(self, outputs):
        num_pixels = num_batches = 0
        total_loss = psnr_val = lpips_val = 0.0
        top1 = top5 = 0

        for output in outputs:
            num_pixels += output["N"]
            num_batches += 1
            total_loss += output["loss"] * output["N"]
            top1 += output["top1"]
            top5 += output["top5"]
            psnr_val += output.get("psnr", 0.0)
            lpips_val += output.get("lpips", 0.0)

        self.log("val_loss", total_loss / num_pixels)
        self.log("val_top1_acc", top1 / num_pixels, prog_bar=True)
        self.log("val_top5_acc", top5 / num_pixels, prog_bar=True)
        self.log("val_psnr", psnr_val / num_batches, prog_bar=True)
        self.log("val_lpips", lpips_val / num_batches, prog_bar=True)
        self.colorize_val_images()
        
    def colorize_val_images(self):
        example_sets = [
            ("a [red,green,blue,purple] car parked on a rainy street", 4269, [
                "a red car parked on a rainy street",
                "a green car parked on a rainy street",
                "a blue car parked on a rainy street",
                "a purple car parked on a rainy street",
            ]),

            ("a woman posing and riding on a [red,green,blue,purple] motorcycle", 7150, [
                "a woman posing and riding on a red motorcycle",
                "a woman posing and riding on a green motorcycle",
                "a woman posing and riding on a blue motorcycle",
                "a woman posing and riding on a purple motorcycle",
            ]),

            ("a bird is flying against a [red,gray,blue,yellow] sky", 768, [
                "a bird is flying against a red sky",
                "a bird is flying against a gray sky",
                "a bird is flying against a blue sky",
                "a bird is flying against a yellow sky",
            ]),

            ("a small dog sits on a [red,green,blue] sofa", 335, [
                "a small dog sits on a red sofa",
                "a small dog sits on a green sofa",
                "a small dog sits on a blue sofa",
            ]),
        ]

        dataset = self.val_dataloader().dataset
        for (title, index, captions) in example_sets:
            examples = []
            for caption in captions:
                this = dataset[index]
                this["caption"] = dataset.tokenize_caption(caption)
                this["caption_len"] = len(this["caption"])
                examples.append(this)
            
            batch = color_collate_fn(examples)
            images = batch["images"].to(self.device)
            captions = batch["captions"].to(self.device)
            Ls = batch["Ls"].to(self.device)
            scores = self(images, captions, batch["captions_l"])
            pred = self.lab2rgb(Ls, scores)
            pred = F.interpolate(pred, scale_factor=4, mode="bilinear")
            grid = make_grid(pred, nrow=8)
            self.logger.experiment.add_image(f"caption: {title}", grid, self.current_epoch)

    def test_step(self, batch, batch_index):
        targets = batch["targets"]
        scores = self(batch["images"], batch["captions"], batch["captions_l"])
        topk_pred = scores.topk(5, dim=1).indices == targets.unsqueeze(1)
        B, K, H, W = topk_pred.shape
        topk_pred = topk_pred.reshape(B, K, H*W)
        top1 = topk_pred[:, 0, :].half().mean(dim=1)
        top5 = topk_pred.half().sum(dim=1).mean(dim=1)
        upsampled = F.interpolate(scores, scale_factor=4, mode="bilinear")

        rgbs = torch.round(255 * batch["rgbs"])
        pred = torch.round(255 * self.lab2rgb(batch["Ls"], upsampled))
        mse = torch.mean(torch.abs(pred - rgbs)**2, dim=(1,2,3))
        psnr_vals = 10*torch.log10(255**2 / (mse + 1e-7))
        rgbs = rgbs / (255. / 2.) - 1.
        pred_norm = pred / (255. / 2.) - 1.
        lpips_vals = self.loss_fn_alex(pred_norm, rgbs).flatten()
        
        output_dir = osp.abspath(osp.expanduser(self.config["output"]))
        image_dir = osp.join(output_dir, "jpg")
        if not osp.isdir(image_dir):
          os.makedirs(image_dir)
        
        output = []
        test_data = self.test_dataloader().dataset
        for i in range(B):
            index = batch["indexes"][i]
            caption = test_data.captions[index]["caption"]
            image = pred[i].cpu().permute(1,2,0).numpy()
            image = image.astype(np.uint8)
            image_path = osp.join(image_dir, f"val_{index:05d}.jpg")
            io.imsave(image_path, image)

            output.append((
                index,
                caption,
                pretty_acc(top1[i].item()),
                pretty_acc(top5[i].item()),
                round(psnr_vals[i].item(), 2),
                round(lpips_vals[i].item(), 4),
            ))

        return output

    def test_epoch_end(self, outputs):
        output_dir = osp.abspath(osp.expanduser(self.config["output"]))
        output_file = osp.join(output_dir, "results.csv")
        with open(output_file, "w") as f:
            f.write("idx,caption,top1,top5,psnr,lpips\n")
            for batch_output in outputs:
                batch_output = sorted(batch_output, key=lambda x: x[0])
                for output in batch_output:
                    line = ",".join([str(x) for x in output]) + "\n"
                    f.write(line)
