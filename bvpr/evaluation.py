import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_squared_error

from bvpr.util import make_mask


__all__ = (
    "compute_iou",
    "compute_thresholded",
    "compute_pixel_acc",
    "psnr",
)


def compute_iou(input, target, size=None):
    assert(target.shape[-2:] == input.shape[-2:])
    if size is not None:
        mask = make_mask(size, target.size()[-2:])
        input[~mask] = 0
    B = input.size(0)
    temp = (input * target)
    intersection = temp.view(B, -1).sum(1)
    union = ((input + target) - temp).view(B, -1).sum(1)
    return intersection, union


def compute_thresholded(predicted, target, thresholds, size=None):
    batch_size, num_thresholds = predicted.size(0), len(thresholds)
    intersection = torch.zeros(batch_size, num_thresholds)
    union = torch.zeros(batch_size, num_thresholds)

    for (idx, threshold) in enumerate(thresholds):
        thresholded = (predicted > threshold).float().data
        I, U = compute_iou(thresholded, target, size)
        intersection[:, idx], union[:, idx] = I, U

    return intersection, union


def compute_pixel_acc(scores, gold):
    num_pixels = torch.sum(gold >= 0)
    topk_pred = scores.topk(5, dim=1).indices == gold.unsqueeze(1)
    top1 = topk_pred[:, 0, :, :].sum().item()
    top5 = topk_pred.sum().item()
    return top1, top5, num_pixels


def psnr(predicted, target, data_range=255., eps=1e-7):
    mse = mean_squared_error(predicted, target)
    return 10 * torch.log10(data_range**2 / (mse + eps))
