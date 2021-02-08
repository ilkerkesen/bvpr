import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bvpr.util import make_mask


__all__ = (
    "compute_iou",
    "compute_thresholded",
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
