import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from bvpr.util import make_mask


__all__ = (
    "compute_iou",
    "compute_thresholded",
)


def compute_iou(mask, target):
    assert(target.shape[-2:] == mask.shape[-2:])
    B = mask.size(0)
    temp = (mask * target)
    intersection = temp.view(B, -1).sum(1)
    union = ((mask + target) - temp).view(B, -1).sum(1)
    return intersection, union


def compute_thresholded(mask, target, thresholds):
    batch_size, num_thresholds = predicted.size(0), len(thresholds)
    intersection = torch.zeros(batch_size, num_thresholds)
    union = torch.zeros(batch_size, num_thresholds)

    for (idx, threshold) in enumerate(thresholds):
        thresholded = (mask > threshold).float().data
        intersection[:, idx], union[:, idx] = compute_iou(thresholded, target)
        
    return intersection, union