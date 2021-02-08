# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from bvpr.util import make_mask


__all__ = (
    "IoULoss",
    "MaskedBCELoss",
    "MaskedBCEWithLogitsLoss",
    "MaskedMultiScaleBCELoss",
)


class IoULoss(nn.Module):
    """
    Creates a criterion that computes the Intersection over Union (IoU)
    between a segmentation mask and its ground truth.
    Rahman, M.A. and Wang, Y:
    Optimizing Intersection-Over-Union in Deep Neural Networks for
    Image Segmentation. International Symposium on Visual Computing (2016)
    http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
    """

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target, size=None):
        if size is not None:
            mask = make_mask(size, target.size()[-2:])
            input[~mask] = 0
        intersection = (input * target).sum()
        union = ((input + target) - (input * target)).sum()
        iou = intersection / union
        iou_dual = input.size(0) - iou
        if self.size_average:
            iou_dual = iou_dual / input.size(0)
        return iou_dual


class MaskedBCELoss(nn.BCELoss):
    """
    Same with BCE, but it takes advantage of usage of output masks.
    """

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', ignore_index=-1):
        super(MaskedBCELoss, self).__init__(
            weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target, size=None):
        ypred, ygold = input, target
        if size is not None:
            mask = make_mask(size, ygold.size()[-2:])
            ypred, ygold = input[mask], target[mask]
        return F.binary_cross_entropy(
            ypred, ygold, weight=self.weight, reduction=self.reduction)


class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    Same with BCE, but it takes advantage of usage of output masks.
    """

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', ignore_index=-1):
        super(MaskedBCEWithLogitsLoss, self).__init__(
            weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target, size=None):
        if size is not None:
            mask = make_mask(size, target.size()[-2:])
            input, target = input[mask], target[mask]
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            weight=self.weight,
            reduction=self.reduction,
        )


class MaskedMultiScaleBCELoss(nn.BCELoss):
    """
    Same with BCE, but it takes advantage of usage of output masks.
    """

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', ignore_index=-1):
        super(MaskedMultiScaleBCELoss, self).__init__(
            weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, logits_list, target, sizes):
        iter_loss = 0
        ratio = 1 / ( 2 ** (len(logits_list) - 1))
        for logits in logits_list:
            if sizes is not None:
                scaled_sizes = [(int(t[0] * ratio), int(t[1] * ratio)) for t in sizes]
                if ratio < 1.0:
                    resize_shape = (int(target.size(2) * ratio), int(target.size(3) * ratio))
                    resized_target = F.interpolate(target, resize_shape, mode='nearest')
                else:
                    resized_target = target
                mask = make_mask(scaled_sizes, resized_target.size()[-2:])
                ypred, ygold = logits[mask], resized_target[mask]
            iter_loss += ratio * F.binary_cross_entropy_with_logits(
                ypred, ygold,
                weight=self.weight,
                reduction=self.reduction)
            if ratio == 0.5:
                ratio = 1.0
            else:
                ratio = ratio * 2
        return iter_loss
