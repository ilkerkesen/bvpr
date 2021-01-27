from collections import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class DownsizeImage(object):
    """Downsize image if it is larger than the size has been specified"""
    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        nc = 3 if img.dim() == 3 else 1
        scale = min(self.size / im_h, self.size / im_w)
        if scale >= 1.0:
            return Variable(img).view(-1, nc, im_h, im_w)

        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = F.upsample(
            Variable(img).view(-1, nc, im_h, im_w),
            size=(resized_h, resized_w),
            mode='bilinear').squeeze().data
        return out


class PadBottomRight(object):
    """Handle the zero padding by placing image the top left corner"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        c = 1
        h, w = img.shape[-2:]
        if img.dim() in (3, 4):
            c = img.shape[-3]
        padded = torch.zeros((c, self.size, self.size))
        padded[:, :h, :w] = img
        return padded