from bvpr.util import annealed_mean
from collections import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import color
from torchvision.transforms import functional as F_
from torchvision import transforms as tr


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
    def __init__(self, size, pad_value=0):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, img):
        c = 1
        h, w = img.shape[-2:]
        if img.dim() in (3, 4):
            c = img.shape[-3]
        padded = torch.ones((c, self.size, self.size)) * self.pad_value
        padded[:, :h, :w] = img
        return padded


class ABColorDiscretizer(object):
    def __init__(self, min_val=-120):
        super().__init__()
        self.min_val = min_val

    def __call__(self, image):
        bin_size = 10
        min_val = self.min_val
        grid_dim = 25

        quantized = torch.round((image - min_val) / bin_size)
        discrete = quantized[0, :, :] * grid_dim + quantized[1, :, :]
        return discrete.long()


class LAB2RGB(object):
    def __init__(self, ab_kernel=None, device="cuda:0", mode="eval"):
        self.device = torch.device(device)
        self.ab_kernel = ab_kernel.to(self.device)
        self.mode = mode
        assert mode in ("eval", "demo")

    def __call__(self, L, scores, T=1.0):
        probs = F.softmax(scores, dim=1)
        probs = annealed_mean(probs, T=T)
        ab_pred = F.conv2d(probs, self.ab_kernel)
        predicted = torch.cat([L, ab_pred], dim=1)
        predicted = predicted.permute(0, 2, 3, 1).cpu().numpy()
        predicted = color.lab2rgb(predicted)
        if self.mode == "eval":
            predicted = torch.tensor(predicted, device=self.device)
            predicted = predicted.permute(0, -1, 1, 2)
        elif self.mode == "demo":
            predicted = predicted.squeeze(0)
        return predicted


class ExtToTensor(tr.ToTensor):
    def __call__(self, image, mask):
        image = F_.to_tensor(image)
        mask = F_.to_tensor(mask)
        return image, mask


class ExtCenterCrop(tr.CenterCrop):
    def forward(self, image, mask):
        image = super().forward(image)
        mask = super().forward(mask)
        return image, mask


class ExtRandomCrop(tr.RandomCrop):
    def forward(self, image, mask):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            mask = F.pad(mask, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(image)
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            image = F.pad(image, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, self.fill, self.padding_mode)
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            image = F.pad(image, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return image, mask
    

class ExtResize(tr.Resize):
    def forward(self, image, mask):
        image = super().forward(image)
        mask = super().forward(mask)
        return image, mask
    

class ExtCompose(tr.Compose):
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask