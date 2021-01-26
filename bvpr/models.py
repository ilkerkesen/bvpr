# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from bvpr.submodules import *
from torchvision.models import resnet50


class LSTMCNNBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = self.setup_submodule(config, "image_encoder")
        self.text_encoder = self.setup_submodule(config, "text_encoder")
        num_channels = self.image_encoder.num_channels
        hidden_size = self.text_encoder.config["hidden_size"]
        self.mask_predictor = MaskPredictor(
            config["mask_predictor"], in_channels=num_channels+hidden_size)
        self.config = config

    def setup_submodule(self, model_config, submodule, **kwargs):
        config = model_config[submodule]
        submodule_class = eval(config["name"])
        return submodule_class(config)

    def forward(self, img, seq, size=None):
        # vis = self.image_encoder(img, size)
        vis = self.image_encoder(img)
        txt = self.text_encoder(seq)
        B, L  = txt.size()
        B, C, H, W = vis.size()
        txt = txt.reshape(B,L,1,1).expand(B,L,H,W)
        vistxt = torch.cat([vis, txt], dim=1)
        mask = self.deconvnet(vistxt, image_size=img.shape)
        return torch.sigmoid(mask)