# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from bvpr.submodules import *
from bvpr.util import sizes2scales, scales2sizes


class LSTMCNNBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = ImageEncoder(
            config["image_encoder"],
            config["use_location_embeddings"])
        self.text_encoder = self.setup_submodule(config, "text_encoder")
        num_channels = self.image_encoder.num_channels
        hidden_size = self.text_encoder.config["hidden_size"]
        self.mask_predictor = MaskPredictor(
            config["mask_predictor"],
            in_channels=num_channels+hidden_size,
            num_layers=self.image_encoder.num_downsample,
            multiscale=config["multiscale"],
        )
        self.config = config

    def setup_submodule(self, model_config, submodule, **kwargs):
        config = model_config[submodule]
        submodule_class = eval(config["name"], **kwargs)
        return submodule_class(config)

    def forward(self, image, phrase, size=None):
        vis = self.image_encoder(image, size)
        _, (hiddens, _) = self.text_encoder(phrase)
        txt = hiddens.squeeze(0)
        B, L = txt.size()
        B, C, H, W = vis.size()
        txt = txt.reshape(B, L, 1, 1).expand(B, L, H, W)
        joint = torch.cat([vis, txt], dim=1)
        outputs = self.mask_predictor(joint, image_size=image.shape)
        return outputs


class SegmentationModel(nn.Module):
    NUM_CLASSES = 1
    
    """Our main model."""
    def __init__(self, config):
        super().__init__()
        self.image_encoder = ImageEncoder(
            config["image_encoder"],
            config["use_location_embeddings"],
        )
        self.text_encoder = self.setup_submodule(config, "text_encoder")
        num_channels = self.image_encoder.num_channels
        hidden_size = self.text_encoder.config["hidden_size"]
        self.multimodal_encoder = MultimodalEncoder(
            config["multimodal_encoder"],
            in_channels=num_channels,
            text_embedding_dim=hidden_size,
        )
        self.mask_predictor = MaskPredictor(
            config["mask_predictor"],
            in_channels=config["multimodal_encoder"]["num_kernels"],
            num_layers=self.image_encoder.num_downsample,
            multiscale=config["multiscale"],
            num_classes=self.NUM_CLASSES,
        )
        self.config = config

    def setup_submodule(self, model_config, submodule, **kwargs):
        config = model_config[submodule]
        submodule_class = eval(config["name"], **kwargs)
        return submodule_class(config)

    def forward(self, image, phrase, size=None):
        scale = sizes2scales(size, image.size())
        vis = self.image_encoder(image, scales2sizes(scale, image.size()))
        txt = self.text_encoder(phrase)
        joint = self.multimodal_encoder(vis, txt, scale)
        outputs = self.mask_predictor(joint, image_size=image.shape)
        return outputs


class ColorizationModel(SegmentationModel):
    NUM_CLASSES = 3