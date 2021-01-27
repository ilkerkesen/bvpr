import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as _resnet18
from torchvision.transforms import Normalize
from torchtext.vocab import GloVe


GLOVE_DIM = 300


__all__ = (
    "resnet18",
    "LSTMEncoder",
    "MaskPredictor",
)


def get_glove_vectors(corpus):
    glove = GloVe(name='840B', dim=300) #, cache=glove_cache)
    vectors = np.zeros((len(corpus.dictionary), 300))
    count = 0
    for word, idx in corpus.dictionary.word2idx.items():
        vector = None
        if word == "<pad>":
            continue
        if word == "<unk>":
            word = "unk"
            vector = glove.vectors[glove.stoi[word]]
        if not (word in glove.stoi):
            count += 1
            vector = np.random.randn(300)
        if vector is None:
            vector = glove.vectors[glove.stoi[word]]
        vectors[idx, :] = vector

    return torch.tensor(vectors, dtype=torch.float)


def resnet18(config):
    net = _resnet18(pretrained=config["pretrained"], progress=True)
    layers = list(net.children())
    net = nn.Sequential(*layers[:4+config["num_layers"]-1])
    if config["pretrained"] and config["freeze"]:
       for par in net.parameters():
            par.requires_grad = False
    out_channels = 2**(5+config["num_layers"]-1)
    net.num_channels = out_channels
    net.config = config
    return net

class CBR(nn.Module):
    """(conv => BN => Relu)"""
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class CBRTranspose(nn.Module):
    """(deconv => BN => Relu)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                pdrop=0.0):
        super(CBRTranspose, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(pdrop)
        self.pdrop = pdrop

    def forward(self, x, output_size=None):
        if self.pdrop > 0:
            x = self.dropout(x)
        x = self.deconv(x, output_size=output_size)
        x = self.bnorm(x)
        x = self.act(x)
        return x


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.get("glove", False):
            config["embedding_dim"] = GLOVE_DIM
            vectors = get_glove_vectors(config["corpus"])
            self.embedding = nn.Embedding.from_pretrained(vectors)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=config["num_embeddings"],
                embedding_dim=config["embedding_dim"],
                padding_idx=0)
        self.lstm = nn.LSTM(config["embedding_dim"], config["hidden_size"])
        self.config = config

    def forward(self, x):
        return self.lstm(self.embedding(x))


class MaskPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        self.layers  = nn.ModuleList()
        self.loss_output_layers = nn.ModuleList()
        layer_kwargs = {
            "kernel_size": config["kernel_size"],
            "stride": config["stride"],
            "padding": config["padding"],
        }
        self.config = config

        for i in range(config["num_layers"]-1):
            _in_channels = config["num_channels"] if i > 0 else in_channels
            self.layers.append(
                CBRTranspose(
                    **layer_kwargs,
                    in_channels=_in_channels,
                    out_channels=config["num_channels"]))
            if i == 0:
                continue  # FIXME: why?
            self.loss_output_layers.append(
                nn.ConvTranspose2d(
                    **layer_kwargs,
                    in_channels=in_channels,
                    out_channels=1))
        self.layers.append(
            nn.ConvTranspose2d(
                **layer_kwargs,
                in_channels=config["num_channels"],
                out_channels=1))

    def forward(self, x, image_size=None):
        B, _, _, _ = x.size()
        y = x
        outputs = []

        multiscale_on = self.config.get("multiscale", False)
        last_layer_indx =  len(self.layers) - int(multiscale_on)
        if image_size is not None:
            _, _, H, W = image_size

        for i, layer in enumerate(self.layers[:last_layer_indx]):
            output_size = None
            if image_size is not None:
                frac = 2**(self.config["num_layers"]-i-1)
                output_size = (B, self.config["num_channels"], H // frac, W // frac)
                output_size2 = (B, 1, output_size[2] * 2, output_size[3] * 2)

            y = layer(y, output_size=output_size)
            if multiscale_on and i < last_layer_indx - 1:
                outputs.append(
                    self.loss_output_layers[i](y, output_size=output_size2))

        if multiscale_on:
            outputs.append(self.layers[-1](y, output_size=(B, 1, H, W)))
            return outputs
        else:
            return [y]