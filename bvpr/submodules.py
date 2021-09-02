import os
import os.path as osp
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet18, resnet50, resnet101
from torchtext.vocab import GloVe
from transformers import AutoModel
# import clip

from bvpr.util import add_batch_location_embeddings
from bvpr.util import MOBILENET_SIZE_MAP
from bvpr.util import inf_clamp
from bvpr.extra import deeplab
from bvpr.extra.char_bert.model import CharacterBertModel


GLOVE_DIM = 300
W2V_DIM = 300


__all__ = (
    "LSTMEncoder",
    "BERTEncoder",
    "CharBERTEncoder",
    "MaskPredictor",
    "ImageEncoder",
    "MultimodalEncoder",
    "SegmentationHead",
    "CaptionEncoder",
)


class HalfConv2d(nn.Conv2d):
    def forward(self, x):
        x = super().forward(x)
        x = inf_clamp(x)
        return x


def get_glove_cache():
    path = osp.split(osp.realpath(__file__))[0]
    path = osp.abspath(osp.join(path, "..", ".vector_cache"))
    return path


def get_glove_vectors(corpus, cache=get_glove_cache()):
    glove = GloVe(name='840B', dim=300, cache=cache) 
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


class CBR(nn.Module):
    """(conv => BN => Relu)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            HalfConv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding),
            nn.BatchNorm2d(out_channels),
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
        x = self.bnorm(inf_clamp(x))
        x = self.act(x)
        return x


class FiLMLayer(nn.Module):
    def __init__(self, in_features, out_features, embedding_dim, **kwargs):
        super().__init__()
        self.dense = nn.Linear(embedding_dim, in_features)
        self.gamma_layer = nn.Linear(in_features, out_features)
        self.beta_layer = nn.Linear(in_features, out_features)

    def forward(self, x, c):
        c = self.dense(c)
        gamma = self.gamma_layer(c).view(x.size(0), x.size(1), 1, 1)
        beta = self.beta_layer(c).view(x.size(0), x.size(1), 1, 1)
        return gamma * x + beta


class BatchConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                       padding=0, dilation=1):
        super(BatchConv2DLayer, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, weight, bias=None):
        assert x.shape[0] == weight.shape[0]
        if bias is not None:
            assert bias.shape[0] == weight.shape[0]

        b_i, b_j, c, h, w = x.shape
        out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
        out_channels = weight.shape[1]
        weight = weight.view([-1] + list(weight.shape[2:]))
        out = F.conv2d(
            out,
            weight=weight,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            groups=b_i,
            padding=self.padding)
        out = out.view(b_j, b_i, out_channels, out.shape[-2], out.shape[-1])
        out = out.permute([1, 0, 2, 3, 4])
        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3).unsqueeze(3)
        return out.squeeze(1)


class BatchConv2DKernelFromText(nn.Module):
    def __init__(self, in_channels, out_channels, text_dim, kernel_size,
        stride=1, padding=0, dilation=1, pdrop=0.1):
        super(BatchConv2DKernelFromText, self).__init__()
        self.bconv = BatchConv2DLayer(
            in_channels,
            out_channels,
            stride,
            padding)
        self.dense = nn.Linear(
            in_features=text_dim,
            out_features=kernel_size**2 * in_channels * out_channels,
            bias=True)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.dropout = nn.Dropout(p=pdrop)

    def forward(self, feature_map, text_embedding, return_kernel=False):
        B, C, H, W = feature_map.size()
        weight = self.dropout(text_embedding)
        weight = inf_clamp(self.dense(weight))
        weight = weight.view(B, C, C, self.kernel_size, self.kernel_size)
        weight = F.normalize(weight)
        if return_kernel:
            return self.bconv(feature_map.unsqueeze(1), weight), weight
        return self.bconv(feature_map.unsqueeze(1), weight)


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.get("glove", False):
            config["embedding_dim"] = GLOVE_DIM
            vectors = get_glove_vectors(config["corpus"])
            self.embedding = nn.Embedding.from_pretrained(vectors)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=len(config["corpus"].dictionary.word2idx),
                embedding_dim=config["embedding_dim"],
                padding_idx=0)
        self.lstm = nn.LSTM(
            config["embedding_dim"],
            config["hidden_size"],
            bidirectional=config.get("bidirectional", False),
            batch_first=config.get("batch_first", False),
        )
        self.config = config

    @property
    def hidden_size(self):
        return self.config.get("hidden_size", 256)

    def process_hidden(self, output):
        hidden = inf_clamp(output[1][0])
        L, B, T = hidden.size()
        hidden = hidden.transpose(0, 1).reshape(B, -1)
        return hidden

    def forward(self, x, x_l=None):
        embed = self.embedding(x)
        embed = inf_clamp(embed)
        if x_l is not None:
            batch_first = self.config.get("batch_first", False)
            embed = pack_padded_sequence(embed, x_l, batch_first=batch_first)
        return self.lstm(embed)


class CaptionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.get("w2v", False):
            config["embedding_dim"] = W2V_DIM
            self.embedding = nn.Embedding.from_pretrained(config["vectors"])
            del config["vectors"]
        else:
            self.embedding = nn.Embedding(
                num_embeddings=config["num_embeddings"],
                embedding_dim=config["embedding_dim"],
                padding_idx=0)
        self.lstm = nn.LSTM(
            config["embedding_dim"],
            config["hidden_size"],
            bidirectional=config.get("bidirectional", False),
            batch_first=True)
        self.config = config

    @property
    def hidden_size(self):
        return self.config.get("hidden_size", 256)

    def forward(self, x, x_l):
        embed = pack_padded_sequence(self.embedding(x), x_l, batch_first=True) 
        return self.lstm(embed)


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.init_bert()
        if config.get("freeze", True):
            for p in self.bert.parameters():
                p.requires_grad = False
        self.config = config

    def init_bert(self):
        self.bert = AutoModel.from_pretrained("bert-base-uncased")

    @property
    def hidden_size(self):
        return 768

    def process_hidden(self, output):
        return output.last_hidden_state[:, 0, :]

    def forward(self, x, x_l):
        output = self.bert(input_ids=x, attention_mask=x_l)
        return output


class CharBERTEncoder(BERTEncoder):
    def init_bert(self):
        cache_dir = osp.expanduser("~/.cache/char_bert")
        checkpoint_path = osp.join(cache_dir, 'general_character_bert/')
        self.bert = CharacterBertModel.from_pretrained(checkpoint_path)

    def process_hidden(self, output):
        return output[0][:, 0, :]

    def forward(self, x, x_l):
        return self.bert(x)


class MaskPredictor(nn.Module):
    def __init__(
            self, config, in_channels, num_layers, multiscale, num_classes=1):
        super().__init__()
        self.layers  = nn.ModuleList()
        self.loss_output_layers = nn.ModuleList()
        layer_kwargs = {
            "kernel_size": config["kernel_size"],
            "stride": config["stride"],
            "padding": config["padding"],
        }
        self.config = config
        self.multiscale = multiscale
        self.num_layers = num_layers
        self.num_classes = num_classes

        for i in range(num_layers-1):
            _in_channels = config["num_channels"] if i > 0 else in_channels
            self.layers.append(
                CBRTranspose(
                    **layer_kwargs,
                    in_channels=_in_channels,
                    out_channels=config["num_channels"]))
            if self.multiscale:
                self.loss_output_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=config["num_channels"],
                        out_channels=self.num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ))
        self.layers.append(
            nn.ConvTranspose2d(
                **layer_kwargs,
                in_channels=config["num_channels"],
                out_channels=self.num_classes))

    def forward(self, x, image_size=None):
        B, _, _, _ = x.size()
        y = x
        outputs = []

        multiscale_on = self.multiscale
        if image_size is not None:
            _, _, H, W = image_size

        for i, layer in enumerate(self.layers):
            output_size = None
            if image_size is not None:
                frac = 2**(self.num_layers-i-1)
                num_channels = self.config["num_channels"]
                if len(self.layers)-1 == i:
                    num_channels = 1
                output_size = (B, self.config["num_channels"], H // frac, W // frac)
                output_size2 = (B, 1, output_size[2] * 2, output_size[3] * 2)

            y = layer(y, output_size=output_size)
            if multiscale_on and i != len(self.layers)-1:
                outputs.append(
                    self.loss_output_layers[i](y))

        outputs.append(y)
        return outputs


class ImageEncoder(nn.Module):
    def __init__(self, config, use_location_embeddings=True):
        super().__init__()
        self.config = config
        self.use_location_embeddings = use_location_embeddings
        setup_func = eval("self.setup_{}".format(config["name"]))
        setup_func(config)
        if config["freeze"]:
            for par in self.model.parameters():
                par.requires_grad = False

    def setup_resnet18(self, config):
        net = resnet18(pretrained=config["pretrained"], progress=True)
        layers = list(net.children())
        self.model = nn.Sequential(*layers[:4+config["num_layers"]])
        self.num_downsample = 1 + config["num_layers"]
        self.num_channels = 64 * 2**(config["num_layers"]-1)
        self.num_channels += 8 * self.use_location_embeddings

    def setup_resnet50(self, config):
        net = resnet50(pretrained=config["pretrained"], progress=True)
        layers = list(net.children())
        self.model = nn.Sequential(*layers[:4+config["num_layers"]])
        self.num_downsample = 1 + config["num_layers"]
        self.num_channels = 256 * 2**(config["num_layers"]-1)
        self.num_channels += 8 * self.use_location_embeddings

    def setup_resnet101(self, config):
        net = resnet101(pretrained=config["pretrained"], progress=True)
        layers = list(net.children())
        self.model = nn.Sequential(*layers[:4+config["num_layers"]])
        self.num_downsample = 1 + config["num_layers"]
        self.num_channels = 256 * 2**(config["num_layers"]-1)
        self.num_channels += 8 * self.use_location_embeddings

    def setup_deeplabv3plus_resnet50(self, config):
        cache_dir = osp.abspath(osp.expanduser("~/.cache/torch/checkpoints"))
        filename = "best_deeplabv3plus_resnet50_voc_os16.pth"
        filepath = osp.join(cache_dir, filename)
        ckpt = torch.load(filepath)
        model = deeplab.deeplabv3plus_resnet50()
        model.load_state_dict(ckpt["model_state"])
        layers = list(model.backbone.children())
        num_layers = config["num_layers"]
        self.model = nn.Sequential(*layers[:4+num_layers])

        self.num_downsample = None
        if num_layers < 3:
            self.num_downsample = 2
        else:
            self.num_downsample = 3

        self.num_channels = min(256 * 2**(num_layers-1), 2048)
        self.num_channels += 8 * self.use_location_embeddings

    def setup_deeplabv3plus_resnet101(self, config):
        cache_dir = osp.abspath(osp.expanduser("~/.cache/torch/checkpoints"))
        filename = "best_deeplabv3plus_resnet101_voc_os16.pth"
        filepath = osp.join(cache_dir, filename)
        ckpt = torch.load(filepath)
        model = deeplab.deeplabv3plus_resnet101()
        model.load_state_dict(ckpt["model_state"])
        layers = list(model.backbone.children())
        num_layers = config["num_layers"]
        self.model = nn.Sequential(*layers[:4+num_layers])

        if num_layers == 1:
            self.num_downsample = 2
        else:
            self.num_downsample = 3

        self.num_channels = min(256 * 2**(num_layers-1), 2048)
        self.num_channels += 8 * self.use_location_embeddings

    def setup_clip_resnet101(self, config):
        model, _ = clip.load("RN101")
        num_layers = config["num_layers"]
        layers = list(model.visual.children())
        self.model = nn.Sequential(*layers[:8+num_layers])

        self.num_channels = 64
        if num_layers > 0:
            self.num_channels = 256 * 2**(num_layers-1)

        self.num_downsample = 2
        if num_layers > 1:
            self.num_downsample += num_layers - 1

        self.num_channels += 8 * self.use_location_embeddings

    def setup_hub_deeplabv3(self, config):
        model = torch.hub.load(
            'pytorch/vision:v0.6.0',
            'deeplabv3_resnet101',
            pretrained=True,
        )
        layers = list(model.backbone.children())
        num_layers = config["num_layers"]
        self.model = nn.Sequential(*layers[:4+num_layers])

        self.num_downsample = 2
        if num_layers > 2:
            self.num_downsample += 1

        if num_layers > 0:
            self.num_channels = 256 * 2**(num_layers-1)
        else:
            self.num_channels = 64 
        self.num_channels += 8 * self.use_location_embeddings

    def setup_mobilenetv2(self, config):
        model = torch.hub.load(
            "pytorch/vision:v0.8.2",
            "mobilenet_v2",
            pretrained=True)
        num_layers = config["num_layers"]
        layers = list(model.features.children())
        self.model = nn.Sequential(*layers[:num_layers])
        self.num_downsample = MOBILENET_SIZE_MAP[num_layers-1][1]
        self.num_channels = MOBILENET_SIZE_MAP[num_layers-1][0]
        self.num_channels += 8 * self.use_location_embeddings

    def forward(self, x, size=None):
        y = self.model(x)
        if self.use_location_embeddings:
            y = add_batch_location_embeddings(y, size)
        return y


class MultimodalEncoder(nn.Module):
    def __init__(self, config, in_channels, text_embedding_dim):
        super().__init__()
        self.config = deepcopy(config)
        self.config["text_embedding_dim"] = text_embedding_dim
        self.config["in_channels"] = in_channels
        self.bottom_up = BottomUpEncoder(self.config)
        self.top_down = TopDownEncoder(self.config)

    def forward(self, visual, textual):
        hidden_size = self.config["text_embedding_dim"] // self.config["num_layers"]
        parted = [
            textual[:, i*hidden_size:(i+1)*hidden_size]
            for i in range(self.config["num_layers"])
        ]
        y = self.bottom_up(visual, parted)
        y = self.top_down(y, parted)
        return y[-1]


class BottomUpEncoder(nn.Module):
    LAYER_DICT = {
        "conv": BatchConv2DKernelFromText,
        "film": FiLMLayer,
    }

    def __init__(self, config):
        super().__init__()
        self.use_language = config["bottomup"]
        self.top_down = config["topdown"]

        # setup layers (Conv2d->BatchNorm2d->ReLU)
        self.layers = nn.ModuleList()
        for i in range(config["num_layers"]):
            in_channels = out_channels = config["num_kernels"]
            if i == 0:
                in_channels = config["in_channels"]
            self.layers.append(
                CBR(
                    in_channels=2*in_channels,
                    out_channels=out_channels,
                    kernel_size=config["kernel_size"],
                    stride=config["stride"],
                    padding=config["kernel_size"] // 2,
                    # dilation=config["dilation"],
                ))

        # setup conditional layers (text kernels, FiLM, or nothing)
        self.conditional_layers = nn.ModuleList()
        if self.use_language:
            layer_func = self.LAYER_DICT[config["layer"]]
            conv_args = {
                "kernel_size": config["text_kernel_size"],
                "stride": 1,
                "padding": config["text_kernel_size"] // 2,
                "dilation": config["dilation"],
            }
        else:
            layer_func = UnconditionalLayer
        in_channels, num_kernels = config["in_channels"], config["num_kernels"]
        for i in range(config["num_layers"]):
            text_dim = config["text_embedding_dim"] // config["num_layers"]
            if i == config["num_layers"]:
                text_dim = config["text_embedding_dim"] % text_dim
            num_channels = in_channels if i == 0 else num_kernels
            self.conditional_layers.append(
                layer_func(
                   num_channels,  # in_channels/in_features
                   num_channels,  # out_channels/out_features
                   text_dim,
                   kernel_size=config["text_kernel_size"],
                   stride=1,  # FIXME
                   padding=config["text_kernel_size"] // 2,  # FIXME
                   dilation=config["dilation"],
                ))

        # to prevent information leakage in case of only bottom-up processing
        if config["bottomup"] and not config["topdown"]:
            self.visual_layers = nn.ModuleList()
            for i in range(config["num_layers"]-1):
                in_channels = out_channels = config["num_kernels"]
                if i == 0:
                    in_channels = config["in_channels"]
                self.visual_layers.append(
                    CBR(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=config["kernel_size"],
                        stride=config["stride"],
                        padding=config["kernel_size"] // 2,
                        # dilation=config["dilation"],
                    ))

        self.config = deepcopy(config)

    def forward(self, vis, txt):
        packed = zip(txt, self.layers, self.conditional_layers)
        num_layers = len(self.layers)
        vis = inf_clamp(vis)
        outputs, visual_branch = [vis], [vis]
        for i, (embedding, layer, conditional_layer) in enumerate(packed):
            previous = outputs[-1]
            feature_map = conditional_layer(previous, embedding)
            feature_map = inf_clamp(feature_map)
            feature_map = layer(torch.cat([previous, feature_map], dim=1))
            feature_map = inf_clamp(feature_map)
            outputs.append(feature_map)

            # to prevent leakage in case of only bottom-up processing
            if i > num_layers-2:
                continue
            if not self.config["topdown"] and self.config["bottomup"]:
                visual_layer = self.visual_layers[i]
                visual_feature_map = visual_layer(visual_branch[-1])
                visual_feature_map = inf_clamp(visual_feature_map)
                visual_branch.append(visual_feature_map)

        if not self.config["topdown"] and self.config["bottomup"]:
            outputs = visual_branch + [outputs[-1]]
        return outputs


class TopDownEncoder(nn.Module):
    LAYER_DICT = {
        "conv": BatchConv2DKernelFromText,
        "film": FiLMLayer,
    }

    def __init__(self, config):
        super().__init__()
        self.use_language = config["topdown"]

        # setup layers
        self.layers = nn.ModuleList()
        for i in range(config["num_layers"]):
            in_channels = 2 * config["num_kernels"]
            if i == 0:
                in_channels = config["num_kernels"]
            self.layers.append(
                CBRTranspose(
                    in_channels=in_channels,
                    out_channels=config["num_kernels"],
                    kernel_size=config["kernel_size"],
                    stride=config["stride"],
                    padding=config["kernel_size"] // 2,
                    # dilation=config["dilation"],
                )
            )

        # setup conditional layers
        self.conditional_layers = nn.ModuleList()
        if self.use_language:
            layer_func = self.LAYER_DICT[config["layer"]]
        else:
            layer_func = UnconditionalLayer
        for i in range(config["num_layers"]):
            text_dim = config["text_embedding_dim"] // config["num_layers"]
            if i == config["num_layers"]:
                text_dim = config["text_embedding_dim"] % text_dim
            self.conditional_layers.append(
               layer_func(
                   config["num_kernels"],
                   config["num_kernels"],
                   text_dim,
                   kernel_size=config["text_kernel_size"],
                   stride=1,  # FIXME
                   padding=config["text_kernel_size"] // 2,  # FIXME
                   dilation=config["dilation"],
               ))

        self.config = deepcopy(config)

    def forward(self, bottom_up_outputs, txt):
        output, top_down_outputs = None, []
        packed = zip(txt[::-1], self.layers, self.conditional_layers)
        for i, (embedding, layer, conditional_layer) in enumerate(packed):
            j = -i-1
            bottomup_output = bottom_up_outputs[j]
            output_size = bottom_up_outputs[j-1].size()
            B, C, H, W = bottomup_output.size()
            feature_map = conditional_layer(bottomup_output, embedding)
            feature_map = inf_clamp(feature_map)
            input = feature_map
            if i != 0:
                input = torch.cat([output, feature_map], dim=1)
            output = layer(input, output_size=output_size)
            output = inf_clamp(output)
            top_down_outputs.append(output)
        return top_down_outputs


class UnconditionalLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feature_map, condition):
        return feature_map


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class InfClamp(nn.Module):
    def forward(self, x):
        return inf_clamp(x)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        clamp = InfClamp()
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, clamp, upsampling, activation)

    def forward(self, *args, image_size=None, **kwargs):
        return super().forward(*args, **kwargs)
