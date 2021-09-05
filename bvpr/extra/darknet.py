# darknet implementation
# transcribed from https://github.com/luogen1996/MCN

import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from skimage import io


class DarknetConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        stride = kwargs.get('stride')
        if stride == (2, 2):  # valid padding
            padding = (0, 0)
        else:  # same padding
            kernel_size = kwargs.get('kernel_size', args[2])
            if isinstance(kernel_size, tuple):
                padding = (
                    kernel_size[0] // 2,
                    kernel_size[1] // 2,
                )
            elif isinstance(kernel_size, int):
                padding = kernel_size // 2
                padding = (padding, padding)

        new_kwargs = {}
        new_kwargs['padding'] = padding
        new_kwargs.update(kwargs)
        super().__init__(*args, **new_kwargs)


class DarknetCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = DarknetConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ResLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layer1 = DarknetCBR(num_channels, num_channels // 2, (1,1))
        self.layer2 = DarknetCBR(num_channels // 2, num_channels, (3,3))

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        return x + y


class DarknetBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_blocks):
        layers = []
        padder = nn.ZeroPad2d((1,0,1,0))
        layers.append(padder)
        layers.append(DarknetCBR(
            in_channels,
            out_channels,
            (3,3),
            stride=(2, 2),
        ))

        for i in range(num_blocks):
            layers.append(ResLayer(out_channels))

        super().__init__(*layers)


def init_darknet(checkpoint_path=None):
    darknet = nn.Sequential(
        DarknetCBR(3, 32, (3, 3)),
        DarknetBlock(32, 64, 1),
        DarknetBlock(64, 128, 2),
        DarknetBlock(128, 256, 8),
        DarknetBlock(256, 512, 8),
        DarknetBlock(512, 1024, 4),
        DarknetCBR(1024, 512, (1,1)),
        DarknetCBR(512, 1024, (3,3)),
        DarknetCBR(1024, 512, (1,1)),
        DarknetCBR(512, 1024, (3,3)),
        DarknetCBR(1024, 512, (1,1)),
        DarknetCBR(512, 1024, (3,3)),
    )

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        darknet.load_state_dict(state_dict, strict=False)
    return darknet


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    image = io.imread(osp.expanduser("~/image.jpg"))
    image = image[np.newaxis, ...] / 255.
    image = image.astype(np.float32)
    tensor = torch.tensor(image).permute(0, 3, 1, 2)

    checkpoint_path = osp.expanduser("~/.cache/darknet/darknet_pretrained.pth")
    model = init_darknet(checkpoint_path=checkpoint_path)
    model.to(device)
    model.eval()
    output = model(tensor.to(device))