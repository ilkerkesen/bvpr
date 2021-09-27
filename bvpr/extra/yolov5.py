import torch
import torch.nn as nn
from math import log2


YOLO_LAYER_CONF = {
    'yolov5m': [
        (48, 1),
        (96, 2),
        (96, 2),
        (192, 3),
        (192, 3),
        (384, 4),
        (384, 4),
        (768, 5),
        (768, 5),
        (768, 5),
        (384, 5),
        (384, 4),
        (768, 4),
        (384, 4),
        (192, 4),
        (192, 3),
        (384, 3),
        (192, 3),
        (192, 4),
        (384, 4),
        (384, 4),
        (384, 5),
        (768, 5),
        (768, 5),
    ],

    'yolov5l': [
        (64, 1),
        (128, 2),
        (128, 2),
        (256, 3),
        (256, 3),
        (512, 4),
        (512, 4),
        (1024, 5),
        (1024, 5),
        (1024, 5),
        (512, 5),
        (512, 4),
        (1024, 4),
        (512, 4),
        (256, 4),
        (256, 3),
        (512, 3),
        (256, 3),
        (256, 4),
        (512, 4),
        (512, 4),
        (512, 5),
        (1024, 5),
        (1024, 5),
    ],

    'yolov5x': [
        (80, 1),
        (160, 2),
        (160, 2),
        (320, 3),
        (320, 3),
        (640, 4),
        (640, 4),
        (1280, 5),
        (1280, 5),
        (1280, 5),
        (640, 5),
        (640, 4),
        (1280, 4),
        (640, 4),
        (320, 4),
        (320, 3),
        (640, 3),
        (320, 3),
        (320, 4),
        (640, 4),
        (640, 4),
        (640, 5),
        (1280, 5),
        (1280, 5),
    ],
}


class YOLOBackbone(nn.Module):
    def __init__(self, layers, save):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.save = save

    def forward(self, x):
        y, dt = [], []  # outputs
        for m in self.layers:
            if m.f != -1:  # if not from previous layer
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x