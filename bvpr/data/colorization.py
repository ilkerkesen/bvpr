import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions


class ColorizationDataset(Dataset):
    def __init__(
        self, data_root, transform=None, split="train", max_query_len=-1,
        year=2014):
        data_root = osp.abspath(osp.expanduser(data_root))
        image_path = osp.join(data_root, f"{split}{year}") 
        json_filename = f"captions_{split}{year}_.json"
        json_path = osp.join(data_root, "annotations", json_filename)
        self.coco_dataset = CocoCaptions(image_path, json_path)

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, index):
        return self.coco_dataset[index]