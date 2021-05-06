import os
import os.path as osp
from functools import reduce

import json
import numpy as np
from tqdm import tqdm
from skimage import io, color
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions
from torchvision import transforms as tr
from torchtext.data import get_tokenizer

from bvpr.data.corpus import Corpus
from bvpr.util import prior_boosting
from bvpr.data.transform import ABColorDiscretizer


UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'


class ColorizationDataset(Dataset):
    def __init__(
        self, data_root, split="train", max_query_len=-1, year=2014,
            min_occur=5, transform=None, L_transform=None, ab_transform=None,
            tokenize=True, reduce_colors=True, features_dir=None, **kwargs):
        self.data_root = osp.abspath(osp.expanduser(data_root))
        self.split = split
        self.year = year
        self.image_dir = osp.join(self.data_root, f"{self.split}{self.year}") 
        self.features_dir = features_dir
        if features_dir is not None:
            features_dir = osp.abspath(osp.expanduser(features_dir))
            self.features_dir = osp.join(features_dir, f"{split}{year}")
        self.tokenizer = get_tokenizer("basic_english")
        self.corpus = self.load_corpus(min_occur=min_occur)
        self.reduce_colors = reduce_colors
        priors_path = osp.join(self.data_root, "coco_train_224_ab_probs.npy")
        self.raw_priors = torch.from_numpy(
            prior_boosting(priors_path, 1.0, 0.5)).float()
        self.num_colors = self.raw_priors.numel()
        self.ab_mask = self.raw_priors > 0
        self.color2index = -torch.ones(self.num_colors).long()
        self.color2index[self.ab_mask] = torch.arange(self.ab_mask.sum())
        self.index2color = torch.arange(self.num_colors)[self.ab_mask]
        self.priors = self.raw_priors[self.raw_priors > 0]
        self.load_data()

        self.L_transform = L_transform
        self.ab_transform = ab_transform
        self.transform = transform
        self.tokenize = tokenize

    @property
    def use_features(self):
        return self.features_dir is not None

    def load_corpus(self, min_occur):
        corpus_file = osp.join(self.data_root, f"corpus-{min_occur}.pth")
        if osp.isfile(corpus_file):
            return torch.load(corpus_file)

        json_filename = f"captions_train{self.year}.json"
        json_path = osp.join(self.data_root, "annotations", json_filename)
        with open(json_path, "r") as f:
            json_data = json.load(f)

        count_dict = dict()
        for entry in tqdm(json_data["annotations"]):
            words = self.tokenizer(entry["caption"])
            for word in words:
                count_dict[word] = 1 + count_dict.get(word, 0)

        corpus = Corpus()
        corpus.dictionary.add_word(PAD_TOKEN)
        corpus.dictionary.add_word(UNK_TOKEN)
        corpus.dictionary.add_word(SOS_TOKEN)
        for (word, count) in tqdm(count_dict.items()):
            if count >= min_occur:
                corpus.dictionary.add_word(word)
        torch.save(corpus, corpus_file)
        return corpus

    def load_data(self):
        json_filename = f"captions_{self.split}{self.year}.json"
        json_path = osp.join(self.data_root, "annotations", json_filename)
        with open(json_path, "r") as f:
            json_data = json.load(f)

        self.image_dict = dict()
        for image in json_data["images"]:
            self.image_dict[image["id"]] = {
                "file_name": image["file_name"],
                "height": image["height"],
                "width": image["width"],
            }

        self.captions = json_data["annotations"]

    def read_rgb_image(self, image_file):
        image_path = osp.abspath(osp.join(self.image_dir, image_file))
        if not osp.exists(image_path):
            return (None, None, None)
        image = io.imread(image_path)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        return image

    def tokenize_caption(self, caption):
        tokens = [SOS_TOKEN] + self.tokenizer(caption)
        w2i = self.corpus.dictionary.word2idx
        tokens = [w2i.get(t, w2i[UNK_TOKEN]) for t in tokens]
        return torch.tensor(tokens)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        item = self.captions[index]
        caption = item["caption"]
        if self.tokenize:
            caption = self.tokenize_caption(caption)
        
        image_path = self.image_dict[item["image_id"]]["file_name"]
        image = self.read_rgb_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        im_h, im_w = image.shape[:2]
        size = torch.tensor([im_h, im_w])

        L = image[:, :, 0]
        if self.L_transform is not None:
            L = self.L_transform(L)

        ab = image[:, :, 1:]
        if self.ab_transform is not None:
            ab = self.ab_transform(ab)
            if self.reduce_colors:
                ab = self.color2index[ab]
        return L, ab, size, caption

