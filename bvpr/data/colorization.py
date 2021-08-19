import os
import os.path as osp
from functools import reduce

import json
import cv2
import h5py
import pickle
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
        im_h, im_w = image.shape[1:]
        size = torch.tensor([im_h, im_w])

        if self.L_transform is not None:
            L = self.L_transform(image)

        ab = image
        if self.ab_transform is not None:
            ab = self.ab_transform(ab)
            if self.reduce_colors:
                ab = self.color2index[ab]
        return L, ab, size, caption


class ColorizationReferenceDataset(Dataset):
    def __init__(self, data_root, split="train", max_query_len=20,
                 transform=None, L_transform=None, ab_transform=None,
                 reduce_colors=False, **kwargs):
        self.data_root = osp.abspath(osp.expanduser(data_root))
        self.split = split
        self.reduce_colors = reduce_colors
        self.data_file_path = osp.join(self.data_root, "coco_colors.h5")
        self.features_file_path = osp.join(self.data_root, "image_features.h5")
        self.priors_path = osp.join(
            self.data_root, "coco_priors_onehot_625.npy")
        self.raw_priors = torch.from_numpy(
            prior_boosting(self.priors_path, 1.0, 0.5)).float()
        self.num_colors = self.raw_priors.numel()
        self.data_file = h5py.File(self.data_file_path, "r")
        self.features_file = h5py.File(self.features_file_path, "r")
        self.lookup_enc = LookupEncode(
            osp.join(self.data_root, "full_lab_grid_10.npy"))
        self.vocab = pickle.load(
            open(osp.join(self.data_root, "coco_colors_vocab.p"), "rb"))
        self.embeddings = pickle.load(
            open(osp.join(self.data_root, "w2v_embeddings_colors.p"), "rb"),
            encoding="iso-8859-1")
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float)

        self.ab_mask = self.raw_priors > 0
        self.color2index = -torch.ones(self.num_colors).long()
        self.color2index[self.ab_mask] = torch.arange(self.ab_mask.sum())
        self.index2color = torch.arange(self.num_colors)[self.ab_mask]
        self.priors = self.raw_priors[self.raw_priors > 0]

    def __len__(self):
        return self.data_file[f"{self.split}_words"].shape[0]

    def __getitem__(self, index):
        image = self.data_file[f"{self.split}_ims"][index]
        target = cvrgb2lab(image)[::4, ::4, 1:]  # FIXME: this is not mine!
        target = self.lookup_enc.encode_points(target)
        target = torch.tensor(target).long()
        features = self.features_file[f"{self.split}_features"][index]
        caption = self.data_file[f"{self.split}_words"][index]
        caption_len = self.data_file[f"{self.split}_length"][index]

        if self.reduce_colors:
            target = self.color2index[target]

        return (
            torch.tensor(features),
            torch.tensor(caption.astype("long")),
            caption_len,
            target,
        )


class BaseColorsDataset(Dataset):
    def __init__(self, data_root, split="train", max_query_len=20,
                 transform=None, L_transform=None, ab_transform=None,
                 raw_L_transform=None, rgb_transform=None, tokenize=True,
                 reduce_colors=True, min_occur=5, gamma=0.5, **kwargs):
        super().__init__()
        self.data_root = osp.abspath(osp.expanduser(data_root))
        self.split = split
        self.tokenize = tokenize
        self.reduce_colors = reduce_colors
        self.json_file = osp.join(data_root, "dataset.json")
        self.image_dir = osp.join(data_root, "jpg")
        self.tokenizer = get_tokenizer("basic_english")
        self.transform = transform
        self.L_transform = L_transform
        self.ab_transform = ab_transform
        self.raw_L_transform = raw_L_transform
        self.rgb_transform = rgb_transform
        self.gamma = gamma
        self.load_data()
        self.corpus = self.load_corpus(min_occur=min_occur)
        self.load_priors()
        self.setup_discritizer()

    def load_priors(self, prior_set="coco"):
        if prior_set == "coco":
            self.load_coco_priors()
        else:
            pass

    def load_coco_priors(self):
        priors_path = osp.join(self.data_root, "priors-56x56.npy")
        self.raw_priors = torch.from_numpy(
            prior_boosting(priors_path, 1.0, gamma)).float()
        self.raw_priors = self.raw_priors.flatten()
        self.num_colors = self.raw_priors.numel()
        self.ab_mask = self.raw_priors > 0
        self.color2index = -torch.ones(self.num_colors).long()
        self.color2index[self.ab_mask] = torch.arange(self.ab_mask.sum())
        self.index2color = torch.arange(self.num_colors)[self.ab_mask]
        self.priors = self.raw_priors[self.raw_priors > 0]

    def setup_discretizer(self, prior_set="coco"):
        if prior_set == "coco":
            self.discretizer = ABColorDiscretizer()
        elif prior_set == "imagenet":
            pass

    def load_data(self):
        with open(self.json_file, "r") as f:
            self.json_data = json.load(f)
        self.captions = [x for x in self.json_data if x["split"] == self.split]

    def load_corpus(self, min_occur):
        corpus_file = osp.join(self.data_root, f"corpus-{min_occur}.pth")
        if osp.isfile(corpus_file):
            return torch.load(corpus_file)

        train_sentences = [
            x["caption"]
            for x in self.json_data
            if x["split"] == "train"
        ]

        count_dict = dict()
        for sentence in train_sentences:
            tokens = self.tokenizer(sentence.lower())
            for token in tokens:
                count_dict[token] = 1 + count_dict.get(token, 0)

        corpus = Corpus()
        corpus.dictionary.add_word(PAD_TOKEN)
        corpus.dictionary.add_word(UNK_TOKEN)
        corpus.dictionary.add_word(SOS_TOKEN)

        words = sorted([
            (word, count)
            for (word, count) in count_dict.items()
            if count >= min_occur
        ], key=lambda x: x[1])
        words = [word for (word, _) in words]
        for word in words:
            corpus.dictionary.add_word(word)

        torch.save(corpus, corpus_file)
        return corpus

    def read_rgb_image(self, image_filename):
        image_path = osp.join(self.image_dir, image_filename)
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

        image_filename = item["image_file"]
        image = self.read_rgb_image(image_filename)
        if self.transform is not None:
            image = self.transform(image)

        L = ab = image
        if self.L_transform is not None:
            L = self.L_transform(image)

        if self.ab_transform is not None:
            ab = self.ab_transform(image)
            ab = self.ab_discretizer(ab)
            ab = self.reduce_colors(ab)

        raw_L = None
        if self.raw_L_transform is not None:
            raw_L = self.raw_L_transform(image)

        rgb = None
        if self.rgb_transform is not None:
            rgb = self.rgb_transform(image)

        return {
            "input_image": L,
            "caption": caption,
            "caption_len": len(caption),
            "target": ab,
            "L": raw_L,
            "rgb": rgb,
            "index" : index,
        }


class FlowersDataset(BaseColorsDataset):
    pass


class ColorsDataset(BaseColorsDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = pickle.load(
            open(osp.join(self.data_root, "w2v_embeddings_colors.p"), "rb"),
            encoding="iso-8859-1")
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float)

    def load_corpus(self, min_occur):
        corpus_file = osp.join(self.data_root, "corpus-0.pth")
        return torch.load(corpus_file)


class COCODataset(BaseColorsDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_dir = osp.join(self.data_root, "images", self.split)


class LookupEncode(object):
    '''Encode points using lookups'''
    def __init__(self, km_filepath=''):
        self.cc = np.load(km_filepath)
        self.offset = np.abs(np.amin(self.cc)) + 17 # add to get rid of negative numbers
        self.x_mult = 59 # differentiate x from y
        self.labels = {}
        for idx, (x,y) in enumerate(self.cc):
            x += self.offset
            x *= self.x_mult
            y += self.offset
            self.labels[x+y] = idx

    # returns bsz x 224 x 224 of bin labels (625 possible labels)
    def encode_points(self, pts_nd, grid_width=10):

        pts_flt = pts_nd.reshape((-1, 2))

        # round AB coordinates to nearest grid tick
        pgrid = np.round(pts_flt / grid_width) * grid_width

        # get single number by applying offsets
        pvals = pgrid + self.offset
        pvals = pvals[:, 0] * self.x_mult + pvals[:, 1]

        labels = np.zeros(pvals.shape,dtype='int32')

        # lookup in label index and assign values
        for k in self.labels:
            labels[pvals == k] = self.labels[k]

        return labels.reshape(pts_nd.shape[:-1])

    # return lab grid marks from probability distribution over bins
    def decode_points(self, pts_enc):
        print(pts_enc)
        return pts_enc.dot(self.cc)


def cvrgb2lab(img_rgb):
    cv_im_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype('float32')
    cv_im_lab[:, :, 0] *= (100. / 255)
    cv_im_lab[:, :, 1:] -= 128.
    return cv_im_lab
