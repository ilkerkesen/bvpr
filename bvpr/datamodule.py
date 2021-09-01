import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as ts
import pytorch_lightning as pl
from skimage import color
from torchvision.transforms.transforms import Grayscale, Lambda

from bvpr.data.transform import ABColorDiscretizer, PadBottomRight, DownsizeImage
from bvpr.data.refexp import ReferDataset
from bvpr.data.colorization import ColorsDataset, FlowersDataset, COCODataset


MAX_IMAGE_SIZE = 640


def make_input_transform(normalizer, image_dim):
    return ts.Compose([
        ts.ToTensor(),
        normalizer,
        DownsizeImage(image_dim),
        PadBottomRight(image_dim),
    ])


def make_L_transform(normalizer, image_dim):
    return ts.Compose([
        # ts.Grayscale(num_output_channels=3),
        ts.Lambda(lambda x: x.permute(1, 2, 0)),
        ts.Lambda(lambda x: color.rgb2lab(x)),
        ts.Lambda(lambda x: np.stack([x[:, :, 0]] * 3, -1)),
        ts.ToTensor(),
        normalizer,
    ])


def make_raw_L_transform(image_dim):
    return ts.Compose([
        ts.Resize(image_dim),
        ts.Lambda(lambda x: x.permute(1, 2, 0)),
        ts.Lambda(lambda x: color.rgb2lab(x)[:, :, :1]),
        ts.ToTensor(),
    ])


def make_ab_transform(image_dim, ab_minval=-120):
    # discretizer = ABColorDiscretizer(ab_minval)
    return ts.Compose([
        ts.Resize(image_dim),
        ts.Lambda(lambda x: x.permute(1, 2, 0)),
        ts.Lambda(lambda x: color.rgb2lab(x)),
        ts.Lambda(lambda x: x[:, :, 1:]),
        # ts.ToTensor(),
        # discretizer,
    ])


def make_rgb_transform(image_dim, crop_transform):
    return ts.Compose([
        ts.ToTensor(),
        ts.Resize(image_dim),
        crop_transform(image_dim),
        # ts.Lambda(lambda x: x.permute(1, 2, 0)),
        # ts.Lambda(lambda x: color.rgb2lab(x)),
    ])


class SegmentationDataModule(pl.LightningDataModule):
    TEST_SPLITS = {
        "unc": ("val", "testA", "testB"),
        "unc+": ("val", "testA", "testB"),
        "referit": ("test",),
        "gref": ("val",),
    }

    def __init__(self, config):
        super().__init__()

        normalizer = ts.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        train_image_dim = config["image_size"]
        val_image_dim = MAX_IMAGE_SIZE

        try:
            text_encoder = config["model"]["text_encoder"]["name"]
        except KeyError:
            text_encoder = "LSTMEncoder"

        self.use_bert = False
        if text_encoder == "BERTEncoder":
            self.use_bert = True

        self.train_image_transform = ts.Compose([
            ts.ToTensor(),
            normalizer,
            DownsizeImage(train_image_dim),
            PadBottomRight(train_image_dim),
        ])

        self.train_mask_transform = ts.Compose([
            DownsizeImage(train_image_dim),
            PadBottomRight(train_image_dim),
        ])

        self.val_image_transform = ts.Compose([
            ts.ToTensor(),
            normalizer,
            DownsizeImage(val_image_dim),
            PadBottomRight(val_image_dim),
        ])

        self.val_mask_transform = ts.Compose([
            DownsizeImage(val_image_dim),
            PadBottomRight(val_image_dim),
        ])

        self.config = config

    @property
    def dataset_name(self):
        return self.config["dataset"]["dataset"]

    def setup(self, stage=None):
        train_split, val_split = "train", "val"
        if self.dataset_name == "referit":
            train_split, val_split = "trainval", "test"

        if stage == "fit" or stage is None:
            self.train_data = ReferDataset(
                split=train_split,
                transform=self.train_image_transform,
                mask_transform=self.train_mask_transform,
                use_bert=self.use_bert,
                **self.config["dataset"]
            )

            self.val_data = ReferDataset(
                split=val_split,
                transform=self.val_image_transform,
                mask_transform=self.val_mask_transform,
                use_bert=self.use_bert,
                **self.config["dataset"]
            )

        self.test_datasplits = []
        if stage == "test" or stage is None:
            for split in self.TEST_SPLITS[self.dataset_name]:
                test_data = ReferDataset(
                    split=split,
                    transform=self.val_image_transform,
                    mask_transform=self.val_mask_transform,
                    use_bert=self.use_bert,
                    **self.config["dataset"]
                )
                self.test_datasplits.append(test_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            collate_fn=segmentation_collate_fn,
            **self.config["loader"])

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=False,
            collate_fn=segmentation_collate_fn,
            **self.config["loader"])

    def test_dataloader(self):
        dataloaders = []
        for test_data in self.test_datasplits:
            dataloader = DataLoader(
                test_data,
                collate_fn=segmentation_collate_fn,
                **self.config["loader"],
            )
            dataloaders.append(dataloader)
        return dataloaders


def segmentation_collate_fn(batch):
    batch = [bi for bi in batch if bi["input"] is not None]
    batch = sorted(batch, key=lambda x: len(x["text"]), reverse=True)
    pack = lambda i: torch.cat([bi[i].unsqueeze(0) for bi in batch], 0)
    input, target, size = tuple(pack(i) for i in ("input", "target", "size"))

    batchsize, longest = len(batch), max([len(x["text"]) for x in batch])
    use_bert = True if batch[0]["text_l"] is not None else False
    if not use_bert:
        text = torch.zeros((longest, batchsize), dtype=torch.long)
        text_l = None
        for (i,bi) in enumerate(batch):
            sent = bi["text"]
            text[-len(sent):, i] = sent
    else:
        text = torch.zeros((batchsize, longest), dtype=torch.long)
        text_l = torch.zeros((batchsize, longest), dtype=torch.long)
        for (i,bi) in enumerate(batch):
            text[i, :len(bi["text"])] = torch.tensor(bi["text"])
            text_l[i, :len(bi["text_l"])] = torch.tensor(bi["text_l"])
    return {
        "input": input.half(),
        "text": text,
        "size": size,
        "target": target.half(),
        "text_l": text_l,
        "index": [bi["index"] for bi in batch],
    }


def color_collate_fn(batch):
    batch = sorted(batch, key=lambda x: x["caption_len"], reverse=True)
    visual = torch.cat([bi["input_image"].unsqueeze(0) for bi in batch], dim=0)
    B, L = len(batch), batch[0]["caption_len"]
    captions = torch.zeros(B, L, dtype=torch.long)
    for i, bi in enumerate(batch):
        captions[i, :bi["caption_len"]] = bi["caption"]
    captions_l = [bi["caption_len"] for bi in batch]
    targets = torch.cat([bi["target"].unsqueeze(0) for bi in batch], dim=0)
    
    # prepare soft targets
    soft_targets = None
    if batch[0]["soft_target"] is not None:
        soft_targets = [bi["soft_target"].unsqueeze(0) for bi in batch]
        soft_targets = torch.cat(soft_targets, dim=0)

    # for validation / testing
    Ls = rgbs = None
    if batch[0]["rgb"] is not None:
        Ls = torch.cat([bi["L"].unsqueeze(0) for bi in batch], dim=0)
        rgbs = torch.cat([bi["rgb"].unsqueeze(0) for bi in batch], dim=0)
    indexes = [bi["index"] for bi in batch]

    return {
        "images": visual,
        "captions": captions,
        "captions_l": captions_l,
        "targets": targets,
        "soft_targets": soft_targets,
        "Ls": Ls,
        "rgbs": rgbs,
        "indexes": indexes,
    }


class ColorizationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        normalizer = ts.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        image_dim = 224
        self.L_transform = make_L_transform(normalizer, image_dim)
        self.raw_L_transform = make_raw_L_transform(image_dim // 4)
        self.ab_transform = make_ab_transform(image_dim // 4, -120)
        self.train_transform = make_rgb_transform(image_dim, ts.RandomCrop)
        self.val_transform = make_rgb_transform(image_dim, ts.CenterCrop)
        self.rgb_transform = ts.Resize(image_dim // 4)
        self.demo_raw_L_transform = make_raw_L_transform(image_dim)
        self.demo_rgb_transform = ts.Resize(image_dim)
        self.test_raw_L_transform = make_raw_L_transform(image_dim)
        self.test_rgb_transform = ts.Resize(image_dim)

        self.dataset_class = ColorsDataset
        self.val_split = self.test_split = "val"
        if config["dataset"]["dataset"] == "flowers":
            self.dataset_class = FlowersDataset
            self.val_split = self.test_split = "test"
        elif config["dataset"]["dataset"] == "coco":
            self.dataset_class = COCODataset

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = self.dataset_class(
                split="train",
                transform=self.train_transform,
                L_transform=self.L_transform,
                ab_transform=self.ab_transform,
                raw_L_transform=self.raw_L_transform,
                rgb_transform=self.rgb_transform,
                **self.config["dataset"]
            )

            self.val_data = self.dataset_class(
                split=self.val_split,
                transform=self.val_transform,
                L_transform=self.L_transform,
                ab_transform=self.ab_transform,
                raw_L_transform=self.raw_L_transform,
                rgb_transform=self.rgb_transform,
                **self.config["dataset"]
            )

        if stage == "test" or stage is None:
            self.test_data = self.dataset_class(
                split=self.test_split,
                transform=self.val_transform,
                L_transform=self.L_transform,
                ab_transform=self.ab_transform,
                raw_L_transform=self.test_raw_L_transform,
                rgb_transform=self.test_rgb_transform,
                **self.config["dataset"]
            )

        if stage == "demo" or stage is None:
            self.demo_data = self.dataset_class(
                split=self.test_split,
                transform=self.val_transform,
                L_transform=self.L_transform,
                ab_transform=self.ab_transform,
                raw_L_transform=self.demo_raw_L_transform,
                rgb_transform=self.demo_rgb_transform,
                **self.config["dataset"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            collate_fn=color_collate_fn,
            **self.config["loader"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=False,
            collate_fn=color_collate_fn,
            **self.config["loader"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            shuffle=False,
            collate_fn=color_collate_fn,
            **self.config["loader"],
        )
