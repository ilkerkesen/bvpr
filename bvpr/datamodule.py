import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as ts
import pytorch_lightning as pl
from skimage import color

from bvpr.data.transform import ABColorDiscretizer, PadBottomRight, DownsizeImage
from bvpr.data.refexp import ReferDataset
from bvpr.data.colorization import ColorizationDataset


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
        ts.Lambda(lambda x: np.stack([x / 100.] * 3, axis=-1)),
        ts.ToTensor(),
        normalizer,
    ])


def make_ab_transform(image_dim):
    return ts.Compose([
        ts.ToTensor(),
        ABColorDiscretizer(),
    ])


def make_lab_transform(image_dim, crop_transform):
    return ts.Compose([
        ts.ToTensor(),
        ts.Resize(image_dim),
        crop_transform(image_dim),
        ts.Lambda(lambda x: x.permute(1, 2, 0)),
        ts.Lambda(lambda x: color.rgb2lab(x)),
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
                **self.config["dataset"]
            )

            self.val_data = ReferDataset(
                split=val_split,
                transform=self.val_image_transform,
                mask_transform=self.val_mask_transform,
                **self.config["dataset"]
            )

        self.test_datasplits = []
        if stage == "test" or stage is None:
            for split in self.TEST_SPLITS[self.dataset_name]:
                test_data = ReferDataset(
                    split=split,
                    transform=self.val_image_transform,
                    mask_transform=self.val_mask_transform,
                    **self.config["dataset"]
                )
                self.test_datasplits.append(test_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            collate_fn=collate_fn(),
            **self.config["loader"])

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=False,
            collate_fn=collate_fn(),
            **self.config["loader"])

    def test_dataloader(self):
        dataloaders = []
        for test_data in self.test_datasplits:
            dataloader = DataLoader(
                test_data,
                collate_fn=collate_fn(),
                **self.config["loader"],
            )
            dataloaders.append(dataloader)
        return dataloaders


def collate_fn(task="segmentation"):
    def collate_fn(batch):
        batch = [bi for bi in batch if bi[0] is not None]
        pack = lambda i: torch.cat([bi[i].unsqueeze(0) for bi in batch], 0)
        input, target, size = tuple(pack(i) for i in range(len(batch[0])-1))
        batchsize = len(batch)
        longest = max([len(x[-1]) for x in batch])
        text = torch.zeros((longest, batchsize), dtype=torch.long)
        for (i,bi) in enumerate(batch):
            sent = bi[-1]
            text[-len(sent):, i] = sent
        if task == "segmentation":
            target = target.float()
        elif task == "colorization":
            target = target.long()
        return input.float(), text, size, target
    return collate_fn


class ColorizationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        image_dim = config["image_size"]
        normalizer = ts.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.train_lab_transform = make_lab_transform(image_dim, ts.RandomCrop)
        self.val_lab_transform = make_lab_transform(image_dim, ts.CenterCrop)
        self.train_L_transform = make_L_transform(normalizer, image_dim)
        self.val_L_transform = make_L_transform(normalizer, image_dim)
        self.train_ab_transform = make_ab_transform(image_dim)
        self.val_ab_transform = make_ab_transform(image_dim)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = ColorizationDataset(
                split="train",
                transform=self.train_lab_transform,
                L_transform=self.train_L_transform,
                ab_transform=self.train_ab_transform,
                **self.config["dataset"]
            )

            self.val_data = ColorizationDataset(
                split="val",
                transform=self.val_lab_transform,
                L_transform=self.val_L_transform,
                ab_transform=self.val_ab_transform,
                **self.config["dataset"]
            )

        if stage == "test" or stage is None:
            self.test_data = ColorizationDataset(
                split="val",
                transform=self.val_lab_transform,
                L_transform=self.val_L_transform,
                ab_transform=self.val_ab_transform,
                **self.config["dataset"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            collate_fn=collate_fn("colorization"),
            **self.config["loader"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            collate_fn=collate_fn("colorization"),
            **self.config["loader"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            collate_fn=collate_fn("colorization"),
            **self.config["loader"],
        )
