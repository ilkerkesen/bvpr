import torch
from torch.utils.data import DataLoader
from torchvision import transforms as ts
import pytorch_lightning as pl

from bvpr.data.transform import PadBottomRight, DownsizeImage
from bvpr.data.refexp import ReferDataset
from bvpr.data.colorization import ColorizationDataset


MAX_IMAGE_SIZE = 640


class SegmentationDataModule(pl.LightningDataModule):
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

        self.train_image_transform = self.val_image_transform
        self.train_mask_transform = self.val_mask_transform

        self.config = config

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = ReferDataset(
                split="train",
                transform=self.train_image_transform,
                mask_transform=self.train_mask_transform,
                **self.config["dataset"]
            )

            self.val_data = ReferDataset(
                split="val",
                transform=self.val_image_transform,
                mask_transform=self.val_mask_transform,
                **self.config["dataset"]
            )

        if stage == "test" or stage is None:
            self.test_data = ReferDataset(
                split=self.config.get("test_split", "testA"),
                transform=self.val_image_transform,
                mask_transform=self.val_mask_transform,
                **self.config["dataset"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            collate_fn=collate_fn,
            **self.config["loader"])

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=False,
            collate_fn=collate_fn,
            **self.config["loader"])

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            collate_fn=collate_fn,
            **self.config["loader"])


def collate_fn(unsorted_batch):
    batch = sorted(unsorted_batch, key=lambda i: len(i[-1]), reverse=True)
    pack = lambda i: torch.cat([bi[i].unsqueeze(0) for bi in batch], 0)
    img, mask, size = tuple(pack(i) for i in range(len(batch[0])-1))
    batchsize = len(batch)
    longest = len(batch[0][-1])
    text = torch.zeros((longest, batchsize), dtype=torch.long)
    for (i,bi) in enumerate(batch):
        sent = bi[-1]
        text[-len(sent):, i] = sent
    return img.float(), text, size, mask.float()


class ColorizationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.train_data

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            collate_fn=colorization_collate_fn,
            **self.config["loader"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            collate_fn=colorization_collate_fn,
            **self.config["loader"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            collate_fn=colorization_collate_fn,
            **self.config["loader"],
        )
