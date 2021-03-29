import torch
from torch.utils.data import DataLoader
from torchvision import transforms as ts
import pytorch_lightning as pl

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


def make_ab_transform(image_dim):
    return ts.Compose([
        ts.ToTensor(),
        ABColorDiscretizer(),
        DownsizeImage(image_dim),
        PadBottomRight(image_dim, pad_value=-1),
    ])


class SegmentationDataModule(pl.LightningDataModule):
    TEST_SPLITS = {
        "unc": ("testA", "testB"),
        "unc+": ("testA", "testB"),
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

        normalizer = ts.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        train_image_dim = config["image_size"]
        inference_image_dim = MAX_IMAGE_SIZE

        self.train_L_transform = make_input_transform(
            normalizer, train_image_dim)
        self.inference_L_transform = make_input_transform(
            normalizer, inference_image_dim)

        self.train_ab_transform = make_ab_transform(train_image_dim)
        self.inference_ab_transform = make_ab_transform(inference_image_dim)

        self.config = config

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = ColorizationDataset(
                split="train",
                L_transform=self.train_L_transform,
                ab_transform=self.train_ab_transform,
                **self.config["dataset"]
            )

            self.val_data = ColorizationDataset(
                split="val",
                L_transform=self.inference_L_transform,
                ab_transform=self.inference_ab_transform,
                **self.config["dataset"]
            )

        if stage == "test" or stage is None:
            self.test_data = ColorizationDataset(
                split="test",
                L_transform=self.inference_L_transform,
                ab_transform=self.inference_ab_transform,
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
