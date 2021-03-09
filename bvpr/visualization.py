import os.path as osp
from copy import deepcopy

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import numpy as np
from scipy.spatial.distance import euclidean, cosine

from bvpr import data
from bvpr.util import process_config
from bvpr.models import SegmentationModel as Model
from bvpr.datamodule import SegmentationDataModule as DataModule

DATASET_NAME = "unc"
DATASET_PATH = "~/data/refexp/data"
CHECKPOINT_PATH = "~/model.ckpt"
DEVICE = "cuda:0"
THRESHOLD = 0.40


class SegmentationDemo(object):
    def __init__(self, checkpoint_path, threshold, data_path,
                 dataset="unc", device="cuda:0"):
        data_config = {
            "image_size": 640,

            "dataset": {
                "dataset": dataset,
                "data_root": data_path,
            },

            # just in case
            "loader": {
                "num_workers": 0,
                "pin_memory": False,
                "batch_size": 1,
            }
        }

        self.checkpoint_path = osp.abspath(osp.expanduser(checkpoint_path))
        self.threshold = threshold
        self.device = torch.device(device)

        # load data
        self.data_module = DataModule(data_config)
        self.data_module.setup(stage="test")

        # load model
        ckpt = torch.load(self.checkpoint_path)
        state_dict = ckpt["state_dict"].items()
        state_dict = {".".join(k.split(".")[1:]): v for (k, v) in state_dict}
        model_config = ckpt["hyper_parameters"]["model"]
        self.model = Model(model_config).to(device)
        self.model.load_state_dict(state_dict)

    def get_data(self, example_id, datasplit_id=0, custom_phrase=None):
        test_data = self.data_module.test_datasplits[datasplit_id]
        image_file, mask_file, phrase = test_data.images[example_id]
        phrase = phrase if custom_phrase is None else custom_phrase
        image, size = test_data.read_image(image_file)
        gold = torch.load(osp.join(test_data.mask_dir, mask_file))
        return image, phrase, size, gold

    def predict(self, img, txt, size):
        h, w = size.flatten().numpy()
        predicted = self.model(img, txt, size)[-1].squeeze()
        predicted = torch.sigmoid(predicted) >= self.threshold
        predicted = predicted[:h, :w].float().detach().cpu()
        return predicted

    def visualize(self, example_id, datasplit_id=0, custom_phrase=None):
        test_data = self.data_module.test_datasplits[datasplit_id]
        image, phrase, size, gold = self.get_data(
            example_id,
            datasplit_id,
            custom_phrase)
        img = test_data.transform(image).to(self.device).unsqueeze(0)
        txt = test_data.tokenize_phrase(phrase).to(self.device).unsqueeze(1)
        size = size.unsqueeze(0) if size is not None else size
        predicted = self.predict(img, txt, size)

        scale = 5.
        fig = plt.figure(figsize=(3 * scale, scale))
        grid = ImageGrid(
            fig, 111, nrows_ncols=(1,3), axes_pad=0.1)

        images = [
            ("image", image),
            ("ground truth mask", gold),
            ("predicted mask", predicted),
        ]

        for ax, (title, im) in zip(grid, images):
            ax.axis('off')
            ax.set_title(title)
            ax.imshow(im)
        plt.show()
        print(f"phrase: {phrase}")


class LanguageFiltersDemo(SegmentationDemo):
    def visualize(self, example_id, datasplit_id=0, custom_phrase=None,
                  method="remove", cmap=None, normalize=False):
        test_data = self.data_module.test_datasplits[datasplit_id]
        image, phrase, size, gold = self.get_data(
            example_id,
            datasplit_id,
            custom_phrase)
        phrases = []
        for idx in range(len(phrase.split())):
            this = phrase.split()
            if method == "replace":
                this[idx] = "UNK"
            elif method == "remove":
                this.pop(idx) 
            this = " ".join(this)
            if this == "":
                this = "UNK"
            phrases.append(this)

        original = self.generate_partitions(phrase)
        manipulated = []
        heatmap = np.zeros((len(phrases), 3*len(original)))
        for i, phrase_ in enumerate(phrases):
            this_manipulated = self.generate_partitions(phrase_)
            manipulated.append(this_manipulated) 
            for j, (a, b) in enumerate(zip(original, this_manipulated)):
                a_ = a.detach().cpu().numpy()
                b_ = b.detach().cpu().numpy()
                val = euclidean(a_, b_) / np.sqrt(a_.size)
                heatmap[i, j] = val

        multimodal_encoder = self.model.multimodal_encoder
        bottom_up_layers = multimodal_encoder.bottom_up.conditional_layers
        top_down_layers = multimodal_encoder.top_down.conditional_layers[::-1]
        bottom_up_original = []
        top_down_original = []

        for i, (layer, txt) in enumerate(zip(bottom_up_layers, original)):
            bottom_up_original.append(layer.dense(txt))

        for i, phrase_ in enumerate(phrases):
            this_manipulated = [bottom_up_layers[j].dense(manipulated[i][j])
                                for j in range(len(bottom_up_original))]
            for j, (a,b)  in enumerate(zip(bottom_up_original, this_manipulated)):
                a_ = a.detach().cpu().numpy()
                b_ = b.detach().cpu().numpy()
                val = euclidean(a_, b_) / np.sqrt(a_.size)
                heatmap[i, j+len(original)] = val
 
        for i, (layer, txt) in enumerate(zip(top_down_layers, original)):
            top_down_original.append(layer.dense(txt))

        for i, phrase_ in enumerate(phrases):
            this_manipulated = [top_down_layers[j].dense(manipulated[i][j])
                                for j in range(len(top_down_original))]
            for j, (a,b) in enumerate(zip(top_down_original, this_manipulated)):
                a_ = a.detach().cpu().numpy()
                b_ = b.detach().cpu().numpy()
                val = euclidean(a_, b_) / np.sqrt(a_.size)
                heatmap[i, j+2*len(original)] = val
         
        if normalize:
            minv, maxv = heatmap.min(), heatmap.max()
            heatmap = (heatmap - minv) / (maxv - minv)

        scale = 5.
        fig = plt.figure(figsize=(3 * scale, scale))
        grid = ImageGrid(
            fig, 111, nrows_ncols=(1,3), axes_pad=0.1)

        images = [
            ("partitions", heatmap[:, :len(original)]),
            ("bottom-up filters", heatmap[:, len(original):2*len(original)]),
            ("top-down filters", heatmap[:, 2*len(original):]),
        ]

        words = phrase.split()
        for ax, (title, im) in zip(grid, images):
            ax.set_yticks(np.arange(len(words)))
            ax.set_yticklabels(words)
            # ax.axis('off')
            ax.set_title(title)
            ax.imshow(im, interpolation="nearest", cmap=cmap)
        plt.show()

        # fig = plt.figure()
        # plt.imshow(heatmap, cmap=cmap, interpolation='nearest', aspect="equal")
        # words = phrase.split()
        # plt.yticks(np.arange(len(words)), words)
        # plt.show()
        print(f"phrase: {phrase}")

    def generate_partitions(self, phrase, datasplit_id=0):
        data = self.data_module.test_datasplits[datasplit_id]
        text_encoder = self.model.text_encoder
        multimodal_encoder = self.model.multimodal_encoder
        num_layers = multimodal_encoder.config["num_layers"]
        hidden_size = text_encoder.config["hidden_size"] // num_layers
        txt = data.tokenize_phrase(phrase).to(self.device).unsqueeze(1)
        txt = text_encoder(txt)[1][0].squeeze(0)
        txt = [
            txt[:, i*hidden_size:(i+1)*hidden_size]
            for i in range(num_layers)]
        return txt

    def to_numpy(self, tensors):
        return [t.detach().cpu().numpy() for t in tensors]