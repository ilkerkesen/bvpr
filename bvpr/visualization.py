import os.path as osp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import euclidean
from torchvision import transforms as ts
from skimage import color
import cv2

from bvpr.models import ColorizationBaseline, SegmentationModel, ColorizationModel
from bvpr.datamodule import SegmentationDataModule, ColorizationDataModule
from bvpr.data.transform import LAB2RGB
from bvpr.data.colorization import cvrgb2lab
from bvpr.util import sizes2scales, scales2sizes, annealed_mean
from bvpr.evaluation import compute_pixel_acc
from bvpr.data.transform import LAB2RGB


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
        self.data_module = SegmentationDataModule(data_config)
        self.data_module.setup(stage="test")

        # load model
        ckpt = torch.load(self.checkpoint_path)
        state_dict = ckpt["state_dict"].items()
        state_dict = {".".join(k.split(".")[1:]): v for (k, v) in state_dict}
        model_config = ckpt["hyper_parameters"]["model"]
        self.model = SegmentationModel(model_config).to(device)
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
        return (image, gold, predicted)


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


class MaximumActivatedPatchesDemo(SegmentationDemo):
    def visualize(self, example_id, datasplit_id=0, custom_phrase=None,
                  thresholded=False):
        test_data = self.data_module.test_datasplits[datasplit_id]
        image, phrase, size, gold = self.get_data(
            example_id,
            datasplit_id,
            custom_phrase)
        img = test_data.transform(image).to(self.device).unsqueeze(0)
        txt = test_data.tokenize_phrase(phrase).to(self.device).unsqueeze(1)
        size = size.unsqueeze(0)

        model = self.model
        scale = sizes2scales(size, img.size())
        vis = model.image_encoder(img, scales2sizes(scale, img.size()))
        txt_ = model.text_encoder(txt)
        txt_ = txt_[1][0].squeeze(0)
        num_layers = model.multimodal_encoder.config["num_layers"]
        text_dim = model.multimodal_encoder.config["text_embedding_dim"]
        hidden_size = text_dim // num_layers
        parted = [
            txt_[:, i*hidden_size:(i+1)*hidden_size]
            for i in range(num_layers)
        ]
        bottom_up_outputs = model.multimodal_encoder.bottom_up(vis, parted, scale)
        top_down_outputs = model.multimodal_encoder.top_down(bottom_up_outputs, parted)
        bottom_up_outputs = [self.get_attention_map(x, size, thresholded=thresholded)
                             for x in bottom_up_outputs]
        top_down_outputs = [self.get_attention_map(x, size, thresholded=thresholded)
                           for x in top_down_outputs][::-1]
        top_down_outputs.append(bottom_up_outputs[-1])

        scale = 15.
        fig = plt.figure(figsize=(3 * scale, scale))
        grid = ImageGrid(
            fig, 111, nrows_ncols=(1+len(bottom_up_outputs), 3), direction="column", axes_pad=(0.1, 0.3))

        images = [("image", image)]
        for i, output in enumerate(bottom_up_outputs):
            images.append((f"bottom-up #{i}", output))
        images.append(("predicted mask", self.predict(img, txt, size)))
        for i, output in enumerate(top_down_outputs):
            images.append((f"top-down #{i}", output))
        images.append(("true mask", gold))

        # FIXME: I should not have need something like this,
        #   but, matplotlib/ImageGrid behaves unresaonable.
        for i in range(len(bottom_up_outputs)):
            images.append(("dummy", torch.zeros(bottom_up_outputs[i].shape)))

        for ax in grid:
            ax.axis('off')
        for ax, (title, im) in zip(grid, images):
            if im is not None:
                ax.set_title(title)
                ax.imshow(im)
        plt.show()
        print(f"phrase: {phrase}")

    def get_attention_map(self, feature_map, size, thresholded=False):
        image_size = self.data_module.config["image_size"]
        predicted = feature_map.sum(dim=1).unsqueeze(0)
        predicted = F.interpolate(predicted, image_size)
        predicted = predicted.view(image_size, image_size)
        h, w = size.flatten().numpy()
        predicted = predicted[:h, :w]
        if thresholded:
            predicted = predicted >= self.threshold
        return predicted.float().detach().cpu()


class WordRemovalActivationDemo(SegmentationDemo):
    def visualize(self, example_id, datasplit_id=0, custom_phrase=None,
                  thresholded=False, method="remove"):
        test_data = self.data_module.test_datasplits[datasplit_id]
        image, phrase, size, gold = self.get_data(
            example_id,
            datasplit_id,
            custom_phrase)
        img = test_data.transform(image).to(self.device).unsqueeze(0)
        txt = test_data.tokenize_phrase(phrase).to(self.device).unsqueeze(1)
        size = size.unsqueeze(0)

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

        bu, td, pred = self.generate_feature_maps(img, phrase, size, datasplit_id)
        heatmaps = np.empty((2, len(bu), len(phrases)), dtype=object)
        predicted = [("predicted", pred)]
        for i, phrase_ in enumerate(phrases):
            bu_, td_, pred_ = self.generate_feature_maps(img, phrase_, size, datasplit_id)
            bu_diff = [self.process_differences(o, o_, size) for (o,o_) in zip(bu, bu_)]
            td_diff = [self.process_differences(o, o_, size) for (o,o_) in zip(td, td_)]
            heatmaps[0,:,i] = bu_diff
            heatmaps[1,:,i] = td_diff
            predicted.append(("predicted", pred_))

        scale = 15.
        fig = plt.figure(figsize=(3 * scale, scale))
        grid = ImageGrid(
            fig, 111, nrows_ncols=(1+len(bu), 2*len(phrases)), direction="column", axes_pad=(0.1, 0.3))

        images = []
        image_ = ("image", image)
        # predicted = ("predicted", self.predict(img, txt, size))
        for i in range(len(phrases)):
            images.append(image_)
            for j in range(len(bu)):
                images.append(("...", heatmaps[0,j,i]))
            images.append(predicted[i])
            for j in range(len(td)):
                images.append(("...", heatmaps[1,j,i]))

        for ax in grid:
            ax.axis('off')
        for ax, (title, im) in zip(grid, images):
            ax.axis('off')
            ax.set_title(title)
            ax.imshow(im)
        plt.show()
        print(f"phrase: {phrase}")

    def process_differences(self, t1, t2, size):
        diff = torch.mean((t1-t2)**2, dim=1)
        normalized = self.normalize(diff).unsqueeze(0)
        image_size = self.data_module.config["image_size"]
        t = F.interpolate(normalized, image_size)
        t = t.view(image_size, image_size)
        h, w = size.flatten().numpy()
        return t[:h, :w]

    def normalize(self, arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    def generate_feature_maps(self, img, phrase, size, datasplit_id):
        test_data = self.data_module.test_datasplits[datasplit_id]
        txt = test_data.tokenize_phrase(phrase).to(self.device).unsqueeze(1)
        model = self.model
        scale = sizes2scales(size, img.size())
        vis = model.image_encoder(img, scales2sizes(scale, img.size()))
        txt_ = model.text_encoder(txt)
        txt_ = txt_[1][0].squeeze(0)
        num_layers = model.multimodal_encoder.config["num_layers"]
        text_dim = model.multimodal_encoder.config["text_embedding_dim"]
        hidden_size = text_dim // num_layers
        parted = [
            txt_[:, i*hidden_size:(i+1)*hidden_size]
            for i in range(num_layers)
        ]
        bu = model.multimodal_encoder.bottom_up(vis, parted, scale)
        td = model.multimodal_encoder.top_down(bu, parted)[::-1]
        image_size = img.size()
        h, w = size.flatten().numpy()
        predicted = model.mask_predictor(td[0], image_size=image_size)[-1]
        predicted = predicted.squeeze()
        predicted = torch.sigmoid(predicted) >= self.threshold
        predicted = predicted[:h, :w].float().detach().cpu()
        bu = [x.detach().cpu() for x in bu]
        td = [x.detach().cpu() for x in td]
        td.append(bu[-1])
        return bu, td, predicted


class ColorizationDemo(object):
    def __init__(self, checkpoint_path, dataset, data_path, device="cuda:0", K=1, prior_set="imagenet"):
        data_config = {
            "image_size": 224,

            "dataset": {
                "dataset": dataset,
                "data_root": data_path,
                "K": K,
                "prior_set": prior_set,
            },

            # just in case
            "loader": {
                "num_workers": 0,
                "pin_memory": False,
                "batch_size": 1,
            }
        }

        self.checkpoint_path = osp.abspath(osp.expanduser(checkpoint_path))
        self.device = torch.device(device)

        # load data
        self.data_module = ColorizationDataModule(data_config)
        self.data_module.setup()

        # load model
        ckpt = torch.load(self.checkpoint_path)
        state_dict = ckpt["state_dict"].items()
        self.weights = ckpt["state_dict"].get("criterion.weights", None)
        state_dict = {".".join(k.split(".")[1:]): v for (k, v) in state_dict if k.startswith("model")}
        model_config = ckpt["hyper_parameters"]["model"]
        # model_config["text_encoder"]["vectors"] = self.data_module.train_data.embeddings
        if model_config["architecture"] == "ColorizationBaseline":
            Model = ColorizationBaseline
            model_config["network"]["glove"] = False
        else:
            Model = ColorizationModel
            model_config["text_encoder"]["glove"] = False
        self.model = Model(model_config).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.lab2rgb = LAB2RGB(
            ab_kernel=self.data_module.train_data.ab_kernel,
            device=device,
            mode="demo",
        )
        self.demo_data = self.data_module.demo_data

    def get_data(self, index, custom_caption=None):
        data = self.demo_data
        example = data[index]
        if custom_caption is not None:
            example["caption"] = data.tokenize_caption(custom_caption)    
            example["caption_len"] = len(example["caption"])
        example["raw_caption"] = data.captions[index]["caption"]
        return example

    def predict(self, features, caption, caption_l):
        device = self.device
        features = features.to(device).unsqueeze(0)
        caption = caption.to(device).unsqueeze(0)
        scores = self.model(features, caption, [caption_l])
        scores = scores.float().detach().cpu()
        return scores

    def visualize(self, example_id, custom_caption=None, averaged=True, T=1.0):
        this = self.get_data(example_id, custom_caption)
        scores = self.predict(
            this["input_image"], this["caption"], this["caption_len"])
        upsampled = F.interpolate(scores, scale_factor=4, mode="bilinear")
        colorized = self.lab2rgb(
            this["L"].unsqueeze(0).to(self.device),
            upsampled.to(self.device),
            T=T)

        scale = 10.
        fig = plt.figure(figsize=(3 * scale, scale))
        grid = ImageGrid(
            fig, 111, nrows_ncols=(2, 3), axes_pad=0.5)

        L = np.stack([this["L"].squeeze(0)] * 3, axis=-1) / 100.
        gold = this["rgb"].permute(1, 2, 0)
        targets = this["target"].unsqueeze(0)
        topk_pred = scores.topk(5, dim=1).indices == targets
        topk_pred = topk_pred.float()
        top1_pred = topk_pred[:, :1, :, :]
        top5_pred = topk_pred.sum(dim=1, keepdim=True)
        top1_pred = F.interpolate(top1_pred, scale_factor=4, mode='nearest')[0, 0]
        top5_pred = F.interpolate(top5_pred, scale_factor=4, mode='nearest')[0, 0]

        images = [
            ("input", L),
            ("ground truth", gold),
            ("colorized", colorized),
            ("top-1", top1_pred),
            ("top-5", top5_pred),
        ]

        for ax, (title, im) in zip(grid, images):
            ax.axis('off')
            ax.set_title(title)
            ax.imshow(im)
        plt.show()

        caption = this["raw_caption"]
        if custom_caption is not None:
            caption = custom_caption
        print(f"caption: {caption}")
        top1_acc = round(100 * top1_pred.mean().item(), 2)
        top5_acc = round(100 * top5_pred.mean().item(), 2)
        print(f"top1={top1_acc}%, top5={top5_acc}%")
