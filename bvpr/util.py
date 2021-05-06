import os.path as osp
from copy import deepcopy

import numpy as np
import torch
import pytorch_lightning as pl


MOBILENET_SIZE_MAP = [
    (32, 1),
    (16, 1),
    (24, 2),
    (24, 2),
    (32, 3),
    (32, 3),
    (32, 3),
    (64, 4),
    (64, 4),
    (64, 4),
    (64, 4),
    (96, 4),
    (96, 4),
    (96, 4),
    (160, 5),
    (160, 5),
    (160, 5),
    (320, 5),
    (1280, 5)]


__all__ = (
    "process_config",
)


def process_config(cfg, dataset):
    cfg = deepcopy(cfg)
    encoder = cfg["image_encoder"]["name"]
    encoder_num_layers = cfg["image_encoder"]["num_layers"]
    predictor_num_layers = 3
    if encoder == "deeplabv3":
        if encoder_num_layers < 3:
            predictor_num_layers = 2
        elif encoder_num_layers < 5:
            predictor_num_layers = 3
    elif encoder == "resnet18":
        predictor_num_layers = encoder_num_layers + 1
    elif encoder == "mobilenetv2":
        predictor_num_layers = MOBILENET_SIZE_MAP[encoder_num_layers-1][1]
    cfg["text_encoder"]["corpus"] = dataset.corpus
    cfg["mask_predictor"]["num_layers"] = predictor_num_layers
    return cfg


def make_mask(real, downsized):
    B = len(real)
    dh, dw = downsized
    mask = torch.zeros(B, 1, dh, dw, dtype=torch.bool)
    for i in range(B):
        this_h, this_w = real[i]
        mask[i, 0, :this_h, :this_w] = 1
    return mask


def create_callbacks(config, log_dir):
    checkpoints_path = osp.join(log_dir, "checkpoints")
    config["checkpoint"]["filepath"] = osp.join(checkpoints_path, "{epoch:03d}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**config["checkpoint"])
    last_ckpt = osp.join(checkpoints_path, "last.ckpt")
    last_ckpt = last_ckpt if osp.isfile(last_ckpt) else None
    ckpt_path = config["trainer"]["resume_from_checkpoint"]

    if last_ckpt is not None and ckpt_path is not None:
        raise Exception("resume checkpoint passed (last.ckpt exists already)")

    ckpt_path = last_ckpt if ckpt_path is None else ckpt_path
    if ckpt_path is not None and not osp.isfile(ckpt_path):
        raise Exception("ckpt does not exist at {}".format(ckpt_path))

    return [checkpoint_callback], ckpt_path


def generate_spatial_batch(featmap_H, featmap_W):
    """Generate additional visual coordinates feature maps.
    Function taken from
    https://github.com/chenxi116/TF-phrasecut-public/blob/master/util/processing_tools.py#L5
    and slightly modified
    """

    spatial_batch_val = np.zeros(
        (1, 8, featmap_H, featmap_W), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w + 1) / featmap_W * 2 - 1
            xctr = (xmin + xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h + 1) / featmap_H * 2 - 1
            yctr = (ymin + ymax) / 2
            spatial_batch_val[0, :, h, w] = (
                [xmin, ymin, xmax, ymax,
                xctr, yctr, 1 / featmap_W, 1 / featmap_H])
    return torch.from_numpy(spatial_batch_val)


def make_batch_location_embeddings(size, mapsize):
    B = size.shape[0]
    locs = []
    for i in range(B):
        # locs.append(make_instance_location_embeddings(size[i,:], mapsize))
        locs.append(generate_spatial_batch(mapsize[0], mapsize[1]))
    return torch.cat(locs, 0).float()


def add_batch_location_embeddings(vis, size):
    mapsize = vis.size()[-2:]
    loc = make_batch_location_embeddings(size, mapsize)
    loc = loc.to(vis.device)
    return torch.cat((vis, loc), 1)

  
def scale2size(scale, tensor_size):
    tensor_h, tensor_w = tensor_size
    scale_h, scale_w = scale
    scaled_h, scaled_w = scale_h * tensor_h, scale_w * tensor_w
    return torch.round(torch.tensor([[scaled_h, scaled_w]])).int()


def scales2sizes(scales, tensor_size):
    processed_size = tensor_size[-2:]
    batch_size = tensor_size[0]
    sizes = [scale2size(scales[i], processed_size) for i in range(batch_size)]
    sizes = torch.cat(sizes, 0)
    return sizes


def sizes2scales(sizes, tensor_size):
    processed_size = tensor_size[-2:]
    batch_size = tensor_size[0]
    scales = [size2scale(
        sizes[i], processed_size) for i in range(batch_size)]
    scales = torch.cat(scales, 0)
    return scales


def size2scale(raw_size, processed_size):
    im_h, im_w = raw_size[0].item(), raw_size[1].item()
    new_h, new_w = processed_size
    if new_h >= im_h and new_w >= im_w:
        ratio_h = im_h / float(new_h)
        ratio_w = im_w / float(new_w)
    elif im_h >= im_w:
        ratio_h = 1.0
        ratio_w = ((new_h / float(im_h)) * im_w) / new_w
    else:
        ratio_w = 1.0
        ratio_h = ((new_w / float(im_w)) * im_h) / new_h
    return torch.tensor([[ratio_h, ratio_w]])


def prior_boosting(prior_file, alpha, gamma):
    prior_probs = np.load(prior_file)

    # define uniform probability
    uni_probs = np.zeros_like(prior_probs)
    uni_probs[prior_probs!=0] = 1.
    uni_probs = uni_probs/np.sum(uni_probs)

    # convex combination of empirical prior and uniform distribution       
    prior_mix = (1-gamma)*prior_probs + gamma*uni_probs

    # set prior factor
    prior_factor = prior_mix**-alpha
    prior_factor[prior_factor==np.inf] = 0. # mask out unused classes
    prior_factor = prior_factor/np.sum(prior_probs*prior_factor) # re-normalize

    # implied empirical prior
    # implied_prior = prior_probs*prior_factor
    # implied_prior = implied_prior/np.sum(implied_prior) # re-normalize
    return prior_factor


def annealed_mean(z, T=1.0, dim=1):
    num = torch.exp(torch.log(z) / T)
    den = torch.sum(num, dim=1)
    return num / den
