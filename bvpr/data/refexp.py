# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import sys
from urllib.request import urlretrieve, urlopen
from urllib.error import HTTPError, URLError
import tempfile
from skimage import io
from skimage.transform import resize
import json
import csv
import uuid
import tqdm
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
from referit import REFER
import torch.utils.data as data
from referit.refer import mask as cocomask
from utils.corpus import Corpus
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pathlib import Path

import cv2
import requests
from PIL import Image
from io import BytesIO


class DatasetNotFoundError(Exception):
    pass


class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'clevr': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'clevr-ref+', 'split_by':'ccvl'}
        }
    }

    def __init__(self, data_root, split_root=None, dataset='referit',
                 transform=None, mask_transform=None,
                 split='train', max_query_len=-1, bertencoding=False,
                 bert=False, corpus_file=None, features_path=None):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        if split_root is None:
            self.split_root = osp.join(self.data_root, '..', 'processed')
        self.dataset = dataset
        self.query_len = max_query_len
        self.corpus = Corpus()
        self.transform = transform
        self.mask_transform = mask_transform
        self.split = split
        self.bertencoding = bertencoding
        self.bert = bert
        if features_path is not None and osp.exists(features_path):
            self.features_path = osp.abspath(features_path)
        else:
            self.features_path = None

        self.dataset_root = self.data_root
        self.im_dir = osp.join(self.data_root, 'images')
        self.mask_dir = osp.join(self.split_root, 'mask')
        self.split_dir = osp.join(self.dataset_root, 'splits')

        if self.dataset != 'referit' and self.dataset != "clevr":
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'train2014')
            self.mask_dir = osp.join(self.split_root, self.dataset, 'mask')
        elif self.dataset == "clevr":
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'clevr', self.split)
            self.mask_dir = osp.join(
                self.split_root, self.dataset, 'mask', self.split)
        elif self.dataset == "referit":
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'referit', 'images')
            self.mask_dir = osp.join(
                self.dataset_root, 'images', 'referit', 'mask')

        if not self.exists_dataset():
            self.process_dataset()

        dataset_path = osp.join(self.split_root, self.dataset)
        corpus_path = osp.join(dataset_path, 'corpus.pth')
        if corpus_file is not None:
            corpus_path = osp.abspath(corpus_file)

        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        self.corpus = torch.load(corpus_path)

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def process_dataset(self):
        if self.dataset not in self.SUPPORTED_DATASETS:
            raise DatasetNotFoundError(
                'Dataset {0} is not supported by this loader'.format(
                    self.dataset))

        dataset_folder = osp.join(self.split_root, self.dataset)
        if not osp.exists(dataset_folder):
            os.makedirs(dataset_folder)

        if self.dataset == 'referit':
            data_func = self.process_referit
        elif self.dataset == "clevr":
            data_func = self.process_clevr
        else:
            data_func = self.process_coco

        splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        for split in splits:
            print('Processing {0}: {1} set'.format(self.dataset, split))
            data_func(split, dataset_folder)

    def process_referit(self, setname, dataset_folder):
        split_dataset = []

        query_file = osp.join(
            self.split_dir, 'referit',
            'referit_query_{0}.json'.format(setname))
        vocab_file = osp.join(self.split_dir, 'vocabulary_referit.txt')

        query_dict = json.load(open(query_file))
        im_list = query_dict.keys()

        if len(self.corpus) == 0:
            print('Saving dataset corpus dictionary...')
            corpus_file = osp.join(self.split_root, self.dataset, 'corpus.pth')
            self.corpus.load_file(vocab_file)
            torch.save(self.corpus, corpus_file)

        for name in tqdm.tqdm(im_list):
            im_filename = name.split('_', 1)[0] + '.jpg'
            if im_filename in ['19579.jpg', '17975.jpg', '19575.jpg']:
                continue
            if osp.exists(osp.join(self.im_dir, im_filename)):
                mask_mat_filename = osp.join(self.mask_dir, name + '.mat')
                mask_pth_filename = osp.join(self.mask_dir, name + '.pth')
                if osp.exists(mask_mat_filename):
                    mask = sio.loadmat(mask_mat_filename)['segimg_t'] == 0
                    mask = mask.astype(np.float64)
                    mask = torch.from_numpy(mask)
                    torch.save(mask, mask_pth_filename)
                    os.remove(mask_mat_filename)
                for query in query_dict[name]:
                    split_dataset.append((im_filename, name + '.pth', query))

        output_file = '{0}_{1}.pth'.format(self.dataset, setname)
        torch.save(split_dataset, osp.join(dataset_folder, output_file))

    def process_coco(self, setname, dataset_folder):
        split_dataset = []
        vocab_file = osp.join(self.split_dir, 'vocabulary_Gref.txt')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.bertencoding:
            tokenizer = BertTokenizer.from_pretrained(osp.join(Path().absolute(), 'utils/bert_repo/bert-large-uncased-vocab.txt'))
            model = BertModel.from_pretrained(osp.join(Path().absolute(), 'utils/bert_repo/bert-large-uncased.tar.gz'))
            model.to(device)
            model.eval()
        if self.bert:
            tokenizer = BertTokenizer.from_pretrained(osp.join(Path().absolute(), 'utils/bert_repo/bert-base-uncased-vocab.txt'))

        refer = REFER(
            self.data_root, **(
                self.SUPPORTED_DATASETS[self.dataset]['params']))

        refs = [refer.refs[ref_id] for ref_id in refer.refs
                if refer.refs[ref_id]['split'] == setname]

        refs = sorted(refs, key=lambda x: x['file_name'])

        if len(self.corpus) == 0:
            print('Saving dataset corpus dictionary...')
            corpus_file = osp.join(self.split_root, self.dataset, 'corpus.pth')
            self.corpus.load_file(vocab_file)
            torch.save(self.corpus, corpus_file)

        if not osp.exists(self.mask_dir):
            os.makedirs(self.mask_dir)
        maxlen = 0
        for ref in tqdm.tqdm(refs):
            img_filename = 'COCO_train2014_{0}.jpg'.format(
                str(ref['image_id']).zfill(12))

            if osp.exists(osp.join(self.im_dir, img_filename)):
                h, w, _ = io.imread(osp.join(self.im_dir, img_filename)).shape
                seg = refer.anns[ref['ann_id']]['segmentation']
                rle = cocomask.frPyObjects(seg, h, w)
                mask = np.max(cocomask.decode(rle), axis=2).astype(np.float32)
                mask = torch.from_numpy(mask)
                mask_file = str(ref['ann_id']) + '.pth'
                mask_filename = osp.join(self.mask_dir, mask_file)
                if not osp.exists(mask_filename):
                    torch.save(mask, mask_filename)
                for sentence in ref['sentences']:
                    if self.bertencoding:
                        tokenized_text = tokenizer.tokenize(sentence['sent'])
                        tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
                        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                        segments_ids = [0] * len(indexed_tokens)
                        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
                        segments_tensors = torch.tensor([segments_ids]).to(device)
                        with torch.no_grad():
                            encoded_layers, pooled = model(tokens_tensor, segments_tensors, output_all_encoded_layers=False)
                            ref_embedding = pooled.unsqueeze().cpu()
                        split_dataset.append((
                            img_filename, mask_file, ref_embedding))
                    elif self.bert:
                        tokenized_text = tokenizer.tokenize(sentence['sent'])
                        if len(tokenized_text) > maxlen:
                            maxlen = len(tokenized_text)
                        if len(tokenized_text) > self.query_len:
                            tokenized_text = tokenized_text[:self.query_len]

                        tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
                        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                        segment_ids = [0] * len(indexed_tokens)
                        input_mask = [1] * len(indexed_tokens)
                        padding = [0] * (self.query_len + 2 - len(input_mask))
                        indexed_tokens += padding
                        input_mask += padding
                        segment_ids += padding
                        split_dataset.append((
                            img_filename, mask_file, indexed_tokens, segment_ids, input_mask))
                    else:
                        split_dataset.append((
                            img_filename, mask_file, sentence['sent']))
        print("Maxlen: ", maxlen)

        output_file = '{0}_{1}.pth'.format(self.dataset, setname)
        torch.save(split_dataset, osp.join(dataset_folder, output_file))

    def process_clevr(self, setname, dataset_folder):
        split_dataset = []
        scenes_file = osp.join(
            self.data_root, self.dataset, 'scenes',
            'clevr_ref+_{}_scenes.json'.format(setname))
        with open(scenes_file, 'r') as f:
            scenes = json.load(f)["scenes"]

        refexps_file = osp.join(
            self.data_root, self.dataset, 'refexps',
            'clevr_ref+_{}_refexps.json'.format(setname))
        with open(refexps_file, 'r') as f:
            refexps = json.load(f)["refexps"]

        im_dir = osp.join(self.dataset_root, 'images', 'clevr', setname)
        mask_dir = osp.join(self.split_root, self.dataset, 'mask', setname)
        if not osp.exists(mask_dir):
            os.makedirs(mask_dir)

        inds = list(set([x["image_index"] for x in refexps]))
        is_sorted = all(inds[i] <= inds[i + 1] for i in range(len(inds)-1))
        if not is_sorted:
            raise Exception("Unexpected error (unsorted dict entries)")

        construct_corpus = True if len(self.corpus) == 0 else False
        if construct_corpus:
            self.corpus.add_to_corpus("<pad> <go> <eos> <unk>")

        maxlen = 0
        for entry in tqdm.tqdm(refexps):
            objects = entry['program'][-1]['_output']
            phrase = entry['refexp'].lower()
            if construct_corpus:
                self.corpus.add_to_corpus(phrase)
            img, img_file = entry["image"], entry['image_filename']
            scene_id = entry["image_index"]
            scene = scenes[scene_id]
            mask = self.get_mask_from_refexp(scene, objects)
            mask = torch.tensor(mask, dtype=torch.float32)
            mask_file = "-".join([img] + [str(x) for x in objects]) + ".pth"
            mask_path = osp.join(mask_dir, mask_file)
            if not osp.exists(mask_path):
                torch.save(mask, mask_path)
            split_dataset.append([img_file, mask_file, phrase])

            num_tokens = len(phrase.split())
            if num_tokens > maxlen:
                maxlen = num_tokens

        if setname == "train" and construct_corpus:
            print('Saving dataset corpus dictionary...')
            corpus_file = osp.join(self.split_root, self.dataset, 'corpus.pth')
            torch.save(self.corpus, corpus_file)

        output_file = '{0}_{1}.pth'.format(self.dataset, setname)
        torch.save(split_dataset, osp.join(dataset_folder, output_file))

    def get_mask_from_refexp(self, sce, obj_list, height=-1, width=-1):
        heatmap = np.zeros((320,480))

        def from_imgdensestr_to_imgarray(imgstr):
            img = []
            cur = 0
            for num in imgstr.split(','):
                num = int(num)
                img += [cur]*num;
                cur = 1-cur
            img = np.asarray(img).reshape((320,480))
            return img

        for objid in obj_list:
            obj_mask = sce['obj_mask'][str(objid+1)]
            mask_img = from_imgdensestr_to_imgarray(obj_mask)
            heatmap += mask_img
        if height !=-1 and width !=-1:
            heatmap = resize(heatmap, (width, height))
        return heatmap

    def read_image(self, img_file):
        img_path = osp.join(self.im_dir, img_file)
        if not osp.exists(img_path):
            return (None, None)
        img = cv2.imread(img_path)
        if img is None:
            return (None, None)
        im_h, im_w = img.shape[:2]
        size = torch.tensor([im_h, im_w])
        if len(img.shape) == 3:
            # bgr2rgb conversion
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img[:,:,[0,2]] = img[:,:,[2,0]]
        else:
            img = np.stack([img] * 3)
        return (img, size)

    def read_features(self, img_file):
        filename = osp.splitext(img_file)[0]+'.pth'
        filepath = osp.join(self.features_path, filename)
        if not osp.exists(filepath):
            return (None, None)
        data_dict = torch.load(filepath, map_location="cpu")
        data, size = data_dict["data"], data_dict["size"]
        return (data, size)

    def pull_item(self, idx):
        if self.bert:
            img_file, mask_file, phrase, segment, lang_mask = self.images[idx]
        else:
            img_file, mask_file, phrase = self.images[idx]

        if self.features_path is None:
            data, size = self.read_image(img_file)
        else:
            data, size = self.read_features(img_file)

        mask_path = osp.join(self.mask_dir, mask_file)
        mask = torch.load(mask_path)

        if self.bert:
            return data, mask, phrase, segment, lang_mask
        else:
            return data, mask, size, phrase

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.bert:
            data, mask, phrase, segment, lang_mask = self.pull_item(idx)
        else:
            data, mask, size, phrase = self.pull_item(idx)
        if self.transform is not None and self.features_path is None:
            data = self.transform(data)
        if self.mask_transform is not None:
            # mask = mask.byte() * 255
            mask = self.mask_transform(mask)
        if not (self.bertencoding or self.bert):
            phrase = self.tokenize_phrase(phrase)
        if self.bert:
            #print("Sizes: ")
            #for t_ in [img, mask, size_mask, torch.tensor(phrase), torch.tensor(segment), torch.tensor(lang_mask)]:
            #    print("Size: ", t_.size())
            return img, mask, size_mask, torch.tensor(phrase), torch.tensor(segment), torch.tensor(lang_mask)
        else:
            return data, mask, size, phrase


def collate_fn(unsorted_batch):
    batch = sorted(unsorted_batch, key=lambda i: len(i[-1]), reverse=True)
    pack = lambda i: torch.cat([bi[i].unsqueeze(0) for bi in batch], 0)
    img, mask, size = tuple(pack(i) for i in range(len(batch[0])-1))
    batchsize = len(batch)
    longest = len(batch[0][-1])
    text = torch.zeros((batchsize, longest), dtype=torch.long)
    for (i,bi) in enumerate(batch):
        sent = bi[-1]
        text[i, -len(sent):] = sent
    return img.float(), mask.float(), size, text