# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import cv2
from skimage import io
from skimage.transform import resize
import json
import tqdm
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
from referit import REFER
import torch.utils.data as data
from referit.refer import mask as cocomask
from transformers import AutoTokenizer, BertTokenizer

from bvpr.data.corpus import Corpus
from bvpr.extra.char_bert.utils import CharacterIndexer


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
        'gref-umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'},
        },
        'clevr': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'clevr-ref+', 'split_by':'ccvl'}
        }
    }

    def __init__(self, data_root, split_root=None, dataset='referit',
                 transform=None, mask_transform=None, split='train',
                 max_query_len=-1, text_encoder="LSTMEncoder",
                 corpus_file=None, features_path=None):
        self.images = []
        self.data_root = osp.expanduser(data_root)
        self.split_root = split_root
        if split_root is None:
            self.split_root = osp.join(self.data_root, '..', 'processed')
            self.split_root = osp.abspath(osp.expanduser(self.split_root))
        self.dataset = dataset
        self.query_len = max_query_len
        self.corpus = Corpus()
        self.transform = transform
        self.mask_transform = mask_transform
        self.split = split
        self.text_encoder = text_encoder
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

        if self.text_encoder == "BERTEncoder":
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif self.text_encoder == "RobertaEncoder":
            self.bert_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        elif self.text_encoder == "CharBERTEncoder":
            cache_dir = osp.expanduser("~/.cache/char_bert")
            checkpoint_path = osp.join(cache_dir, "bert-base-uncased/")
            self.bert_tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
            self.char_indexer = CharacterIndexer()

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
                h, w = io.imread(osp.join(self.im_dir, img_filename)).shape[:2]
                seg = refer.anns[ref['ann_id']]['segmentation']
                rle = cocomask.frPyObjects(seg, h, w)
                mask = np.max(cocomask.decode(rle), axis=2).astype(np.float32)
                mask = torch.from_numpy(mask)
                mask_file = str(ref['ann_id']) + '.pth'
                mask_filename = osp.join(self.mask_dir, mask_file)
                if not osp.exists(mask_filename):
                    torch.save(mask, mask_filename)
                for sentence in ref['sentences']:
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
        img_file, mask_file, phrase = self.images[idx]
        data, size = self.read_image(img_file)
        mask_path = osp.join(self.mask_dir, mask_file)
        mask = torch.load(mask_path)
        return data, mask, size, phrase

    def tokenize_phrase(self, phrase):
        if self.text_encoder == "LSTMEncoder":
            return self.corpus.tokenize(phrase, self.query_len), None
        elif self.text_encoder in ("BERTEncoder", "RobertaEncoder"):
            input_dict = self.bert_tokenizer(phrase)
            return input_dict["input_ids"], input_dict["attention_mask"]
        elif self.text_encoder == "CharBERTEncoder":
            tokens = self.bert_tokenizer.basic_tokenizer.tokenize(phrase)
            char_ids = self.char_indexer.tokens_to_indices(tokens)
            return char_ids, None

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data, mask, size, phrase = self.pull_item(idx)
        if self.transform is not None and self.features_path is None:
            data = self.transform(data)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        text, text_l = self.tokenize_phrase(phrase)
        return {
            "input": data,
            "target": mask,
            "size": size,
            "text": text,
            "text_l": text_l,
            "index": idx,
        }