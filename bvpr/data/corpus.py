# -*- coding: utf-8 -*-

"""
Language-related data loading helper functions and class wrappers.
"""

import re
import torch
import codecs

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, a):
        if isinstance(a, int):
            return self.idx2word[a]
        elif isinstance(a, list):
            return [self.idx2word[x] for x in a]
        elif isinstance(a, str):
            return self.word2idx[a]
        else:
            raise TypeError("Query word/index argument must be int or str")

    def __contains__(self, word):
        return word in self.word2idx


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def set_max_len(self, value):
        self.max_len = value

    def load_file(self, filename):
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                self.add_to_corpus(line)
        self.dictionary.add_word(UNK_TOKEN)
        self.dictionary.add_word(PAD_TOKEN)

    def add_to_corpus(self, line):
        """Tokenizes a text line."""
        # Add words to the dictionary
        words = line.split()
        # tokens = len(words)
        for word in words:
            word = word.lower()
            self.dictionary.add_word(word)

    def tokenize(self, line, max_len=20):
        # Tokenize line contents
        words = SENTENCE_SPLIT_REGEX.split(line.strip())
        words = [w.lower() for w in words if len(w) > 0]
        words = list(filter(lambda w: w != " ", words))

        if words[-1] == '.':
            words = words[:-1]

        if max_len > 0:
            if len(words) > max_len:
                words = words[:max_len]
            elif len(words) < max_len:
                words = [PAD_TOKEN] * (max_len - len(words)) + words

        tokens = len(words)
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            if word not in self.dictionary:
                word = UNK_TOKEN
            ids[token] = self.dictionary[word]
            token += 1

        return ids
    
    def tokenize_with_words(self, line, max_len=20):
        # Tokenize line contents
        words = SENTENCE_SPLIT_REGEX.split(line.strip())
        words = [w.lower() for w in words if len(w) > 0]
        words = list(filter(lambda w: w != " ", words))

        if words[-1] == '.':
            words = words[:-1]

        if max_len > 0:
            if len(words) > max_len:
                words = words[:max_len]
            elif len(words) < max_len:
                words = [PAD_TOKEN] * (max_len - len(words)) + words

        tokens = len(words)
        ids = torch.LongTensor(tokens)
        token = 0
        words_ = []
        for word in words:
            if word not in self.dictionary:
                word = UNK_TOKEN
            words_.append(word)
            ids[token] = self.dictionary[word]
            token += 1

        return ids, words_

    def __len__(self):
        return len(self.dictionary)