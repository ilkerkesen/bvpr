import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.init import kaiming_normal, kaiming_uniform

from bvpr.submodules import get_glove_vectors


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_params(m.weight)


class CaptionEncoder(nn.Module):
    def __init__(self, word_embedding_dim, hidden_dim, vocab_size, glove,
                 corpus=None):
        super(CaptionEncoder, self).__init__()
        if glove:
            vectors = get_glove_vectors(corpus)
            self.embedding = nn.Embedding.from_pretrained(vectors)
        else:
            self.embedding = nn.Embedding(vocab_size, word_embedding_dim)
        # self.embedding.weight.data.copy_(
        #     torch.from_numpy(train_vocab_embeddings))
        self.hidden_size = hidden_dim
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, captions, lens):
        bsz, max_len = captions.size()
        embeds = self.dropout(self.embedding(captions))

        lens, indices = torch.sort(lens, 0, True)
        _, (enc_hids, _) = self.lstm(
            pack(embeds[indices], lens.tolist(), batch_first=True))
        enc_hids = torch.cat( (enc_hids[0], enc_hids[1]), 1)
        _, _indices = torch.sort(indices, 0)
        enc_hids = enc_hids[_indices]

        return enc_hids


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    How this layer works : 
    x = Variable(torch.randn(2, 64, 32 ,32))       
    gammas = Variable(torch.randn(2, 64)) # gammas and betas have to be 64 
    betas = Variable(torch.randn(2, 64))           
    y = film(x, gammas, betas)
    print y.size()
    y is : [2, 64, 32, 32]

    """
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class FilMedResBlock(nn.Module):
    expansion = 1
    '''
    A much simplified version
    '''
    def __init__(self, in_dim, out_dim, stride=1, padding=1, dilation=1):
        super(FilMedResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=stride,
            padding=1, dilation=dilation)  # bias=False? check what perez did
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.film = FiLM()
        init_modules(self.modules())

    def forward(self, x, gammas, betas):
        out = x
        out = F.relu(self.conv1(out))
        out = self.bn2(F.relu(self.conv2(out)))
        out = F.relu(self.film(out, gammas, betas))
        out += x
        return out


class AutocolorizeResnet(nn.Module):
    def __init__(self, vocab_size, feature_dim=(512, 28, 28), d_hid=256,
                 d_emb=300, num_modules=4, num_classes=625, glove=True,
                 corpus=None):
        super().__init__()
        self.num_modules = num_modules
        self.n_lstm_hidden = d_hid
        self.block = FilMedResBlock
        self.in_dim = feature_dim[0]
        self.num_classes = num_classes
        dilations = [1, 1, 1, 1]
        self.caption_encoder = CaptionEncoder(
            d_emb, d_hid, vocab_size, glove, corpus)

        # self.function_modules = {}
        # for fn_num in range(self.num_modules):
        # self.add_module(str(fn_num), mod)
        # self.function_modules[fn_num] = mod

        self.mod1 = self.block(self.in_dim, self.in_dim, dilations[0])
        self.mod2 = self.block(self.in_dim, self.in_dim, dilations[1])
        self.mod3 = self.block(self.in_dim, self.in_dim, dilations[2])
        self.mod4 = self.block(self.in_dim, self.in_dim, dilations[3])

        self.dense_film_1 = nn.Linear(self.n_lstm_hidden*2, self.in_dim*2) 
        self.dense_film_2 = nn.Linear(self.n_lstm_hidden*2, self.in_dim*2)
        self.dense_film_3 = nn.Linear(self.n_lstm_hidden*2, self.in_dim*2)
        # out = x # 2x512x28x28
        # out = F.relu(self.conv1(out)) # 2x512x28x28
        self.dense_film_4 = nn.Linear(self.n_lstm_hidden*2, self.in_dim*2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.classifier = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, dilation=1)    

    def forward(self, x, captions, caption_lens):
        caption_features = self.caption_encoder(captions, caption_lens)
        # out = F.relu(self.bn1(self.conv1(x)))

        dense_film_1 = self.dense_film_1(caption_features)
        dense_film_2 = self.dense_film_2(caption_features)
        dense_film_3 = self.dense_film_3(caption_features)
        dense_film_4 = self.dense_film_4(caption_features) # bsz * 128

        gammas1, betas1 = torch.split(dense_film_1, self.in_dim, dim=-1)
        gammas2, betas2 = torch.split(dense_film_2, self.in_dim, dim=-1)
        gammas3, betas3 = torch.split(dense_film_3, self.in_dim, dim=-1)
        gammas4, betas4 = torch.split(dense_film_4, self.in_dim, dim=-1)

        out = self.mod1(x, gammas1, betas1)  # out is 2x512x28x28
        out = self.mod2(out, gammas2, betas2)  # out is 2x512x28x28
        out = self.mod3(out, gammas3, betas3)
        out_last = self.mod4(out, gammas4, betas4)

        out = self.upsample(out_last)
        out = self.classifier(out)
        return out_last, out
