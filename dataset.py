import torch 
import numpy as np 
import codecs
import pandas
import jieba
from collections import OrderedDict

class THUNewsDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_file, csv, max_seq_length):
        super(THUNewsDataset, self).__init__()
        self.vocab = {}
        self.max_seq_length = max_seq_length
        self.X = []
        self.Y = []
        self.load_vocab(vocab_file)
        self.load_data(csv)

    def load_vocab(self, vocab_file):
        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            # for i, l in enumerate(f.read().splitlines()):
            #     self.vocab[l] = i
            lines = f.read().splitlines()
            print('lines number: ', len(lines))
            s = OrderedDict().fromkeys(lines)
            print('set size: ', len(s))
            for i, l in enumerate(s.keys()):
                self.vocab[l] = i
            print('len of vocab:', len(self.vocab))
    
    def load_data(self, csv):
        df = pandas.read_csv(csv)
        for i in range(len(df)):
            text = df.at[i,'text']
            text = text.split('/')
            text = text[:self.max_seq_length]
            if len(text) < self.max_seq_length:
                text = text + [self.vocab['PAD']] * \
                        (self.max_seq_length - len(text))
            # idx_list = [self.vocab[x] for x in text]
            idx_list = []
            for word in text:
                if word in self.vocab:
                    idx_list.append(self.vocab[word])
                else:
                    idx_list.append(self.vocab['UNK'])
            
            self.X.append(idx_list)
            self.Y.append(int(df.at[i, 'class']))

    def sentence2idx(self, sentence):
        seg_list = list(jieba.cut(sentence))
        idx_list = []
        for word in seg_list:
            if word in self.vocab:
                idx_list.append(self.vocab[word])
            else:
                idx_list.append(self.vocab['UNK'])
        return idx_list
    
    def idx2words(self, idx):
        words = []
        for i in idx:
            for word, index in self.vocab.items():
                if index == i:
                    words.append(word)
        return words
    
    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.Y[index])

    def __len__(self):
        return len(self.Y)

