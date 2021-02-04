# -*- coding:utf-8 -*-

import os
import jieba
import pickle
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from embedding import Word2Vec1Gram
from dataset import PairTextDataset

class PairTextDataProcessor(DataLoader):

    def __init__(self, embed):
        self.embed = embed

    def prepare_dataset(self, df, sample_rate=None):
        print(df.head())
        if sample_rate:
            df1 = df[df['label'] == 1].sample(frac=sample_rate['1'], )
            print(f'label 1: extract {len(df1)} samples')
            df0 = df[df['label'] == 0].sample(frac=sample_rate['0'], )
            print(f'label 0: extract {len(df0)} samples')
            df = pd.concat([df1, df0], axis=0)
        return df

    def process(self, dataset):
        """
        先对文本进行分词，再用word2vec转化为词向量
        :param dataset: idx, text1，text2，label
        :return: list1, list2, labels
        """
        sentence_list_left = []
        sentence_list_right = []
        labels = []
        print(len(dataset), 'samples found.')
        for idx, line in tqdm(dataset.iterrows()):
            label = line['label']
            s1 = self.embed.get_sentence_words(line['s1'])
            s2 = self.embed.get_sentence_words(line['s2'])
            _, s1_vectors = self.embed.get_sentence_embedding(s1)
            _, s2_vectors = self.embed.get_sentence_embedding(s2)

            sentence_list_left.append(s1_vectors)
            sentence_list_right.append(s2_vectors)
            labels.append(label)

        return [sentence_list_left, sentence_list_right, labels]


if __name__=='__main__':


    # loading w2v
    word2vec_path = "./data/financial_ali_dataset_1gram/atec_nlp_text_corpus_1gram_w2v_sg1_size32.txt"
    w2v = Word2Vec1Gram(word2vec_path,max_seq_len=40,is_padding=True)
    data_loader = PairTextDataProcessor(w2v)

    # loading data
    sample_rate = {'1':1, '0':1}
    datadir = './data/'
    train_set_path = os.path.join(datadir, 'test_sample.tsv')
    test_set_path = os.path.join(datadir, 'train_sample.tsv')

    todo_list = [train_set_path, test_set_path]
    DEBUG = True

    for datafile in todo_list:
        if DEBUG:
            df = data_loader.prepare_dataset(pd.read_csv(datafile, sep='\t'), sample_rate=sample_rate)
            dataset = data_loader.process(df)
            with open(datafile+'.pkl', "wb") as dstf:
                pickle.dump(dataset, dstf)
        else:
            with open(datafile+'.pkl', "rb") as f:
               dataset = pickle.load(f)

        pairdata = PairTextDataset(dataset)
        print(pairdata[0][0].shape)
        print(pairdata[0][1].shape)
        print(pairdata[0][2])



