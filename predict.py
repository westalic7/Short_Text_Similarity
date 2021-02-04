# -*- coding:utf-8 -*-


import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, auc, roc_curve
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from data_processor import PairTextDataProcessor
from dataset import PairTextDataset
from embedding import Word2Vec1Gram
from model import SiameseLSTM,Esim
from utils import get_parameter_number


class EvalConfig(object):
    def __init__(self):
        # w2v
        self.word2vec_path = "./data/financial_ali_dataset_1gram/atec_nlp_text_corpus_1gram_w2v_sg1_size32.txt"
        self.w2v = Word2Vec1Gram(self.word2vec_path, max_seq_len=40, is_padding=True)
        self.data_loader = PairTextDataProcessor(self.w2v)
        self.embedding_size = self.w2v.embedding_size

        # data
        self.test_set_path = './data/20201112_test_2000.tsv.pkl'
        self.test_dataset = self._reader(self.test_set_path)

        # eval
        self.batch_size = 32
        self.model_func = SiameseLSTM
        self.save_model_path = os.path.join('./data/metrics_siamese/', 'base_1gram_SiameseLSTM_ce.pt')

        # device
        self.core_gpu = 0
        self.gpu_count = 1
        self.use_multi_gpu = 1

        self._info()

    def _reader(self, data_path):
        if data_path.split('.')[-1] == 'tsv':
            df = self.data_loader.prepare_dataset(pd.read_csv(data_path, sep='\t'), sample_rate={'1': 1, '0': 1})
            dataset = self.data_loader.process(df)
        elif data_path.split('.')[-1] == 'pkl':
            dataset = pickle.load(open(data_path, "rb"))
        return PairTextDataset(dataset)

    def _info(self):
        print(f'*** Data: {self.test_set_path}')
        print(f'*** Embeding: {self.w2v}')
        print(f'*** Model: {self.model_func}')
        print(f'*** Eval batch size: {self.batch_size}')


class Evaluation(EvalConfig):

    def __init__(self):
        super(Evaluation, self).__init__()

    def init_gpu(self):
        self.device = torch.device("cuda:{}".format(self.core_gpu) if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True if self.device == 'cuda' else False
        self.cuda = True if torch.cuda.is_available() else False
        print(f'*** Training on device: {self.device}')

    def init_data(self):
        self.valid_data_loader = DataLoader(dataset=self.test_dataset,
                                            batch_size=self.batch_size)
        print(f'*** Eval dataset get samples: {len(self.test_dataset)}')

    def init_model(self):
        self.model = self.model_func(self.embedding_size)
        self.model.load_state_dict(torch.load(self.save_model_path))
        print(f'*** Model: {self.model}')

    def eval(self):
        self.init_gpu()
        self.init_data()
        self.init_model()

        self.model.eval()
        predict_result = []
        for idx, (s1, s2, target) in tqdm(enumerate(self.valid_data_loader), desc='*** Eval epochs'):
            if self.cuda:
                s1 = s1.float().cuda()
                s2 = s2.float().cuda()
            y_pred = self.model(s1, s2)
            # res
            pred = y_pred.argmax(dim=-1)
            predict_result.extend(list(zip(pred.tolist(), target.tolist())))
        return predict_result

    def get_metrics(self, df):
        print(classification_report(df['target'], df['predict'], digits=3))


if __name__ == '__main__':
    el = Evaluation()
    predict_result = el.eval()
    df = pd.DataFrame(predict_result)
    # df = pd.read_csv('./data/badcase.tsv',sep='\t')
    df.columns = ['predict', 'target']
    df.to_csv('./predict.tsv',sep='\t')
    el.get_metrics(df)