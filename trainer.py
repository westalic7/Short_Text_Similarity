# -*- coding:utf-8 -*-

import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
# import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
# from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from data_processor import PairTextDataProcessor
from dataset import PairTextDataset
from embedding import Word2Vec1Gram
from log import Logger
from model import *
from utils import get_parameter_number

CWD = os.path.dirname(__file__)


class Config(object):
    def __init__(self):
        # w2v
        self.word2vec_path = "./data/financial_ali_dataset_1gram/atec_nlp_text_corpus_1gram_w2v_sg1_size32.txt"
        self.w2v = Word2Vec1Gram(self.word2vec_path, max_seq_len=40, is_padding=True)
        self.data_loader = PairTextDataProcessor(self.w2v)
        self.embedding_size = self.w2v.embedding_size

        # data
        self.train_set_path = './data/train_sample.tsv.pkl'
        self.test_set_path = './data/test_sample.tsv.pkl'
        self.train_dataset = self._reader(self.train_set_path)
        self.test_dataset = self._reader(self.test_set_path)

        # train
        self.random_seed = 1234
        self.epochs = 20
        self.batch_size = 256
        self.lr = 1E-3
        self.model_func = BiMPM
        self.loss_func = CrossEntropyLoss
        self.save_model_path = f'./data/base_1gram_{self.model_func.__name__}_ce.pt'
        self.badcase_save_path = f'./data/badcase_{self.model_func.__name__}.tsv'

        # device
        self.core_gpu = 0
        self.gpu_count = 1
        self.use_multi_gpu = 1

        # logger
        self.log_file_path = f'./data/trainer_{self.model_func.__name__}_' \
                             f'{time.strftime("%Y%m%d%H%M%S", time.localtime())}.log'
        self.log = Logger(self.log_file_path)

        self._info()

    def _reader(self, data_path):
        if data_path.split('.')[-1] == 'tsv':
            df = self.data_loader.prepare_dataset(pd.read_csv(data_path, sep='\t'), sample_rate={'1': 1, '0': 1})
            dataset = self.data_loader.process(df)
        elif data_path.split('.')[-1] == 'pkl':
            dataset = pickle.load(open(data_path, "rb"))
        return PairTextDataset(dataset)

    def _info(self):
        self.log.logger.info(f'*** Data: {self.train_set_path}, {self.test_set_path}')
        self.log.logger.info(f'*** Embeding: {self.w2v}')
        self.log.logger.info(f'*** Model: {self.model_func}')

        self.log.logger.info(f'*** Train epochs: {self.epochs}')
        self.log.logger.info(f'*** Train batch size: {self.batch_size}')
        self.log.logger.info(f'*** Test batch size: {self.batch_size}')


class Trainer(object):
    def __init__(self, args):
        self.DEBUG = False
        self.DEBUG_NUM = 100
        self.args = args
        self.log = self.args.log
        self.loss_fn = self.args.loss_func(reduction='sum')
        self.badcase_save_path = self.args.badcase_save_path
        self.epochs = self.args.epochs
        self.model_func = self.args.model_func
        self.metrics = 'loss'
        self.lr = self.args.lr
        self.metrics_config = {'loss': float('+inf'), 'f1': float('-inf')}

    def init_seed(self):
        random.seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed_all(self.args.random_seed)

    def init_gpu(self):
        self.device = torch.device("cuda:{}".format(self.args.core_gpu) if torch.cuda.is_available() else "cpu")
        self.args.gpu_count = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True if self.device == 'cuda' else False
        self.cuda = True if torch.cuda.is_available() else False
        self.log.logger.info(f'*** Found gpu num: {self.args.gpu_count}')
        self.log.logger.info(f'*** Training on device: {self.device}')

    def init_data(self):
        self.train_data_loader = DataLoader(dataset=self.args.train_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=True)
        self.valid_data_loader = DataLoader(dataset=self.args.test_dataset,
                                            batch_size=self.args.batch_size)
        self.log.logger.info(f'*** Train dataset get samples: {len(self.args.train_dataset)}')
        self.log.logger.info(f'*** Test  dataset get samples: {len(self.args.test_dataset)}')

    def init_model(self):
        self.model = self.model_func(embed_size=self.args.embedding_size)
        self.model.to(self.device)
        self.log.logger.info(f'*** Model: {self.model}')

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001, alpha=0.9)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.log.logger.info(f'*** Optimizer: {self.optimizer}')

    def freeze_model_parameters(self):
        # freeze
        for param in self.model.parameters():
            param.requires_grad = True
        # for param in self.model.embedding_layer.parameters():
        #     param.requires_grad = False
        params_info = get_parameter_number(self.model)
        self.log.logger.info(f'*** Parameters: {params_info}')

    def train(self):
        # 构建初始种子
        self.init_seed()
        # 初始化gpu
        self.init_gpu()
        # 初始化训练验证数据集
        self.init_data()
        # 初始化训练模型
        self.init_model()
        # 初始化优化器
        self.init_optimizer()
        # 冻结网络层
        self.freeze_model_parameters()

        # lr更新策略
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)

        best_metrics_score = self.metrics_config[self.metrics]
        for epoch in range(self.epochs):
            self.model.train()

            total_loss, total_num = 0., 0.
            for idx, (s1, s2, target) in tqdm(enumerate(self.train_data_loader), desc='*** Train epochs'):

                if self.cuda:
                    s1 = s1.to(self.device)
                    s2 = s2.to(self.device)
                    target = target.to(self.device)

                y_pred = self.model(s1, s2)
                # loss
                loss = self.loss_fn(y_pred, target.long())

                # 获取NER最终的loss
                total_loss += float(loss)
                total_num += target.shape[0]

                loss.backward()

                # 反向传播和参数更新
                self.optimizer.step()
                self.optimizer.zero_grad()

                if idx > self.DEBUG_NUM and self.DEBUG:
                    break

            # lr update
            scheduler.step()
            self.lr = scheduler.get_lr()[0]

            train_loss_avg = total_loss / total_num

            # 验证集metrics
            valid_loss_avg = self.validation()

            # update model and save checkpoint
            current_metrics_score = valid_loss_avg if self.metrics == 'loss' else -valid_loss_avg
            update_flag = current_metrics_score < best_metrics_score
            if update_flag:
                best_metrics_score = current_metrics_score
                torch.save(self.model.state_dict(), self.args.save_model_path)

            self.log.logger.info("| epoch: {:2d} | lr: {:.6f} | - train -| loss: {:4.4f} "
                                 "| - valid -| loss: {:4.4f}| {}".format(epoch + 1,
                                                                         self.lr if self.lr else 1.,
                                                                         train_loss_avg,
                                                                         valid_loss_avg,
                                                                         '*' if update_flag else ''))

    def validation(self, ):
        """
        计算验证集metrics
        """
        self.model.eval()
        predict_result = []
        total_loss, total_num = 0., 0.
        for idx, (s1, s2, target) in tqdm(enumerate(self.valid_data_loader), desc='*** Validation epochs'):
            if self.cuda:
                s1 = s1.to(self.device)
                s2 = s2.to(self.device)
                target = target.to(self.device)

            y_pred = self.model(s1, s2)
            # loss
            loss = self.loss_fn(y_pred, target.long())

            # total loss
            total_loss += float(loss)
            total_num += target.shape[0]

            if self.badcase_save_path:
                pred = y_pred.argmax(dim=-1)
                predict_result.extend(list(zip(pred.tolist(), target.tolist())))
            if idx > self.DEBUG_NUM and self.DEBUG:
                break

        if self.badcase_save_path:
            res = pd.DataFrame(predict_result)
            res.columns = ['predict', 'target']
            res.to_csv(self.badcase_save_path, sep='\t', index=False)
            self.log.logger.info(classification_report(res['target'], res['predict'], digits=3))

        avgloss = total_loss / total_num
        return avgloss


if __name__ == '__main__':
    args = Config()
    tr = Trainer(args)
    tr.train()
