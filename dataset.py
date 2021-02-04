# -*- coding:utf-8 -*-


import torch
from torch.utils.data.dataset import Dataset


class PairTextDataset(Dataset):

    def __init__(self, data):
        self._sentence_left = data[0]
        self._sentence_right = data[1]
        self._labels = data[2]

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return (torch.tensor(self._sentence_left[index], dtype=torch.float32),
                torch.tensor(self._sentence_right[index], dtype=torch.float32),
                torch.tensor(self._labels[index], dtype=torch.int)
                )
