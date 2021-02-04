# -*- coding:utf-8 -*-


import torch
import torch.nn as nn


class SiameseLSTM(nn.Module):
    """
    this version use trained word vector
    """

    def __init__(self, embed_size=300):
        super(SiameseLSTM, self).__init__()

        self.embed_len = embed_size

        # lstm units
        self.dropout = 0.5
        self.lstm_params_init = False
        self.bidirectional = True
        self.rnn_layer_num = 1
        self.hidden_size = 32
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn_layer = nn.LSTM(self.embed_len,
                                 self.hidden_size,
                                 self.rnn_layer_num,
                                 batch_first=True,
                                 bidirectional=self.bidirectional)
        self.h0 = self.init_hidden((2 * self.rnn_layer_num, 1, self.hidden_size))
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, 2),
            nn.Softmax(dim=-1)
        )

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def distance_method(self, *input):
        return torch.exp(-torch.norm(input[0] - input[1], p=2, dim=1, keepdim=True))

    def rnn_encode(self, x):
        x, (h, c) = self.rnn_layer(x)
        return x

    def forward(self, *input):
        # batch_size, sequence_len, embedding_size
        s1 = input[0]
        s2 = input[1]

        # encoding
        encoding1 = self.rnn_encode(s1)
        encoding2 = self.rnn_encode(s2)

        # similarity
        sim = self.distance_method(encoding1, encoding2).squeeze()
        similarity = self.fc(sim)
        return similarity
