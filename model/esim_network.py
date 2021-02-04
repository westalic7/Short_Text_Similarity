# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class Esim(nn.Module):
    """
    this version use trained word vector
    """
    def __init__(self, embed_size=300, hidden_size=32):
        super(Esim, self).__init__()

        self.embed_len = embed_size

        self.dropout = 0.5
        self.hidden_size = hidden_size
        self.embeds_dim = embed_size
        self.linear_size = 128
        self.rnn_layer_num = 1
        # num_word = 20000
        # self.embeds = nn.Embedding(num_word, self.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim,
                             self.hidden_size,
                             num_layers= self.rnn_layer_num,
                             batch_first=True,
                             bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size * 8,
                             self.hidden_size,
                             num_layers= self.rnn_layer_num,
                             batch_first=True,
                             bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, self.linear_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout),
            # nn.Linear(self.linear_size, self.linear_size),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(self.linear_size),
            # nn.Dropout(self.dropout),
            nn.Linear(self.linear_size, 2),
        )

    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, *input):
        # batch_size * seq_len
        sent1, sent2 = input[0], input[1]
        mask1, mask2 = sent1.sum(dim=-1).eq(0), sent2.sum(dim=-1).eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = sent1
        x2 = sent2
        # x1 = self.bn_embeds(sent1.transpose(1, 2)).transpose(1, 2)
        # x2 = self.bn_embeds(sent2.transpose(1, 2)).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity


