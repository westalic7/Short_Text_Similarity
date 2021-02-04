# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchPyramid(nn.Module):
    """
    this version use trained word vector
    """
    def __init__(self, embed_size=300, hidden_size=32):
        super(MatchPyramid, self).__init__()

        self.embed_len = embed_size
        self.inner_chanel = 16
        self.fc_hidden_size = 32
        self.dropout = 0.5

        self.cnn_pyramid = nn.Sequential(
            # conv1
            # nn.BatchNorm2d(1),
            nn.Conv2d(1, self.inner_chanel, (3, 3), stride=1,padding=(1,1)),
            nn.BatchNorm2d(self.inner_chanel),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2, padding=(1, 1)),

            # conv2
            # nn.BatchNorm2d(self.inner_chanel),
            nn.Conv2d(self.inner_chanel, self.inner_chanel * 2, (3, 3), stride=1,padding=(1,1)),
            nn.BatchNorm2d(self.inner_chanel*2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2, padding=(1, 1)),

            # conv3
            nn.Conv2d(self.inner_chanel * 2, self.inner_chanel * 4, (3, 3), stride=1,padding=(1,1)),
            nn.BatchNorm2d(self.inner_chanel*4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2, padding=(1, 1)),

            nn.Conv2d(self.inner_chanel * 4, self.inner_chanel * 8, (5, 5)),
            nn.BatchNorm2d(self.inner_chanel*8),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.inner_chanel * 8, 2),
        )

    def forward(self, *input):
        # batch_size * seq_len
        sent1, sent2 = input[0], input[1]
        # mask1, mask2 = sent1.sum(dim=-1).eq(0), sent2.sum(dim=-1).eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = sent1
        x2 = sent2

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        x = torch.matmul(x1, x2.transpose(1, 2)).unsqueeze(1)

        # batch_size * chanel * seq_len * dim => batch_size * chanel
        x = self.cnn_pyramid(x)
        x = torch.flatten(x,1,-1)
        similarity = self.fc(x)

        return similarity


