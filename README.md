# Short_Text_Similarity

基于Pytorch实现的SiameseNet、ESIM、BIMPM、MatchPyramid文本相似度模型，可快速开始模型训练。

## 项目简介

1.采用已训练好的词向量，转换文本为向量输入模型；（本项目不包含词向量训练过程）

2.所有模型使用相同输入输出，可配置参数自由切换

## 测试环境

- python 3.7
- pytorch 1.6
- sklearn

## 测试结果

**Financial_ali**

测试数据集包含2类标签，同义句非同义句

训练集:35000/验证集:2000

| Model        | Acc   | Parameters |
| ------------ | ----- | ---------- |
| SiameseLSTM  | 0.676 | 17218      |
| Abcnn        | 0.670 | 66344      |
| Esim         | 0.733 | 125122     |
| BiMPM        | 0.739 | 43202      |
| MatchPyramid | 0.653 | 57522      |

## Todo

1. 其他类型数据集
2. 其他Embedding方案
3. 字向量和词向量融合