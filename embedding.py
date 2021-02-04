# -*- coding:utf-8 -*-

import numpy as np
from gensim.models import KeyedVectors

class Word2VecEmbed(object):

    def __init__(self, word_vector_file_path, max_seq_len=None, is_padding=True):
        self.w2v_fpath = word_vector_file_path
        self.w2v_dict = KeyedVectors.load_word2vec_format(self.w2v_fpath, binary=False)
        self.embedding_size = self.w2v_dict.__dict__['vectors'][0].size
        self.max_seq_len = max_seq_len
        self.is_padding = is_padding

    def get_word_embedding(self, word):
        return self.w2v_dict.get_vector(word)

    def get_sentence_embedding(self, words):
        sentence = []
        vectors = []

        # check max_seq_len and padding
        if self.max_seq_len:
            if len(words) >= self.max_seq_len:
                words = words[:self.max_seq_len]
            else:
                words = words + (self.max_seq_len - len(words)) * ["PAD"]

        # word2vec
        for word in words:
            try:
                vector = self.w2v_dict.get_vector(word)
                vectors.append(vector)
            except:
                # 添加 "pad" 和 "UNK",
                if word == "PAD":
                    vectors.append(np.zeros(self.embedding_size).tolist())
                else:
                    # print(word, "不存在于词向量中")
                    word = 'UNK'
                    vectors.append(np.random.randn(self.embedding_size).tolist())
            finally:
                sentence.append(word)

        assert len(sentence) == len(vectors)

        return sentence, np.array(vectors)

    def get_sentence_words(self):
        pass

    def preprocess(self, sentence):
        pass


class Word2Vec1Gram(Word2VecEmbed):
    name = ''

    def __init__(self, word_vector_file_path, max_seq_len=None, is_padding=True):
        super(Word2Vec1Gram, self).__init__(word_vector_file_path=word_vector_file_path,
                                            max_seq_len=max_seq_len,
                                            is_padding=is_padding)

    def get_sentence_words(self, text):
        return [x for x in text]
