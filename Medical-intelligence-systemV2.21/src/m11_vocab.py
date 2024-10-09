import os
import sys

# 设置项目的root路径 方便后续代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入项目代码文件包
from collections import Counter
import numpy as np
import torch
import torch.nn as nn

from d10_config import word_vector_model_path
from gensim.models import word2vec

# 构建词典类 相当于做了一个容器（word2idx idx2word） 等待添加单词
# 说白了：把字典读到内存中 构建出一个word2id id2word两张表
class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {} # 字典
        self.word2count = Counter()
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.reserved[:]  # 相当于copy作用 # 列表
        # 如果预训练词向量存在 则后续直接载入模型； 否则设置为None
        self.embedding_matrix = None

    # 向 词典类 中 增加单词
    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)

        # 因引入Counter()工具包 直接执行update()更新即可 【统计单词词频】
        self.word2count.update(words)

    # 若已提前预训练好了 词向量 则执行类内函数加载 词向量
    def load_embeddings(self, word_vector_model_path):
        # 直接下载预训练词向量模型
        wv_model = word2vec.Word2Vec.load(word_vector_model_path)
        # 从模型中直接提取 词嵌入矩阵
        self.embedding_matrix = wv_model.wv.vectors

    # 根据id值item, 读取字典中的单词
    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    # 获取字典的当前长度值（等效单词总数）
    def __len__(self):
        return len(self.index2word)

    # 获取字典的当前的单词数
    def size(self):
        return len(self.index2word)


if __name__ == '__main__':

    # 初始化 单词容器类 : 等待别人添加单词
    vocab = Vocab()
    print('vocab对象-->', vocab)

    vocab.add_words(['我', '爱', '长江', '黄河', '火花塞', '曲轴'])
    for i in vocab:
       print(i, end=' ')

    print('vocab.size()-->', vocab.size())
    print('vocab.embedding_matrix-->', vocab.embedding_matrix)

    print('Vocab类测试 End')

