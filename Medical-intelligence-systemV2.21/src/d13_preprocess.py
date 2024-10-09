import re
import jieba
import pandas as pd
from gensim.models.word2vec import LineSentence, Word2Vec
import numpy as np
import os
import sys

# 设置项目的root路径 方便后续代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入文本预处理的配置信息config1
from d11_config1 import *  # 配置模块

# 导入多核CPU并行处理函数
from d12_multi_proc_utils import *

# jieba载入自定义切词表
jieba.load_userdict(user_dict_path)


# 根据 max_len 和 vocab 填充<START> <STOP> <PAD> <UNK>
def pad_proc(sentence, max_len, word_to_id):

    # 1 按空格统计切分出词
    words = sentence.strip().split(' ')

    # 2 截取规定长度的词数
    words = words[:max_len]

    # 3 填充<UNK>
    sentence = [w if w in word_to_id else '<UNK>' for w in words]

    # 4 填充<START> <END>
    sentence = ['<START>'] + sentence + ['<STOP>']

    # 5 判断长度，填充<PAD>
    sentence = sentence + ['<PAD>'] * (max_len - len(words))

    return ' '.join(sentence)


# 从文件中加载停用词
def load_stop_words(stop_word_path):
    # 加载停用词(程序调用)
    # stop_word_path: 停用词的文件存储路径
    # 打开文件
    f = open(stop_word_path, 'r', encoding='utf-8')

    # 读取所有行
    stop_words = f.readlines()

    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words

# 加载停用词
stop_words = load_stop_words(stop_words_path)


# 清洗文本的函数
def clean_sentence(sentence):
    # 特殊符号去除(被sentence_proc调用)
    # sentence: 待处理的字符串
    if isinstance(sentence, str):

        # 删除1. 2. 3. 这些标题 - 前后不是数字
        r = re.compile("\D(\d\.)\D")
        sentence = r.sub("", sentence)

        # 删除带括号的 进口 海外
        r = re.compile(r"[(（]进口[)）]|\(海外\)")
        sentence = r.sub("", sentence)

        # 删除除了汉字数字字母和，！？。.- 以外的字符
        r = re.compile("[^，！？。\.\-\u4e00-\u9fa5_a-zA-Z0-9]")

        # 用中文输入法下的，！？来替换英文输入法下的,!?
        sentence = sentence.replace(",", "，")
        sentence = sentence.replace("!", "！")
        sentence = sentence.replace("?", "？") 
        sentence = r.sub("", sentence)

        # 删除 车主说 技师说 语音 图片 你好 您好
        r = re.compile(r"车主说|技师说|语音|图片|你好|您好")
        sentence = r.sub("", sentence)
        return sentence
    else:
        return ''


# 句子分好词后，过滤停用词
def filter_stopwords(seg_list):
    # 过滤一句切好词的话中的停用词(被sentence_proc调用)
    # seg_list: 切好词的列表 [word1 ,word2 .......]
    # 首先去掉多余空字符
    words = [word for word in seg_list if word]
    # 去掉停用词
    return [word for word in words if word not in stop_words]


# 预处理的单条语句函数
def sentence_proc(sentence):
    # 预处理模块(处理一条句子，被sentences_proc调用)
    # sentence:待处理字符串
    # 清除无用词
    sentence = clean_sentence(sentence)
    # 切词，默认精确模式，全模式cut参数cut_all=True
    words = jieba.cut(sentence)
    # 过滤停用词
    words = filter_stopwords(words)
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


# 预处理的多条语句函数
def sentences_proc(df):
    # 预处理模块(处理一个句子列表，对每个句子调用sentence_proc操作)
    # df: 数据集
    # 批量预处理 训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)
        
    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(sentence_proc)

    return df


# 构建用于数据加载+预处理的函数（只需要执行一次）
def build_dataset(train_raw_data_path, test_raw_data_path):
    # 主要函数，用于数据加载+预处理(只需执行一次)
    # 1. 加载原始数据
    print('1. 加载原始数据')
    print(train_raw_data_path)
    train_df = pd.read_csv(train_raw_data_path, engine='python', encoding='utf-8')  # 必须utf-8
    test_df = pd.read_csv(test_raw_data_path, engine='python', encoding='utf-8')  # 必须utf-8
    # print('原始csv读出的表格数据类型为: ', type(train_df))  # <class 'pandas.core.frame.DataFrame'>
    # print('第一行Question列: ', train_df['Question'][0])
    print('原始训练集行数 {}, 测试集行数 {}'.format(len(train_df), len(test_df)))  # 82943, 20000
    print('\n')

    # 2. 空值去除（对于一行数据，任意列只要有空值就去掉该行）
    print('2. 空值去除（对于一行数据，任意列只要有空值就去掉该行）')
    train_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)
    print('空值去除后训练集行数 {}, 测试集行数 {}'.format(len(train_df), len(test_df)))  # 82871, 20000
    print('\n')

    # 3. 多线程, 批量数据预处理(对每个句子执行sentence_proc，清除无用词，切词，过滤停用词，再用空格拼接为一个字符串)
    print('3. 多进程, 批量数据预处理(对每个句子执行sentence_proc，清除无用词，切词，过滤停用词，再用空格拼接为一个字符串)')
    train_df = parallelize(train_df, sentences_proc)
    test_df = parallelize(test_df, sentences_proc)
    print('\n')
    print('sentences_proc has done!')

    # 4. 合并训练测试集，用于训练词向量
    print('4. 合并训练集、测试集，用于训练词向量')
    # 新建一列，按行堆积
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    train_df['Y'] = train_df[['Report']]
    # 新建一列，按行堆积
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 5. 保存分割处理好的train_seg_data.csv、test_set_data.csv
    print('5. 保存处理好的 train_seg_data.csv、test_set_data.csv')
    # 把建立的列merged去掉，该列对于神经网络无用，只用于训练词向量
    train_df = train_df.drop(['Question'], axis=1)
    train_df = train_df.drop(['Dialogue'], axis=1)
    train_df = train_df.drop(['Brand'], axis=1)
    train_df = train_df.drop(['Model'], axis=1)
    train_df = train_df.drop(['Report'], axis=1)
    train_df = train_df.drop(['QID'], axis=1)

    test_df = test_df.drop(['Question'], axis=1)
    test_df = test_df.drop(['Dialogue'], axis=1)
    test_df = test_df.drop(['Brand'], axis=1)
    test_df = test_df.drop(['Model'], axis=1)
    test_df = test_df.drop(['QID'], axis=1)

    # 将处理后的数据存入持久化文件
    test_df.to_csv(test_seg_path, index=None, header=False)

    train_df['data'] = train_df[['X', 'Y']].apply(lambda x: '<sep>'.join(x), axis=1)
    train_df = train_df.drop(['X'], axis=1)
    train_df = train_df.drop(['Y'], axis=1)
    train_df.to_csv(train_seg_path, index=None, header=False)
    print('The csv_file has saved!')
    print('\n')

    print('6. 后续工作是将第5步的结果文件进行适当处理， 并保存为.txt文件')
    print('本程序代码的所有所有工作执行完毕')


if __name__ == '__main__':
    build_dataset(train_raw_data_path, test_raw_data_path)
    print('数据预处理 进行文本清洗、分词、生成2个seg文件 End')


