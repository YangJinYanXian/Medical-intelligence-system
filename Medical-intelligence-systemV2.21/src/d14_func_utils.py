import os
import sys
from collections import Counter
# 导入系统工具包

# 设置项目root路径，方便后续文件导入
import pandas as pd

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import numpy as np
import time
import heapq
import random
import pathlib
import d10_config
import torch

# 装饰器函数，统计函数耗时
def timer(module):
    def wrapper(func):
        # func: 一个函数名，下面的计时代码就是计算这个func函数的耗时
        def cal_time(*args, **kwargs):
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for ', module)
            return res
        return cal_time

    return wrapper

@timer(module='test a demo program')
def demo01_time_test():
    print('test Begin ...')
    s = 0
    for i in range(100000000):
        s += i
    print('s = ', s)
    print('test End ...')


# 文本按空格切分, 返回结果列表
def simple_tokenizer(text):
    return text.split()

def demo02_test_simple_tokenizer():
    sentence = '技师说：你好，以前也出现过该故障吗？|技师说：缸压多少有没有测量一下?|车主' \
               '说：没有过|车主说：没测缸压|技师说：测量一下缸压 看一四缸缸压是否偏低|车主' \
               '说：用电脑测，只是14缸缺火|车主说：[语音]|车主说：[语音]|技师说：点火线圈  火' \
               '花塞 喷油嘴不用干活  直接和二三缸对倒一下  跑一段在测量一下故障码进行排除|车主' \
               '说：[语音]|>车主>说：' \
               '主说：[语音]|车主说：[语音]|车主说：师傅还在吗|技师说：调一下喷油嘴  测一下缸' \
               '压  都正常则为>发动机电脑板问题|车主说：[语音]|车主说：[语音]|车主说：[语音]|技' \
               '师说：这个影响不大的|技师说：缸压八个以上正常|车主说：[语音]|技师说：所以说让你测量' \
               '缸压  只要缸压正常则没有问题|车主说：[语音]|车主说：[语音]|技师说：可以点击头像关注我  ' \
               '有什么问题随时询问  一定真诚用心为你解决|车主说：师傅，谢谢了|技师说：不用客气'

    sentence = 'abc def hijk'

    res = simple_tokenizer(sentence)
    print('res -->', res)
    print('res length -->', len(res))

    # res type --> <class 'list'>
    # print('res type -->', type(res))

'''
# 1
Counter类: 目的是用来跟踪值出现的次数。
它是一个无序的容器类型，以字典的键值对形式存储，
    其中元素作为key，其计数作为value。
    计数值可以是任意的Interger（包括0和负数）。
Counter类和其他语言的bags或multisets很相似。

# 2
以字典计数的方式，统计一段文本中,单词的出现次数
数据格式 [ ['奔驰', 'ML500', '排气', '凸轮轴', ... ], ['111', '2222', '3333' ...], []...]
数据格式特点： 列表里面套列表
'''
def count_words(counter, text):
    # counter：计数器类对象
    # text：待统计的文本
    for sentence in text:
        for word in sentence:
            counter[word] += 1

def demo03_test_count_words():

    # 实例化 一个 计数器类 对象
    # counter = Counter()
    # text = ['以前也出现过该故障吗？一下一下一下缸压多少有没有测量一下?', '14缸缺火点火线圈火花塞喷油嘴不用干活直接和二三缸对倒一下跑一段在测量一下故障码']
    # count_words(counter, text)
    # for w, c in counter.most_common(100):
    #     print(w, ': ', c)

    text2 = [['奔驰', 'ML500', '排气', '凸轮轴', '111', '111'], ['111', '2222', '3333']]
    counter2 = Counter() # 实例化一个计数器类对象
    count_words(counter2, text2)
    for w, c in counter2.most_common(100):
        print(w, ': ', c)

    '''
        111 :  2
        奔驰 :  1
        ML500 :  1
        排气 :  1
        凸轮轴 :  1
        2222 :  1
        3333 :  1 
    '''

# 对一个批次batch_size个样本, 按照x_len字段长度 排序, 并返回排序后的结果
# 1 如果4个样本 先把4个样本的x y x_len y_len OOV len_OOV 放在一起
# 2 然后在按照x_len进行排序
# 说白了，就是把x集中在一起进行排序 把y集中在一起进行排序 把oov集中在一起 进行排序
def sort_batch_by_len(data_batch):

    # 初始化一个结果字典, 其中包含的6个字段都是未来数据迭代器中的6个字段
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}

    # 遍历批次数据, 分别将6个字段数据按照字典key值, 添加进各自的列表中
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    # print('res_tmp--->', res)

    '''
    # 把4个x 4个y 4个OOV 4个len_OOV放在一起
    res_tmp - --> {
        'x': [[1, 110, 46, 690], [1, 110, 46, 690, 81], [1, 110], [1, 110, 9, 3, 543, 2211, 33, 556, 633, 91]],
        'x_len': [4, 5, 2, 10],
        'y': [[1, 181, 21, 115, 19, 758, 4, 9, 51], [1, 181, 21, 115, 19, 758, 4, 9, 51],
              [1, 181, 21, 115, 19, 758, 4, 9, 51], [1, 181, 21, 115, 19, 758, 4, 9, 51]], 
        'y_len': [9, 9, 9, 9],
        'OOV': [['B107C14', '入位', '台板'], ['B107C14', '入位', '台板', '入位2'], ['B107C14', '入位', '台板', '入位2'],
                ['B107C14', '入位', '台板', '入位2']], 
        'len_OOV': [3, 4, 4, 4] }
  
    '''


    '''
    print('\nres[x_len]--->', res['x_len'])
    a1 = np.array(res['x_len'])
    a2 = a1.argsort() # 返回排序后的下标，默认从小到大排列
    print( 'a2--->', a2, type(a2) )
    a3 = a2[::-1]
    print('a3--->', a3, type(a3))
    a4 = a3.tolist()
    print('a4--->', a4, type(a4))
    
    # res[x_len] - --> [4, 5, 2, 10]
    # a2 - --> [2 0 1 3] <class 'numpy.ndarray'>
    # a3 - --> [3 1 0 2] <class 'numpy.ndarray'>
    # a4 - --> [3, 1, 0, 2] <class 'list'>
    '''

    # 以 x_len 字段大小进行排序, 并返回下标结果 的 列表 （也即是返回排序后的下标，4个下标形成一个列表）
    sorted_indices = np.array( res['x_len'] ).argsort()[::-1].tolist()

    # 返回的data_batch 依然保持字典类型
    data_batch = { name: [ _tensor[i] for i in sorted_indices ]   for name, _tensor in res.items() }

    # data_dict2= {}
    # for name, __tensor in res.items():
    #
    #     tmplist = [] # 排序后的字段形成列表
    #     for i in sorted_indices:
    #         tmplist.append(__tensor[i])
    #
    #     data_dict2[name] =  tmplist # 添加到字典中

    return data_batch


def demo04_test_sort_batch_by_len():

    # mydf2 = {'x': ['11', '111', '1111'], 'y': [2, 22, 22], 'OOV': [3, 33, 333], 'len_OOV': [1, 1, 2]}
    # df = pd.DataFrame(data=d)

    data_batch = [{'x':[1,110, 46, 690],
                   'x_len':4,
                   'y': [1, 181, 21, 115, 19, 758, 4, 9, 51],
                   'y_len': 25,
                   'OOV': ['B107C14', '入位', '台板'],
                   'len_OOV': 3},

                  {'x': [1, 110, 46, 690, 81],
                   'x_len': 5,
                   'y': [1, 181, 21, 115, 19, 758, 4, 9, 51],
                   'y_len': 25,
                   'OOV': ['B107C14', '入位', '台板', '入位2'],
                   'len_OOV': 4},

                  {'x': [1, 110],
                   'x_len': 2,
                   'y': [1, 181, 21, 115, 19, 758, 4, 9, 51],
                   'y_len': 25,
                   'OOV': ['B107C14', '入位', '台板', '入位2'],
                   'len_OOV': 4},

                  {'x': [1, 110,9,3,543,2211,33,556,633,91],
                   'x_len': 10,
                   'y': [1, 181, 21, 115, 19, 758, 4, 9, 51],
                   'y_len': 25,
                   'OOV': ['B107C14', '入位', '台板', '入位2'],
                   'len_OOV': 4}]

    print('mydata_batch 排序之前\n', data_batch)

    mydata_batch = sort_batch_by_len(data_batch)

    print('mydata_batch 排序之后\n', mydata_batch)

    ''' # 排序以后的结果
    {'x': [[1, 110, 9, 3, 543, 2211, 33, 556, 633, 91], [1, 110, 46, 690, 81], [1, 110, 46, 690], [1, 110]], 
     'x_len': [10, 5, 4, 2], 
     'y': [[1, 181, 21, 115, 19, 758, 4, 9, 51], [1, 181, 21, 115, 19, 758, 4, 9, 51], [1, 181, 21, 115, 19, 758, 4, 9, 51], [1, 181, 21, 115, 19, 758, 4, 9, 51]], 
     'y_len': [9, 9, 9, 9], 
     'OOV': [['B107C14', '入位', '台板', '入位2'], ['B107C14', '入位', '台板', '入位2'], ['B107C14', '入位', '台板'], ['B107C14', '入位', '台板', '入位2']], 
     'len_OOV': [4, 4, 3, 4]}
    '''


# 原始文本source 映射成 ids 张量
# 输入一个中文文本 和 word2indx字典
# 1 若中文token 在   word2index中，把token的index直接添加到列表ids中
# 2 若中文token 不在 word2index中，说明是oov单词，
#                   把oov单词添加到列表oovs中，再生成一个oov单词的index，添加到列表ids中
# 3 返回列表ids 和 列表oovs
def source2ids(source_words, vocab):
    # source_words：原始中文文本
    # vocab：本质上是word_to_id字典
    ids = []
    oovs = []
    unk_id = vocab.UNK
    # 遍历文本进行映射操作
    for w in source_words:
        i = vocab[w]
        # 如果w是oov单词
        if i == unk_id:
            # 如果w是oov，不在oov列表中，则添加到oovs中
            if w not in oovs:
                oovs.append(w)
            # 索引0对应第一个source document OOV  索引1对应第二个source document OOV
            oov_num = oovs.index(w)
            # 在本项目中 索引vocab_size对应对一个source document OOV ， vocab_size+1对应第二个source document OOV
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)

    # 返回映射后的列表 以及oov单词列表
    return ids, oovs

def dm06_test_source2ids():
    mylist = ['雨刷器', '不', '喷', '玻璃水', '一个多月', '，', '以为', '冻', '，', '后来', '暖', '库放', '一天', '，', '不', '喷', '。', '现在', '抬', '喷水',
     '开关', '没声', '雨刷器', '干', '刮', '三下', '，', '怎么回事', '？', '，', '需要', '检查一下', '保险丝', '，', '没', '问题', '喷水', '电机', '坏',
     '。', '毛病', '不', '及时处理', '会', '部件', '影响', '？', '没有', '影响', '，', '影响', '使用', '换', '喷水', '电机', '大概', '费用', '？', '五十',
     '左右', 'Ok', '，', '十分', '感谢', '！', '！', '！', '太', '客气', '。', '谢谢', '支持']



# 摘要文本映射成数字化ids张量
'''
对样本的摘要，进行数字化映射。
1 若在vocab字典中 则 word2id 后加入 ids列表 中
2 若不在 vocab 字典中 则判断是否在原文中
    若在原文中，编个号 添加到 ids列表 中
    不在原文中，无法处理 忽略成 unk_id 
'''
def abstract2ids(abstract_words, vocab, source_oovs):
    # abstract_words： 摘要中文文本
    # vocab ：word2index： 映射字典
    # source_oovs：原始文本中的oov单词

    ids = []
    unk_id = vocab.UNK
    # 遍历 摘要中文文本

    for w in abstract_words:

        i = vocab[w]

        # 若w是OOV单词
        if i == unk_id:
            # 若w 是source document OOV单词
            if w in source_oovs:
                # 对于这样的w 计算出映射id，加入到ids中
                vocab_idx = vocab.size() + source_oovs.index(w)
                ids.append(vocab_idx)
            # 若w 不是source document OOV单词
            else:
                # 对于这样的w只能替换成UNK
                # 这样的单词 不在vocab-word2index字典中，也不在source_oovs，是没有方法处理
                ids.append(unk_id)

        # 若w本身在字典中
        else:
            ids.append(i)

    # 返回映射列表
    return ids


'''
应用场景：模型训练完毕，预测出来的id需要转换成文本 
将输出张量结果 映射成 文本  ， inde2word
1 若id在vocab中 则直接id2word
2 若id不在vocab中，判断是否在source_oovs列表中 
        若在source_oovs列表中，则根据source_oov_idx 2 word
        若不在source_oovs列表中，则异常
        
注意：
    1 每一篇文档，都有不同长度的oov单词；这个是PGN模型数据处理的难点
        对每一篇文档进行摘要的时候，都要先进行oov单词生成的
    这就涉及到整个摘要的流程与seq2seq摘要的流程不同之处
    
    2 seq2seq模型 与 PGN模型 生成摘要的异同
        1 输入一段sorce文本，对sorce进行encode编码，求中间语义张量C；
        2 然后一个时间步一个时间步的生成摘要文本（eg：最大摘要长度40个，超过40没有结束就强制结束）
            2-1 attention求预测单词的注意力权重分布，注意力结果表示
            2-2 然后再decode解码求预测单词
            
            在2-2环节中，PGN模型生成摘要文本的范围可以在字典中，也可以在oov列表中
            
        3 所以PGN在解码的时，是需要原文source中的OOV单词列表；
            也就是每一段文本进行摘要先生成各自的oov单词列表
'''
def outputids2words(id_list, source_oovs, vocab):
    # id_list：输出的数字化结果
    # source_oovs：原始文本的oov
    # vocab：index2word的字典

    words = []
    # i对应的单词可能在index2word中，可能在oov中，也可能是其他
    for i in id_list:
        try:
            # 因 w 可能是unk
            w = vocab.index2word[i]  # might be [UNK]

        # 若w是OOV单词
        except IndexError:
            assert_msg = "Error: 无法在词典中找到该id值."
            assert source_oovs is not None, assert_msg
            # 若 i 对应的单词是一个 source document OOV单词
            source_oov_idx = i - vocab.size()
            try:
                # 若能成功取到，则w是在原文中的单词（source document OOV单词）
                w = source_oovs[source_oov_idx]
                # 若i对应的单词，不在index2word中，也不在source_oovs中 ，只能抛出异常
            except ValueError:  # i doesn't correspond to an source oov
                raise ValueError('Error: 模型生成的ID: %i, 原始文本中的OOV ID: %i 但是当前样本中只有 %i 个OOVs'
                                 % (i, source_oov_idx, len(source_oovs)) )

        # 向结果列表中添加原始字符
        words.append(w)

    # 空格连接成字符串返回
    return ' '.join(words)


from io import StringIO
import math
# 辅助函数：字符界面显示堆
def show_tree(tree, total_width=36, fill=' '):
    output = StringIO()
    last_row = -1
    for i, n in enumerate(tree):
        if i:
            row = int(math.floor(math.log(i + 1, 2)))
        else:
            row = 0
        if row != last_row:
            output.write('\n')
        columns = 2 ** row
        col_width = int(math.floor((total_width * 1.0) / columns))
        output.write(str(n).center(col_width, fill))
        last_row = row
    print(output.getvalue())
    print('-' * total_width)
    print()
    return


# 创建小顶堆，包含k个点的特殊二叉树，始终保持二叉树中最小值在root根节点
# 向小顶堆数据结构添加元素的函数add2heap(heap, item, k)
# 容器
#   1 [1, 2, 3] 控制数据如何进，如何出来
#   2 k=3 容量
#   3 应用场景 1000个数 ====》 把最大的3个数给沉淀下来
def add2heap(heap, item, k):
    # Maintain a heap with k nodes and the smallest one as root
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)


# 构建一个k = 3的小顶推
def demo05_heap_k3():
    myheap = []
    heapq.heapify(myheap)
    k = 3
    for i in range(15):
        item = random.randint(10, 100)
        print('coming item', item, end=' ')
        add2heap(myheap, item, k)
        print(myheap)


# 将文本张量中所有OOV单词的id, 全部替换成<UNK>对应的id
# eg: [11,22,44,20008,20009,5] --> [11,22,44,3,3,5]
def replace_oovs(in_tensor, vocab):

    # 在Pytorch1.5.0以及更早的版本中, torch.full()默认返回float类型
    # 在Pytorch1.7.0最新版本中, torch.full()会将bool返回成torch.bool, 会将integer返回成torch.long.
    # 上面一行代码在Pytorch1.6.0版本中会报错, 因为必须指定成long类型, 如下面代码所示
    # print('in_tensor.shape--->', in_tensor.shape)

    # 1 创建一个张量 形状通过in_tensor， 内容为vocab.UNK
    oov_token = torch.full(in_tensor.shape, vocab.UNK, dtype=torch.long).to(d10_config.DEVICE)

    # 2 torch.where( condition，a，b)： 如果满足条件，则选择a，否则选择b作为输出
    # 2 判断 in_tensor 中的每一个元素，如果值大于（20004-1）则用oov_token， 否则使用in_tensor
    # 2 达到效果：把in_tensor中的 oov单词id 替换成 <UNK>对应的id
    out_tensor = torch.where( in_tensor > len(vocab) - 1, oov_token, in_tensor)
    return out_tensor

def dm06_test_replace_oovs():

    in_tensor = torch.tensor([[11, 12, 20013, 20014, 15, 16, 17, 18, 19],
                             [21, 22, 23, 24, 20025, 20026, 27, 28, 29],
                             [31, 32, 33, 34, 35, 20036, 37, 38, 39]], dtype = torch.long)

    print('in_tensor-->', in_tensor.shape )

    print('in_tensor中oov单词替换之前-->\n', in_tensor)
    oov_token = torch.full(in_tensor.shape, 3, dtype=torch.long).to(d10_config.DEVICE)

    # in_tensor矩阵中的每一个元素 进行 条件判断,  如果大于20003 则对应元素替换成vocab.UNK 3
    out_tensor = torch.where( in_tensor > (20004-1), oov_token, in_tensor)
    print('\n in_tensor中oov单词替换后--> \n', out_tensor)
    pass


# 获取 模型训练中 若干超参数的函数
def config_info(config):
    # get some config information
    info = 'model_name = {}, pointer = {}, coverage = {}, fine_tune = {}, scheduled_sampling = {}, weight_tying = {},' + 'source = {}  '
    return (info.format(config.model_name, config.pointer, config.coverage, config.fine_tune, config.scheduled_sampling, config.weight_tying, config.source))


class Beam(object):
    def __init__(self, tokens, log_probs, decoder_states, coverage_vector):
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.coverage_vector = coverage_vector

    def extend(self, token, log_prob, decoder_states, coverage_vector):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    coverage_vector=coverage_vector)

    def seq_score(self):
        # This function calculate the score of the current sequence.
        len_Y = len(self.tokens)
        # Lenth normalization
        ln = (5 + len_Y) ** d10_config.alpha / (5 + 1) ** d10_config.alpha
        # Coverage normalization
        cn = d10_config.beta * torch.sum(torch.log(d10_config.eps + torch.where(self.coverage_vector < 1.0,
                                                                                self.coverage_vector,
                                                                                torch.ones((1, self.coverage_vector.shape[1])).to(torch.device(d10_config.DEVICE)))))

        score = sum(self.log_probs) / ln + cn
        return score

    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    def __le__(self, other):
        return self.seq_score() <= other.seq_score()


class ScheduledSampler():
    def __init__(self, phases):
        self.phases = phases
        self.scheduled_probs = [i / (self.phases - 1) for i in range(self.phases)]

    def teacher_forcing(self, phase):
        # According to a certain probability to choose whether to execute teacher_forcing 
        sampling_prob = random.random()
        if sampling_prob >= self.scheduled_probs[phase]:
            return True
        else:
            return False


if __name__ == '__main__':

    # demo01_time_test()
    # demo02_test_simple_tokenizer()
    # demo03_test_count_words()
    # demo04_test_sort_batch_by_len()
    # demo05_heap_k3()
    dm06_test_replace_oovs()
    print('func_utils 工具函数 end')

