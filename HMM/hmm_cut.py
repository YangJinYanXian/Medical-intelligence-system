# 实现使用hmm模型进行分词

import numpy as np
import os
import pickle


class HMM(object):
    def __init__(self):
        self.pi = None
        self.A = None # 转移矩阵
        self.B = None # 发射概率矩阵
        self.model_file = 'hmm_model.pkl'
        self.state_list = ['B', 'M', 'E', 'S'] # 表示的是每个在词语中的位置
        self.load_param = False

    def try_load_model(self, trained):
        if trained:
            with open(self.model_file, 'rb') as f:
                self.A = pickle.load(f)
                self.B = pickle.load(f)
                self.pi = pickle.load(f)
                self.load_param = True
        else:
            self.A = {}
            self.B = {}
            self.pi = {}
            self.load_param = False

    def train(self, path):
        self.try_load_model(False)

        def init_parameters():
            for state in self.state_list:
                self.A[state] = { s: 0.0 for s in self.state_list}
                self.B[state] = {}
                self.pi[state] = 0.0

        def generate_state(text):
            state_seq = []
            if len(text) == 1:
                state_seq.append('S')
            else:
                state_seq += ['B'] + ['M']*(len(text)-2) + ['E']

            return state_seq

        init_parameters()

        with(open(path, encoding='utf8')) as f:
            for line in f:
                # 更 高 地 举起 邓小平理论 的 伟大 旗帜”这句话及其对应状态序列“SSSBEBMMMESBEBE
                line = line.strip()
                if not line:
                    continue

                # 把一行的每个字 生成列表
                word_list = [i  for i in line if i != ' ']

                # 文本按照空格进行分割
                line_sequences = line.split()

                line_states = []
                for seq in line_sequences:
                    # 取出每个单词，转成对应的状态
                    line_states.extend(generate_state(seq))

                assert len(word_list) == len(line_states)

                # 根据一行中B M S E计算A B pi初始值
                # SSSBEBMMMESBEBE
                # word_list [更 高 地 举 起 邓 小 平 理 论 的 伟 大 旗 帜]
                for idx, state in enumerate(line_states):
                    if idx == 0:
                        self.pi[state] += 1

                    else:
                        self.A[line_states[idx-1]][state] += 1

                    self.B[line_states[idx]][word_list[idx]] = \
                        self.B[line_states[idx]].get(word_list[idx], 0) + 1.0

            # print(self.B)
            # 计算pi初始概率
            self.pi = {k: np.log(v / np.sum(list(self.pi.values()))) if v != 0 else -3.14e+100 for k, v in self.pi.items()}

            # 计算转移概率
            self.A = { k:{ k1 : np.log(v1/np.sum(list(v.values()))) if v1 != 0 else -3.14e+100 for k1, v1 in v.items()} for k, v in self.A.items()}

            # 计算发射概率
            self.B = { k: {k1: np.log(v1 / np.sum(list(v.values()))) if v1 != 0 else -3.14e+100 for k1, v1 in v.items()} for k, v in self.B.items()}

            # print(self.pi)
            # print(self.A)
            # print(self.B)

            # 保存模型
            with open(self.model_file, 'wb') as pkl:
                pickle.dump(self.A, pkl)
                pickle.dump(self.B, pkl)
                pickle.dump(self.pi, pkl)

            return self

    def viterbi(self, text, states, pi, A, B):
        """
        维特比算法实现
        :param text: 待分词文本
        :param states:  状态列表
        :param pi: 初始概率向量
        :param A:   状态转移概率矩阵
        :param B:  观测概率矩阵
        :return:   最大概率,预测状态序列
        """
        delta = [{}] # 不同时刻前向概率
        psi = {}
        # 初始化delta psi
        for state in states:
            delta[0][state] = pi[state] + B[state].get(text[0], 0)# 第0时刻前向概率
            psi[state] = [state]
        # psi {'B':['B'], 'M':['M'], 'S':['S'], 'E':['E']}

        for t in range(1, len(text)):
            delta.append({})
            newpsi = {}

            for state in states:
                (prob, state_sequence) = max([(delta[t-1][state0] + A[state0].get(state, 0), state0) for state0 in states])
                delta[t][state] = prob + B[state].get(text[t], -3.14e100)
                newpsi[state]  = psi[state_sequence] + [state]
            # newpsi
            psi = newpsi

        (prob, state_sequence) = max([(delta[len(text)-1][state], state) for state in states])

        return prob, psi[state_sequence]



    def cut(self, text):
        if not self.load_param:
            self.try_load_model(os.path.exists(self.model_file))

        prob, pos_list = self.viterbi(text, self.state_list, self.pi, self.A, self.B)
        begin, next_idx = 0, 0
        # 分词
        for i, char in enumerate(text):
            pos = pos_list[i] # B M S E中的某个值
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield  text[begin: i+1]
                next_idx = i + 1
            elif pos == 'S':
                yield  char
                next_idx = i + 1

        if next_idx < len(text):
            yield text[next_idx:]


def load_article(fname):
    with open(fname, encoding='utf-8') as file:
        aritle = []
        for line in file:
            aritle.append(line.strip())

    return aritle


def to_region(segmentation):
    """
    把分词结果转成区间
    :param segmentation:  分词后的数据
    :return:  区间
    """
    import re
    region = []
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):
        end = start + len(word)
        region.append((start, end))
        start = end

    return region

def prf(gold, pred):
    """
    计算精确率 召回率 f1
    :param gold: 真实值
    :param pred: 预测值
    :return:
    """
    A, B = set(to_region(gold)), set(to_region(pred))
    A_size = len(A)
    B_size = len(B)
    A_cap_B_size = len(A & B) # TP
    p, r = A_cap_B_size/B_size, A_cap_B_size/A_size
    return p, r, 2*p*r/(p+r)



if __name__ == '__main__':
    hmm = HMM()
    # hmm.train('./HMMTrainSet.txt')
    # print(list(hmm.cut('商品和服务')))
    # print(list(hmm.cut('项目的研究')))
    # print(list(hmm.cut('研究生命起源')))
    # print(list(hmm.cut('中文博大精深!')))
    # print(list(hmm.cut('这是一个非常棒的方案!')))
    # print(list(hmm.cut('武汉市长江大桥')))
    # print(list(hmm.cut('普京与川普通话')))
    # print(list(hmm.cut('四川普通话')))
    # print(list(hmm.cut('小明硕士毕业于中国科学院计算所，后在日本京都大学深造')))
    # print(list(hmm.cut('改判被告人死刑立即执行')))
    # print(list(hmm.cut('检察院检察长')))
    # print(list(hmm.cut('中共中央总书记、国家主席')))

    hmm.try_load_model(True)

    article = load_article('./test1_org.txt')
    pred = "  ".join(list(hmm.cut(article[0])))
    # print(pred)
    gold = load_article('./test1_cut.txt')[0]
    # print(gold)
    print("精确率:%.5f, 召回率:%.5f, F1:%.5f" % prf(gold, pred))
