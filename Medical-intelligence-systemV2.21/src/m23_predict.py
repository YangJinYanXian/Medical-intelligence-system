import random
import os
import sys
import torch
import jieba

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import d10_config
from m13_model import PGN
from m12_dataset import PairDataset
from d14_func_utils import source2ids, outputids2words, Beam, timer, add2heap, replace_oovs


# 创建预测类
class Predict():

    # @timer(module='initalize predicter')
    def __init__(self):
        self.DEVICE = d10_config.DEVICE

        dataset = PairDataset(d10_config.train_data_path,
                              max_enc_len=d10_config.max_enc_len,
                              max_dec_len=d10_config.max_dec_len,
                              truncate_enc=d10_config.truncate_enc,
                              truncate_dec=d10_config.truncate_dec)

        self.vocab = dataset.build_vocab(embed_file=d10_config.embed_file)

        self.model = PGN(self.vocab)
        self.stop_word = list(set([self.vocab[x.strip()] for x in open(d10_config.stop_word_file).readlines()]))

        self.model.load_state_dict(torch.load(d10_config.model_save_path, map_location=lambda storage, loc:storage), False)
        # 导入已经训练好的模型 并转移到GPU上
        # self.model.load_state_dict(torch.load(config.model_save_path), map_location=torch.device('cpu'))


        # self.model = self.model.to(self.DEVICE)
        self.model.to(self.DEVICE)

        print('设备是否gpu异常')

    # 编写预测函数的代码
    @timer(module='doing prediction')
    def predict(self, text, tokenize=True):
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))

        # 将原始文本映射成数字化张量
        x, oov = source2ids(text, self.vocab)
        x = torch.tensor(x).to(self.DEVICE)

        # 获取oov的长度 和 paddiing_mask张量
        len_oovs = torch.tensor([len(oov)]).to(self.DEVICE)
        x_padding_masks = torch.ne(x, 0).byte().float()

        # 利用贪心解码函数得到摘要结果
        summary = self.greedy_search(x.unsqueeze(0),
                                     max_sum_len=d10_config.max_dec_steps,
                                     len_oovs=len_oovs,
                                     x_padding_masks=x_padding_masks)

        # 将得到的摘要数字化张量 转换成自然语言文本
        summary = outputids2words(summary, oov, self.vocab)

        # 删除掉特殊字符 SOS EOS ，去除空字符
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()


    # 编写贪心解码的类内函数
    def greedy_search(self, x, max_sum_len, len_oovs, x_padding_masks):
        # Get encoder output and states.Call encoder forward propagation

        # 模型编码
        encoder_output, encoder_states = self.model.encoder(replace_oovs(x, self.vocab))

        # 用encode的hidden state来初始化 decoder中的 hidden state
        decoder_states = self.model.reduce_state(encoder_states)

        # Initialize decoder's input at time step 0 with the SOS token.
        # 利用 sos 作为解码器的初始化输入字符
        x_t = torch.ones(1) * self.vocab.SOS
        x_t = x_t.to(self.DEVICE, dtype=torch.int64)
        summary = [self.vocab.SOS] # 初始化摘要 用SOS作为起始

        # Generate hypothesis with maximum decode step.
        # 循环解码 最多解码max_sum_len步
        while int(x_t.item()) != (self.vocab.EOS) and len(summary) < max_sum_len:

            context_vector, attention_weights = self.model.attention(decoder_states, encoder_output, x_padding_masks)

            p_vocab, decoder_states, p_gen = self.model.decoder(x_t.unsqueeze(1), decoder_states, context_vector)

            final_dist = self.model.get_final_distribution(x, p_gen, p_vocab, attention_weights, torch.max(len_oovs))

            # Get next token with maximum probability.
            # 以贪心解码的策略进行预测字符
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
            decoder_word_idx = x_t.item()

            # 预测的字符添加进结果摘要中
            summary.append(decoder_word_idx)
            x_t = replace_oovs(x_t, self.vocab)

        return summary


if __name__ == "__main__":
    print('实例化Predict对象, 构建dataset和vocab......')
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))

    # Randomly pick a sample in test set to predict.
    # 随机 从测试集中抽取一条样本进行预测
    with open(d10_config.val_data_path, 'r') as test:
        picked = random.choice(list(test))
        source, ref = picked.strip().split('<SEP>')

    print('原始文本source: ', source, '\n')
    print('******************************************')
    print('人工智能摘要ref: ', ref, '\n')
    print('******************************************')

    greedy_prediction = pred.predict(source.split())
    print('贪心预测摘要greedy: ', greedy_prediction, '\n')



