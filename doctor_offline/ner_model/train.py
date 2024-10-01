import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import BertTokenizer
from evaluate import evaluate

from bilstm_crf import NER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_batch_inputs(data, labels, tokenizer):

    # 函数需要返回一个按照内容长度从大到小排序过的，sentence 和 label, 还要返回 sentence 长度
    # 将批次数据的输入和标签值分开，并计算批次的输入长度
    data_inputs, data_length, data_labels = [], [], []
    for data_input, data_label in zip(data, labels):

        # 对输入句子进行编码
        data_input_encode = tokenizer.encode(data_input,
                                             return_tensors='pt',
                                             add_special_tokens=False)
        data_input_encode = data_input_encode.to(device)
        data_inputs.append(data_input_encode.squeeze())

        # 去除多余空格，计算句子长度
        data_input = ''.join(data_input.split())
        data_length.append(len(data_input))

        # 将标签转换为张量
        data_labels.append(torch.tensor(data_label, device=device))

    # 对一个批次的内容按照长度从大到小排序，符号表示降序
    sorted_index = np.argsort(-np.asarray(data_length))

    # 根据长度的索引进行排序
    sorted_inputs, sorted_labels, sorted_length = [], [], []
    for index in sorted_index:
        sorted_inputs.append(data_inputs[index])
        sorted_labels.append(data_labels[index])
        sorted_length.append(data_length[index])

    # 对张量进行填充，使其变成长度一样的张量
    print(sorted_inputs, )
    pad_inputs = pad_sequence(sorted_inputs)
    print(pad_inputs)

    return pad_inputs, sorted_labels, sorted_length


label_to_index = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}


def train():
    # 读取训练集
    train_data = load_from_disk('ner_data/bilstm_crf_data_aidoc')['train']

    # tokenizer
    tokenizer = BertTokenizer(vocab_file='ner_data/bilstm_crf_vocab_aidoc.txt')

    # 构建模型
    model = NER(vocab_size=tokenizer.vocab_size, label_num=len(label_to_index))#.cuda(device)

    # 批次大小
    batch_size = 16
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    # 训练轮数
    num_epoch = 700

    # train history
    train_history_list = []

    # valid history
    valid_history_list = []

    def start_train(data_inputs, data_labels, tokenizer):
        # 对数据进行填充补齐
        pad_inputs, sorted_labels, sorted_length = pad_batch_inputs(data_inputs, data_labels, tokenizer)

        # 计算损失
        loss = model(pad_inputs, sorted_labels, sorted_length)#实际调用的是NER中forward函数

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计损失值
        nonlocal total_loss
        total_loss += loss.item()


    for epoch in range(num_epoch):
        # total_loss
        total_loss = 0.0

        train_data.map(start_train, input_columns=['data_inputs', 'data_labels'],
                       batched = True,
                       batch_size=batch_size,
                       fn_kwargs={'tokenizer': tokenizer},
                       desc='epoch: %d' % (epoch + 1))
        print('epoch: %d loss: %.3f' % (epoch + 1, total_loss))

        # evaluate train data
        train_eval_result = evaluate(model, tokenizer, train_data)
        train_eval_result.append(total_loss)
        train_history_list.append(train_eval_result)

        model.save_model('model/BiLSTM-CRF-%d.bin' % (epoch + 1))

if __name__ == '__main__':
    train()