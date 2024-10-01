import os
import torch
import torch.nn as nn

from RNN_MODEL import RNN
from bert_chinese_encode import get_bert_encode_for_single

# 超参数
MODEL_PATH = './BERT_RNN.pth'
n_hidden = 128
input_size = 768
n_categories = 2

# 实例化rnn模型
rnn = RNN(input_size, n_hidden, n_categories)
rnn.load_state_dict(torch.load(MODEL_PATH))


# 定义一个_test函数
def _test(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[1]):
        output, hidden = rnn(line_tensor[0][i].unsqueeze(0), hidden)

    return output

# 定义predict函数
def predict(text_tensor):
    # 不自动求梯度
    with torch.no_grad():
        output = _test(get_bert_encode_for_single(text_tensor))
        _, topi = output.topk(1, 1)

        return topi.item()


# 批量处理函数
def batch_predict(input_path_noreview, output_path_reviewed):
    # 批量处理noreview目录下的所有的csv文件，对每个csv文件中的症状描述进行审核
    # 审核完成后写入到reviewed目录
    csv_list = os.listdir(input_path_noreview)

    for csv in csv_list:
        with open(os.path.join(input_path_noreview, csv), 'r') as fr:
            with open(os.path.join(output_path_reviewed, csv), 'w') as fw:
                input_lines = fr.readlines()
                for input_line in input_lines:
                    print(csv, input_line)
                    res = predict(input_line)

                    if res: # 结果是1，把文本写入到文件中
                        fw.write(input_line+'\n')
                    else:
                        pass


if __name__ == '__main__':
    input_path = '../structured/noreview/'
    output_path = '../structured/reviewed/'
    batch_predict(input_path, output_path)




