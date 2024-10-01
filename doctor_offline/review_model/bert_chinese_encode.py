import torch
import torch.nn as nn

# 从本地加载
# root用户从本地加载
source = '/root/.cache/torch/hub/huggingface_pytorch-transformers_main'
# 普通用户xxx从本地加载
# source = '/home/xxx/.cache/torch/hub/huggingface_pytorch-transformers_main'
# 直接使用预训练的bert中文模型
model_name = 'bert-base-chinese'
# 通过torch.hub获得已经训练好的bert-base-chinese模型
model = torch.hub.load(source, 'model', model_name, source='local')
# 获得对应的字符映射器, 它将把中文的每个字映射成一个数字
tokenizer = torch.hub.load(source, 'tokenizer', model_name, source='local')

# # 从github加载
# model_name = "bert-base-chinese"
# source = "huggingface/pytorch-transformers"
# model = torch.hub.load(source, 'model', model_name, source='github')
# tokenizer = torch.hub.load(source, 'tokenizer', model_name, source='github')

def get_bert_encode_for_single(text):
    """
    使用bert-base-chinese模型对文本进行编码
    :param text:  输入的文本
    :return: 编码后的张量
    """
    # 公国tokenizer对文本进行编号
    indexed_tokens = tokenizer.encode(text)[1: -1]
    # print('text 编号: ',indexed_tokens, type(indexed_tokens))
    # [101, 872, 1962, 8024, 4886, 2456, 5664, 102]
    # 把列表转成张量
    tokens_tensor = torch.LongTensor([indexed_tokens])

    # 不自动进行梯度计算
    with torch.no_grad():
        output = model(tokens_tensor)
        # print('model的输出: ', output)

    return output[0]


if __name__ == '__main__':
    text = "你好，福建舰"
    outputs = get_bert_encode_for_single(text)
    # print('text编码:', outputs)