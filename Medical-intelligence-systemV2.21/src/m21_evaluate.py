import os
import sys
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from m12_dataset import collate_fn, PairDataset, SampleDataset
import d10_config
from m13_model import PGN

# 编写评估函数的代码
def evaluate(model, val_data, epoch):
    print('开始进行评估 validating')
    val_loss = []
    with torch.no_grad():

        DEVICE = d10_config.DEVICE
        # pin_memory=True是对GPU机器的优化设置
        val_dataloader = DataLoader(dataset=val_data,
                                    batch_size=d10_config.batch_size,
                                    shuffle=True,
                                    # pin_memory=True,
                                    drop_last=True,
                                    collate_fn=collate_fn)

        # 遍历测试数据进行评估
        # for batch, data in enumerate(tqdm(val_dataloader)):
        for batch, data in enumerate(val_dataloader):
            x, y, x_len, y_len, oov, len_oovs = data
            if d10_config.is_cuda:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)
            total_num = len(val_dataloader)

            loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=total_num, teacher_forcing=True)
            val_loss.append(loss.item())
            if (batch == 3):
                print('验证集训练3次 就提前退出')
                break

    # 返回整个测试集的平均损失值
    return np.mean(val_loss)


if __name__ == '__main__':

    print('从配置文件中获取参数信息', d10_config.max_vocab_size)

    # 1 训练集 数据对
    train_dataset = PairDataset(d10_config.train_data_path,
                                max_enc_len=d10_config.max_enc_len,
                                max_dec_len=d10_config.max_dec_len,
                                truncate_enc=d10_config.truncate_enc,
                                truncate_dec=d10_config.truncate_dec)

    # 2 创建模型的单词字典
    # vocab 有97733个 使用了20000个+4个特殊字符 其他的都是oov单词
    vocab = train_dataset.build_vocab(embed_file=d10_config.embed_file)
    print('vocab--->', vocab.size())

    # 3 实例化PGN类对象
    model = PGN(vocab)
    print('model--->', model)

    DEVICE = d10_config.DEVICE

    val_dataset = PairDataset(d10_config.val_data_path,
                          max_enc_len=d10_config.max_enc_len,
                          max_dec_len=d10_config.max_dec_len,
                          truncate_enc=d10_config.truncate_enc,
                          truncate_dec=d10_config.truncate_dec)

    val_dataset = SampleDataset(val_dataset.pairs, vocab)


    # 4 加载已经训练好的模型
    model.load_state_dict(torch.load(d10_config.model_save_path, map_location=lambda storage, loc:storage), False)
    print('model-->2', model)

    # 5 模型预测
    loss = evaluate(model, val_dataset, epoch=1)
    print('loss--->', loss)

