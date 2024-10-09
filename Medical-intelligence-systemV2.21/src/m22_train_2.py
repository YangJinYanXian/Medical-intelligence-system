import pickle
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tensorboardX import SummaryWriter

from m12_dataset import PairDataset
from m13_model import PGN
import d10_config
from m21_evaluate import evaluate
from m12_dataset import collate_fn, SampleDataset
from d14_func_utils import ScheduledSampler, config_info


# 模型训练函数思路分析 train(dataset, val_dataset, v, start_epoch=0):
# 1 加载数据集 train_data = SampleDataset(dataset.pairs, v) / val_data val_dataset
# 数据迭代器train_dataloader DataLoader(dataset=train_data, batch_size=d10_config.batch_size, shuffle=True, collate_fn=collate_fn)
# 2 实例化pgn mymodel   DEVICE=d10_config.DEVICE / mymodel=PGN(v) / mymodel.to(DEVICE)
# 3 实例化优化器 myoptimizer optim.Adam(, lr=d10_config.learning_rate)
    # 验证集损失值val_losses=100000000
# 4 外层for循环控制轮次 for epoch in (range(1, d10_config.epochs+1)):
    # 每个轮次辅助参数(config_info(d10_config)) / 每个批次损失列表 batch_losses=[]  批次数num_batches=len(train_dataloader)
# 5 内层for循环控制批次数 for batch, data in enumerate(tqdm(train_dataloader), start=1):
    # 5-1 解包data处理to_device x, y, x_len, y_len, oov, len_oovs = data
    # if d10_config.is_cuda:  x / y / x_len / len_oovs = len_oovs.to(DEVICE)
    # 模型模式 mymodel.train()
    # 5-2 给模型喂数据 myloss = mymodel(x, x_len, y, len_oovs, batch=batch, num_batches=num_batches, teacher_forcing=True)
    # 5-3 梯度清零 # 反向传播 # 梯度裁剪 # 梯度更新
    # clip_grad_norm_(mymodel.encoder.parameters(), d10_config.max_grad_norm) / decoder / attention
    # 5-4 辅助打印控制
    # batch_losses.append(myloss.item())
    # print('epoch:%d, batch:%d, average loss:%.6f ' % (epoch, batch, np.mean(batch_losses)) )
# 6 每个轮次求平均损失 验证集评估损失
    # epoch_loss = np.mean(batch_losses) # avg_val_loss = evaluate(mymodel, val_data, epoch)
    # print('epoch:%d, training loss:%.4f, avg_val_loss loss:%.4f' %(epoch, epoch_loss, avg_val_loss))

    # 更新一下有更小损失值的模型
    # if (avg_val_loss < val_losses): torch.save(mymodel.state_dict(), d10_config.model_save_path) val_losses = avg_val_loss
    # 评估损失小的模型保存 with open(d10_config.losses_path, 'wb') as f: pickle.dump(val_losses, f)
def train(dataset, val_dataset, v, start_epoch=0):

    print("加载数据 train_data val_data ... ")
    train_data = SampleDataset(dataset.pairs, v)
    val_data = SampleDataset(val_dataset.pairs, v)

    # 实例化PGN类对象
    DEVICE = d10_config.DEVICE
    mymodel = PGN(v)
    mymodel = mymodel.to(DEVICE)

    print("初始化优化器 optimizer ...")
    # 定义模型训练的优化器
    myoptimizer = optim.Adam(mymodel.parameters(), lr=d10_config.learning_rate)

    # 定义训练集的数据迭代器
    train_dataloader = DataLoader(dataset=train_data, batch_size=d10_config.batch_size, shuffle=True, collate_fn=collate_fn)

    # 验证集上的损失值初始化为一个大整数
    val_losses = 100000000.0

    # 根据配置文件config.py中的设置 对整个数据集进行一定轮次的迭代训练
    for epoch in (range(1, d10_config.epochs+1)): # 外层for循环 控制epoch多少轮

        # 每一个epoch之前打印模型训练的相关参数信息
        print('\n', config_info(d10_config))

        # 初始化每一个batch损失值的存放列表
        batch_losses = []  # Get loss of each batch.
        num_batches = len(train_dataloader)

        # 针对每一个epoch 按照batch_size读取数据迭代训练
        # for batch, data in enumerate(tqdm(train_dataloader), start=1 ): # 内层for循环 控制多少批次数
        for batch, data in enumerate(train_dataloader, start=1):  # 内层for循环 控制多少批次数
            x, y, x_len, y_len, oov, len_oovs = data

            # 如果配置GPU 则加速训练
            if d10_config.is_cuda:  # Training with GPUs.
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)

            # 设置模型进入训练模型
            mymodel.train()

            # 老三样中第一步： 梯度清零
            myoptimizer.zero_grad()

            # 利用模型进行训练 并返回损失值
            myloss = mymodel(x, x_len, y, len_oovs, batch=batch, num_batches=num_batches, teacher_forcing=True)

            batch_losses.append(myloss.item())

            # 老三样的第二步 反向传播
            myloss.backward()

            # 为了防止梯度爆炸而进行梯度裁剪
            # Do gradient clipping to prevent gradient explosion.
            clip_grad_norm_(mymodel.encoder.parameters(), d10_config.max_grad_norm)
            clip_grad_norm_(mymodel.decoder.parameters(), d10_config.max_grad_norm)
            clip_grad_norm_(mymodel.attention.parameters(), d10_config.max_grad_norm)

            # 老三样中的 参数更新
            myoptimizer.step()

            # 每隔100个batch记录一下损失信息
            if batch % 1 == 0:
                print('epoch:%d, batch:%d, average loss:%.6f ' % (epoch, batch, np.mean(batch_losses)) )

            if batch == 3:
                print('训练 3次迭代 就退出 ', flush=True)
                break

        # Calculate average loss over all batches in an epoch.
        # 将一个epoch中所有batch的平均损失值 作为这个epoch的损失值
        epoch_loss = np.mean(batch_losses)

        # 结束每一个epoch训练后，直接在验证集上跑一下模型的效果
        avg_val_loss = evaluate(mymodel, val_data, epoch)

        print('epoch:%d, training loss:%.4f, avg_val_loss loss:%.4f' %(epoch, epoch_loss, avg_val_loss))

        # 更新一下有更小损失值的模型
        if (avg_val_loss < val_losses):
            torch.save(mymodel.state_dict(), d10_config.model_save_path)
            val_losses = avg_val_loss

            # 将更小的损失值写入文件中保存
            with open(d10_config.losses_path, 'wb') as f:
                pickle.dump(val_losses, f)


if __name__ == "__main__":
    # Prepare dataset for training.
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('DEVICE: ', DEVICE)

    # 构建训练用的数据集
    dataset = PairDataset(d10_config.train_data_path,
                          max_enc_len=d10_config.max_enc_len,
                          max_dec_len=d10_config.max_dec_len,
                          truncate_enc=d10_config.truncate_enc,
                          truncate_dec=d10_config.truncate_dec)

    # 构建测试用的数据集
    val_dataset = PairDataset(d10_config.val_data_path,
                              max_enc_len=d10_config.max_enc_len,
                              max_dec_len=d10_config.max_dec_len,
                              truncate_enc=d10_config.truncate_enc,
                              truncate_dec=d10_config.truncate_dec)

    # vocab 20004个
    # 创建模型的单词字典
    vocab = dataset.build_vocab(embed_file=d10_config.embed_file)

    # 调用训练函数进行模型训练
    train(dataset, val_dataset, vocab, start_epoch=0)
    print('pgn模型训练 End')







