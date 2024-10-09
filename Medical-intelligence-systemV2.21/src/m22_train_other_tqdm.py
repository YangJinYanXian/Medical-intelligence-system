
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import numpy as np
from tqdm import tqdm
import d10_config

'''
1 什么是TQDM？是一个快速的，易扩展的进度条提示模块
2 安装  tqdm==4.62.2
    普通环境安装：     pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/
    Anaconda下安装：    conda install -c conda-forge tqdm
3 三大使用方式

4 拓展阅读
    https://www.freesion.com/article/5916402868/
'''

# 方式1： 基于迭代类型 向tqdm中传入迭代类型即可
import time
def dm01_test_tqdm():
    print('基于迭代类型 Begin')
    text = ""
    for char in tqdm( ["a", "b", "c", "d", "e"], desc="mydescAAAAAA"):
        time.sleep(1)
        text = text + char
        # print(text)

# 方式2： 手动更新进度
# 注意 进度条是异步的，若程序中有sleep睡眠函数，进度条消失以后，还会有不友好的提示
def dm02_test_tqdm():
    # 使用with语句来控制tqdm的更新
    print('使用with语句来控制tqdm的更新')
    with tqdm(total=100) as pbar:
        for i in range(20):
            time.sleep(1)
            pbar.update(10)  # 每次更新的多少

    print('*' * 20, '\n')


def dm03_test_tqdm():
    # 当然也可以不用with，将tqdm赋值给一个变量
    print('不使用with语句进行更新')
    pbar = tqdm(total=100)
    for i in range(10):
        time.sleep(1)   # 时间需要运行10次
        pbar.update(20) # 5次更新完
    pbar.close()  # ！！ 注意这样使用之后必须调用del 或者close方法删除该变量


def dm04_test_tqdm():
    start_epoch = 0
    num_batches = 700
    d10_config.epochs = 5
    num_epochs = len ( range(start_epoch, d10_config.epochs))
    # print('num_epochs:', num_epochs)

    epoch_loss = 0.93

    with tqdm(total=d10_config.epochs) as epoch_progress: # 根据多少轮epochs进行进度条控制
        for epoch in range(start_epoch, d10_config.epochs):

            with tqdm(total=num_batches // 100) as batch_progress: # 根据每轮批次数进行进度条控制
                # print('sleep 1')
                for batch in range(701):
                    time.sleep(0.01)
                    if (batch % 100) == 0:
                        batch_progress.set_description(f'Epoch {epoch}')
                        batch_progress.set_postfix(Batch=batch, Loss=batch)
                        batch_progress.update()

        # time.sleep(1)
        # 设置/修改进度条的提示，字符串后自动添加 “:”
        epoch_progress.set_description(f'Epoch {epoch}')
        # 设置/修改进度条后的提示信息
        epoch_progress.set_postfix(Loss=epoch_loss)
        epoch_progress.update()


# 注意每个epoch之间打屏幕，进度条效果就不好
def dm05_test_tqdm():
    start_epoch = 0
    num_batches = 700
    d10_config.epochs = 5
    num_epochs = len ( range(start_epoch, d10_config.epochs))
    # print('num_epochs:', num_epochs)

    epoch_loss = 0.93

    # with tqdm(total=d10_config.epochs) as epoch_progress:
    for epoch in range(start_epoch, d10_config.epochs):

        with tqdm(total=num_batches // 100) as batch_progress:
            # print('sleep 1')
            for batch in range(701):
                time.sleep(0.01)
                if (batch % 100) == 0:
                    batch_progress.set_description(f'Epoch {epoch}')
                    batch_progress.set_postfix(Batch=batch, Loss= batch)
                    batch_progress.update()

        # time.sleep(1)
        # # 设置/修改进度条的提示，字符串后自动添加 “:”
        # epoch_progress.set_description(f'Epoch {epoch}')
        # # 设置/修改进度条后的提示信息
        # epoch_progress.set_postfix(Loss=epoch_loss)
        # epoch_progress.update()
        # print('epoch %d' %epoch, flush=True) # 实验1
        # print('epoch %d' % epoch)  # 实验2


if __name__ == '__main__':
    # dm01_test_tqdm()
    # dm02_test_tqdm()
    # dm03_test_tqdm()
    # dm04_test_tqdm()
    dm05_test_tqdm()
    print('tqdm End')







