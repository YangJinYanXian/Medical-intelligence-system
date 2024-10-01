import torch
import torch.nn as nn
import random
import pandas as pd
from bert_chinese_encode import get_bert_encode_for_single
from RNN_MODEL import RNN
import math
import time


def randomTrainingExample(train_data):
    """
    随机选择一条数据，进行编码
    :param train_data:
    :return:
    """
    # 获取一行数据 标签
    category, text = random.choice(train_data)

    # 对文本进行编码
    text_tensor = get_bert_encode_for_single(text)

    # 把标签封装成张量
    category_tensor = torch.tensor([int(category)])

    return category, text, category_tensor, text_tensor


def train(category_tensor, line_tensor):
    """

    :param category_tensor: 标签
    :param line_tensor: 文本对应的编码
    :return:
    """
    # 初始化隐藏层
    hidden = rnn.initHidden()

    # 梯度归0
    rnn.zero_grad()

    # # 循环取出每个字对应的张量
    for i in range(line_tensor.size()[1]):
        output, hidden = rnn(line_tensor[0][i].unsqueeze(0), hidden)

    # 计算损失值
    loss = criterion(output, category_tensor)

    # 反向传播
    loss.backward()

    # 更新参数
    for p in rnn.parameters():
        p.data.add_(-lr_rate, p.grad.data)

    return output, loss.item()


def valid(category_tensor, line_tensor):
    """模型验证函数, category_tensor代表类别张量, line_tensor代表编码后的文本张量"""
    # 初始化隐层
    hidden = rnn.initHidden()
    # 验证模型不自动求解梯度
    with torch.no_grad():
        # 遍历line_tensor中的每一个字的张量表示
        for i in range(line_tensor.size()[1]):
            # 然后将其输入到rnn模型中, 因为模型要求是输入必须是二维张量, 因此需要拓展一个维度, 循环调用rnn直到最后一个字
            output, hidden = rnn(line_tensor[0][i].unsqueeze(0), hidden)
        # 获得损失
        loss = criterion(output, category_tensor)
     # 返回结果和损失的值
    return output, loss.item()


# 构建时间计算函数
def timeSince(since):
    "获得每次打印的训练耗时, since是训练开始时间"
    # 获得当前时间
    now = time.time()
    # 获得时间差，就是训练耗时
    s = now - since
    # 将秒转化为分钟, 并取整
    m = math.floor(s / 60)
    # 计算剩下不够凑成1分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)

def main():
    # 设置迭代次数为50000步
    n_iters = 50000

    # 打印间隔为1000步
    plot_every = 1000

    # 初始化打印间隔中训练和验证的损失和准确率
    train_current_loss = 0
    train_current_acc = 0
    valid_current_loss = 0
    valid_current_acc = 0

    # 初始化盛装每次打印间隔的平均损失和准确率
    all_train_losses = []
    all_train_acc = []
    all_test_losses = []
    all_test_acc = []

    # 获取开始时间戳
    start = time.time()

    for iter in range(1, n_iters+1):
        # 分别获取一条训练数据和一条验证数据
        category, text, category_tensor, text_tensor = randomTrainingExample(train_data[:9000])
        category_test, text_test, category_tensor_test, text_tensor_test = randomTrainingExample(train_data[9000:])

        # 训练验证
        train_output, train_loss = train(category_tensor, text_tensor)
        valid_output, valid_loss = valid(category_tensor_test, text_tensor_test)

        # 累计 损失值 准确率
        train_current_loss += train_loss
        train_current_acc += (train_output.argmax(1) == category_tensor).sum().item()

        valid_current_loss += valid_loss
        valid_current_acc += (valid_output.argmax(1) == category_tensor_test).sum().item()

        # 每个1000次 打印输入
        if iter % plot_every == 0:
            train_average_loss = train_current_loss / plot_every
            train_average_acc = train_current_acc / plot_every

            valid_average_loss = valid_current_loss /plot_every
            valid_average_acc = valid_current_acc / plot_every

            # 打印迭代步, 耗时, 训练损失和准确率, 验证损失和准确率
            print("Iter:", iter, "|", "TimeSince:", timeSince(start))
            print("Train Loss:", train_average_loss, "|", "Train Acc:", train_average_acc)
            print("Valid Loss:", valid_average_loss, "|", "Valid Acc:", valid_average_acc)

            # 保存结果到列表中，方便画图
            all_train_losses.append(train_average_loss)
            all_train_acc.append(train_average_acc)

            all_test_losses.append(valid_average_loss)
            all_test_acc.append(valid_average_acc)

            # 把中间结果 归零
            train_current_loss = 0
            train_current_acc = 0
            valid_current_loss = 0
            valid_current_acc = 0

    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.plot(all_train_losses, label="Train Loss")
    plt.plot(all_test_losses, color="red", label="Valid Loss")
    plt.legend(loc='upper left')
    plt.savefig("./loss.png")

    plt.figure(1)
    plt.plot(all_train_acc, label="Train Acc")
    plt.plot(all_test_acc, color="red", label="Valid Acc")
    plt.legend(loc='upper left')
    plt.savefig("./acc.png")

    # 保存路径
    MODEL_PATH = './BERT_RNN.pth'
    # 保存模型参数
    torch.save(rnn.state_dict(), MODEL_PATH)


if __name__ == '__main__':
    # 读取数据
    train_data_path = "./train_data.csv"
    train_data = pd.read_csv(train_data_path, header=None, sep="\t")

    # 转换数据到列表形式
    train_data = train_data.values.tolist()

    # # 选择10条数据进行查看
    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample(train_data)
        print('category =', category, '/ line =', line, line_tensor.shape)

    # 实现train函数
    # 定义Loss
    criterion = nn.NLLLoss() # 圆括号不要掉了
    # learning rate
    lr_rate = 0.005

    # 定义参数
    input_size = 768  # bert模型输出的维度
    hidden_size = 128  # 自定义的
    n_categories = 2  # 类别数量

    # input = torch.rand(1, input_size)  # 随机生成符合形状的张量
    # hidden = torch.rand(1, hidden_size)

    rnn = RNN(input_size, hidden_size, n_categories)  # 实例化模型
    # outputs, hidden = rnn(input, hidden)

    main()
