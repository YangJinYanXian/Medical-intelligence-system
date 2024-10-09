import torch

'''
PyTorch数据发散（scatter/scatter_add）与聚集（Gather）操作

1 scatter_(input, dim=1, index, src)
  input.scatter_(dim, index, src)
      (a)将src中数据 (b)根据index中的索引 (c)按照dim的方向填进input中
      一个典型的用标量来修改张量的一个例子
      scatter() 一般可用来对标签进行 one-hot 编码

2 问题本质
    copy数据 需要指出 src[i][j] -> myc[m][n] i*j与m*n的关系通用通过label进行表达 所以变得晦涩难懂了！

3 拓展阅读
    https://zhuanlan.zhihu.com/p/158993858    # 有画图
    https://www.jianshu.com/p/6ce1ad89417f    # 

4 发散操作 ===> 单词分布[8,20004] + 权重分布([8,300]中8句话，每一句有oov，最大oov是10个单词)-->生成最终分布[8,20014]
          ===> 这样就相当于 生成 最终单词分布时，根据字典生成单词，有可以从原文source中copy原文oov单词
          
5 能力提升：根据 技术代码 提升 业务流能力，根据 技术代码 提升 论文复现能力
'''

def dm01_test_scatter():

    batch_size = 4  # 4个样本
    class_num = 10  # 10分类

    #10分类- 4个样本的类别标签Y
    # label2 = torch.LongTensor(batch_size, 1).random_() % class_num
    # print('\n10分类，4个样本的Y，label2--->\n',label2) # label2[4, 1]

    label = torch.tensor([[6], [0], [2], [3]])
    print('\n10分类，4个样本的Y label--->\n', label) # label[4, 1]

    # 用200的值，根据label索引，按照dim=1(按照行) 去修改torch.zeros的值
    # myc = torch.zeros(batch_size, class_num, dtype=torch.long)  # myc[4, 10]
    myc = torch.zeros(4, 10, dtype=torch.long)  # myc[4, 10]
    print('myc--->\n', myc)

    myc = torch.zeros(batch_size, class_num, dtype=torch.long).scatter_(1, label, 200)
    print(myc)
    pass

    # tensor([[0., 0., 0., 0., 0., 0., 200., 0., 0., 0.],
    #         [200., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 200., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 200., 0., 0., 0., 0., 0., 0.]])


def dm02_test_scatter():
    import numpy as np
    np.set_printoptions(suppress=True)

    batch_size = 4  # 4个样本
    class_num = 10  # 10分类

    # 10分类- 4个样本的类别标签Y
    # label2 = torch.LongTensor(batch_size, 1).random_() % class_num
    # print('\n10分类，4个样本的Y，label2--->\n', label2)

    label = torch.tensor([[6], [0], [2], [3]])
    print('\n10分类，4个样本的Y label--->\n', label)

    # 用200的值，根据label索引，按照dim=1(按照行) 去修改torch.zeros的值
    myc = torch.zeros(batch_size, class_num, dtype=torch.float32)
    print('myc--->\n', myc)

    myc = torch.zeros(batch_size, class_num, dtype=torch.float32).scatter_(1, label, 200)
    print(myc)

    # tensor([[0., 0., 0., 0., 0., 0., 200., 0., 0., 0.],
    #         [200., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 200., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 200., 0., 0., 0., 0., 0., 0.]])

    x_id_data = torch.tensor(  [[0, 1, 2],
                                [1, 2, 3],
                                [2, 3, 4],
                                [4, 5, 6]])

    weight = torch.tensor([[0.1, 0.1, 0.2],
                        [0.1, 0.2, 0.3],
                        [0.2, 0.3, 0.4],
                        [0.4, 0.5, 0.6]], dtype=torch.float32)

    # 把p_x的值，按照x的索引，累加到myc中
    final_distribution = myc.scatter_add_(dim=1, index=x_id_data, src=weight)


    print('final_distribution--->\n')
    print('{:2s}'.format(repr(final_distribution.numpy())))
    # print('{:2f}'.format(final_distribution.numpy()) )


# 聚集操作
# 作用： 最终概率分布[8, 20004+10], 根据真实值的id（作为索引）
#       把对应的概率给找到做损失，损失函数希望：真实值对应的位置的概率最大即可
def dm03_test_gather():
    class_num = 10
    batch_size = 4
    # label = torch.LongTensor(batch_size, 1).random_() % class_num

    label = torch.tensor([[6], [0], [3], [2]])
    # print('label:', label)

    # 用1的值，根据label索引，按照dim=1行的方式，去修改torch.zeros的值
    myc = torch.zeros(batch_size, class_num).scatter_(1, label, 100)
    print('myc-->\n', myc)

    # tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    #         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])


    # 聚集操作：把myc的值 按照label指定的位置，copy给 target_probs
    y_label = torch.tensor([[6], [0], [3], [2]])                 # 实验1
    # label = torch.tensor( [[0, 1], [1, 2], [2, 3], [3, 4]] ) # 实验2
    print('\n每个时间步有预测值和真实值 打印真实值label-->\n', label)
    print('\n下面根据真实值的索引id，找到最终单词分布表对应位置的概率值，我们希望真实值位置对应的概率值越大越好\n')
    target_probs = torch.gather(myc, 1, y_label)
    print('标签y的真实概率值：target_probs--->', target_probs)


    # 发散操作：用0.88的值， 根据[[2],[3]]的索引， 去修改torch.zeros的值
    # z = torch.zeros(2, 4).scatter_(1, torch.LongTensor([[2], [3]]), 0.88)
    # print("z---> \n", z)

    print('demo04_test_scatter End')


def dm04_test_torch_stack():
    # sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
    step_losses = []
    a1 = torch.tensor([1], dtype=torch.float32)
    a2 = torch.tensor([2], dtype=torch.float32)
    a3 = torch.tensor([3], dtype=torch.float32)
    step_losses.append(a1)
    step_losses.append(a2)
    step_losses.append(a3)

    myb1 = torch.stack(step_losses, 1) # myb1：1*3 torch.stack()函数 dim=1 会按照行的方向堆积元素
    myb2 = torch.stack(step_losses, 0) # myb2：3*1

    print('\nmyb1--->', myb1, myb1.shape, type(myb1))
    print('\nmyb2--->', myb2, myb2.shape, type(myb2))

    sample_losses = torch.sum(myb1, 1)
    print('sample_losses--->', sample_losses)

    '''
        myb1---> tensor([[1., 2., 3.]]) torch.Size([1, 3]) <class 'torch.Tensor'>
    
        myb2---> tensor([[1.],
                [2.],
                [3.]]) torch.Size([3, 1]) <class 'torch.Tensor'>
        sample_losses---> tensor([6.])
    '''



if __name__ == '__main__':

    # 发散操作
    # dm01_test_scatter()
    dm02_test_scatter()

    # 聚集操作
    dm03_test_gather()

    # dm04_test_torch_stack()

    print('model_other End')





# def dm05_test_set():
    # self.stop_word = list(set([self.vocab[x.strip()] for x in open(d10_config.stop_word_file).readlines()]))

    # myset = set()
    # myfile = open(d10_config.stop_word_file)
    # for x in myfile.readlines():
    #     a = x.strip()
    #     b = self.vocab [a]
    #     myset.add(b)
    #
    #
    # mylist = list(myset)
    # print(mylist)
    # pass

