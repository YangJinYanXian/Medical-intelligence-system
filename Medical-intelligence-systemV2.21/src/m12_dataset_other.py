# 导入工具包
import torch
from torch.utils.data import DataLoader, Dataset


'''
在 PyTorch 中，使用 torch.utils.data.DataLoader 类可以实现批量的数据集加载，
训练模型中经常常用，其功能也比较强度大

使用Pytorch自定义读取数据时步骤如下：
1）创建Dataset对象
    可编写继承Dataset的类，并且覆写__getitem__和__len__方法 
     
2）将Dataset对象作为参数传递到Dataloader中

'''

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.sample_number = len(self.y)

    def __len__(self):
        return self.sample_number

    def __getitem__(self, idx):
        # 修正 idx  [0, idx]范围为
        idx = min(max(idx, 0), self.sample_number - 1)
        # 返回一组样本
        return self.x[idx], self.y[idx], torch.tensor([1.,2.,3.,4.,5.])
        # return self.x[idx]


def dm01_test_dataloader():
    # 构造数据集
    x = torch.arange(21).reshape(21, 1) # [21,1]
    y = torch.arange(21)        # [21, ]
    print('x--->', x, x.shape)  # 注意x原始数据是二维的 torch.Size([21, 1])
    print('y--->', y, y.shape)  # 注意y原始数据是一维的 torch.Size([21])

    # 初始化数据集
    dataset = MyDataset(x, y)
    # 初始化数据加载器
    # dataloader = DataLoader(dataset, shuffle=True, batch_size=8, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=3, drop_last=True)
    for tx, ty, tz in  dataloader :
        print('batch_size=3 的情况下，查看tx、ty的数据形状', tx.shape, ty.shape, tz.shape)
        print('具体内容数据tx-->', tx)
        print('具体内容数据ty-->', ty)
        print('具体内容数据tz-->', tz)
        break

'''
0 概念：有关数据的二次处理
    collate_fn 参数用于接收用于传递的一个函数(回调函数，函数参数做参数)。
    DataLoader 会从数据集中获得一个批次的数据，然后将该批次数据再传递到 collate_fn 指向的函数中进行二次处理。

1  collate_fn回调函数调用时机：DataLoader(..., ..., collate_fn = callback回调函数入口地址 )，

2 collate_fn回调函数的处理机制中的 数据格式传递规则
    若没有二次处理机制，dataloader会做默认处理
        eg: batch_size=4
        eg: dataset有3个返回值return x, y, z;  dataloader返回做3个列表，添加到各自的列表中
        eg: [x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]

    如果启用二次处理机制，dataloader不会做上面的处理，而是机械的拼接(x,y,z)成一个元组，
                        然后再根据batch_size=4，添加到列表中 [(x1,y1,z1),(x2,y2,z2),(...),(...)]
        eg: [(x1,y1,z1), (x2,y2,z3), (x3,y3,z3), (x4,y4,z4)]

'''

# 此函数对数据集进行二次处理
# data数据格式 根据dataloader的getitem函数，返回元素的个数
# 返回1个元素data [x1, x2, x3]
# 返回2个元素data [(x1,y1), (x2,y2), (x3,y3)]
# 返回3个元素data [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]
def collate_fn_secondary_processing(data):
    print('data--->', data)
    feature = []
    target = []
    for x, y, z in data:
        feature.append(x.tolist())
        target.append(y.item())
    feature = torch.tensor(feature)
    target = torch.tensor(target) + 200
    return feature, target, z


def dm02_test_dataloader_collate_fn():
    # 构造数据集
    x = torch.arange(9).reshape(9, 1)
    y = torch.arange(100, 109)
    print('x--->', x, x.shape)  # torch.Size([16, 1])
    print('y--->', y, y.shape)  # torch.Size([16])
    # 初始化数据集
    dataset = MyDataset(x, y)
    # 初始化数据加载器
    dataloader = DataLoader(dataset,
                            # shuffle=True,
                            batch_size=3,
                            collate_fn=collate_fn_secondary_processing)

    print('注意函数collate_fn_secondary_processing的调用时机')

    for tx, ty, tz in dataloader:
        print('tx--->', tx) #
        print('ty--->', ty)  #
        print('tz--->', tz)  #
        break


if __name__ == '__main__':

    # 1 DataLoader的基本使用
    # dm01_test_dataloader()

    # 2 DataLoader的collate_fn参数
    dm02_test_dataloader_collate_fn()
    print('pytorch DataLoarder 演示完毕')
