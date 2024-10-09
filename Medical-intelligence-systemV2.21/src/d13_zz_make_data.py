import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

'''
本文件使用方法：

1 先执行d15_make_data.py文件 把csv转成text
    函数功能：
        把<sep>小写换成大写
        csv 换成 txt

2 通过cmd命令 查看文件行   
    cmd命令：
    wc -l train.txt   # 测试文件有多少行
    wc -l test.txt


3 通过tail命令 head命令 制作 7000行训练集txt 1281行验证集txt
    tail -n 12871 train.txt > dev.txt
    head -n 70000 train.txt > train2.txt
    mv train2.txt train.txt
'''

def my_make_text():

    # 打开最终的结果文件
    train_writer = open('../data/train.txt', 'w', encoding='utf-8')
    test_writer = open('../data/test.txt', 'w', encoding='utf-8')

    # 对训练数据做处理, 将article和abstract中间用'<SEP>'分隔
    n = 0
    with open('../data/train_seg_data.csv', 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            line = line.strip().strip('\n')
            article, abstract = line.split('<sep>')
            text = article + '<SEP>' + abstract + '\n'
            train_writer.write(text)
            n += 1

    print('train n=', n)
    n = 0

    # 对测试数据做处理, 仅将文件存储格式从.csv换成.txt
    with open('../data/test_seg_data.csv', 'r', encoding='utf-8') as f2:
        for line in f2.readlines():
            line = line.strip().strip('\n')
            text = line + '\n'
            test_writer.write(text)
            n += 1

    print('test n=', n)


if __name__ == '__main__':
    my_make_text()

    print('制作txt文件 End')

