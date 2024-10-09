import os
import sys

# 配置文件：为了数据预处理

# 项目文件路径配置模块
# 设置项目的root路径 方便后续代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
# print(root_path)

# 原始数据文件存储路径
train_raw_data_path = os.path.join(root_path, 'data', 'train.csv')
test_raw_data_path = os.path.join(root_path, 'data', 'test.csv')

# 停用词和用户自定义字典 的路径
stop_words_path = os.path.join(root_path, 'data', 'stopwords.txt')
user_dict_path = os.path.join(root_path, 'data', 'user_dict.txt')

# 预处理+切分后的训练数据路径、测试数据路径
train_seg_path = os.path.join(root_path, 'data', 'train_seg_data.csv')
test_seg_path = os.path.join(root_path, 'data', 'test_seg_data.csv')

# 经过第一轮预处理后的最终数据集
train_data_path = os.path.join(root_path, 'data', 'train.txt')
test_data_path = os.path.join(root_path, 'data', 'test.txt')

# 词向量模型的路径
word_vector_path = os.path.join(root_path, 'data', 'wv', 'word2vec.model')

