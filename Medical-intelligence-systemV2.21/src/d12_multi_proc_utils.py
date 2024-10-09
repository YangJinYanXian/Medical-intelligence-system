import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool

# cpu 数量
cores = cpu_count()

# 确定分块的个数
partitions = cores
# print(cores)

# 定义并行处理的函数
def parallelize(df, func):

    # 数据切分
    data_split = np.array_split(df, partitions)

    # 初始化线程池
    pool = Pool(cores)

    # 数据分发给不同的进程池执行 合并处理的结果
    data = pd.concat(pool.map(func, data_split))

    # 关闭线程池
    pool.close()

    # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    pool.join()

    # 返回处理好的结果数据
    return data


