import os
import sys
from rouge import Rouge

# 设置项目的root路径，方便项目文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
print('root_path--->', root_path)

# from server.s721_predict import Predict
from m23_predict import Predict
from d14_func_utils import timer
import d10_config


# 构建 ROUGE 评估类
class RougeEval():
    def __init__(self, path):
        self.path = path
        self.rouge = Rouge()    # 实例化Rouge对象
        self.sources = []       # 原文source，对soruce进行预测生成摘要，存放在 hypos列表 中
        self.hypos = []         # 假定摘要 - 待生成的机器摘要存储空间，也就是机器摘要
        self.refs = []          # 人工摘要 - 从文件中读取人工摘要
        self.process()          # 函数调用 预处理 init函数中调用process()

    # 从文件中读取 原文source、人工摘要ref, 并显示样本个数
    def process(self):
        print('Reading from ', self.path)
        with open(self.path, 'r') as test:
            for line in test:
                source, ref = line.strip().split('<SEP>')
                ref = ref.replace('。', '.')
                self.sources.append(source)
                self.refs.append(ref)

        print('self.refs[]包含的样本数: ', len(self.refs))
        print(f'Test set contains {len(self.sources)} samples.')

    # 产生机器摘要，根据source生成机器摘要，存放在hypos列表中
    @timer('building hypotheses')
    def build_hypos(self, predict):
        print('Building hypotheses.')
        count = 0
        for source in self.sources:
            count += 1
            if count % 10 == 0:
                print('count=', count)

            # 调用模型输入原始文本，产生摘要
            myres = predict.predict(source.split())
            # print(myres)
            self.hypos.append(myres)

    # 获取平均分数的函数
    def get_average(self):
        assert len(self.hypos) > 0, '需要首先构建hypotheses。Build hypotheses first!'
        print('Calculating average rouge scores.')
        # 输入机器摘要 、人工摘要 求rouge平均分数
        return self.rouge.get_scores(self.hypos, self.refs, avg=True)


if __name__ == '__main__':

    # 真实的测试机是val_data_path: dev.txt(3000条)
    print('实例化Rouge对象 ... ')
    my_rougeeval = RougeEval(d10_config.val_data_path100)

    print('实例化Predict对象 ... ')
    predict = Predict()

    # 利用模型对article进行预测
    print('利用模型对article进行摘要, 并通过Rouge对象进行评估 ... ')
    my_rougeeval.build_hypos(predict)

    # 将预测结果和标签abstract进行ROUGE规则计算
    print('开始用Rouge规则进行评估 ... ')
    result = my_rougeeval.get_average()
    print('rouge1: ', result['rouge-1'])
    print('rouge2: ', result['rouge-2'])
    print('rougeL: ', result['rouge-l'])

    print('\nresult-->\n', result)

    # 最后将计算评估结果写入文件中
    print('将评估结果写入结果文件中 ... ')
    with open('../eval_result/rouge_result.txt', 'a') as f:
        for r, metrics in result.items():
            f.write(r + '\n')
            for metric, value in metrics.items():
                f.write(metric + ': ' + str(value * 100) + '\n')

    print('产生机器摘要 评价机器摘要 End')


''' 
    result-->
     {'rouge-1': {'r': 0.18300240432773573, 'p': 0.244203019374846, 'f': 0.19141200254477497}, 
      'rouge-2': {'r': 0.014409722091920232, 'p': 0.014262259225645951, 'f': 0.013145457783872834}, 
      'rouge-l': {'r': 0.165243957964494, 'p': 0.2243682805842249, 'f': 0.17355817268990162}}
    将评估结果写入结果文件中 ... 


    # 写文件后 显示如下 
    rouge-1
    r: 29.059043018724147
    p: 41.315438899466606
    f: 30.715481766348407
    rouge-2
    r: 10.55261475764138
    p: 12.736431322799739
    f: 10.038140028495691
    rouge-l
    r: 26.806683753327416
    p: 38.08388072940074
    f: 28.270743901662097

'''
