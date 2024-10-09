import os
import sys
from rouge import Rouge

def dm01_test_rouge():

    # 人工摘要 S1: police killed the gunman
    # 人工摘要 S2: the gunman was shot down by police
    # 机器摘要 C1: police ended the gunman
    # 机器摘要 C2: the gunman murdered police
    s1 = 'police killed the gunman'
    c1 = 'police ended the gunman'

    s2 = 'the gunman was shot down by police'
    c2 = 'the gunman murdered police'

    # 计算字粒度的rouge-1、rouge-2、rouge-L
    rouge = Rouge()
    rouge_scores1 = rouge.get_scores(hyps=c1, refs=s1, avg=True)
    rouge_scores2 = rouge.get_scores(hyps=c2, refs=s2, avg=True)
    rouge_scores3 = rouge.get_scores(hyps=[c1, c2], refs=[s1, s2], avg=False)
    rouge_scores4 = rouge.get_scores(hyps=[c1, c2], refs=[s1, s2], avg=True) # avg=Ture 求平均

    print('\nrouge_scores1-->\n', rouge_scores1)
    print('\nrouge_scores2-->\n', rouge_scores2)
    print('\nrouge_scores3-->\n', rouge_scores3)
    print('\nrouge_scores4-->\n', rouge_scores4)

''' 
    rouge_scores1-->
     {'rouge-1': {'r': 0.75, 'p': 0.75, 'f': 0.749999995}, 
      'rouge-2': {'r': 0.3333333333333333, 'p': 0.3333333333333333, 'f': 0.3333333283333334}, 
      'rouge-l': {'r': 0.75, 'p': 0.75, 'f': 0.749999995}}
    
    rouge_scores2-->
     {'rouge-1': {'r': 0.42857142857142855, 'p': 0.75, 'f': 0.5454545408264463}, 
      'rouge-2': {'r': 0.16666666666666666, 'p': 0.3333333333333333, 'f': 0.22222221777777784}, 
      'rouge-l': {'r': 0.42857142857142855, 'p': 0.75, 'f': 0.5454545408264463}}
      
    rouge_scores3-->
     [{'rouge-1': {'r': 0.75, 'p': 0.75, 'f': 0.749999995}, 
       'rouge-2': {'r': 0.3333333333333333, 'p': 0.3333333333333333, 'f': 0.3333333283333334}, 
       'rouge-l': {'r': 0.75, 'p': 0.75, 'f': 0.749999995}}, 
      
      {'rouge-1': {'r': 0.42857142857142855, 'p': 0.75, 'f': 0.5454545408264463}, 
       'rouge-2': {'r': 0.16666666666666666, 'p': 0.3333333333333333, 'f': 0.22222221777777784}, 
       'rouge-l': {'r': 0.42857142857142855, 'p': 0.75, 'f': 0.5454545408264463}}]  
         
    rouge_scores4-->
     {'rouge-1': {'r': 0.5892857142857143, 'p': 0.75, 'f': 0.6477272679132231}, 
      'rouge-2': {'r': 0.25, 'p': 0.3333333333333333, 'f': 0.2777777730555556}, 
      'rouge-l': {'r': 0.5892857142857143, 'p': 0.75, 'f': 0.6477272679132231}}

    测试Rouge指标
'''

if __name__ == '__main__':
    dm01_test_rouge()
    print('测试Rouge Api测试 End')

