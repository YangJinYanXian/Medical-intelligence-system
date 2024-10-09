## **改进V1.2.2**

------

#### 改进点 使用生成式模型

- 第一: 采用PGN新的架构, 力求可以将描述病况细节的单词直接生成摘要.
- 第二: 如果模型具备第一点的能力, 可以copy原始文本的单词, 那么将极大的解决seq2seq的OOV问题.
- 第三: 引入机制来控制和跟踪原始文本的重复范围, 减少生成摘要的重复问题.

<img src=".\site\PGN.png" width = "100%" />



PGN架构的组成和运行机制**（1个指针，三个分布）**

-   GN架构图, 基本上是在seq2seq架构基础上多了一层
-   先计算出来一个指针p_gen
    -   用(1 - p_gen)乘以Attention Distribution, 得到原始文本的信息
    -   用p_gen乘以Vocabulary Distribution, 得到生成文本的信息
    -   这两部分加和, 得到Final Distribution

### 数据处理

##### 模型生成摘要文本的流程

```
1 输入一段sorce文本，对sorce进行encode编码，求中间语义张量C、最后一个隐藏层输出；
2 1个时间步1个时间步的生成摘要文本（eg：最大摘要长度40个，超过40没有结束就强制结束）
    2-1 attention求预测单词的注意力权重分布，注意力结果表示
    2-2 然后再decode解码 求 【预测单词的分布-最后的单词分布】
    
    在2-2环节中，PGN模型生成摘要文本的范围可以在字典中，也可以在oov列表中
    
3 所以PGN在解码的时，是需要原文source中的OOV单词；
    也就是每一段文本进行摘要先生成各自的oov单词列表
```

##### PGN数据的特殊性分析

产生的摘要单词如果不在字典中（ eg：20004个单词中的一个），则PGN模型就无法从原文中copy单词

为了让PGN模型能从原文中copy单词，需要把原文中的单词，组成oov列表

结论：每一篇原始文档如果要提取摘要，都是需要source2id，返回两个列表：word2id对应的列表，oov列表

这个工作需要通过DataLoader的自定义函数collate_fn函数来实现。

##### 工具函数

    原始文本映射成ids张量的函数source2ids(source_words, vocab)
    摘要文本映射成数字化ids函数abstract2ids(abstract_words, vocab, source_oovs)
    将输出张量ids结果映射成文本函数outputids2words(id_list, source_oovs, vocab)
    
    向小顶堆数据结构添加元素的函数add2heap(heap, item, k)
    
    将文本中OOV单词替换成的函数replace_oovs(in_tensor, vocab)
    获取模型中超参数信息的函数config_info(config)
### 模型评估

目标：同时观察 训练集 和 验证集 的损失变化情况，为训练模型提供决策

思路：

- 创建PGN模型对象，初始化字典类

- 加载已经训练好的模型

- 调用评估函数，对验证器数据进行摘要，并计算验证集上的loss

  - DataLoader方式，按照加载验证集数据

  - 循环送入model进行评估，并进行损失计算

    - ```
       遍历测试数据进行评估
      for batch, data in enumerate(tqdm(val_dataloader)):
          x, y, x_len, y_len, oov, len_oovs = data
          if d10_config.is_cuda:
              x = x.to(DEVICE)
              y = y.to(DEVICE)
              x_len = x_len.to(DEVICE)
              len_oovs = len_oovs.to(DEVICE)
          total_num = len(val_dataloader)
      
          loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=total_num, teacher_forcing=True)
      ```



### 模型训练

-   思路：
    -   创建PGN模型对象，初始化字典类，加载训练集DataLoader、验证器DataLoader
    -   外层for循环 控制epoch多少轮，内层for循环 控制多少批次数，进行模型训练
        -   去一个批次的数据
        -   梯度清零
        -   给模型喂批次数据，求损失
        -   反向传播、更新梯度
    -   其他辅助打印信息：每100个批次辅助信息损失信息；每轮训练打印损失信息



模型训练 1个epoch 1小时



<img src=".\site\train1.png" width = "100%" />

### 5 模型预测

-   思路：
    -   随机取一条测试数据
    -   将原始文本映射成数字化张量（source2ids()) 得到word2id列表、oov列表
    -   将数据通过encode进行编码，得到中间语义张量C、最后隐藏层输出
    -   对数据按照时间步进行解码，生成摘要信息
        -   每一个时间步：通过注意力机制，查询张量q的注意力权重、q的注意力结果表示
        -   每一个时间步：通过decode解码，求查询张量的最终单词分布
        -   每一个时间步：通过贪心策略，进行字符预测
        -   最终循环结束得到：解码后的摘要数据
    -   解码后的摘要数据进行id2word，文本进行显示



## **初代V1.1.15**

------



## 效果图：

<img src=".\xiaoguotu.png" width = "60%" />


## 流程图：

<img src=".\site\整体架构.svg" width = "150%" />

##  目录路径：

<img src=".\site\image-2.png" width = "60%" />


### 1、 linux端部署梳理

#### 1 确定所有服务都关闭，以防干扰

<img src=".\site\image-1.png" width = "80%" />

#### 2 把代码上传到工作目录

```shell
/root/ai13bj
```



#### 3 配置werobot服务

```shell
/root/ai13bj/wr.py # werobot 公众号后台程序，进行转发，把用户的消息发送到主要逻辑服务
```

在/etc/supervisord.conf添加以下内容

```properties
[program:werobot]
# python的路径是激活虚拟环境后，通过 which python 命令查找的
# /root/ai13bj/ 项目路径，以实际为准
command=/root/anaconda3/envs/ai_doctor/bin/python wr.py
directory=/root/ai13bj/
stopsignal=QUIT               ; signal used to kill process (default TERM)
stopasgroup=false             ; send stop signal to the UNIX process group (default false)
killasgroup=false             ; SIGKILL the UNIX process group (def false)
stdout_logfile=/var/log/werobot_out      ; stdout log path, NONE for none; default AUTO
stdout_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
stderr_logfile=/var/log/werobot_error     ; stderr log path, NONE for none; default AUTO
stderr_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
```

```shell
# 通过命令启动supervisord
systemctl start supervisord
# 进入supervisorctl中
status #命令 查看当前运行的
werobot                          RUNNING   pid 6064, uptime 0:05:51
supervisor> 
```

此时向公众号发送消息得到的回复：“机器人客服正在休息，请稍后再试...”

经过分析，发现在发送post请求时发生异常，原因时请求的主要逻辑服务没有启动



#### 4 启动主逻辑服务

```properties
# 配置/etc/supervisord.conf
# gunicorn的路径是激活虚拟环境后，通过 which gunicorn 命令查找的
# /root/ai13bj/ 项目路径，以实际为准
[program:main_serve]
command=/root/anaconda3/envs/ai_doctor/bin/gunicorn -w 1 -b 0.0.0.0:5000 app:app                    ; the program (relative uses PATH, can take args)
directory=/root/ai13bj/doctor_online/main_serve
stopsignal=QUIT               ; signal used to kill process (default TERM)
stopasgroup=false             ; send stop signal to the UNIX process group (default false)
killasgroup=false             ; SIGKILL the UNIX process group (def false)
stdout_logfile=/var/log/main_serve_out      ; stdout log path, NONE for none; default AUTO
stdout_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
stderr_logfile=/var/log/main_serve_error        ; stderr log path, NONE for none; default AUTO
stderr_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)

# 重启supervisor服务
(ai_doctor) root@aliyun1:~/ai13bj# systemctl restart supervisord
(ai_doctor) root@aliyun1:~/ai13bj# supervisorctl 
main_serve                       RUNNING   pid 6143, uptime 0:00:08
werobot                          RUNNING   pid 6144, uptime 0:00:08
supervisor> 
```

此时向公众号发送消息，微信界面中得到错误：500 Internal Server Error

查看日志

```shell
(ai_doctor) root@aliyun1:~/ai13bj# cat /var/log/main_serve_error 
[2022-07-01 16:15:26 +0800] [6143] [INFO] Starting gunicorn 20.1.0
[2022-07-01 16:15:26 +0800] [6143] [INFO] Listening at: http://0.0.0.0:5000 (6143)
[2022-07-01 16:15:26 +0800] [6143] [INFO] Using worker: sync
[2022-07-01 16:15:26 +0800] [6146] [INFO] Booting worker with pid: 6146
[2022-07-01 16:16:12,414] ERROR in app: Exception on /v1/main_serve/ [POST]
Traceback (most recent call last):
  File "/root/anaconda3/envs/ai_doctor/lib/python3.8/site-packages/redis/connection.py", line 609, in connect
    sock = self.retry.call_with_retry(
  File "/root/anaconda3/envs/ai_doctor/lib/python3.8/site-packages/redis/retry.py", line 46, in call_with_retry
    return do()
  File "/root/anaconda3/envs/ai_doctor/lib/python3.8/site-packages/redis/connection.py", line 610, in <lambda>
    lambda: self._connect(), lambda error: self.disconnect(error)
  File "/root/anaconda3/envs/ai_doctor/lib/python3.8/site-packages/redis/connection.py", line 675, in _connect
    raise err
  File "/root/anaconda3/envs/ai_doctor/lib/python3.8/site-packages/redis/connection.py", line 663, in _connect
    sock.connect(socket_address)
ConnectionRefusedError: [Errno 111] Connection refused

redis.exceptions.ConnectionError: Error 111 connecting to 0.0.0.0:6379. Connection refused.

# 原因就是redis没启动
```

#### 5 启动redis

第一种直接在命令行中启动，终端挂了，服务就挂了

```properties
# 配置/etc/supervisord.conf，添加以下内容
[program:redis]
command=redis-server

# 重启supervisor服务
(ai_doctor) root@aliyun1:~/ai13bj# systemctl restart supervisord
(ai_doctor) root@aliyun1:~/ai13bj# supervisorctl 
main_serve                       RUNNING   pid 6309, uptime 0:00:11
redis                            RUNNING   pid 6310, uptime 0:00:11
werobot                          RUNNING   pid 6311, uptime 0:00:11
supervisor> 
```

继续向公众号发送消息，此时随便发送的字符串可以得到unit闲聊机器人返回的信息。并且发送疾病症状的消息，也被闲聊。

也就是并么有走到正常功能的分支。进一步查看main_serve的日志

```shell
(ai_doctor) root@aliyun1:~/ai13bj# tail -F /var/log/main_serve_error 
  File "/root/anaconda3/envs/ai_doctor/lib/python3.8/site-packages/neo4j/io/__init__.py", line 692, in _acquire
    connection = self.opener(address, timeout)
  File "/root/anaconda3/envs/ai_doctor/lib/python3.8/site-packages/neo4j/io/__init__.py", line 817, in opener
    return Bolt.open(
  File "/root/anaconda3/envs/ai_doctor/lib/python3.8/site-packages/neo4j/io/__init__.py", line 319, in open
    s, pool_config.protocol_version, handshake, data = connect(
  File "/root/anaconda3/envs/ai_doctor/lib/python3.8/site-packages/neo4j/io/__init__.py", line 1391, in connect
    raise ServiceUnavailable(
neo4j.exceptions.ServiceUnavailable: Couldn't connect to 0.0.0.0:7687 (resolved to ('0.0.0.0:7687',)):
Failed to establish connection to ResolvedIPv4Address(('0.0.0.0', 7687)) (reason [Errno 111] Connection refused)

# 7687 是neo4j的端口
```

#### 6 配置neo4j

```properties
# 配置/etc/supervisord.conf，添加以下内容
[program:neo4j]
command=neo4j console
user=neo4j
autostart=true
autorestart=unexpected
startsecs=30
startretries=999
priorities=90
exitcodes=0,1,2
stopsignal=SIGTERM
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile=/var/log/neo4j/neo4j.out
stdout_logfile_backups=10
stderr_capture_maxbytes=20MB

# 重启neo4j
(ai_doctor) root@aliyun1:~/ai13bj# systemctl restart supervisord
(ai_doctor) root@aliyun1:~/ai13bj# supervisorctl 
main_serve                       RUNNING   pid 6364, uptime 0:00:04
neo4j                            STARTING  
redis                            RUNNING   pid 6366, uptime 0:00:04
werobot                          RUNNING   pid 6365, uptime 0:00:04
supervisor> status
main_serve                       RUNNING   pid 6364, uptime 0:00:15
neo4j                            RUNNING   pid 6367, uptime 0:00:45  
redis                            RUNNING   pid 6366, uptime 0:00:15
werobot                          RUNNING   pid 6365, uptime 0:00:15

```

#### 7 配置句子相关模型

```properties
# 配置/etc/supervisord.conf
# gunicorn的路径是激活虚拟环境后，通过 which gunicor 命令查找的
# /root/ai13bj/ 项目路径，以实际为准
[program:bert_serve]
command=/root/anaconda3/envs/ai_doctor/bin/gunicorn -w 1 -b 0.0.0.0:5001 app:app                    ; the program (relative uses PATH, can take args)
directory=/root/ai13bj/doctor_online/bert_serve
stopsignal=QUIT               ; signal used to kill process (default TERM)
stopasgroup=false             ; send stop signal to the UNIX process group (default false)
killasgroup=false             ; SIGKILL the UNIX process group (def false)
stdout_logfile=/var/log/bert_serve_out      ; stdout log path, NONE for none; default AUTO
stdout_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
stderr_logfile=/var/log/bert_serve_error        ; stderr log path, NONE for none; default AUTO
stderr_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)

# 重启supervisord
(ai_doctor) root@aliyun1:~/ai13bj# systemctl restart supervisord
(ai_doctor) root@aliyun1:~/ai13bj# supervisorctl 
bert_serve                       RUNNING   pid 6543, uptime 0:00:05
main_serve                       RUNNING   pid 6542, uptime 0:00:05
neo4j                            STARTING  
redis                            RUNNING   pid 6544, uptime 0:00:05
werobot                          RUNNING   pid 6545, uptime 0:00:05
supervisor> 


```

#### 8 通过命令查看端口监听情况

```shell
(ai_doctor) root@aliyun1:~/ai13bj# lsof -i:5000
COMMAND   PID USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
gunicorn 6542 root    5u  IPv4 3729945      0t0  TCP *:commplex-main (LISTEN)
gunicorn 6596 root    5u  IPv4 3729945      0t0  TCP *:commplex-main (LISTEN)
(ai_doctor) root@aliyun1:~/ai13bj# lsof -i:5001
COMMAND   PID USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
gunicorn 6543 root    5u  IPv4 3729956      0t0  TCP *:commplex-link (LISTEN)
gunicorn 6603 root    5u  IPv4 3729956      0t0  TCP *:commplex-link (LISTEN)
(ai_doctor) root@aliyun1:~/ai13bj# lsof -i:80
COMMAND    PID USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
AliYunDun 1267 root   17u  IPv4   21117      0t0  TCP iZt4nfthzkc0agfos0w2epZ:50504->100.103.15.60:http (ESTABLISHED)
python    6545 root    3u  IPv4 3731155      0t0  TCP *:http (LISTEN)
(ai_doctor) root@aliyun1:~/ai13bj# lsof -i:6379
COMMAND    PID USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
redis-ser 6544 root    4u  IPv6 3731136      0t0  TCP *:6379 (LISTEN)
redis-ser 6544 root    5u  IPv4 3731137      0t0  TCP *:6379 (LISTEN)
(ai_doctor) root@aliyun1:~/ai13bj# lsof -i:7687
COMMAND  PID  USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
java    6546 neo4j  258u  IPv4 3729966      0t0  TCP *:7687 (LISTEN)

# 使用的端口都是正常被监听的，所以服务是正常的
```

### 2、离线部分代码流程

三个模型

**第一个模型**

doctor_offline/ner_model(命名实体识别模型，BiLSTM+CRF)

第一步准备预料库：build_vocab.py load_corpus.py encode_label.py

第二步构建模型：bilstm_crf.py

第三步训练模型：train.py

第四步评估模型：evaluate.py

第五步应用模型进行实体提取：entity_extract.py 会把unstructrued/norecognite/xxx.txt进行实体提取，把结果写入到structured/noreview/xxx.csv（这里就包括爬虫爬取的结构化数据和通过模型提取的结构化数据）

**第二个模型**

doctor_offline/review_model(命名实体审核模型，Bert预训练+RNN)

第一步构造bert预训练模型：bert_chinese_encode.py

第二步构造RNN模型：RNN_MODEL.py

第三步训练模型：train.py

第四步应用模型：predict.py 会把 structured/noreview/xxx.csv文件进行审核，结果写入到structrued/reviewed/xxx.csv

接着通过 neo4j_write.py把structrued/reviewed/xxx.csv写入到图数据库中

**第三个模型**

doctor_online/ber_serve（句子相关模型，Bert预训练+微调的模型）这个模型是离线训练，在在线流程中使用的

第一步构造bert预训练模型：bert_chinese_encode.py

第二个构造微调模型：finetuning_net.py

第三步训练模型：train.py

第四步部署模型：app.py 通过gunicorn部署（supervisord配置），就提供了在线接口 '/v1/recognition/'

第四步测试模型：test.py

### 3、在线部分代码流程

第一步：部署werobot服务，启动 项目/wr.py （supervisord配置）

第二步：部署主要逻辑服务，启动 项目/doctor_online/main_serve/app.py（supervisord配置）

第三步：启动redis neo4j（supervisord配置）



