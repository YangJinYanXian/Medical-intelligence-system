# 导入系统工具包
import os
import sys

# 设置项目的root路径, 方便后续相关代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入若干工具包
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入项目中的相关代码文件
import d10_config
from d14_func_utils import timer, replace_oovs
from m11_vocab import Vocab
from m12_dataset import PairDataset, SampleDataset, collate_fn
from torch.utils.data import DataLoader

# 编码器类 Encoder 实现思路分析  Encoder(20004, 512, 512, 0)
# init函数 (self, vocab_size, embed_size, hidden_size, rnn_drop=0)
    # super() 词嵌入层self.embedding 隐藏层属性self.hidden_size  # 双向LSTM self.lstm
    # nn.LSTM(embed_size, hidden_size, bidirectional=True, dropout=rnn_drop, batch_first=True)
# forward函数 (self, x)
    # 数据经过词嵌入层 [8,133]->[8,133,512]
    # 数据经过双向lstm [8,133,512] --> output[8,133,1024] hidden([2,8,512],[2,8,512])

    # 每一句话(1个样本)最后的隐藏层输出、细胞状态输出：有一个hn、一个cn
    # 每个hn有2个   从左到右的hn[1,8,512] 从右向左的hn[1,8,512] 所以：[2,8,512]
    # 每个cn有2个   从左到右的cn[1,8,512] 从右向左的cn[1,8,512] 所以：[2,8,512]

    # 返回 output:[8,251,1024] hidden：tuple(hn,cn) ([2,8,512], [2,8,512])

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_drop=0):
        super(Encoder, self).__init__()

        # 词嵌入层采用跟随模型一起训练的模式
        self.embedding = nn.Embedding(vocab_size, embed_size)   # nn.Embedding(20004,512)
        self.hidden_size = hidden_size      # 512

        # 编码器的主体采用单层, 双向LSTM结构  # nn.LSTM(512, 512)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, dropout=rnn_drop, batch_first=True)

    def forward(self, x):

        # [batch_size, seq_len] eg:[8,133] ---> [8,133,512]
        embedded = self.embedding(x)

        # output, (hn, cn) = nn.LSTM(input, (h0, c0))
        # eg: [8,133,512] --> [8,133,1024]
        # 注意是双向LSTM，hidden_size是512，输出为2*512=1024
        output, hidden = self.lstm(embedded)

        # 若有8句话(8个样本), 每个样本251个单词, 经过双向LSTM处理后, 特征变化为:
        # 每一句话(1个样本)最后的隐藏层输出、细胞状态输出：有一个hn、一个cn
        # 每个hn有2个   从左到右的hn[1,8,512] 从右向左的hn[1,8,512] 所以：[2,8,512]
        # 每个cn有2个   从左到右的cn[1,8,512] 从右向左的cn[1,8,512] 所以：[2,8,512]
        # 最终返回值：output: [8,251,1024]    hidden：tuple(hn,cn) eg: ( [2,8,512], [2,8,512] )
        return output, hidden #

'''
1 输入：
    encode编码函数输入一个[8,133] 
2 输出：
    myoutput[8,133,1024]    中间语义张量C(双向)
    hn: [2,8,512]           每一句话的隐藏层状态(双向)
    cn: [2,8,512]           每一句话的细胞状态(双向)
'''
# 1 实例化Encoder对象 myEncoder
# 2 准备数据 torch.ones((8, 133), dtype=torch.long)
# 3 给模型喂数据   myoutput myhidden[0] myhidden[1]
def dm01_test_Encoder():

    # 1 实例化Encoder对象
    myEncoder = Encoder(vocab_size=20004, embed_size=512, hidden_size=512)
    print('myEncoder-->', myEncoder)

    # 2 准备数据
    myx = torch.ones((8, 133), dtype=torch.long)

    # 3 给模型喂数据
    myoutput, myhidden = myEncoder(myx)
    print('注意返回值：')
    print('myoutput--->', myoutput.shape)
    print('myhidden[0]--->', myhidden[0].shape)
    print('myhidden[1]--->', myhidden[1].shape)

    ''' 
        myoutput--> torch.Size([8, 133, 1024])    # 双向特征 512 + 512 = 1024
        myhidden[0]--> torch.Size([2, 8, 512])      # 隐藏层 返回值是touble {tuple:2}
        myhidden[1]--> torch.Size([2, 8, 512])
    '''


# 构建注意力类 Attention(nn.Module) 实现思路分析
# init函数 (self, hidden_units)
    # super()  self.Wh  self.Ws  self.v
    # 对最后一个隐藏层输出(hn cn)进行线性变换层 self.Wh  nn.Linear(2 * hidden_units, 2 * hidden_units, bias=False)
    # 对所有隐藏层输出hn-中间语义张量c进行线性变换层 self.Ws  公式1
    # 对加和后的数据箭头1方向进行变换层  self.v nn.Linear(2 * hidden_units, 1, bias=False) 公式2

# forward函数 (self, decoder_states, encoder_output, x_padding_masks)
    # 数据形状(decoder_states(h:[1,8,512] c:[1,8,512]), encoder_output[8,203,1024], x_padding_masks[8,203])

    # 对decoder_states双向元组拆 再进行cat操作 s_t = torch.cat([h_dec, c_dec], dim=2)
    # 按照hi的维度扩展St的维度 # eg:[8,1,1024] -> [8,203,1024] s_t.transpose(0, 1) s_t.expand_as(encoder_output).contiguous()
    # encoder_output经过Wh层 [8,203,1024] -> [8,203,1024] encoder_features = self.Wh(encoder_output.contiguous())
    # s_t经过Ws层 [8,203,1024] -> [8,203,1024] decoder_features = self.Ws(s_t)
    # 公式1 加和 attn_inputs = encoder_features + decoder_features
    # 公式2 执行tanh运算和一个全连接层的运算  score = self.v(torch.tanh(attn_inputs))
    # 转成概率分布 [8,203,1] -> [8,203]  attention_weights = F.softmax(score, dim=1).squeeze(2)

    # 对pad权重进行修正
    # 对pad权重屏蔽 [8,203] -> [8,203] attention_weights = attention_weights * x_padding_masks
    # 求每一句真实句子长度 #[8,203] -> [8,1] normalization_factor = attention_weights.sum(1, keepdim=True)
    # 最权重修正 # [8,203] -> [8,203] attention_weights = attention_weights / normalization_factor

    # 做bmm运算 (8,1,203) * (8,203,1024) -> (8,1,1024)
    # context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output) / context_vector.squeeze(1)
    # 返回 context_vector: [8,1024] [batchsize,2*hidden_units特征数] attention_weights[8,203]
class Attention(nn.Module):
    def __init__(self, hidden_units):

        super(Attention, self).__init__()

        # 定义前向传播层, 对应论文中的公式1中的Wh, Ws
        self.Wh = nn.Linear(2 * hidden_units, 2 * hidden_units, bias=False) # Wh: nn.Linear(2*512, 2*512) --> (1024, 1024)
        self.Ws = nn.Linear(2 * hidden_units, 2 * hidden_units) # Ws: nn.Linear(2*512, 2*512)  ---> (1024, 1024)

        # 定义全连接层, 对应论文中的公式1中最外层的v
        # V: nn.Linear(2*512, 1), nn.Linear(1024, 1)
        self.v = nn.Linear(2 * hidden_units, 1, bias=False)

    # # q != K = V
    # q：decoder_states  解码器的隐藏层状态
    # k：encoder_output  编码结果 中间语义张量
    # v：encoder_output  编码结果 中间语义张量
    def forward(self, decoder_states, encoder_output, x_padding_masks):
        # Define forward propagation for the attention network.
        h_dec, c_dec = decoder_states # h:[1,8,512] c:[1,8,512]

        # 将两个张量在最后一个维度拼接, 得到deocder state St: (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2) # [1,8,512]+[1,8,512] ---> [1,8,1024]

        # 将batch_size置于第一个维度上: (batch_size, 1, 2*hidden_units)
        s_t = s_t.transpose(0, 1)   #[1,8,1024] ---> [8,1,1024]

        # 按照 hi 的维度扩展St的维度: (batch_size, seq_length, 2*hidden_units) # eg:[8,1,1024] -> [8,203,1024]
        s_t = s_t.expand_as(encoder_output).contiguous() # 调用contiguous，重新copy一份让内存变得连续

        # 根据论文中的公式1来计算et, 总共有三步
        # 第一步: 分别经历各自的全连接层矩阵乘法
        # Wh * h_i: (batch_size, seq_length, 2*hidden_units)
        # eg:encoder_output数据形状变化: [8,203,1024] -> [8,203,1024]
        encoder_features = self.Wh(encoder_output.contiguous())
        # Ws * s_t: (batch_size, seq_length, 2*hidden_units)
        # eg: s_t的数据形状变化: [8,203,1024] -> [8,203,1024]
        decoder_features = self.Ws(s_t)

        # 第二步: 两部分执行加和运算
        # (batch_size, seq_length, 2*hidden_units) eg:[8,203,1024] + [8,203,1024] -> [8,203,1024]
        attn_inputs = encoder_features + decoder_features

        # 第三步: 执行tanh运算和一个全连接层的运算
        # (batch_size, seq_length, 1)
        # [8,203,1024] * [8,1024,1] -> [8,203,1]
        score = self.v(torch.tanh(attn_inputs)) # eg: attn_inputs:[8,203,1024] -> [8,203,1]

        # 得到score后, 执行论文中的公式2
        # (batch_size, seq_length) eg: [8,203,1] -> [8,203]
        attention_weights = F.softmax(score, dim=1).squeeze(2)

        # 添加一步执行padding mask的运算, 将编码器端无效的PAD字符全部遮掩掉 eg: [8,203] * [8, 203] -> [8,203]
        attention_weights = attention_weights * x_padding_masks

        # 整个注意力层执行一次正则化操作 （归一化）
        # 注意：attention_weights做了mask操作后，权重分布发生变化了，需要重新做归一化
        normalization_factor = attention_weights.sum(1, keepdim=True) # [8,203] -> [8,1]
        attention_weights = attention_weights / normalization_factor #[8,203] -> [8,203]
        
        # 执行论文中的公式3,将上一步得到的attention distributon应用在encoder hidden states上,得到context_vector
        # bmm运算：(batch_size, 1, seq_length) * (batch_size, seq_length, hidden_units) ===> (batch_size, 1, 2*hidden_units)
        # (8, 1, 203) * (8, 203, 1024) ===> (8, 1,1024)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output)
        # (batch_size, 2*hidden_units) [8,1,1024] ->[8,1024]
        context_vector = context_vector.squeeze(1)

        # context_vector: [8,1024] [batchsize,1,2*hidden_units特征数]   attention_weights[8,203] [batchsize,seq_len]
        return context_vector, attention_weights

'''
# 注意力机制的测试函数

1 输入查询张量q：hn[1, 8, 512]，cn[1, 8, 512] 得到 1-1, 1-2
    1-1: q的注意力权重分布：[8,203]
            也就是输入1个q，得到1个203个单词的注意力权重分布，输入8个词得到8个203个注意力权重分布
    1-2: q的注意力结果表示[8,1024]
            也就是输入1个q，得到1个targe单词；输入8个q得到8个target
    
2 从q的查询张量，融合k信息，融合了v信息，q查询张量带了一个"有色眼镜"，更加注意关键信息！
    q：hn[1, 8, 512]，cn[1, 8, 512]相当于q:[8,1024]  ====> 得到注意结果表示还是 context_vector:[8,1024] 
    把context_vector交给解码器Decoder，要比把q交给解码器Decoder效果要好的多！

3 注意力机制的本质：把精力放在事务聚焦的地方
    本机制结合反向传播，可快速的寻找到哪些特征对自己有用！从而快速的提高神经网络的推理质量！
    
'''

# 1 实例化注意力机制对象 myAttention
# 2 准备数据 decoder_states ([1,8,512],[1,8,512])  encoder_output [8,203,512*2]  x_padding_masks [8,203]
# 3 给模型qkv计算 --> # context_vector[8,1024]  attention_weights[8,203]
# 4 打印 注意力权重分布 注意力结果表示
def dm02_test_Attention():

    torch.set_printoptions(sci_mode=False)

    # 1 实例化注意力机制对象 myAttention
    myAttention = Attention(512)
    print('myAttention->', myAttention)

    # 2 准备数据 decoder_states ([1,8,512],[1,8,512])  encoder_output [8,203,512*2]  x_padding_masks [8,203]
    decoder_states = (torch.randn(1, 8, 512), torch.randn(1, 8, 512))
    encoder_output = torch.randn(8, 203, 512 * 2)
    x_padding_masks = torch.randn(8, 203)

    # 3 给模型qkv计算 --> # context_vector[8,1024]  attention_weights[8,203]
    context_vector, attention_weights = myAttention(decoder_states, encoder_output, x_padding_masks)

    # 4 打印 注意力权重分布 注意力结果表示
    print('context_vector-->', context_vector.shape)        #  torch.Size([8, 1024])
    print('attention_weights-->', attention_weights.shape)  # torch.Size([8, 203])
    pass


# 构建解码器 Decoder(nn.Module) 实现思路分析
# init函数 (self, vocab_size, embed_size, hidden_size, enc_hidden_size=None) # super()
    # 参数数据提示: vocab_size:20004  embed_size:512 hidden_size:512
    # 词嵌入层self.embedding 属性self.vocab_size 属性self.hidden_size self.lstm层
    # 公式4 线性层self.W1 (512*3,512) nn.Linear(self.hidden_size*3, self.hidden_size)
    # 公式4-5 线性层self.W2 (512,20004) nn.Linear(self.hidden_size, vocab_size) # (512,20004)
    # 公式8 self.w_gen #(1024+1024+512, 1) nn.Linear(self.hidden_size*4 + embed_size,1) / if d10_config.pointer:

# forward函数 (self, x_t, decoder_states, context_vector)
# 每个时间步数据x_t[8,1], 上一个时间步状态decoder_states ([1,8,512],[1,8,512], 注意力q context_vector[8,1024]
# 1 对x_t 经过词嵌入层 再进行lstm操作 ：
    # 1-1 x_t 经过词嵌入self.embedding(x_t)得到 decoder_emb
    # 1-2 lstm(x_t[8,1,512], decoder_states([1,8,512],[1,8,512]))->decoder_output[8,1,512], decoder_states([1,8,512],[1,8,512])
# 2 根据公式4 调整decoder_states维度与context_vector, 再经过2个线性层
    # decoder_states调整维度[8,1,512]->[8,512]  decoder_output.view(-1, d10_config.hidden_size)
    # concat_vector ：[8,512],[8,1024]->[8,512*3] torch.cat([decoder_output, context_vector], dim=-1)
    # 经过线性层self.W1得到 FF1_out [8,512*3]->[8,512]  self.W1(concat_vector)
    # 经过线性层self.W2得到 FF2_out [8,512] ->[8,20004] self.W2(FF1_out)
# 3 根据公式5 转化为vocabulary distribution P_vocab概率分布 p_vocab = F.softmax(FF2_out, dim=1)
# 4 根据公式8
    # 准备s_t decoder_states ([1,8,512],[1,8,512]) / s_t=torch.cat([h_dec,c_dec],dim=2)
    # 准备x_gen cat context_vector[8,1024] s_t [8,1024] decoder_emb[8,512] -> x_gen[8,2560] 512*5
        # x_gen = torch.cat([context_vector, s_t.squeeze(0), decoder_emb.squeeze(1)], dim=-1)
    # 经过线性层[8,2560]->[8,1] p_gen = torch.sigmoid( self.w_gen(x_gen) ) # 注意是sigmoid 把数据转到01区间
# 5 返回  p_vocab[8,20004], decoder_states([1,8,512],[1,8,512]), p_gen[8,1]
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden_size=None):
        super(Decoder, self).__init__()

        # 解码器端也采用跟随模型一起训练的方式, 得到词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size) # (20004, 512)
        self.vocab_size = vocab_size        # 20004
        self.hidden_size = hidden_size      # 512

        # 解码器的主体结构采用单向LSTM, 区别于编码器端的双向LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True) # (512, 512, batch_first=True)

        # 因为要将decoder hidden state 和 context vector进行拼接, 因此需要3倍的hidden_size维度设置
        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size) # (512*3, 512) = (1536, 512)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)           # (512, 20004)

        # 若 有指针 P_gen
        if d10_config.pointer:
            # 因为要根据论文中的公式8进行运算, 所谓输入维度上匹配的是4 * hidden_size + embed_size
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1) # (512*4+512, 1) = (2560, 1)

    def forward(self, x_t, decoder_states, context_vector):
        # 首先计算Decoder的前向传播输出张量  # x_t:[8, 1] ---> decoder_emb:[8, 1, 512]
        decoder_emb = self.embedding(x_t)

        # lstm( [8,1,512], ([1,8,512],[1,8,512]) ) -> decoder_output [8,1,512], decoder_states ([1,8,512],[1,8,512])
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # 接下来就是论文中的 公式4 的计算.
        # decoder_output :view操作 [8,1,512] -> [8, 512]
        decoder_output = decoder_output.view(-1, d10_config.hidden_size)
        # 将 context vector 和 decoder state 进行拼接, (batch_size, 3*hidden_units)
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1) # concat_vector：[8, 512*3]

        # 经历两个全连接层V和V'后,再进行softmax运算, 得到 vocabulary distribution
        # (batch_size, hidden_units)   eg: [8,1536] * [1536,512] -> [8,512]
        FF1_out = self.W1(concat_vector)
        # (batch_size, vocab_size)  eg: [8,512] ->[8,20004]
        FF2_out = self.W2(FF1_out)

        # 公式5：转化为vocabulary distribution P_vocab概率分布
        # (batch_size, vocab_size)  eg: [8,20004] ->[8,20004]
        p_vocab = F.softmax(FF2_out, dim=1)

        # 构造decoder state s_t. eg: decoder_states ([1,8,512],[1,8,512])
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_units) eg: ( [1,8,512],[1,8,512] ) -> [1,8,1024]
        s_t = torch.cat([h_dec, c_dec], dim=2)

        # p_gen是通过 context vector h_t, decoder state s_t, decoder input x_t, 三个部分共同计算出来的.
        # 下面的部分是计算论文中的 公式8
        p_gen = None
        if d10_config.pointer:
            # 这里面采用了直接拼接3部分输入张量, 然后经历一个共同的全连接层w_gen, 和原始论文的计算不同.
            # 这也给了大家提示, 可以提高模型的复杂度, 完全模拟原始论文中的3个全连接层来实现代码.
            # context_vector:[8,1024]  s_t:[1,8,1024]  decoder_emb:[8,1,512]  -> x_gen: [8, 2560]
            x_gen = torch.cat([context_vector, s_t.squeeze(0), decoder_emb.squeeze(1)], dim=-1)
            # w_gan_linear [8,2560]*[2560,1] -> [8, 1]
            p_gen = torch.sigmoid( self.w_gen(x_gen) )  # 注意是：二分类的sigmoid （两个分布的权重）

        return p_vocab, decoder_states, p_gen

'''
解码器业务单元测试函数
1 输入：
        decoder input x_t ：解码器端的摘要文本 eg：一个样本输入1个字符 8个样本输入8个字符
        decoder state s_t ：lstm解码器最后的隐藏层输出hn cn eg:([1, 8, 512], [1, 8, 512]
        context vector h_t：编码器端的注意力结果表示 eg: 8个查询张量有8个结果表示 [8, 1024]
2 输出：
        摘要文本的概率分布 p_vocab ：          torch.Size([8, 20004])
        本次操作的隐藏层输出 decoder_states ：  torch.Size([1, 8, 512]) torch.Size([1, 8, 512])
        原始文本和摘要文本产生概率 generation probability(p_gen pgn指针) ： torch.Size([8, 1])
        
'''

''' 
# 数据形状 myDecoder([8,1], decoder_states([1, 8, 512], [1, 8, 512], context_vector[8,1024]->
#               p_vocab[8,1024], decoder_states([1,8,512],[1,8,512]), p_gen[8,1]
p_vocab, decoder_states, p_gen = myDecoder(x_t, decoder_states, context_vector)   
'''


# 数据形状 myDecoder(x_t[8,1], decoder_states([1, 8, 512], [1, 8, 512], context_vector[8,1024]->
    #               p_vocab[8,1024], decoder_states([1,8,512],[1,8,512]), p_gen[8,1]
# 1 实例化解码器对象
# 2 准备数据  x_t[8,1]  decoder_states([1,8,512],[1,8,512])  context_vector[8,1024]
# 3  给模型喂数据 单个时间步解码 p_vocab[8,1024], decoder_states([1,8,512],[1,8,512]), p_gen[8,1]
def dm03_test_Decoder():

    vocab_size ,embed_size,hidden_size = 20004, 512, 512

    # 1 实例化解码器对象
    myDecoder = Decoder(vocab_size, embed_size, hidden_size)

    # 2 准备数据  x_t[8,1]  decoder_states([1,8,512],[1,8,512])  context_vector[8,1024]
    # 2-1 每个时间步数据x_t[8,1] 类型torch.long
    x_t = torch.ones([8, 1], dtype=torch.long) # 8个样本，每个样本的其实开始start

    # 2-2  上一个时间步状态 decoder_states ([1,8,512],[1,8,512],
    decoder_states_hn = torch.randn([1, 8, 512])
    decoder_states_cn = torch.randn([1, 8, 512])
    decoder_states = (decoder_states_hn, decoder_states_cn) #  ([1, 8, 512], [1, 8, 512]
    # 2-3 注意力q context_vector[8,1024]
    context_vector = torch.randn( [8, 1024] )

    # 3 给模型喂数据 单个时间步解码 p_vocab[8,1024], decoder_states([1,8,512],[1,8,512]), p_gen[8,1]
    p_vocab, decoder_states, p_gen = myDecoder(x_t, decoder_states, context_vector)
    print('摘要文本的概率分布 p_vocab', p_vocab.shape)
    print('本次操作的隐藏层输出 decoder_states', len (decoder_states),decoder_states[0].shape, decoder_states[1].shape )
    print('原始文本和摘要文本产生概率 generation probability (p_gen) ', p_gen.shape)

    '''
        摘要文本的概率分布p_vocab ：          torch.Size([8, 20004])
        本次操作的隐藏层输出decoder_states ：  torch.Size([1, 8, 512]) torch.Size([1, 8, 512])
        原始文本和摘要文本产生概率generation probability(p_gen pgn指针) ： torch.Size([8, 1])
    '''


# 构造降维加和类 ReduceState(nn.Module), 把双向lstm的hn cn进行降维加和
# init(self)函数 : super()
# forward(self, hidden)
    # h, c = hidden hidden+hn((2,8,512),(2,8,512)) -> ((1,8,512),(1,8,512))
    # 第一个维度相加 hn:[2, 8, 512] -> [1, 8, 512]
    # 第一个维度相加 cn:[2, 8, 512] -> [1, 8, 512]
    # h_reduced = torch.sum(h, dim=0, keepdim=True)
    # c_reduced = torch.sum(c, dim=0, keepdim=True)
    # return (h_reduced, c_reduced)
class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)

# 把隐藏层输出进行数据压缩，压缩过程中保持维度不变 都是三维
#   hn:[2, 8, 512] -> [1, 8, 512]
#   cn:[2, 8, 512] -> [1, 8, 512]
def dm04_test_ReduceState():

    # myreducestate = ReduceState()
    # h, c = myreducestate((torch.randn(2, 8, 512), torch.randn(2, 8, 512)))
    # print('h.shape-->', h.shape)    # hn:[2, 8, 512] -> [1, 8, 512]
    # print('c.shape-->', c.shape)    # cn:[2, 8, 512] -> [1, 8, 512]


    # # 1 实例化Encoder对象
    # myEncoder = Encoder(vocab_size=20004, embed_size=512, hidden_size=512)
    # print('myEncoder-->', myEncoder)
    #
    # # 2 准备数据
    # myx = torch.ones((8, 133), dtype=torch.long)
    #
    # # 3 给模型喂数据
    # myoutput, myhidden = myEncoder(myx)
    # print('注意返回值：')
    # print('myoutput--->', myoutput.shape)
    # print('myhidden[0]--->', myhidden[0].shape)
    # print('myhidden[1]--->', myhidden[1].shape)
    #
    # # hn:[2, 8, 512] -> [1, 8, 512]
    # # cn:[2, 8, 512] -> [1, 8, 512]
    # # myhidden = (torch.randn(2, 8, 512), torch.randn(2, 8, 512))
    # myReduceState = ReduceState()
    # h_reduced, c_reduced = myReduceState(myhidden)
    # print('h_reduced--->', h_reduced.shape)
    # print('c_reduced--->', h_reduced.shape)

    '''
        注意返回值：
        myoutput---> torch.Size([8, 133, 1024])
        h_myhidden---> torch.Size([2, 8, 512])
        c_myhidden---> torch.Size([2, 8, 512])
        h_reduced---> torch.Size([1, 8, 512])
        c_reduced---> torch.Size([1, 8, 512])
    '''


# 构建PGN类 PGN(nn.Module) 实现思路分析
# init函数 (self, v) # super()
    # super # 设置字典对象属性self.v self.DEVICE = d10_config.DEVICE
    # 注意力机制属性 self.attention Attention(512) Attention(d10_config.hidden_size)
    # 编码器属性 self.encoder Encoder(20004,512,512) (len(v), d10_config.embed_size, d10_config.hidden_size)
    # 解码器属性 self.decoder Decoder(20004,512,512) (len(v), d10_config.embed_size, d10_config.hidden_size)
    # 降维加和属性 self.reduce_state  ReduceState()

# forward函数 (self, x, x_len, y, len_oovs, batch, num_batches, teacher_forcing)
    # 数据形状x[8,181] y[8,24] len_oovs:10
# 1 将x中OOV单词替换为<UNK>对应的id, 不对oov单词编码 通过pgen指针从原文中copy / x_copy=replace_oovs(x,self.v)
# 2 求pad数组 x_padding_masks 每个元素与0比较,不相等Ture，相等False # 结果0的位置都是padding位置 torch.ne(x,0).byte().float()
# 3 对x进行Encode编码 对encoder_states进行加和降维
    # self.Encoder() x_copy[8,181] --> encoder_output[8,181,1024], encoder_states([2,8,512],[2,8,512])
    # self.reduce_state() encoder_states([2,8,512],[2,8,512]) --> ([1,8,512],[1,8,512]) 得到 decoder_states

# 4 循环解码 每个时间步 通过注意力层计算 context_vector, attention_weights / p_vocab, decoder_states, p_gen
# 4-1 根据y长度进行循环  x_t = y[:, 0] # x_t[8,] / for t in range(y.shape[1] - 1):
    # 准备输入值和标签 if teacher_forcing:  x_t = y[:, t] / x_t = replace_oovs(x_t, self.v) / y_t = y[:, t + 1]
# 4-2 通过注意力层计算 decoder_states([1,8,512], [1,8,512]), encoder_output[8,181,1024] x_padding_masks[8,181] ->
                # context_vector[8,1024], attention_weights[8,181]
# 4-3 [8,1], ([1,8,512], [1,8,512]), [8,1024] -> p_vocab[8,20004], decoder_states([1,8,512],[1,8,512]), p_gen:[8,1]
    # 通过解码器层self.decoder (x_t.unsqueeze(1), decoder_states, context_vector) --> p_vocab, decoder_states, p_gen

# 4-4 计算final_dist概率分布  self.get_final_distribution
    # x[8,181], p_gen[8,1], p_vocab[8,20004], attention_weights[8,181], torch.max(len_oovs)=15  -> final_dist[8,20015]
# 4-5 第t个时间步的预测结果, 将作为第 t+1 个时间步的输入(若采用Teacher-forcing则会从新覆盖x_t)
    # 非teacher_forciing x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
    # 若没有使用pgn 则y的oov单词需要修改成unk if not d10_config.pointer: y_t = replace_oovs(y_t, self.v)

# 4-6 计算损失
    #  根据真实值y_t的id索引 在final_dist表中 查找到对应概率分布target_probs，
    #  也就是根据真实值的id，查找到对应id的概率，我们希望对应的概率值 越接近真实值越好！
        # target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1)) # [8,20015] -> [8,1]
        # target_probs = target_probs.squeeze(1) # [8, 1] -> [8,]
    # 计算真实值y_t是否是pad 如果是pad产生的损失就mask掉  mask = torch.ne(y_t, 0).byte() # mask[8,]
    # 计算loss防止为零 loss = -torch.log(target_probs + d10_config.eps) # loss[8,] 每个时间步，8个样本预测8个值, 出来8个损失
    # 先遮掩, 再添加损失值 mask = mask.float() loss = loss * mask step_losses.append(loss)

# 5 求每个句子的平均损失
    # 1 求每个样本的 损失 torch.stack(, dim=1) 按照行的方向堆积元素
    # tmp = torch.stack(step_losses, 1)  # eg: 8句摘要，每1句23个时间步 [8,23]
    # sample_losses = torch.sum(torch.stack(step_losses, 1), 1)  # 每句话的平均损失 sample_losses[8,]

    # 2 求每个样本的 句子长度 统计非PAD的字符个数, 作为当前批次序列的有效长度
    # seq_len_mask = torch.ne(y, 0).byte().float()
    # batch_seq_len = torch.sum(seq_len_mask, dim=1)

    # 3 先计算每个样本的平均损失，再求8个样本的平均损失，也就是批次样本损失
    # batch_loss = torch.mean(sample_losses / batch_seq_len)
    # return batch_loss

class PGN(nn.Module):
    def __init__(self, v):
        super(PGN, self).__init__()
        # 初始化字典对象
        self.v = v
        self.DEVICE = d10_config.DEVICE

        # 依次初始化4个类对象
        self.attention = Attention(d10_config.hidden_size)
        self.encoder = Encoder(len(v), d10_config.embed_size, d10_config.hidden_size)
        self.decoder = Decoder(len(v), d10_config.embed_size, d10_config.hidden_size)
        self.reduce_state = ReduceState()

    # 计算 最终分布 的函数
    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights, max_oov):
        # 若参数的形状： x[8, 181], p_gen[8, 1], p_vocab[8, 20004], attention_weights[8, 181], max_oov:10
        if not d10_config.pointer: # 判断是否使用PGN调节单词分布
            return p_vocab

        batch_size = x.size()[0]

        # 进行p_gen概率值的裁剪, 具体取值范围可以调参
        p_gen = torch.clamp(p_gen, 0.001, 0.999)

        # 接下来两行代码是论文中公式9的计算 [8,1]*[8,20004] -> [8,20004] 广播机制 对应位置点乘
        p_vocab_weighted = p_gen * p_vocab
        # (batch_size, seq_len) [8,1] * [8,181] -> [8,181] 广播机制 对应位置点乘
        attention_weighted = (1 - p_gen) * attention_weights

        # 得到扩展后的单词概率分布(extended-vocab probability distribution)
        # extended_size = len(self.v) + max_oovs  eg： extension[8, 10]
        extension = torch.zeros((batch_size, max_oov)).float().to(self.DEVICE)

        # (batch_size, extended_vocab_size) eg:[8,20004],[8,10] = [8,20014]
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)

        # 数据发散（scatter / scatter_add） # 直接赋值a = 1  间接赋值 b = &a *b = 2
        # 根据论文中的公式9, 累加注意力值attention_weighted到对应的单词位置x。按照索引值进行累加
        final_distribution = p_vocab_extended.scatter_add_(dim=1, index=x, src=attention_weighted)

        return final_distribution

    def forward(self, x, x_len, y, len_oovs, batch, num_batches, teacher_forcing):

        # 将 x 中所有OOV单词的id, 全部替换成<UNK>对应的id，返回替换以后的结果 x_copy
        # 好处：PGN模型生成摘要：一部分来自字典数据，一部分oov单词
        x_copy = replace_oovs(x, self.v)   # eg: x  [8, 181]

        # 1 若 x 8个样本最大长度为181，长度不够181 要打padding 0
        # 2 逐元素比较 元素是否不相等；不相等Ture，相等False, 结果：0的位置都是padding位置
        x_padding_masks = torch.ne(x, 0).byte().float()   # eg: x_padding_masks  [8, 181]

        # 第一步: 进行Encoder的编码计算
        # x_copy [8,181] -> encoder_output[8,181,1024], encoder_states ([2,8,512], [2,8,512])
        encoder_output, encoder_states = self.encoder(x_copy)

        # encoder_states([2, 8, 512], [2, 8, 512]) -> ([1,8,512], [1,8,512])
        decoder_states = self.reduce_state(encoder_states)

        # 初始化每一步的损失值
        step_losses = []

        # 第二步：循环解码 每一个时间步都经历注意力的计算 解码器层的计算
        # 初始化解码器的输入, 是 ground truth 中的第一列, 即真实摘要的第一个字符
        x_t = y[:, 0]   # x_t[8,]
        for t in range(y.shape[1] - 1):
            # 如果使用Teacher_forcing, 则每一个时间步用真实标签来指导训练
            if teacher_forcing:
                x_t = y[:, t]   # x_t[8,]

            x_t = replace_oovs(x_t, self.v) # x_t[8,]
            y_t = y[:, t + 1]   #y_t[8,]

            # 通过注意力层计算 context vector
            # decoder_states([1,8,512], [1,8,512]), encoder_output[8,181,1024] x_padding_masks[8,181] ->
            # context_vector[8,1024], attention_weights[8,181]
            context_vector, attention_weights = self.attention(decoder_states, encoder_output, x_padding_masks)
            # print('attention_weights--->','t--->', t, attention_weights)

            # 通过解码器层计算得到 vocab distribution 和hidden states, p_gen
            # [8,1], ([1,8,512], [1,8,512]), [8,1024] -> p_vocab[8,20004], decoder_states([1,8,512],[1,8,512]), p_gen:[8,1]
            p_vocab, decoder_states, p_gen = self.decoder(x_t.unsqueeze(1), decoder_states, context_vector)

            # 得到最终的概率分布 x[8,181],p_gen[8,1],p_vocab[8,20004],attention_weights[8,181],torch.max(len_oovs)=15 -> final_dist[8,20015]
            final_dist = self.get_final_distribution(x, p_gen, p_vocab, attention_weights, torch.max(len_oovs) )

            # 第t个时间步的预测结果, 将作为第 t+1 个时间步的输入(如果采用Teacher-forcing则不同)
            # final_dist[8,20015]->x_t[8,]
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)

            # 重要：模型的损失函数处理
            # Get the probabilities predict by the model for target tokens
            # 根据模型对target tokens的预测, 来获取到预测的概率
            if not d10_config.pointer:
                y_t = replace_oovs(y_t, self.v)  # y_t[8,]

            # 聚集(Gather)操作: 把final_dist的值，按照y_t位置信息，根据dim=1(按照行)，copy给target_probs
            # final_dist[8, 20015] :y_t.unsqueeze(1)[8, 1] -> [8, 1]，
            # 思想：根据 y_t的值索引 找到对应位置概率！ y_t的值索引 对应的概率值 越接近真实值越好！
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1) # [8, 1] -> [8,]

            # 将解码器端的PAD用padding mask遮掩掉, 防止计算loss时的干扰
            mask = torch.ne(y_t, 0).byte() # mask[8,]

            # 为防止计算log(0) 而做的数学上的平滑处理
            loss = -torch.log(target_probs + d10_config.eps) # loss[8,] 每个时间步，8个样本预测8个值, 出来8个损失

            # 先遮掩, 再添加损失值
            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)

        # 1 求每个样本的 损失
        # torch.stack() dim=1 按照行的方向堆积元素
        tmp = torch.stack(step_losses, 1) # tmp 8 * 23  #8句摘要，23个时间步；也就是8句摘要，每句摘要22个单词
        sample_losses = torch.sum(torch.stack(step_losses, 1), 1) # 每句话的平均损失 sample_losses[8,]

        # 2 求每个样本的 句子长度
        # 统计非PAD的字符个数, 作为当前批次序列的有效长度
        seq_len_mask = torch.ne(y, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        # 3 先计算每个样本的平均损失，再求8个样本的平均损失，也就是批次样本损失
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss


def dm05_test_PGN():

    print('从配置文件中获取参数信息', d10_config.max_vocab_size)

    # 1 训练集 数据对
    train_dataset = PairDataset(d10_config.train_data_path,
                                max_enc_len=d10_config.max_enc_len,
                                max_dec_len=d10_config.max_dec_len,
                                truncate_enc=d10_config.truncate_enc,
                                truncate_dec=d10_config.truncate_dec)


    # 2 创建模型的单词字典
    # vocab 有97733个 使用了20000个+4个特殊字符 其他的都是oov单词
    vocab = train_dataset.build_vocab(embed_file=d10_config.embed_file)
    print('vocab--->', vocab.size())

    # 3 创建训练集，加入到DataLoarder中
    train_data = SampleDataset(train_dataset.pairs, vocab)

    # 定义训练集的数据迭代器
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=d10_config.batch_size,
                                  # batch_size=1, # 1 2 8 16 测试
                                  shuffle=True,
                                  collate_fn=collate_fn)

    num_batches = len(train_dataloader)
    print('num_batches--->', num_batches)

    # 4 实例化PGN类对象
    model = PGN(vocab)
    print('model--->', model)

   # 5 模型单batch测试
    for batch, data in enumerate(train_dataloader):
        x, y, x_len, y_len, oov, len_oovs = data
        print('x->', x.shape)
        print('y->', y.shape)
        print('x_len->', x_len)
        print('y_len->', y_len)
        print('oov->', oov)
        print('len_oovs->', len_oovs)

        # 利用模型进行训练 并返回损失值
        loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=num_batches, teacher_forcing=True)
        print('一个批次的平均loss--->', loss)
        if (batch == 10):
            print('10个批次 退出 ')
            break


if __name__ == '__main__':

    dm01_test_Encoder()           # 编码
    # dm02_test_Attention()         # 注意力机制
    # dm03_test_Decoder()             # 解码
    # dm04_test_ReduceState()       # 隐藏层输出降维
    # dm05_test_PGN()
    print('model End')