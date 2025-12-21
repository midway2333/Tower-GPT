import torch
import torch.nn.functional as fc
from torch import nn, Tensor

# 参考: https://github.com/retepViolet/Transformer-


class RoPE_Emb(nn.Module):
    """RoPE位置编码"""
    def __init__(self, d: int, max_len: int=4096, device: str | None=None):
        """
        RoPE位置编码, Tower2 技术下放(doge)
        - d: 模型维度
        - max_len: 最大序列长度
        """
        super().__init__()

        self.d = d
        self.max_len = max_len
        self.device = device

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float().to(device) / d))
        # 计算频率

        self.register_buffer('inv_freq', inv_freq, persistent=False)
        # 注册频率

        self._get_embedding(inv_freq)
        # 预计算

    def _get_embedding(self, inv_freq):
        """预计算位置编码"""
        len_ids = torch.arange(self.max_len, device=self.device)
        # 序列索引

        freqs = torch.outer(len_ids, inv_freq)
        # 计算频率

        emb = torch.cat((freqs, freqs), dim=-1)
        # 复制频率参数, 使复数对共享相同的频率

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        # 频率缓存

    def forward(self) -> tuple:
        """
        生成RoPE位置编码
        """

        self.cos_cached: Tensor
        self.sin_cached: Tensor

        return (
            self.cos_cached,
            self.sin_cached,
        )

def RoPE_rotate(x: Tensor) -> Tensor:
    """
    RoPE旋转操作
    - x: 输入张量
    """
    x1 = x[..., : x.shape[-1] // 2]   # 取前一半维度
    x2 = x[..., x.shape[-1] // 2 :]   # 取后一半维度
    return torch.cat((-x2, x1), dim=-1)   # 拼接

def RoPE_reshape(x: Tensor) -> Tensor:
    """重塑张量形状"""
    batch, head_num, seq_len, dim = x.shape
    x = x.view(batch, head_num, seq_len, dim//2, 2).transpose(4, 3).reshape(batch, head_num, seq_len, dim)

    return x

def RoPE_apply(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, pos_ids: Tensor):
    """
    应用RoPE编码
    - q: query
    - k: key
    - cos: RoPE cos
    - sin: RoPE sin
    - pos_ids: 位置索引
    """
    cos = cos[pos_ids].unsqueeze(0).unsqueeze(0)   # 按位置索引选择cos值
    sin = sin[pos_ids].unsqueeze(0).unsqueeze(0)   # 按位置索引选择sin值

    q = RoPE_reshape(q)
    # 重塑 Query

    k = RoPE_reshape(k)
    # 重塑 Key

    q_embed = (q * cos) + (RoPE_rotate(q) * sin)
    k_embed = (k * cos) + (RoPE_rotate(k) * sin)
    # 应用旋转位置编码

    return q_embed, k_embed


class MHA(nn.Module):
    """多头自注意力层"""
    def __init__(self, d, dk, head_num, device: str | None=None, dropout: float=0.0):
        """
        参数:
        - d: 输入/输出的维度
        - dk: 每个头的维度
        - head_num: 头的数量
        - device: 设备
        - dropout: dropout概率
        """
        super().__init__()
        self.head_num = head_num
        self.dk = dk
        self.dropout = dropout

        self.q_proj = nn.Linear(d, head_num * dk, bias=False)
        self.k_proj = nn.Linear(d, head_num * dk, bias=False)
        self.v_proj = nn.Linear(d, head_num * dk, bias=False)
        self.o_proj = nn.Linear(head_num * dk, d, bias=False)
        # 初始化投影层

        self.rope = RoPE_Emb(dk, max_len=4096, device=device)
        # 初始化 RoPE

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = x.size()

        pos_ids = torch.arange(seq_len, device=x.device)
        # 生成位置索引 [seq_len]

        Q: Tensor = self.q_proj(x)
        K: Tensor = self.k_proj(x)
        V: Tensor = self.v_proj(x)
        # 并行投影, 方便并行计算

        Q = Q.view(batch_size, seq_len, self.head_num, self.dk).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.dk).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.dk).transpose(1, 2)
        # 分成多个注意力头 [batch, seq_len, head_num, dk]

        cos, sin = self.rope()
        Q, K = RoPE_apply(Q, K, cos, sin, pos_ids)

        attn_output = fc.scaled_dot_product_attention(Q, K, V,
            dropout_p=self.dropout if self.training else 0,
            attn_mask=mask,
            is_causal=False,
        )   # 注意力计算

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
        # 合并输出


class decoder(nn.Module):
    def __init__(self, head_num, d, dk, dff, device: str | None=None, dropout: float=0.0):
        """
        参数:
        - head_num: 注意力头数
        - d: 输入/输出的维度
        - dk: 每个头的维度
        - dff: 前馈网络内部层的维度
        - device: 设备
        """
        super().__init__()
        self.dropout = dropout
        self.multi_head = MHA(d, dk, head_num, device=device, dropout=self.dropout)
        # 多头注意力机制

        self.o = nn.Linear(head_num * dk, d)
        # 将多头注意力机制的输出合并成一个单一的输出
        # 输入维度是head_num * dk,输出维度是d

        self.norm1 = nn.LayerNorm(d)   # 层归一化

        self.ffn = nn.Sequential(             # 前馈网络
            nn.Linear(d, dff, bias=False),    # 维度变换
            nn.Dropout(self.dropout),                  # Dropout层
            nn.GELU(),                        # 激活函数
            nn.Linear(dff, d, bias=False),    # 维度变换
        )

        self.norm2 = nn.LayerNorm(d)   # 前馈网络输出层归一化

    def forward(self, inputs: tuple):

        x, mask = inputs  # x为需要处理的数据

        norm_x = self.norm1(x)   # 层归一化
        multi_head_output = self.multi_head(norm_x, mask)
        # 运用多头注意力机制

        residual_and_norm = multi_head_output + x
        norm_residual = self.norm2(residual_and_norm)
        final_output = self.ffn(norm_residual) + residual_and_norm
        # 再次应用层归一化，然后通过前馈神经网络
        # 使用的是 Pre_layer_normalization

        return (final_output, mask)


class Tower_GPT(nn.Module):   # 模型实现
    """语言模型"""
    def __init__(self, decoder_num=8, head_num=8, d=1024, dk=128, dff=4096, vocab_size=32768,   \
        padding_idx=3, device: str='cuda' if torch.cuda.is_available() else 'cpu', init: bool=True, dropout: float=0.0):

        """   
        参数:
        - decoder_num: 解码器的数量
        - head_num: 注意力头的数量
        - d: 输入/输出的维度
        - dk: 每个头的维度
        - dff: 前馈网络内部层的维度
        - vocab_size: 词汇表的大小
        - padding_idx: 填充的索引
        - device: 设备类型
        - init: 是否初始化模型权重
        - dropout: dropout概率
        """

        # 在自带词表中padding_id=3

        super().__init__()
        self.mask = Tensor()
        self.zero_mask = Tensor()
        self.pos_code = Tensor()
        # 初始化三个张量属性

        self.d = d
        self.vocab_size = vocab_size
        self.padding_id = padding_idx
        # 初始化模型的参数

        self.device = device
        # 初始化设备类型

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=d,   \
                                       padding_idx=self.padding_id).to(self.device)
        # 使用独立嵌入层

        self.decoders = nn.Sequential()   # 容器模块

        self.dropout = dropout
        for _ in range(decoder_num):
            self.decoders.append(decoder(head_num, d, dk, dff, self.device, self.dropout)).to(self.device)
            # 添加decoder_num个解码器

        self.last_linear = nn.Linear(d, self.vocab_size, bias=False).to(self.device)
        # 线性层,将解码器的输出映射到词汇表的大小

        self.last_linear.weight = self.embedding.weight
        # 共享weight和embedding权重

        self.softmax = nn.Softmax(dim=-1)   # 转换为概率分布

        if init:   # 初始化模型权重
            self.init_weights()

    def get_mask(self, sequence_len, data):
        padding_idx = self.padding_id

        if not self.training:
        # 检查模型是否在训练模式

            if self.zero_mask is None or sequence_len != self.zero_mask.size(0):
                # 判断输入序列的长度是否与当前掩码长度相等

                self.zero_mask = torch.zeros(sequence_len, sequence_len).to(self.device)
                # 生成全零掩码

            return self.zero_mask
            # 返回新的全零掩码

        if self.mask is None or sequence_len != self.mask.size(0):
        # 判断长度

            self.mask = torch.triu(torch.full((sequence_len, sequence_len),  \
                                        float('-inf')).to(self.device), diagonal=1)
            # 创建上三角掩码,设置掩码遮掩

        padding_mask = torch.zeros_like(data, dtype=torch.float)
        padding_mask[data == padding_idx] = float('-inf')
        padding_mask = padding_mask.unsqueeze(1).expand(-1, data.size(1), -1)
        # 创建一个与data形状相同的全零矩阵
        # 把padding位置设置为-inf
        # 扩展padding_mask维度

        atta_mask = self.mask + padding_mask
        return atta_mask.unsqueeze(1)
        # 返回掩码

    def init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)   # 初始化 embedding
        if self.padding_id is not None:
            self.embedding.weight.data[self.padding_id].zero_()
            # 确保 padding token 的 embedding 为 0

        for module in self.decoders.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # 初始化 decoder 中的线性层

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                # 初始化 LayerNorm

        print("模型权重已初始化")

    def forward(self, x:Tensor):

        x = x.to(self.device)

        sequence_len = x.shape[1]
        # 获取输入张量x的第二个维度的大小

        ebd_x = self.embedding(x)   # 使用嵌入层

        atta_mask = self.get_mask(sequence_len=sequence_len, data=x)
        # 创造掩码

        y, _ = self.decoders((ebd_x, atta_mask))
        # 将带有位置编码的嵌入向量和掩码传递给解码器
        # 解码器返回的y是每个位置的输出向量
        # 将解码器的输出赋值给y,并且忽略注意力权重
        # 忽略注意力掩码,毕竟任务是文本生成

        y = self.last_linear(y)
        # 将解码器的输出通过最后的线性层,得到每个位置的词汇表分布

        return y


if __name__ == '__main__':

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    model = Tower_GPT(decoder_num=12, head_num=8, d=1024, dk=128, dff=4096, vocab_size=32768, padding_idx=3, device='cpu')
    input = torch.tensor([[1, 2, 3], [4, 5, 6]]).to('cpu')
    model.eval()
    output = model(input)
    print(output.shape)
    print(count_parameters(model))
