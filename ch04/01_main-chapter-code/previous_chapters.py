# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""前面章节中实现的、供第 4 章复用的数据处理和注意力模块。

本文件主要包含两部分：
1. ``GPTDatasetV1`` 和 ``create_dataloader_v1``：把文本转换为 GPT 训练所需的
   “输入序列—下一个词元序列”数据。
2. ``MultiHeadAttention``：实现带因果遮罩（只能关注当前位置及其前文）的
   多头自注意力机制。
"""

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """把一整段文本切分成多个固定长度的 GPT 训练样本。

    参数说明：
        txt: 用来制作训练样本的原始文本。
        tokenizer: 负责在文本与词元 ID 之间转换的分词器。
        max_length: 每条输入序列包含的词元数量。
        stride: 滑动窗口每次向后移动的词元数量。
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        # input_ids 保存模型输入，target_ids 保存每个输入对应的预测目标。
        self.input_ids = []
        self.target_ids = []

        # 将整段文本编码成词元（token）ID。
        # allowed_special 允许文本中出现 GPT-2 用来表示文本结束的特殊词元。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口切分词元序列，每个窗口包含 max_length 个词元。
        # stride 是窗口每次向右移动的距离；当 stride < max_length 时，相邻样本会有重叠。
        # 上界使用 len(token_ids) - max_length，是为了给目标序列额外预留一个“下一个词元”。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]

            # 目标序列相对输入向右移动一个位置。例如：
            # 输入 [今, 天, 天, 气] -> 目标 [天, 天, 气, 好]
            # 这样模型在每个位置都学习预测“下一个词元”。
            target_chunk = token_ids[i + 1: i + max_length + 1]

            # DataLoader 需要张量；这里每个张量的形状都是 (max_length,)。
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回当前数据集中样本窗口的数量。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """返回第 idx 个样本，格式为 (输入序列, 目标序列)。"""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """根据原始文本创建可按批次迭代的 DataLoader。

    参数说明：
        txt: 待处理的原始文本。
        batch_size: 每批包含的样本数。
        max_length: 每个样本包含的词元数，也是模型看到的序列长度。
        stride: 滑动窗口的步长。
        shuffle: 是否在每轮迭代前打乱样本顺序。
        drop_last: 是否丢弃数量不足 batch_size 的最后一批样本。
        num_workers: 并行加载数据的子进程数；0 表示在当前进程中加载。

    返回：
        一个 PyTorch DataLoader。迭代它可获得成批的 (input_ids, target_ids)。
    """
    # 使用 GPT-2 的分词规则，使文本与模型使用同一套词元 ID。
    tokenizer = tiktoken.get_encoding("gpt2")

    # 先将文本切分为一个个 (输入序列, 目标序列) 样本。
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 再由 DataLoader 负责打乱、分批和迭代。
    # 每次迭代返回的两个张量形状通常都是 (batch_size, max_length)。
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


class MultiHeadAttention(nn.Module):
    """带因果遮罩的多头自注意力层。

    多个注意力头可以从不同角度学习词元之间的关系；因果遮罩保证当前位置看不到
    后面的词元，从而满足 GPT 从左到右预测文本的要求。

    参数说明：
        d_in: 每个输入词元的特征维度。
        d_out: 注意力层的输出特征维度。
        context_length: 模型支持的最大词元序列长度。
        dropout: 注意力权重的随机丢弃比例。
        num_heads: 并行计算的注意力头数量。
        qkv_bias: Q、K、V 线性变换是否使用偏置项。
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # 每个头分到相同的特征维度，因此总输出维度必须能被头数整除。
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 单个注意力头使用的特征维度。

        # 将同一份输入分别投影为查询（Q）、键（K）和值（V）。
        # 可以直观地理解为：Q 表示“当前词元想找什么”，K 表示“各词元能提供什么”，
        # Q 与 K 的匹配程度决定关注权重，V 则是最终按权重汇总的实际信息。
        # qkv_bias 控制这三个线性投影层是否使用偏置项。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 融合所有注意力头的输出。
        self.dropout = nn.Dropout(dropout)

        # 创建因果遮罩：主对角线上方为 1，表示需要屏蔽“未来”位置。
        # register_buffer 会让 mask 随模型一起移动到 CPU/GPU，但不会把它当作可训练参数。
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """计算多头自注意力。

        x 的形状为 (批大小, 词元数, d_in)，词元数不能超过 context_length；
        返回张量的形状为 (批大小, 词元数, d_out)。
        """
        b, num_tokens, d_in = x.shape

        # 线性投影后，Q、K、V 的形状均为 (b, num_tokens, d_out)。
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 把最后一维 d_out 拆成 num_heads × head_dim，相当于把特征分给多个头：
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 把“头”维度移到词元维度前，便于每个头独立计算注意力：
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Q 与 K 的转置相乘，得到每个词元对其他词元的原始关注分数。
        # attn_scores 的形状为 (b, num_heads, num_tokens, num_tokens)。
        attn_scores = queries @ keys.transpose(2, 3)

        # 模型的实际输入可能短于 context_length，因此只截取当前序列所需的遮罩区域。
        # 这个二维遮罩会自动广播到所有批次和所有注意力头。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 将未来位置的分数设为负无穷；经过 softmax 后，这些位置的权重就会变成 0。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 除以 sqrt(head_dim) 可防止点积数值过大，使 softmax 和训练过程更稳定。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # 随机丢弃部分注意力权重，用于降低训练时的过拟合风险。
        # 调用 model.eval() 切换到推理模式后，Dropout 会自动停止随机丢弃。
        attn_weights = self.dropout(attn_weights)

        # 用注意力权重对 V 做加权求和，再把形状恢复为：
        # (b, num_tokens, num_heads, head_dim)。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 拼接所有头的结果：num_heads × head_dim = d_out。
        # contiguous() 确保转置后的张量在内存中连续，从而可以安全地使用 view。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 通过最终线性层融合各个头的信息，输出形状为 (b, num_tokens, d_out)。
        context_vec = self.out_proj(context_vec)

        return context_vec
