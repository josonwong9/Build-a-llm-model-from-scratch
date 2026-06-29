# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file contains the relevant code from chapter 3 that is going to be used
# in forthcoming chapters.

import torch
import torch.nn as nn


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # 分别学习 query、key、value 三组投影矩阵；它们把输入词向量映射到注意力空间。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Dropout 随机丢弃一部分注意力权重，训练时有助于减轻过拟合。
        self.dropout = nn.Dropout(dropout)  # New
        # 上三角 mask 屏蔽当前位置之后的 token，保证模型只能看见“过去”和“现在”。
        # register_buffer 会把 mask 跟随模型移动到 CPU/GPU，但它不是可训练参数。
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New

    def forward(self, x):
        # x 的形状是 (batch_size, token 数量, 输入维度)。
        b, num_tokens, d_in = x.shape  # New batch dimension b
        # 每个 token 同时生成 key、query、value，后面用它们计算注意力。
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # query 和 key 做点积得到注意力分数；分数越大表示两个 token 越相关。
        attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose
        # 把未来位置的分数设为 -inf，softmax 后这些位置的权重会变成 0。
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 除以 key 维度的平方根可以稳定数值，避免 softmax 过早变得极端。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # New

        # 用注意力权重对 value 加权求和，得到每个 token 的上下文向量。
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 简单版本：直接创建多个独立的 CausalAttention 头。
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        # 拼接多个头的输出后，再用线性层做一次融合。
        self.out_proj = nn.Linear(d_out*num_heads, d_out*num_heads)

    def forward(self, x):
        # 每个头都会从不同的投影角度提取信息，最后在特征维度拼接。
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 总输出维度必须能平均分给每个注意力头。
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # 每个头只处理 d_out 的一部分维度；所有头拼回去后仍是 d_out。
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # b 是 batch 大小，num_tokens 是当前序列长度，d_in 是输入嵌入维度。
        b, num_tokens, d_in = x.shape

        # 一次性生成所有头需要的 key/query/value，暂时仍放在 d_out 这一整维里。
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 每个头内部独立计算注意力分数；最后两个维度表示 token 两两之间的关系。
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        # 只取当前序列长度对应的 mask，避免固定 context_length 比实际输入更长。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # softmax 把分数变成概率分布；每个 token 对可见 token 的权重和为 1。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 先得到每个头的上下文向量，再把 num_heads 维移回 token 维之后。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # transpose 后内存可能不连续，contiguous() 让 view 可以安全重排形状。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
