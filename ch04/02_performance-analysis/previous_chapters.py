# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# 本文件汇总了第 2～4 章实现 GPT 所需的主要代码，既可以被其他程序导入，
# 也可以直接运行，以演示“输入一小段文本，再让未训练的 GPT 续写”的完整流程。

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# 第 2 章：准备训练数据
#####################################


class GPTDatasetV1(Dataset):
    """把一整段文本整理成 GPT 训练所需的“输入—目标”样本对。"""

    def __init__(self, txt, tokenizer, max_length, stride):
        # 每个列表元素都是一个长度为 max_length 的一维 token 张量。
        self.input_ids = []
        self.target_ids = []

        # 一次性把全文转换成 token ID。
        # allowed_special 允许原文中出现 GPT-2 的文本结束标记，而不会触发报错。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 用滑动窗口从全文中截取训练样本。
        # 输入和目标只错开一个位置：模型看到 input_chunk 中的每个 token 后，
        # 应该预测 target_chunk 中对应位置的下一个 token。
        # stride 小于 max_length 时，相邻样本会重叠，可以更充分地利用文本。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # DataLoader 会通过这个方法获知数据集中一共有多少个样本。
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 返回第 idx 组“输入 token 序列、目标 token 序列”。
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """根据原始文本创建可按批次迭代的 DataLoader。"""

    # 使用与 GPT-2 配套的分词器，确保 token ID 与模型词表一致。
    tokenizer = tiktoken.get_encoding("gpt2")

    # 先将文本切分为固定长度的输入和目标序列。
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 再由 DataLoader 负责组装批次、打乱数据以及多进程读取。
    # drop_last=True 会丢弃最后一个不足 batch_size 的批次，使批次形状保持一致。
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# 第 3 章：多头自注意力
#####################################
class MultiHeadAttention(nn.Module):
    """带因果遮罩的多头自注意力层。"""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 每个注意力头分到相同数量的特征，因此总输出维度必须能被头数整除。
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 单个注意力头处理的特征维度

        # 同一份输入分别经过三个线性层，得到查询 Q、键 K、值 V。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 多个注意力头拼接后，再用一个线性层融合各个头的信息。
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # 因果遮罩的主对角线上方为 1，表示这些“未来 token”不能被看到。
        # register_buffer 会让 mask 随模型移动到 CPU/GPU，但不会把它当作可训练参数。
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # x 的形状为 (批次大小, token 数量, 输入特征维度)。
        b, num_tokens, d_in = x.shape

        # 投影后的 Q、K、V 形状均为 (b, num_tokens, d_out)。
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 把最后一个 d_out 维度拆成 num_heads × head_dim，得到多个注意力头：
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 把“注意力头”维度提前，方便每个头独立完成矩阵乘法：
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 每个查询 Q 与所有键 K 做点积，衡量当前 token 应该关注其他 token 的程度。
        # 结果形状为 (b, num_heads, num_tokens, num_tokens)。
        attn_scores = queries @ keys.transpose(2, 3)

        # 当前输入可能短于模型支持的最大上下文，因此只截取实际 token 数量对应的遮罩。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 把未来位置的注意力分数设为负无穷；经过 softmax 后，其权重就会变成 0。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 除以 sqrt(head_dim) 可以防止点积过大，使 softmax 和训练过程更稳定。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 用注意力权重对 V 加权求和，再把维度换回 token 在前的顺序。
        # context_vec 的形状变为 (b, num_tokens, num_heads, head_dim)。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 将所有注意力头重新拼接成 d_out 维，并通过输出投影层融合信息。
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


#####################################
# 第 4 章：组装完整 GPT 模型
#####################################
class LayerNorm(nn.Module):
    """对每个 token 的所有特征做层归一化。"""

    def __init__(self, emb_dim):
        super().__init__()
        # eps 用于避免方差为 0 时出现除零错误。
        self.eps = 1e-5
        # scale 和 shift 是可训练参数，让模型能调整归一化后的缩放与偏移。
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 只沿最后的特征维度统计均值和方差，各 token 之间互不影响。
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GPT 使用的 GELU 激活函数的近似实现。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 与 ReLU 的硬截断不同，GELU 会更平滑地控制输入信息通过的比例。
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """逐 token 处理的两层前馈神经网络。"""

    def __init__(self, cfg):
        super().__init__()
        # 先把特征维度扩大 4 倍以提升表达能力，再压回原来的 emb_dim。
        # 线性层只作用于最后一维，不会混合不同 token 的位置。
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """一个采用“预归一化”结构的 Transformer 解码器块。"""

    def __init__(self, cfg):
        super().__init__()
        # 注意力子层负责让每个 token 汇总它之前的上下文信息。
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        # 前馈子层负责对注意力提取出的特征做进一步变换。
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 第一个子层：层归一化 -> 多头注意力 -> Dropout -> 残差相加。
        # 残差连接保留原始输入，有助于深层网络中的梯度传播。
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # 形状保持为 [batch_size, num_tokens, emb_dim]
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 第二个子层：层归一化 -> 前馈网络 -> Dropout -> 残差相加。
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """由嵌入层、多个 Transformer 块和输出层组成的 GPT 模型。"""

    def __init__(self, cfg):
        super().__init__()
        # token 嵌入表示“这是哪个 token”，位置嵌入表示“它位于序列中的哪里”。
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 串联 n_layers 个结构相同、但参数互不共享的 Transformer 块。
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 最终归一化后，把每个 token 的隐藏特征映射为整个词表上的预测分数。
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx 的形状为 (batch_size, seq_len)，其中每个元素都是一个 token ID。
        batch_size, seq_len = in_idx.shape

        # 将离散的 token ID 和位置编号分别转换成 emb_dim 维向量，再逐元素相加。
        # 位置向量会自动广播到批次中的每个样本。
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # [batch_size, seq_len, emb_dim]

        # 依次经过嵌入 Dropout、全部 Transformer 块和最终层归一化。
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # logits 是未经 softmax 的词表分数，形状为
        # (batch_size, seq_len, vocab_size)。训练时可直接交给交叉熵损失函数。
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """使用贪心解码，让模型逐个生成 max_new_tokens 个新 token。"""

    # idx 形状为 (batch_size, 当前 token 数)，初始内容就是模型的提示词。
    for _ in range(max_new_tokens):

        # 如果当前序列超过模型支持的最大上下文长度，只保留末尾 context_size 个 token。
        # 例如模型最多接收 5 个 token，而当前已有 10 个，就只用最后 5 个进行预测。
        idx_cond = idx[:, -context_size:]

        # 生成阶段不需要反向传播；关闭梯度可减少内存占用和计算开销。
        with torch.no_grad():
            logits = model(idx_cond)

        # 每轮只需要最后一个位置的预测结果：
        # (batch_size, num_tokens, vocab_size) -> (batch_size, vocab_size)。
        logits = logits[:, -1, :]

        # 贪心解码：直接选择分数最高的 token，不进行随机采样。
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 把新 token 接到序列末尾，下一轮再用扩展后的序列继续预测。
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens + 1)

    return idx


if __name__ == "__main__":

    # 采用 GPT-2 small 的主体配置。配置名沿用“124M”，但本实现没有让 token 嵌入层
    # 与输出层共享权重，因此实际参数量会高于 1.24 亿。
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # GPT-2 分词器的词表大小
        "context_length": 1024,  # 模型最多能接收的 token 数量
        "emb_dim": 768,          # 每个 token 的嵌入/隐藏特征维度
        "n_heads": 12,           # 每层多头注意力的头数
        "n_layers": 12,          # Transformer 块的层数
        "drop_rate": 0.1,        # Dropout 概率
        "qkv_bias": False        # Q、K、V 线性层是否使用偏置项
    }

    # 固定随机种子，使模型的随机初始化和示例输出可以复现。
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # 切换到推理模式，关闭 Dropout

    start_context = "Hello, I am"

    # 把提示文本编码成 token ID，并增加批次维度：
    # (num_tokens,) -> (1, num_tokens)。
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # 基于提示词逐个生成 10 个 token。这里的模型只有随机初始化参数，尚未训练，
    # 因此本段代码主要用于验证数据流和张量形状，输出文本通常没有实际语义。
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    # 去掉批次维度并把 token ID 转回人类可读的文本。
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)
