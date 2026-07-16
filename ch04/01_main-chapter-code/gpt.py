# 本文件汇总了第 2～4 章介绍的核心代码，包括数据集构造、
# 多头注意力、GPT 模型，以及最简单的文本生成方法。
# 直接运行本文件即可看到一个完整的“输入文本 -> 模型续写”示例。

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 分别保存模型的输入序列，以及每个输入序列对应的预测目标。
        self.input_ids = []
        self.target_ids = []

        # 将整段文本转换成 token ID。这里允许把 <|endoftext|> 当作特殊 token 使用。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 用滑动窗口把长文本切成多个长度为 max_length 的训练样本。
        # stride 是窗口每次向右移动的距离；当 stride < max_length 时，相邻样本会有重叠。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]

            # 目标序列相对输入序列向右错开一个 token。
            # 例如输入是 [A, B, C]，目标就是 [B, C, D]，模型由此学习“预测下一个 token”。
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # DataLoader 会通过这个方法获取数据集中的样本总数。
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 返回第 idx 个“输入序列、目标序列”对。
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 使用 GPT-2 的分词器，把文本转换为 GPT-2 词表中的 token ID。
    tokenizer = tiktoken.get_encoding("gpt2")

    # 先构造数据集，再由 DataLoader 自动完成分批、打乱等工作。
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # drop_last=True 时会丢弃最后一个不足 batch_size 的批次，
    # 这样每个批次的张量形状都完全一致。
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 每个注意力头分到相同的特征维度，因此总输出维度必须能被头数整除。
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个注意力头负责的特征维度

        # 同一份输入会分别经过三个线性层，得到查询（Q）、键（K）和值（V）。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 融合多个注意力头的输出
        self.dropout = nn.Dropout(dropout)

        # 上三角区域代表“当前位置之后的 token”。将它注册为 buffer 后，
        # 它不会作为参数训练，但会随模型一起移动到 CPU 或 GPU。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # x 的形状为 (批次大小, token 数量, 输入特征维度)。
        b, num_tokens, d_in = x.shape

        # 将每个 token 的输入向量映射成 K、Q、V；三者形状均为
        # (批次大小, token 数量, d_out)。
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 把最后一维拆成“注意力头数 × 每头维度”，相当于把特征分给多个注意力头：
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 把注意力头维度提前，方便每个头独立计算注意力：
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 对每个注意力头计算 Q 与 K 的点积。分数越大，说明两个 token 的相关性越强。
        # 得到的 attn_scores 形状为 (b, num_heads, num_tokens, num_tokens)。
        attn_scores = queries @ keys.transpose(2, 3)

        # 模型实际收到的序列可能短于 context_length，所以只截取本次需要的掩码区域。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 把“未来 token”对应的分数设为负无穷。经过 softmax 后，这些位置的权重会变成 0，
        # 从而保证生成当前位置时只能看到当前位置及其之前的内容（因果注意力）。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 除以维度的平方根可以避免点积过大，让 softmax 和训练过程更稳定。
        # softmax 把每一行分数转换成总和为 1 的注意力权重。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 用注意力权重对 V 做加权求和，再把维度换回 token 在前的顺序：
        # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 将所有注意力头重新拼接成 d_out 维向量。
        # contiguous() 保证转置后的数据在内存中连续，使 view() 可以安全重塑形状。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 再做一次线性映射来融合各个头的信息

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # eps 防止方差为 0 时出现除零错误。
        self.eps = 1e-5

        # scale 和 shift 是可训练参数，让模型能在标准化后重新缩放和平移特征。
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 只沿最后一个维度（每个 token 的特征维度）计算均值和方差；
        # 不混合不同 token，也不混合不同批次中的样本。
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU 是 GPT 使用的平滑激活函数。这里使用 tanh 近似公式，
        # 相比 ReLU 的直接截断，它会更平滑地控制每个输入值通过的比例。
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 先把特征维度扩大 4 倍，经过非线性激活后再缩回原维度。
        # 这个网络分别处理每个 token，用于进一步变换注意力层提取出的特征。
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 一个 Transformer 块由“多头注意力”和“前馈网络”两个子层组成。
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # 残差分支上的 dropout 用于训练时正则化；调用 model.eval() 后会自动关闭。
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 第一个子层：先做层归一化，再计算注意力，最后加回原输入形成残差连接。
        # 残差连接给信息和梯度提供“捷径”，有助于稳定深层网络的训练。
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # 形状仍是 [batch_size, num_tokens, emb_dim]
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 第二个子层：以同样的方式执行“层归一化 -> 前馈网络 -> 残差相加”。
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # token 嵌入把离散的 token ID 转成可学习的连续向量；
        # 位置嵌入则告诉模型每个 token 位于序列中的第几个位置。
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 按配置重复堆叠 n_layers 个结构相同、参数各自独立的 Transformer 块。
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 所有 Transformer 块之后再做一次归一化，最后映射到整个词表。
        # 每个 token 位置都会得到 vocab_size 个分数（logits）。
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx 保存 token ID，形状是 [batch_size, seq_len]。
        batch_size, seq_len = in_idx.shape

        # token 嵌入形状为 [batch_size, seq_len, emb_dim]。
        tok_embeds = self.tok_emb(in_idx)

        # 同一批次中的样本共享位置 0～seq_len-1 的位置嵌入；相加时 PyTorch 会
        # 自动把 [seq_len, emb_dim] 广播到整个 batch。
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # 形状为 [batch_size, seq_len, emb_dim]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # logits 不是概率，而是模型对词表中每个 token 给出的原始分数；
        # 形状为 [batch_size, seq_len, vocab_size]。
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx 是当前已有的 token ID，形状为 (batch_size, 当前序列长度)。
    # 每轮循环只追加一个 token，所以共执行 max_new_tokens 轮。
    for _ in range(max_new_tokens):

        # 如果已有文本超过模型支持的最大上下文长度，只保留最后 context_size 个 token。
        # 例如模型最多接收 5 个 token，而当前已有 10 个，就只用最后 5 个预测下一个 token。
        idx_cond = idx[:, -context_size:]

        # 文本生成只是推理，不需要计算梯度；关闭梯度可以减少内存和计算开销。
        with torch.no_grad():
            logits = model(idx_cond)

        # 模型会给输入序列中的每个位置都输出预测，这里只关心最后一个位置，
        # 因为它代表“下一个 token”的预测结果：
        # (batch_size, token 数, vocab_size) -> (batch_size, vocab_size)。
        logits = logits[:, -1, :]

        # 贪心解码：直接选择分数最高的 token，不进行随机采样。
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 把新 token 追加到已有序列末尾，作为下一轮预测的上下文。
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def main():
    # 参照 GPT-2 small（通常称为 124M）的核心结构配置。
    # 注意：这个教学实现没有让 token 嵌入层与输出层共享权重，所以实际参数量会多于 1.24 亿。
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词表大小
        "context_length": 1024,  # 模型最多能接收的 token 数
        "emb_dim": 768,          # 每个 token 的嵌入维度
        "n_heads": 12,           # 多头注意力的头数
        "n_layers": 12,          # Transformer 块的层数
        "drop_rate": 0.1,        # dropout 比例
        "qkv_bias": False        # Q、K、V 线性层是否使用偏置项
    }

    # 固定随机种子，使随机初始化的模型每次运行时得到一致的结果。
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # 切换到推理模式，关闭 dropout

    start_context = "Hello, I am"

    # 把起始文本编码成 token ID，并在最前面添加 batch 维度：
    # [token 数] -> [1, token 数]。
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # 使用尚未训练、只有随机参数的模型继续生成 10 个 token。
    # 因为模型未训练，输出文本没有实际语义；本示例主要用于验证完整的数据流。
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    # 去掉 batch 维度，再把 token ID 转回人类可读的文本。
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    # 只有直接执行 python gpt.py 时才运行示例；被其他文件 import 时不会运行。
    main()
