import torch
# 导入 PyTorch。
# 本节主要使用 PyTorch 张量、点积、softmax 和矩阵乘法。

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your    (x^1)
     # 第 1 个词元 "Your" 的 3 维嵌入向量。

     [0.55, 0.87, 0.66],  # journey (x^2)
     # 第 2 个词元 "journey" 的 3 维嵌入向量。

     [0.57, 0.85, 0.64],  # starts  (x^3)
     # 第 3 个词元 "starts" 的 3 维嵌入向量。

     [0.22, 0.58, 0.33],  # with    (x^4)
     # 第 4 个词元 "with" 的 3 维嵌入向量。

     [0.77, 0.25, 0.10],  # one     (x^5)
     # 第 5 个词元 "one" 的 3 维嵌入向量。

     [0.05, 0.80, 0.55]]  # step    (x^6)
     # 第 6 个词元 "step" 的 3 维嵌入向量。
)

query = inputs[1]
# 选择第 2 个词元 "journey" 作为 query。
# query 可以理解为：当前我们要为哪个词计算上下文向量。

attn_scores_2 = torch.empty(inputs.shape[0])
# 创建一个长度为 6 的空张量。
# 用来存储 "journey" 和所有词元之间的注意力分数。

for i, x_i in enumerate(inputs):
    # 遍历所有输入词元。

    attn_scores_2[i] = torch.dot(x_i, query)
    # 计算当前词元 x_i 和 query 之间的点积。
    # 点积越大，表示当前词元和 query 越相关。

print("Attention scores for query x^2:", attn_scores_2)
# 打印 "journey" 对所有词元的注意力分数。

res = 0.
# 初始化点积结果，用于手动验证 torch.dot 的计算过程。

for idx, element in enumerate(inputs[0]):
    # 遍历第 1 个词元 "Your" 的每一个维度。
    # element 是当前维度的值，不过这里主要使用 idx 取值。

    res += inputs[0][idx] * query[idx]
    # 将 "Your" 和 "journey" 在相同维度上的值相乘，然后累加。
    # 这就是点积的手动计算方式。

print("Manual dot product:", res)
# 打印手动计算的点积。

print("torch.dot result:", torch.dot(inputs[0], query))
# 打印 torch.dot 计算的点积，用来验证二者是否一致。

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# 简单归一化：每个注意力分数除以所有分数的总和。
# 得到的权重总和为 1。

print("Attention weights by simple normalization:", attn_weights_2_tmp)
# 打印简单归一化得到的注意力权重。

print("Sum:", attn_weights_2_tmp.sum())
# 打印权重总和，应为 1。

def softmax_naive(x):
    # 定义一个简单版 softmax 函数。
    # x 是一组原始分数。

    return torch.exp(x) / torch.exp(x).sum(dim=0)
    # 对每个分数做指数运算，再除以指数总和。
    # 这样可以得到总和为 1 的权重。

attn_weights_2_naive = softmax_naive(attn_scores_2)
# 使用手写 softmax 对注意力分数进行归一化。

print("Attention weights by naive softmax:", attn_weights_2_naive)
# 打印手写 softmax 得到的注意力权重。

print("Sum:", attn_weights_2_naive.sum())
# 打印权重总和，应为 1。

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# 使用 PyTorch 官方 softmax。
# dim=0 表示对这个一维张量整体做 softmax。
# 这是实际开发中更推荐的写法。

print("Attention weights by torch.softmax:", attn_weights_2)
# 打印 PyTorch softmax 得到的注意力权重。

print("Sum:", attn_weights_2.sum())
# 打印权重总和，应为 1。

context_vec_2 = torch.zeros(query.shape)
# 创建一个和 query 形状相同的零向量。
# 用来保存最终的上下文向量。
# query.shape 是 [3]，所以 context_vec_2 也是 3 维向量。

for i, x_i in enumerate(inputs):
    # 遍历所有输入词元。

    context_vec_2 += attn_weights_2[i] * x_i
    # 用第 i 个注意力权重乘以第 i 个输入向量。
    # 然后累加到 context_vec_2。
    # 这就是加权求和。

print("Context vector for x^2:", context_vec_2)
# 打印第 2 个词元 "journey" 的上下文向量。

attn_scores = torch.empty(6, 6)
# 创建一个 6x6 的空矩阵。
# 用来保存所有词元两两之间的注意力分数。

for i, x_i in enumerate(inputs):
    # 外层循环：选择当前 query 词元。

    for j, x_j in enumerate(inputs):
        # 内层循环：选择当前被关注的词元。

        attn_scores[i, j] = torch.dot(x_i, x_j)
        # 计算第 i 个词元和第 j 个词元的点积。
        # 保存到注意力分数矩阵的第 i 行第 j 列。

print("Attention scores by loops:")
print(attn_scores)
# 打印通过双重 for 循环得到的注意力分数矩阵。

attn_scores = inputs @ inputs.T
# 使用矩阵乘法一次性计算所有词元之间的注意力分数。
# inputs 是 [6, 3]，inputs.T 是 [3, 6]。
# 二者相乘得到 [6, 6]。
# 这个结果和上面的双重 for 循环结果一致。

print("Attention scores by matrix multiplication:")
print(attn_scores)
# 打印矩阵乘法得到的注意力分数矩阵。

attn_weights = torch.softmax(attn_scores, dim=-1)
# 对注意力分数矩阵的每一行做 softmax。
# dim=-1 表示沿最后一个维度归一化。
# 对 [6, 6] 矩阵来说，就是每一行分别做 softmax。
# 每一行的总和都会变成 1。

print("Attention weights:")
print(attn_weights)
# 打印所有词元的注意力权重矩阵。

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# 手动计算第 2 行注意力权重的总和。
# 第 2 行对应的是 "journey" 对所有词元的注意力权重。

print("Row 2 sum:", row_2_sum)
# 打印第 2 行权重之和。

print("All row sums:", attn_weights.sum(dim=-1))
# 打印所有行的权重之和。
# 每一行都应该接近 1。

all_context_vecs = attn_weights @ inputs
# 使用注意力权重矩阵乘以输入向量矩阵。
# [6, 6] @ [6, 3] = [6, 3]。
# 得到每个词元对应的上下文向量。

print("All context vectors:")
print(all_context_vecs)
# 打印所有词元的上下文向量。