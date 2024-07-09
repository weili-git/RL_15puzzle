import torch
import torch.nn.functional as F

# 定义原始矩阵
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
])

# 生成mask
mask = matrix > 3

# 生成one-hot编码
one_hot = F.one_hot(matrix, num_classes=9)

# 对符合条件的位置应用mask
one_hot[mask] = torch.zeros_like(one_hot[mask])  # 将符合条件的位置置为全零

print(one_hot)