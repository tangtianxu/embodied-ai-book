import torch
from torch import nn
import torch.nn.functional as F

class LowRankLinear(nn.Module):
    def __init__(self, base_layer, rank):
        super().__init__()
        # 1. 保留基础映射的数值，固定其参数。
        self.base = base_layer
        self.base.requires_grad_(False)
        self.A = nn.Parameter(base_layer.weight.new_empty(rank, base_layer.in_features)
                              .normal_(0.0, 0.01))
        self.B = nn.Parameter(base_layer.weight.new_zeros(base_layer.out_features, rank))

    def forward(self, x):
        # 2. 先形成低秩内部表示，再映射为输出修正。
        correction = F.linear(F.linear(x, self.A), self.B)
        # 3. 原映射与修正相加；仍保留经过它们到输入的求导路径。
        return self.base(x) + correction
