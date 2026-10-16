import torch

def safe_softmax(scores, allow):
    # 步骤1：将可读关系扩展到分数的所有批次和头
    allow = torch.broadcast_to(allow, scores.shape)
    # 步骤2：禁止位置不参与指数归一化
    masked = scores.masked_fill(~allow, float('-inf'))
    # 步骤3：减去最大值，整行禁止时采用零值基准
    maximum = masked.amax(dim=-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    # 步骤4：有效指数除以有效分母
    weights = torch.exp(masked - maximum)
    return weights / weights.sum(-1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).tiny)

import math
import torch
from torch import nn

class TutorialMHA(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.d = width // heads
        # 步骤1：每头矩阵并排存放为一个大投影
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def split(self, x):
        b, t, _ = x.shape
        # 步骤2：[B,T,H*d] -> [B,T,H,d] -> [B,H,T,d]
        return x.reshape(b, t, self.heads, self.d).transpose(1, 2)

    def forward(self, x, context=None, allow=None):
        source = x if context is None else context
        q = self.split(self.q(x))
        k = self.split(self.k(source))
        v = self.split(self.v(source))
        # 步骤3：每个头分别得到 [B,H,Tq,Tk] 的分数
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        if allow is None:
            allow = torch.ones_like(scores, dtype=torch.bool)
        # 步骤4：safe_softmax 见上节，禁止位置不参与归一化
        weights = safe_softmax(scores, allow)
        per_head = weights @ v
        b, _, tq, _ = per_head.shape
        # 步骤5：先换回位置轴，再把每头内容并排放在一起
        merged = per_head.transpose(1, 2).reshape(b, tq, -1)
        return self.out(merged), weights

model = TutorialMHA(width=12, heads=3)
x = torch.randn(2, 3, 12)        # 3 个查询位置
context = torch.randn(2, 7, 12)  # 7 个可读位置
output, weights = model(x, context)
print(output.shape)   # torch.Size([2, 3, 12])
print(weights.shape)  # torch.Size([2, 3, 3, 7])
