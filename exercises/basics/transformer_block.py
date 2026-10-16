import torch
from torch import nn
from attention_mha import TutorialMHA

class TutorialBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = TutorialMHA(width, heads)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x, allow=None):
        # 步骤1：归一化后，按掩码交换位置间信息
        delta, _ = self.attn(self.norm1(x), allow=allow)
        # 步骤2：直接路径上的 x 与注意力修正相加
        x = x + delta
        # 步骤3：同一个 FFN 分别加工每个 token
        delta = self.ffn(self.norm2(x))
        # 步骤4：再加入残差，输出仍是 [B,T,D]
        return x + delta

block = TutorialBlock(width=12, heads=3)
x = torch.randn(2, 5, 12)
allow = torch.ones(5, 5, dtype=torch.bool).tril()
print(block(x, allow).shape)  # torch.Size([2, 5, 12])
