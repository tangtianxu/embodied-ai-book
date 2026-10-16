"""Hand-written MHA and a pre-norm Transformer block; CPU/GPU agnostic.
No nn.MultiheadAttention or functional SDPA is used inside the implementation.
"""
import math
import torch
from torch import nn


def safe_softmax(scores, allow):
    # 步骤1：把允许掩码扩展到分数矩阵
    allow = torch.broadcast_to(allow, scores.shape)
    # 步骤2：禁止位置退出 softmax 的归一化
    masked = scores.masked_fill(~allow, float('-inf'))
    # 步骤3：用最大值稳定指数；全遮蔽行随后单独处理
    maximum = masked.amax(dim=-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    # 步骤4：计算指数并归一化
    exp = torch.exp(masked - maximum)
    return exp / exp.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(exp.dtype).tiny)


class HandMHA(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0):
        super().__init__()
        if d_model % heads:
            raise ValueError('d_model must be divisible by heads')
        self.heads, self.d_head = heads, d_model // heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, self.d_head).transpose(1, 2)

    def forward(self, x, context=None, key_valid=None, causal=False,
                cache=None, use_cache=False, query_valid=None):
        # x: [B,Tq,D], context: [B,Tk,D]. Cache supports self-attention only.
        if (cache is not None or use_cache) and context is not None:
            raise ValueError('This teaching cache supports self-attention only')
        # 步骤1：确定查询与上下文来源
        source = x if context is None else context
        # 步骤2：投影 Q、K、V，并分成多个头
        q = self.split_heads(self.q(x))
        k = self.split_heads(self.k(source))
        v = self.split_heads(self.v(source))
        past = 0 if cache is None else cache[0].shape[-2]
        if cache is not None:
            k = torch.cat([cache[0], k], dim=-2)
            v = torch.cat([cache[1], v], dim=-2)
        b, _, tq, _ = q.shape
        tk = k.shape[-2]
        # 步骤3：组合有效键与因果关系
        allow = torch.ones((b, 1, tq, tk), dtype=torch.bool, device=x.device)
        if key_valid is not None:
            if tuple(key_valid.shape) != (b, tk):
                raise ValueError('key_valid must cover all cached and new keys')
            allow = allow & key_valid[:, None, None, :]
        if causal:
            qi = past + torch.arange(tq, device=x.device)
            ki = torch.arange(tk, device=x.device)
            allow = allow & (ki[None, :] <= qi[:, None])[None, None]
        # Float32 softmax when projected tensors use a lower-precision type.
        calc_type = torch.float64 if q.dtype == torch.float64 else torch.float32
        # 步骤4：计算点积分数并求权重
        scores = q.to(calc_type) @ k.to(calc_type).transpose(-2, -1)
        weights = safe_softmax(scores / math.sqrt(self.d_head), allow)
        # 步骤5：聚合 V，合头并映射输出
        values = self.dropout(weights).to(v.dtype) @ v
        merged = values.transpose(1, 2).contiguous().reshape(b, tq, -1)
        output = self.out(merged)
        if query_valid is not None:
            output = output * query_valid[..., None].to(output.dtype)
        return output, weights, ((k, v) if use_cache else None)


class PreNormBlock(nn.Module):
    def __init__(self, d_model, heads, expansion=4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = HandMHA(d_model, heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, expansion * d_model),
                                 nn.GELU(), nn.Linear(expansion * d_model, d_model))

    def forward(self, x, key_valid=None, causal=False):
        # 步骤1：预归一化后交换 token 信息
        y, _, _ = self.attn(self.norm1(x), key_valid=key_valid, causal=causal)
        # 步骤2：加入注意力残差
        x = x + y
        # 步骤3：加工逐位置特征并加入残差
        return x + self.ffn(self.norm2(x))


class LoRALinear(nn.Module):
    def __init__(self, base, rank=2, alpha=2.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        # Do not put base(x) inside no_grad: gradients may be needed for x.
        # 步骤1：相加固定主干与低秩修正，保留输入梯度
        return self.base(x) + self.scale * (x @ self.a.T @ self.b.T)
