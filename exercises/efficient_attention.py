"""Exact online-softmax arithmetic and integer dot-product references.
NumPy reference only: this is not a FlashAttention GPU kernel or benchmark.
"""
import numpy as np


def online_attention(q, k, v, block=8, allow=None):
    q, k, v = [np.asarray(x, float) for x in (q, k, v)]
    n, d = q.shape
    if block <= 0:
        raise ValueError('block size must be positive')
    mask = np.ones((n,len(k)), bool) if allow is None else np.broadcast_to(allow, (n,len(k)))
    maximum, normalizer = np.full(n, -np.inf), np.zeros(n)
    numerator = np.zeros((n,v.shape[-1]))
    for offset in range(0, len(k), block):
        scores = q @ k[offset:offset+block].T / np.sqrt(d)
        scores = np.where(mask[:,offset:offset+block], scores, -np.inf)
        # 步骤1：合并本块与此前最大分数
        new_max = np.maximum(maximum, scores.max(axis=1))
        # All-masked prefixes are allowed; replace -inf before subtraction.
        safe_max = np.where(np.isfinite(new_max), new_max, 0)
        # 步骤2：把旧累计量换到新尺度
        correction = np.exp(maximum - safe_max)
        # 步骤3：计算本块指数
        p = np.exp(scores - safe_max[:,None])
        # 步骤4：同时更新加权和与分母
        numerator = correction[:,None]*numerator + p @ v[offset:offset+block]
        normalizer = correction*normalizer + p.sum(axis=1)
        maximum = new_max
    return np.divide(numerator, normalizer[:,None], out=np.zeros_like(numerator),
                     where=normalizer[:,None] > 0)


def quantize(x, scale, zero_point=0, qmin=-8, qmax=7):
    if scale <= 0:
        raise ValueError('scale must be positive')
    # 步骤1：按尺度取整，加零点，再限制范围
    return np.clip(np.rint(np.asarray(x)/scale)+zero_point, qmin, qmax).astype(np.int32)


def integer_dot(qx, qw, sx, sw, zx=0, zw=0):
    # 步骤1：减去零点后使用整数累加
    acc = np.dot(np.asarray(qx, np.int64)-zx, np.asarray(qw, np.int64)-zw)
    # 步骤2：用两个尺度恢复实值
    return float(sx*sw*acc), int(acc)


def kv_bytes(layers, batch, tokens, kv_heads, head_dim, bytes_per_value=2):
    return 2*layers*batch*tokens*kv_heads*head_dim*bytes_per_value
