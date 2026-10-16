"""Readable single-head attention, backward, normalization and RoPE references.
Boolean masks always mean True = allowed. All-masked rows return zeros.
"""
import numpy as np


def masked_softmax(scores, allow=None):
    scores = np.asarray(scores, dtype=np.float64)
    if allow is None:
        allow = np.ones(scores.shape, dtype=bool)
    allow = np.broadcast_to(np.asarray(allow, dtype=bool), scores.shape)
    masked = np.where(allow, scores, -np.inf)
    maximum = np.max(masked, axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    exp = np.exp(masked - maximum)
    total = exp.sum(axis=-1, keepdims=True)
    return np.divide(exp, total, out=np.zeros_like(exp), where=total > 0)


def attention(q, k, v, allow=None):
    q, k, v = [np.asarray(a, dtype=np.float64) for a in (q, k, v)]
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("Q/K feature size or K/V sequence size mismatch")
    p = masked_softmax(q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1]), allow)
    return p @ v, p


def attention_backward(q, k, v, p, grad_out):
    """One head, without dropout. Upstream gradient is for output P @ V."""
    # 步骤1：从聚合输出反传到 V 和权重
    grad_v = p.T @ grad_out
    grad_p = grad_out @ v.T
    # 步骤2：通过 softmax 比较各位置相对贡献
    grad_s = p * (grad_p - (grad_p * p).sum(axis=-1, keepdims=True))
    scale = np.sqrt(q.shape[-1])
    return grad_s @ k / scale, grad_s.T @ q / scale, grad_v


def layer_norm(x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
        x.var(axis=-1, keepdims=True) + eps)


def rms_norm(x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)


def rope(x, positions, base=10000.0):
    """Adjacent-pair layout; x is [B,H,T,D], positions is [T]."""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE feature size must be even")
    angle = np.asarray(positions)[:, None] * base ** (-np.arange(0, d, 2) / d)
    c, s = np.cos(angle), np.sin(angle)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * c - odd * s
    out[..., 1::2] = even * s + odd * c
    return out
