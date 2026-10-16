"""Explicit trajectory boundaries and objective gradients, not a full RL agent."""
import numpy as np


def gae(rewards, values, next_values, terminated, truncated, gamma=.9, lam=.8):
    r, v, nv = [np.asarray(x, dtype=float) for x in (rewards, values, next_values)]
    term, trunc = [np.asarray(x, dtype=bool) for x in (terminated, truncated)]
    if not (r.shape == v.shape == nv.shape == term.shape == trunc.shape):
        raise ValueError('all inputs must have identical [T,N] or [T] shapes')
    # 步骤1：真正终止不自举；其余转移保留下一价值
    delta = r + gamma * (~term) * nv - v
    adv = np.zeros_like(r)
    # 步骤2：从轨迹末尾开始递推
    carry = np.zeros_like(r[0])
    for t in range(len(r) - 1, -1, -1):
        # 步骤3：截断跨回合连接，以 γλ 累积残差
        carry = delta[t] + gamma * lam * (~(term[t] | trunc[t])) * carry
        # 步骤4：保存优势，随后构造回报目标
        adv[t] = carry
    return adv, adv + v


def ppo_surrogate(log_new, log_old, advantages, clip=.2):
    log_new, log_old, advantages = [np.asarray(x, float) for x in
                                   (log_new, log_old, advantages)]
    # 步骤1：对同一已采动作比较新旧概率
    ratio = np.exp(log_new - log_old)
    # 步骤2：形成裁剪候选
    clipped = np.clip(ratio, 1 - clip, 1 + clip)
    # 步骤3：按优势符号保留较保守的收益
    objective = np.minimum(ratio * advantages, clipped * advantages)
    active = np.where(advantages >= 0, ratio <= 1 + clip, ratio >= 1 - clip)
    grad_log_new = -np.where(active, ratio * advantages, 0.0) / ratio.size
    return -float(objective.mean()), grad_log_new


def expectile_loss(values, q_targets, tau=.7):
    if not 0 < tau < 1:
        raise ValueError('tau must be in (0,1)')
    # 步骤1：计算已有动作 Q 与当前 V 的差
    residual = np.asarray(q_targets) - np.asarray(values)
    # 步骤2：较高 Q 使用更大的 expectile 权重
    weights = np.where(residual >= 0, tau, 1 - tau)
    return float(np.mean(weights * residual ** 2)), -2 * weights * residual / residual.size


def squash_log_prob(raw, mu, log_std, action_scale=1.0):
    raw, mu, log_std = np.broadcast_arrays(raw, mu, log_std)
    # 步骤1：计算未压缩高斯变量的对数密度
    gaussian = -.5 * ((raw - mu) / np.exp(log_std)) ** 2 - log_std - .5 * np.log(2*np.pi)
    # log(1-tanh(raw)^2), evaluated stably even for large raw.
    # 步骤2：补偿 tanh 压缩产生的体积变化
    log_jacobian = 2 * (np.log(2) - raw - np.logaddexp(0, -2 * raw))
    return (gaussian - log_jacobian - np.log(action_scale)).sum(axis=-1)


def polyak(target, online, tau=.005):
    return (1 - tau) * np.asarray(target) + tau * np.asarray(online)
