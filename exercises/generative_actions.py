"""Action losses, DDPM identities and an actual conditional-flow toy learner."""
import json
from pathlib import Path
import numpy as np


def masked_mse(pred, target, valid):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError('prediction/target shapes differ')
    # 步骤1：确定有效监督位置
    valid = np.broadcast_to(np.asarray(valid, bool), pred.shape)
    # 步骤2：检查有效数量
    count = int(valid.sum())
    if count == 0:
        raise ValueError('empty supervision: skip this batch explicitly')
    # Select first so invalid NaN padding cannot poison the loss.
    # 步骤3：先选择再求误差，排除缺失填充
    residual = pred[valid] - target[valid]
    grad = np.zeros_like(pred, dtype=float)
    grad[valid] = 2 * residual / count
    return float(np.mean(residual ** 2)), grad


def ddpm_pair(clean, noise, alpha_bar):
    if not 0 < alpha_bar < 1:
        raise ValueError('alpha_bar must be strictly between zero and one')
    # 步骤1：从干净动作和已知噪声构造训练输入
    noisy = np.sqrt(alpha_bar) * clean + np.sqrt(1 - alpha_bar) * noise
    # 步骤2：以所采噪声作为标签
    return noisy, noise


def clean_from_noise(noisy, predicted_noise, alpha_bar):
    return (noisy - np.sqrt(1 - alpha_bar) * predicted_noise) / np.sqrt(alpha_bar)


def flow_pair(clean, noise, tau):
    return (1 - tau) * noise + tau * clean, clean - noise


def euler_sample(field, noise, condition, steps=32):
    if steps <= 0:
        raise ValueError('steps must be positive')
    # 步骤1：从新噪声初始化生成样本
    x = np.array(noise, dtype=float, copy=True)
    for i in range(steps):
        # 步骤2：读取当前场，按欧拉步推进
        x += field(x, i / steps, condition) / steps
    return x


def train_toy_flow(seed=13, updates=1600):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(3, 48), nn.SiLU(), nn.Linear(48, 48),
                        nn.SiLU(), nn.Linear(48, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.002)
    def batch(n):
        # 步骤1：采样条件及其干净数据
        c = (torch.randint(0, 2, (n, 1)) * 2 - 1).float()
        clean = c + 0.2 * torch.randn(n, 1)
        # 步骤2：采样噪声和生成时间
        noise, tau = torch.randn(n, 1), torch.rand(n, 1)
        # 步骤3：构造路径点与速度标签
        x, target = (1 - tau) * noise + tau * clean, clean - noise
        # 步骤4：打包网络输入与标签
        return torch.cat([x, tau, c], dim=-1), target
    for _ in range(updates):
        inputs, target = batch(256)
        # 步骤5：计算速度匹配损失
        loss = (net(inputs) - target).square().mean()
        # 步骤6：清梯度、反传并更新参数
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    result = {'seed': seed, 'updates': updates, 'torch': torch.__version__}
    with torch.no_grad():
        inputs, target = batch(8192)
        result['heldout_mse'] = float((net(inputs) - target).square().mean())
        result['zero_field_mse'] = float(target.square().mean())
        for sign in (-1, 1):
            x, c = torch.randn(4096, 1), torch.full((4096, 1), float(sign))
            for i in range(64):
                tau = torch.full_like(x, i / 64)
                x = x + net(torch.cat([x, tau, c], dim=-1)) / 64
            result[str(sign)] = {'mean': float(x.mean()), 'std': float(x.std()),
                                'target_mean': sign, 'target_std': 0.2}
    return result


if __name__ == '__main__':
    result = train_toy_flow()
    Path(__file__).with_name('flow_training_results.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
