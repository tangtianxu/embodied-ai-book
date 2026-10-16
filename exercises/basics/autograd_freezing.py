import torch

def inspect_gradients(stop_encoder_path=False):
    # 步骤1：适配器、动作头可训练，编码器权重固定
    wa = torch.tensor(1.0, requires_grad=True)
    we = torch.tensor(3.0, requires_grad=False)
    wh = torch.tensor(0.5, requires_grad=True)
    # 步骤2：冻结 we，仍可以记录 z 对 u 的依赖
    u = wa * 2.0
    if stop_encoder_path:
        with torch.no_grad():
            z = we * u
    else:
        z = we * u
    # 步骤3：动作头拟合标签 4，计算本次损失的梯度
    prediction = wh * z
    loss = 0.5 * (prediction - 4.0).square()
    loss.backward()
    # 步骤4：None 与数值零有不同含义
    return [None if p.grad is None else p.grad.item()
            for p in (wa, we, wh)]

print(inspect_gradients(False))  # [-3.0, None, -6.0]
print(inspect_gradients(True))   # [None, None, -6.0]
