"""Small teaching adaptation of DiT conditioning; source/terms: ../THIRD_PARTY_NOTICES.md."""
from pathlib import Path
import copy
import json
import platform
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent


class ConditionedBlock(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(width, 2, batch_first=True, dropout=0)
        self.feedforward = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                        nn.Linear(2 * width, width))
        self.modulation = nn.Linear(width, 6 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, condition):
        # A1：条件产生两组偏移、缩放增量和残差门控。
        b1, s1, a1, b2, s2, a2 = self.modulation(condition).chunk(6, dim=-1)
        # A2：缩放增量为0时，1+s保留归一化特征。
        u = (1 + s1[:, None]) * self.norm1(x) + b1[:, None]
        delta, _ = self.attention(u, u, u, need_weights=False)
        # A3：门控为0时，分支修正不改变直接路径。
        x = x + a1[:, None] * delta
        # A4：前馈子层有自己的一组调制和门控。
        u = (1 + s2[:, None]) * self.norm2(x) + b2[:, None]
        return x + a2[:, None] * self.feedforward(u)


class TeachingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition_encoder = nn.Linear(2, 4)
        self.block = ConditionedBlock()
        self.final_norm = nn.LayerNorm(4, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Linear(4, 8)
        self.head = nn.Linear(4, 2)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)

    def forward(self, x, raw_condition):
        c = torch.nn.functional.silu(self.condition_encoder(raw_condition))
        x = self.block(x, c)
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        features = (1 + scale[:, None]) * self.final_norm(x) + shift[:, None]
        return self.head(features)


def norm(parameters):
    gradients = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(gradients))) if gradients else None


def modulation_row_norm(layer, rows):
    pieces = [p.grad[rows].reshape(-1) for p in [layer.weight, layer.bias]]
    return float(torch.linalg.vector_norm(torch.cat(pieces)))


def run(model, inputs, conditions, target):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    steps = []
    for step in range(3):
        # U1：用当前参数重新计算输出与任务误差。
        optimizer.zero_grad(set_to_none=True)
        predicted = model(inputs, conditions)
        loss = 0.5 * (predicted - target).square().sum()
        # U2：更新前分别检查输出头、门控、分支与条件编码器的梯度。
        loss.backward()
        gate_rows = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23])
        other_rows = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19])
        steps.append({
            "backward_index": step, "parameter_updates_completed": step,
            "loss": float(loss.detach()), "prediction": predicted.detach().tolist(),
            "gradient_norms": {
                "output_head": norm(model.head.parameters()),
                "block_gate_rows": modulation_row_norm(model.block.modulation, gate_rows),
                "block_scale_shift_rows": modulation_row_norm(model.block.modulation, other_rows),
                "attention_and_ffn": norm(list(model.block.attention.parameters()) +
                                          list(model.block.feedforward.parameters())),
                "condition_encoder": norm(model.condition_encoder.parameters()),
                "final_modulation": norm(model.final_modulation.parameters()),
            },
            "parameters_with_none_grad": [name for name, p in model.named_parameters()
                                          if p.grad is None],
        })
        # U3：只按当前梯度更新，无动量和权重衰减。
        optimizer.step()
    return steps


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(41)
    inputs = torch.randn(2, 2, 4)
    conditions = torch.tensor([[1., 0.], [0., 1.]])
    target = torch.tensor([[[0.4, -0.6], [0.8, 0.2]],
                           [[-0.3, 0.7], [0.1, -0.5]]])
    regular = TeachingModel()
    zero = copy.deepcopy(regular)
    nn.init.zeros_(zero.head.weight)
    nn.init.zeros_(zero.head.bias)
    initial_state = copy.deepcopy(regular.state_dict())
    zero_initial_state = copy.deepcopy(zero.state_dict())
    c = torch.nn.functional.silu(zero.condition_encoder(conditions))
    torch.testing.assert_close(zero.block(inputs, c), inputs, rtol=0, atol=0)
    regular_steps = run(regular, inputs, conditions, target)
    zero_steps = run(zero, inputs, conditions, target)
    r0 = regular_steps[0]["gradient_norms"]
    assert r0["block_gate_rows"] > 0
    assert r0["block_scale_shift_rows"] == r0["attention_and_ffn"] == r0["condition_encoder"] == 0
    z0, z1, z2 = [row["gradient_norms"] for row in zero_steps]
    assert z0["output_head"] > 0 and all(v == 0 for k, v in z0.items() if k != "output_head")
    assert z1["block_gate_rows"] > 0 and z1["final_modulation"] > 0
    assert z1["block_scale_shift_rows"] == z1["attention_and_ffn"] == z1["condition_encoder"] == 0
    assert all(v > 0 for v in z2.values())
    assert all(not row["parameters_with_none_grad"] for row in regular_steps + zero_steps)
    # F1：同一条件的两个位置共享gamma、beta，反向贡献要在位置上求和。
    features = torch.tensor([[2., 0.5], [1., -1.]])
    incoming = torch.tensor([[-0.1, 0.1], [0.2, -0.3]])
    gamma = torch.ones(2, requires_grad=True)
    beta = torch.zeros(2, requires_grad=True)
    ((gamma * features + beta) * incoming).sum().backward()
    torch.testing.assert_close(gamma.grad, torch.tensor([0., 0.35]), atol=1e-15, rtol=0)
    torch.testing.assert_close(beta.grad, torch.tensor([0.1, -0.2]))
    # F2：一条局部AdaLN残差子层，逐项查看候选修正与门控后的输出。
    local_x = torch.tensor([1., 3., 5.])
    normalized = torch.nn.functional.layer_norm(local_x, (3,), eps=1e-5)
    scale = torch.tensor([0.5, 0., -0.5])
    shift = torch.tensor([0.1, 0.2, -0.1])
    gate = torch.tensor([0.2, -0.1, 0.])
    branch_diagonal = torch.tensor([0.2, -0.1, 0.3])
    modulated = (1 + scale) * normalized + shift
    candidate = branch_diagonal * modulated
    local_output = local_x + gate * candidate
    # F3：零门控会遮住错误的内部缩放，因此另开门控再比较。
    opened_gate = torch.full((3,), 0.1)
    correct_opened = local_x + opened_gate * branch_diagonal * normalized
    wrong_opened = local_x + opened_gate * branch_diagonal * torch.zeros_like(normalized)
    assert float((correct_opened - wrong_opened).abs().max()) > 0.01
    # F4：若候选分支和门控同时全零，二者都没有打开这条乘法路径的梯度。
    dead_gate = torch.zeros(3, requires_grad=True)
    dead_weights = torch.zeros(3, requires_grad=True)
    dead_output = local_x + dead_gate * dead_weights * normalized
    dead_output.sum().backward()
    assert torch.count_nonzero(dead_gate.grad) == torch.count_nonzero(dead_weights.grad) == 0
    (ROOT / "results").mkdir(exist_ok=True)
    with torch.no_grad():
        final_losses = {
            "regular_output_head": float(0.5 * (regular(inputs, conditions) - target).square().sum()),
            "zero_output_head": float(0.5 * (zero(inputs, conditions) - target).square().sum()),
        }
    record = {
        "status": "passed", "python": platform.python_version(), "torch": torch.__version__,
        "scope": "original one-block teaching model, two known inputs and three actual SGD updates per initialization; not full DiT/VLA training",
        "seed": 41, "input_shape": list(inputs.shape), "target_shape": list(target.shape),
        "regular_output_head": regular_steps, "zero_output_head": zero_steps,
        "loss_after_three_completed_updates": final_losses,
        "shared_film": {"features": features.tolist(), "incoming": incoming.tolist(),
                        "gamma_gradient": gamma.grad.tolist(), "beta_gradient": beta.grad.tolist()},
        "local_adaln": {"input": local_x.tolist(), "normalized": normalized.tolist(),
                        "scale_offset": scale.tolist(), "shift": shift.tolist(), "gate": gate.tolist(),
                        "modulated": modulated.tolist(), "candidate": candidate.tolist(),
                        "output": local_output.tolist(),
                        "correct_with_open_gate_zero_scale_shift": correct_opened.tolist(),
                        "wrong_with_open_gate_zero_scale_shift": wrong_opened.tolist()},
        "zero_branch_and_zero_gate": {"gate_gradient": dead_gate.grad.tolist(),
                                     "branch_weights_gradient": dead_weights.grad.tolist()},
    }
    torch.save({"initial_regular": initial_state, "initial_zero_head": zero_initial_state,
                "after_three_regular": regular.state_dict(), "after_three_zero_head": zero.state_dict(),
                "inputs": inputs, "conditions": conditions, "target": target},
               ROOT / "results/reference_models.pt")
    output = ROOT / "results/conditioning_paths.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
