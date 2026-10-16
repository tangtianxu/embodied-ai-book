"""Reuse the original LoRA layer; trace actual updates and frozen/input paths."""
from pathlib import Path
import json
import torch
from torch import nn
from low_rank import LowRankLinear

HOME = Path(__file__).resolve().parent
DTYPE = torch.float64


def make_layer(a=(1., -1.)):
    base = nn.Linear(2, 2, bias=False, dtype=DTYPE)
    with torch.no_grad():
        base.weight.copy_(torch.eye(2, dtype=DTYPE))
    layer = LowRankLinear(base, 1)
    with torch.no_grad():
        layer.A.copy_(torch.tensor([a], dtype=DTYPE))
    return layer


def main():
    (HOME / "results").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    layer = make_layer()
    x = torch.tensor([[1., 2.]], dtype=DTYPE)
    target = torch.tensor([[1., 0.]], dtype=DTYPE)
    original_weight = layer.base.weight.detach().clone()
    original_a = layer.A.detach().clone()
    optimizer = torch.optim.SGD([layer.A, layer.B], lr=.1)
    trace = []
    for step in range(3):
        # L1: fixed pretrained W, nonzero A and initially zero B.
        current_x = x.clone().requires_grad_(True)
        # L2-L3: original forward uses W x + B(A x).
        prediction = layer(current_x)
        loss = .5 * (prediction - target).square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        g = (prediction - target).detach()
        expected_a = layer.B.detach().T @ g.T @ current_x.detach()
        expected_b = g.T @ (current_x.detach() @ layer.A.detach().T)
        expected_x = g @ (layer.base.weight.detach() + layer.B.detach() @ layer.A.detach())
        assert torch.allclose(layer.A.grad, expected_a, atol=1e-14)
        assert torch.allclose(layer.B.grad, expected_b, atol=1e-14)
        assert torch.allclose(current_x.grad, expected_x, atol=1e-14)
        assert layer.base.weight.grad is None
        row = {"updates_completed": step, "A": layer.A.detach().tolist(),
               "B": layer.B.detach().tolist(), "prediction": prediction.detach().tolist(),
               "loss": float(loss.detach()), "grad_A": layer.A.grad.tolist(),
               "grad_B": layer.B.grad.tolist(), "grad_x": current_x.grad.tolist()}
        optimizer.step()
        row["A_after"] = layer.A.detach().tolist()
        row["B_after"] = layer.B.detach().tolist()
        trace.append(row)
    assert torch.equal(layer.base.weight, original_weight)
    assert not torch.equal(layer.A, original_a)
    # L4: merge the linear correction; dense matrix size is unchanged.
    merged_weight = layer.base.weight.detach() + layer.B.detach() @ layer.A.detach()
    merged_output = x @ merged_weight.T
    assert torch.allclose(layer(x), merged_output, atol=1e-14)
    # A nonzero A can still give A x=0; that sample cannot start B learning.
    null_layer = make_layer()
    null_x = torch.tensor([[1., 1.]], dtype=DTYPE)
    (.5 * null_layer(null_x).square().sum()).backward()
    assert torch.equal(null_layer.A.grad, torch.zeros_like(null_layer.A))
    assert torch.equal(null_layer.B.grad, torch.zeros_like(null_layer.B))
    # Wrapping the whole layer removes adapter and input paths alike.
    with torch.no_grad():
        no_graph = layer(x.clone().requires_grad_(True))
    assert not no_graph.requires_grad
    result = {
        "scope": "one synthetic 2x2 linear layer, rank1, three actual SGD updates, no VLA training",
        "dtype": str(DTYPE), "x": x.tolist(), "target": target.tolist(),
        "loss": "0.5 * summed squared error", "learning_rate": .1,
        "trace": trace, "merged_weight": merged_weight.tolist(),
        "merged_prediction": merged_output.tolist(), "base_weight_unchanged": True,
        "nonzero_A_null_input_starts_neither_factor": True,
        "whole_layer_no_grad_output_requires_grad": no_graph.requires_grad,
    }
    (HOME / "results/lora_paths.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"updates": 3, "losses_before_updates": [r["loss"] for r in trace],
                      "merged_equal": True}))


if __name__ == "__main__":
    main()
