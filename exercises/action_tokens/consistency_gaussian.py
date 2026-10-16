"""Consistency distillation with an analytic Gaussian teacher, not a robot VLA.

Uses the boundary skip/output scalings from the pinned authors' implementation.
Adjacent states are advanced by its exact Gaussian flow here. Euler error is a
separate diagnostic; it does not silently contaminate the training labels.
"""
from copy import deepcopy
from pathlib import Path
import json
import math
import torch
from torch import nn

HOME = Path(__file__).resolve().parent
MU, DATA_STD, EPS = 1., .5, .05
DTYPE = torch.float64


def advance(x, high, low):
    return MU + torch.sqrt((DATA_STD**2 + low**2) / (DATA_STD**2 + high**2)) * (x - MU)


def endpoint(x, sigma):
    return advance(x, sigma, torch.full_like(sigma, EPS))


class EndpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(),
                                 nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x, sigma):
        # C0: the boundary holds for every weight value, not just after training.
        skip = DATA_STD**2 / ((sigma - EPS)**2 + DATA_STD**2)
        out = (sigma - EPS) * DATA_STD / torch.sqrt(sigma**2 + DATA_STD**2)
        network_input = torch.cat(
            [x / torch.sqrt(sigma**2 + DATA_STD**2), torch.log(sigma)], dim=-1)
        return skip * x + out * self.net(network_input)


def norm(parameters):
    return math.sqrt(sum(float(p.grad.square().sum()) for p in parameters if p.grad is not None))


def numerical_example():
    xh = torch.tensor([[1.5]], dtype=DTYPE)
    high = torch.tensor([[1.]], dtype=DTYPE)
    low = torch.tensor([[.5]], dtype=DTYPE)
    exact_low = advance(xh, high, low)
    drift = high / (DATA_STD**2 + high**2) * (xh - MU)
    euler_low = xh + (low - high) * drift
    f_high, f_low = endpoint(xh, high), endpoint(exact_low, low)
    assert torch.allclose(f_high, f_low, atol=1e-14)
    # Two shared-noise perturbations of a single clean point are generally not
    # two exact points on the marginal probability-flow trajectory.
    clean, shared_noise = 1.3, .4
    ct_high, ct_low = clean + 1. * shared_noise, clean + .5 * shared_noise
    ct_advanced = float(advance(torch.tensor([[ct_high]], dtype=DTYPE), high, low))
    assert abs(ct_advanced - ct_low) > .01
    return {"high_state": float(xh), "high_sigma": 1., "low_sigma": .5,
            "exact_low_state": float(exact_low), "euler_low_state": float(euler_low),
            "exact_endpoint_from_high": float(f_high),
            "exact_endpoint_from_low": float(f_low),
            "endpoint_after_euler": float(endpoint(euler_low, low)),
            "ct_shared_clean": clean, "ct_shared_noise": shared_noise,
            "ct_high": ct_high, "ct_low": ct_low, "pf_low_from_ct_high": ct_advanced}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(67)
    model = EndpointModel().to(dtype=DTYPE)
    target = deepcopy(model).requires_grad_(False)
    initial = deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    grid = torch.linspace(EPS, 1., 9, dtype=DTYPE)
    z_eval = torch.linspace(-2., 2., 41, dtype=DTYPE).reshape(-1, 1)

    def evaluate():
        with torch.no_grad():
            xx = MU + math.sqrt(DATA_STD**2 + 1.) * z_eval
            ss = torch.ones_like(xx)
            return float(torch.square(model(xx, ss) - endpoint(xx, ss)).mean())

    records, first_steps = [], []
    first_sample_trace = None
    initial_error = evaluate()
    for step in range(2000):
        # C1: sample a high-noise point and one adjacent lower level.
        indices = torch.randint(1, len(grid), (64,))
        high, low = grid[indices, None], grid[indices - 1, None]
        z = torch.randn(64, 1, dtype=DTYPE)
        x_high = MU + torch.sqrt(DATA_STD**2 + high**2) * z
        # C2: fixed teacher gives a state, not a clean action label.
        with torch.no_grad():
            x_low = advance(x_high, high, low)
            reference = target(x_low, low)
        # C3-C4: only online parameters receive this loss gradient.
        prediction = model(x_high, high)
        loss = .5 * torch.square(prediction - reference).mean()
        if step == 0:
            with torch.no_grad():
                network_input = torch.cat(
                    [x_high / torch.sqrt(high**2 + DATA_STD**2), torch.log(high)], dim=-1)
                hidden = model.net[:-1](network_input)
                network_output = model.net[-1](hidden)
                skip = DATA_STD**2 / ((high-EPS)**2 + DATA_STD**2)
                out = (high-EPS)*DATA_STD / torch.sqrt(high**2+DATA_STD**2)
                # One sample's contribution to one shared output weight.
                contributions = (prediction-reference) / 64 * out * hidden[:, 0:1]
                first_sample_trace = {
                    "z": float(z[0, 0]), "high_sigma": float(high[0, 0]),
                    "low_sigma": float(low[0, 0]), "x_high": float(x_high[0, 0]),
                    "teacher_x_low": float(x_low[0, 0]),
                    "c_skip_high": float(skip[0, 0]), "c_out_high": float(out[0, 0]),
                    "network_input_high": network_input[0].tolist(),
                    "network_F_high": float(network_output[0, 0]),
                    "online_endpoint_prediction": float(prediction[0, 0]),
                    "target_endpoint_prediction": float(reference[0, 0]),
                    "analytic_endpoint": float(endpoint(x_high[0:1], high[0:1])),
                    "dloss_dprediction_sample0": float((prediction-reference)[0, 0] / 64),
                    "last_hidden_feature0": float(hidden[0, 0]),
                    "output_weight0_sample0_gradient": float(contributions[0, 0]),
                    "output_weight0_gradient_sum_expected": float(contributions.sum())}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            observed = float(model.net[-1].weight.grad[0, 0])
            assert abs(observed-first_sample_trace["output_weight0_gradient_sum_expected"]) < 1e-14
            first_sample_trace["output_weight0_gradient_observed"] = observed
        assert all(p.grad is None for p in target.parameters())
        gradient_norm = norm(model.parameters())
        assert gradient_norm > 0
        before = float(model.net[0].weight[0, 0].detach())
        if step < 3:
            first_steps.append({"updates_completed": step, "pair_loss": float(loss.detach()),
                                "online_gradient_norm": gradient_norm,
                                "target_has_any_gradient": False, "first_weight_before": before})
        optimizer.step()                           # C5: update online network.
        with torch.no_grad():                     # C6: EMA is not backpropagation.
            for tp, p in zip(target.parameters(), model.parameters()):
                tp.mul_(.95).add_(p, alpha=.05)
        if step < 3:
            first_steps[-1]["first_weight_after"] = float(model.net[0].weight[0, 0].detach())
        if step in (0, 1, 2, 99, 499, 999, 1999):
            records.append({"updates_completed": step + 1, "sampled_pair_loss": float(loss.detach()),
                            "one_call_endpoint_grid_mse": evaluate()})
    final_error = evaluate()
    assert final_error < initial_error
    with torch.no_grad():
        boundary_x = torch.tensor([[-2.], [0.], [3.]], dtype=DTYPE)
        assert torch.equal(model(boundary_x, torch.full_like(boundary_x, EPS)), boundary_x)
        track = []
        for sigma in [1., .75, .5, .25, EPS]:
            ss = torch.tensor([[sigma]], dtype=DTYPE)
            state = advance(torch.tensor([[1.5]], dtype=DTYPE),
                            torch.ones((1, 1), dtype=DTYPE), ss)
            track.append({"sigma": sigma, "state": float(state),
                          "analytic_endpoint": float(endpoint(state, ss)),
                          "learned_endpoint": float(model(state, ss))})
    result = {
        "scope": "one conditional Gaussian; analytic exact teacher; 2000 actual Adam updates; no robot or image model",
        "seed": 67, "mu": MU, "data_std": DATA_STD, "boundary_sigma": EPS,
        "dtype": str(DTYPE), "torch": torch.__version__, "noise_grid": grid.tolist(),
        "batch_size": 64, "ema_decay": .95, "learning_rate": .002,
        "teacher_solver": "exact analytic Gaussian flow, not the authors' Heun image teacher",
        "initial_one_call_grid_mse": initial_error, "final_one_call_grid_mse": final_error,
        "first_sample_trace": first_sample_trace,
        "first_three_updates": first_steps, "training_snapshots": records,
        "same_trajectory": track, "numeric_example": numerical_example(),
        "sampling_scope": "training samples the exact Gaussian marginal; evaluation is an equally weighted deterministic z grid in [-2,2], not Gaussian expected MSE",
    }
    (HOME / "results").mkdir(exist_ok=True)
    (HOME / "results/consistency_gaussian.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"initial": initial, "after_2000": model.state_dict(), "target": target.state_dict()},
               HOME / "results/consistency_models.pt")
    print(json.dumps({"updates": 2000, "initial_endpoint_grid_mse": initial_error,
                      "final_endpoint_grid_mse": final_error}))


if __name__ == "__main__":
    main()
