"""Hand-computable key/value paths; stopping a derivative keeps forward values."""
from pathlib import Path
import json
import math
import torch

HOME = Path(__file__).resolve().parent
DTYPE = torch.float64


def attention_case(route):
    q = torch.tensor(1., dtype=DTYPE, requires_grad=True)
    keys = torch.tensor([0., math.log(3.)], dtype=DTYPE, requires_grad=True)
    values = torch.tensor([0., 2.], dtype=DTYPE, requires_grad=True)
    # K1: use both numerical keys/values; choose their derivative paths separately.
    read_keys = keys.detach() if route in ("k", "both") else keys
    read_values = values.detach() if route in ("v", "both") else values
    weights = torch.softmax(q * read_keys, dim=0)
    prediction = (weights * read_values).sum()
    loss = .5 * prediction.square()
    gq, gk, gv = torch.autograd.grad(loss, [q, keys, values], allow_unused=True)
    # K2: independently derived local gradients before any stopping operation.
    raw_v = prediction.detach() * weights.detach()
    raw_k = prediction.detach() * weights.detach() * (values.detach()-prediction.detach()) * q.detach()
    raw_q = (prediction.detach() * weights.detach()
             * (values.detach()-prediction.detach()) * keys.detach()).sum()
    assert torch.allclose(gq, raw_q, atol=1e-14)
    for computed, expected, stopped in (
        (gk, raw_k, route in ("k", "both")),
        (gv, raw_v, route in ("v", "both")),
    ):
        if stopped:
            assert computed is None
        else:
            assert torch.allclose(computed, expected, atol=1e-14)
    return {
        "q": float(q.detach()), "keys": keys.detach().tolist(), "values": values.detach().tolist(),
        "weights": weights.detach().tolist(), "prediction": float(prediction.detach()),
        "loss": float(loss.detach()), "grad_query": float(gq),
        "grad_keys": None if gk is None else gk.tolist(),
        "grad_values": None if gv is None else gv.tolist(),
    }


def main():
    (HOME / "results").mkdir(parents=True, exist_ok=True)
    rows = {name: attention_case(name) for name in ("none", "k", "v", "both")}
    assert all(row["prediction"] == rows["none"]["prediction"] for row in rows.values())
    result = {
        "scope": "one scalar query and two scalar keys/values; no trained representation",
        "rows": rows,
        "gradient_interpretation": "autodiff derivative of the routed graph; not a finite-difference derivative of the unchanged forward map",
    }
    (HOME / "results/attention_routes.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(rows["none"]))


if __name__ == "__main__":
    main()
