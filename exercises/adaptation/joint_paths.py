"""Two-block attention lesson for joint labels, visibility and gradient routing.

Uses synthetic observation vectors and a six-item vocabulary. It is not pi0.5,
not a pretrained VLM, and not an experiment on preserving semantic knowledge.
"""
from copy import deepcopy
from pathlib import Path
import json
import math
import torch
from torch import nn
import torch.nn.functional as F

HOME = Path(__file__).resolve().parent
DTYPE = torch.float64
WIDTH, BOS, EOS = 2, 0, 5
FLOW_WEIGHT = .5


def masks(prefix_count, discrete_count, action_count, leak=False):
    nb = prefix_count + discrete_count
    backbone = torch.zeros(nb, nb+action_count, dtype=torch.bool)
    backbone[:prefix_count, :prefix_count] = True
    for i in range(discrete_count):
        backbone[prefix_count+i, :prefix_count+i+1] = True
    action = torch.zeros(action_count, nb+action_count, dtype=torch.bool)
    action[:, :prefix_count] = True
    action[:, nb:] = True
    if leak:
        action[:, prefix_count:nb] = True
    return backbone, action


class JointBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("bq", "bk", "bv", "bo", "aq", "ak", "av", "ao"):
            setattr(self, name, nn.Linear(WIDTH, WIDTH, bias=False, dtype=DTYPE))

    def forward(self, xb, xa, prefix_count, route, leak=False):
        kb, vb = self.bk(xb), self.bv(xb)
        ka, va = self.ak(xa), self.av(xa)
        bm, am = masks(prefix_count, len(xb)-prefix_count, len(xa), leak)
        # J2: visibility and derivative routing are distinct operations.
        bs = self.bq(xb) @ torch.cat([kb, ka]).T / math.sqrt(WIDTH)
        bp = torch.softmax(bs.masked_fill(~bm, -torch.inf), dim=-1)
        hb = xb + self.bo(bp @ torch.cat([vb, va]))
        # J3: detach exactly the cross-module key/value paths.
        read_k = kb.detach() if route in ("k", "both") else kb
        read_v = vb.detach() if route in ("v", "both") else vb
        action_keys = torch.cat([read_k, ka])
        action_values = torch.cat([read_v, va])
        score = self.aq(xa) @ action_keys.T / math.sqrt(WIDTH)
        probability = torch.softmax(score.masked_fill(~am, -torch.inf), dim=-1)
        ha = xa + self.ao(probability @ action_values)
        return hb, ha


class JointLesson(nn.Module):
    def __init__(self):
        super().__init__()
        self.words = nn.Embedding(6, WIDTH, dtype=DTYPE)
        self.blocks = nn.ModuleList([JointBlock(), JointBlock()])
        self.token_head = nn.Linear(WIDTH, 6, bias=False, dtype=DTYPE)
        self.action_in = nn.Linear(2, WIDTH, bias=False, dtype=DTYPE)
        self.time_in = nn.Linear(1, WIDTH, bias=False, dtype=DTYPE)
        self.flow_out = nn.Linear(WIDTH, 2, bias=False, dtype=DTYPE)

    def forward(self, sample, routes, leak=False, discrete_override=None):
        # J1: teacher-forced inputs are BOS + all targets except final EOS.
        target_ids = sample["tokens"] if discrete_override is None else discrete_override
        input_ids = torch.tensor([BOS]+target_ids[:-1])
        prefix = torch.tensor(sample["prefix"], dtype=DTYPE)
        xb = torch.cat([prefix, self.words(input_ids)])
        if sample["has_action"]:
            clean = torch.tensor(sample["action"], dtype=DTYPE)
            noise = torch.tensor(sample["noise"], dtype=DTYPE)
            tau = sample["tau"]
            noisy = (1-tau)*noise + tau*clean
            xa = self.action_in(noisy) + self.time_in(torch.full((len(clean), 1), tau, dtype=DTYPE))
        else:
            xa = torch.empty(0, WIDTH, dtype=DTYPE)
        for block, route in zip(self.blocks, routes):
            xb, xa = block(xb, xa, len(prefix), route, leak)
        logits = self.token_head(xb[len(prefix):])
        return logits, self.flow_out(xa)


ROBOT = {"kind": "robot", "prefix": [[1., .1], [-.2, .8]], "tokens": [1, 2, EOS],
         "has_action": True, "action": [[.5, -.2], [.1, .4]],
         "noise": [[-.3, .2], [.7, -.6]], "tau": .4}
VLM = {"kind": "vl_only", "prefix": [[.3, -.8], [.9, .2]], "tokens": [3, 4, EOS],
       "has_action": False}


def loss_terms(model, routes, leak=False, samples=(ROBOT, VLM)):
    # J4: average each objective only over its valid supervised positions.
    token_sum = torch.zeros((), dtype=DTYPE)
    flow_sum = torch.zeros((), dtype=DTYPE)
    count, flow_count = 0, 0
    for sample in samples:
        logits, prediction = model(sample, routes, leak)
        targets = torch.tensor(sample["tokens"])
        token_sum = token_sum + F.cross_entropy(logits, targets, reduction="sum")
        count += len(targets)
        if sample["has_action"]:
            # Same noise -> data convention as accepted section1.10.
            target_velocity = torch.tensor(sample["action"], dtype=DTYPE)-torch.tensor(sample["noise"], dtype=DTYPE)
            flow_sum = flow_sum + (prediction-target_velocity).square().sum()
            flow_count += target_velocity.numel()
    # The VL-only sample has no flow label; it is not a zero-action example.
    flow = .5*flow_sum/flow_count if flow_count else flow_sum
    return token_sum/count, flow


def group(name):
    if name.startswith(("words.", "token_head.")) or any("."+part+"." in name for part in ("bq","bk","bv","bo")):
        return "backbone"
    return "expert"


def gradients(model, routes):
    token, flow = loss_terms(model, routes)
    parameters = list(model.parameters())
    token_g = torch.autograd.grad(token, parameters, retain_graph=True, allow_unused=True)
    flow_g = torch.autograd.grad(flow, parameters, retain_graph=True, allow_unused=True)
    total_g = torch.autograd.grad(token+FLOW_WEIGHT*flow, parameters, allow_unused=True)
    records = {"token": {}, "flow": {}, "total": {}}
    by_parameter = []
    for (name, parameter), gc, gf, gt in zip(model.named_parameters(), token_g, flow_g, total_g):
        zeros = torch.zeros_like(parameter)
        c, f, t = [zeros if g is None else g for g in (gc, gf, gt)]
        assert torch.allclose(t, c+FLOW_WEIGHT*f, atol=1e-14, rtol=1e-12)
        by_parameter.append({"name": name, "group": group(name),
                             "token_gradient_norm": float(c.norm()), "flow_gradient_norm": float(f.norm()),
                             "flow_gradient_is_none": gf is None})
        for label, g in (("token", c), ("flow", f), ("total", t)):
            records[label].setdefault(group(name), 0.)
            records[label][group(name)] += float(g.square().sum())
    for task in records.values():
        for key in task:
            task[key] = math.sqrt(task[key])
    return {"token_loss": float(token.detach()), "flow_loss": float(flow.detach()),
            "gradient_norms": records, "by_parameter": by_parameter}


def train_three(initial, routes):
    model = deepcopy(initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    rows = []
    for step in range(3):
        token, flow = loss_terms(model, routes)
        before = {name: p.detach().clone() for name,p in model.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        # J4: combine objectives, backpropagate once, update the two parameter groups.
        (token+FLOW_WEIGHT*flow).backward()
        optimizer.step()
        change = {"backbone": 0., "expert": 0.}
        for name, p in model.named_parameters():
            change[group(name)] += float((p.detach()-before[name]).square().sum())
        rows.append({"updates_completed_before": step, "token_loss_before": float(token.detach()),
                     "flow_loss_before": float(flow.detach()),
                     "parameter_step_norm": {k:math.sqrt(v) for k,v in change.items()}})
    with torch.no_grad():
        token, flow = loss_terms(model, routes)
        prediction = model(ROBOT, routes)[1]
    return {"trace": rows, "token_loss_after3": float(token), "flow_loss_after3": float(flow),
            "robot_velocity_after3": prediction.tolist()}, model.state_dict()


def main():
    (HOME / "results").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(83)
    initial = JointLesson()
    route_table = {}
    for name,routes in {"joint":["none","none"],"keys_only":["k","k"],
                        "values_only":["v","v"],"insulated":["both","both"],
                        "last_layer_only":["none","both"]}.items():
        route_table[name] = gradients(initial, routes)
    isolated = route_table["insulated"]["gradient_norms"]
    assert isolated["flow"]["backbone"] == 0 and isolated["flow"]["expert"] > 0
    assert isolated["token"]["backbone"] > 0 and isolated["token"]["expert"] == 0
    for name in ("joint","keys_only","values_only","last_layer_only"):
        assert route_table[name]["gradient_norms"]["flow"]["backbone"] > 0
    with torch.no_grad():
        normal = initial(ROBOT, ["none","none"])[1]
        detached = initial(ROBOT, ["both","both"])[1]
        assert torch.equal(normal, detached)       # J3 changes derivatives, not values.
        altered = [2,1,EOS]
        correct_changed = initial(ROBOT, ["both","both"], discrete_override=altered)[1]
        leaky = initial(ROBOT, ["both","both"], leak=True)[1]
        leaky_changed = initial(ROBOT, ["both","both"], leak=True, discrete_override=altered)[1]
        assert torch.equal(detached, correct_changed)
        assert not torch.allclose(leaky, leaky_changed, atol=1e-12)
        token_examples = []
        for sample in (ROBOT, VLM):
            logits, _ = initial(sample, ["both","both"])
            target_ids = torch.tensor(sample["tokens"])
            per_position = F.cross_entropy(logits, target_ids, reduction="none")
            token_examples.append({
                "kind": sample["kind"], "input_ids": [BOS]+sample["tokens"][:-1],
                "target_ids": sample["tokens"],
                "target_probabilities": torch.exp(-per_position).tolist(),
                "negative_log_probabilities": per_position.tolist(),
            })
        # J4: unequal action lengths use element counts; all-VL batches omit flow.
        short_probe = {**ROBOT, "action":[[.9,.2]], "noise":[[-.4,.1]]}
        _, full_flow = loss_terms(initial, ["both","both"], samples=(ROBOT,))
        _, short_flow = loss_terms(initial, ["both","both"], samples=(short_probe,))
        _, combined_flow = loss_terms(initial, ["both","both"], samples=(ROBOT,short_probe))
        _, no_action_flow = loss_terms(initial, ["both","both"], samples=(VLM,))
        assert torch.allclose(combined_flow,(4*full_flow+2*short_flow)/6,atol=1e-14)
        assert float(no_action_flow) == 0.
        aggregation_probe = {
            "scope":"extra loss-reduction probes only, not added to the three-update training batch",
            "valid_action_element_counts":[4,2],
            "individual_flow_losses":[float(full_flow),float(short_flow)],
            "combined_flow_loss":float(combined_flow),
            "all_vl_flow_loss":float(no_action_flow),
        }
    # Updating backbone with CE still changes expert inputs at the next forward.
    changed_base = deepcopy(initial)
    token, _ = loss_terms(changed_base, ["both","both"])
    token.backward()
    with torch.no_grad():
        for name,p in changed_base.named_parameters():
            if group(name) == "backbone" and p.grad is not None:
                p.add_(p.grad, alpha=-.03)
        changed_prediction = changed_base(ROBOT, ["both","both"])[1]
    change_norm = float((changed_prediction-detached).norm())
    assert change_norm > 1e-12
    training, states = {}, {}
    for name,routes in {"joint":["none","none"],"insulated":["both","both"]}.items():
        training[name], states[name] = train_three(initial, routes)
    bm, am = masks(2, 3, 2)
    result = {
        "scope": "synthetic two-block one-head attention, mixed robot/VL labels; no pretrained VLM or semantic-retention experiment",
        "seed": 83, "width": WIDTH, "dtype": str(DTYPE), "samples": [ROBOT,VLM],
        "vocabulary": {"BOS":BOS,"robot_tokens":[1,2],"word_tokens":[3,4],"EOS":EOS},
        "loss_reduction": {"token":"sum over6 valid target positions /6",
                           "flow":"0.5 * sum over4 robot velocity elements /4; VL-only sample omitted",
                           "total":"token +0.5*flow"},
        "attention_allow_mask_robot": torch.cat([bm,am]).int().tolist(),
        "initial_robot_velocity": detached.tolist(),
        "initial_token_examples": token_examples,
        "label_aggregation_probe": aggregation_probe,
        "gradient_routes": route_table, "three_update_comparisons": training,
        "detach_preserves_forward": True,
        "label_dependency": {"correct_mask_delta_norm": float((correct_changed-detached).norm()),
                             "leaky_mask_delta_norm": float((leaky_changed-leaky).norm()),
                             "insulated_but_leaky_prediction":leaky.tolist(),
                             "after_counterfactual_label_swap":leaky_changed.tolist(),
                             "scope":"swap teacher tokens only, holding clean/noise fixed; dependency probe, not a new demonstration"},
        "expert_fixed_base_ce_update_changes_prediction_norm": change_norm,
    }
    (HOME / "results/joint_paths.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    torch.save({"initial":initial.state_dict(),"after3":states},HOME/"results/joint_models.pt")
    print(json.dumps({"flow_to_backbone_insulated":isolated["flow"]["backbone"],
                      "leaky_label_dependency":result["label_dependency"]["leaky_mask_delta_norm"],
                      "base_still_changes_features":change_norm}))


if __name__ == "__main__":
    main()
