"""Trace three original DAgger rounds; the imported legacy program is unchanged.

The narrow/broad BC updates preserve the original RNG sequence for comparison.
This scalar environment has no robot units, images, actuator or safety model.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import numpy as np

BASE = Path(__file__).resolve().parent
LEGACY = Path(__file__).with_name("legacy_train_bc_dagger.py")
LEGACY_SHA256 = hashlib.sha256(LEGACY.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
PROBES = np.array([-.8, 0., .4, .8, 1.2])


def load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_dagger_teaching", LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(legacy, model):
    rows = [legacy.rollout(model, x) for x in np.linspace(-1., .75, 100)]
    return {"success": float(np.mean([r[1] for r in rows])),
            "final_error": float(np.mean([r[2] for r in rows]))}


def distribution(xs):
    xs = np.asarray(xs)
    return {"count": len(xs), "min": float(xs.min()), "max": float(xs.max()),
            "quantiles": np.quantile(xs, [0.25, 0.5, 0.75]).tolist(),
            "near_goal_fraction": float(np.mean(np.abs(xs - .8) < .025))}


def snapshot(model, index, destination):
    target = destination / f"policy_{index}.npz"
    np.savez(target, **{f"p{i}": p for i, p in enumerate(model.p)},
             **{f"adam_m{i}": p for i, p in enumerate(model.m)},
             **{f"adam_v{i}": p for i, p in enumerate(model.v)},
             adam_step=np.array(model.step))
    return {"file": str(target.relative_to(BASE)), "adam_step": model.step,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "probes": [{"x": float(x), "prediction": model.act(x)} for x in PROBES]}


def one_label_gradient_probe(model, x, target):
    # G1: use the queried state and expert label, not the executed learner action.
    prediction, h, z = model.forward(np.array([[x]]))
    error = prediction - target
    # G2: chain the action error through the output tanh to its scalar bias.
    grad_bias = float((2 * error * .15 * (1 - z * z))[0, 0])
    # G3: isolated SGD on that bias illustrates a direction; full rounds use Adam.
    probe = copy.deepcopy(model)
    before = float(error[0, 0] ** 2)
    probe.p[3] -= .1 * grad_bias
    after = (probe.act(x) - target) ** 2
    # Central differences check the formula without advancing Adam or RNG.
    eps = 1e-6
    plus, minus = copy.deepcopy(model), copy.deepcopy(model)
    plus.p[3] += eps
    minus.p[3] -= eps
    finite_difference = ((plus.act(x) - target) ** 2 - (minus.act(x) - target) ** 2) / (2 * eps)
    assert np.isclose(grad_bias, finite_difference, atol=1e-10)
    assert after < before
    return {"x": x, "expert_label": target, "prediction": float(prediction[0, 0]),
            "output_tanh": float(z[0, 0]), "loss_before": before,
            "bias_gradient": grad_bias, "finite_difference": finite_difference,
            "isolated_sgd_lr": .1, "isolated_prediction_after": probe.act(x),
            "isolated_loss_after": after,
            "scope": "copy-only single-label bias probe; not a round update"}


def main():
    legacy = load_legacy()
    result_dir = BASE / "results"
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    # D1: the same 160 expert samples and initial fitted model as the original.
    xs = rng.uniform(-.9, -.5, 160)
    ys = legacy.teacher(xs)
    bc = legacy.Policy(3)
    initial_loss = bc.fit(xs, ys, rng)
    dagger, broad = copy.deepcopy(bc), copy.deepcopy(bc)
    bx, dx, wx = list(xs), list(xs), list(xs)
    data_rows = [{"sample_id": f"init_{i}", "round": 0, "state": float(x),
                  "expert_label": float(y)} for i, (x, y) in enumerate(zip(xs, ys))]
    checkpoints = [snapshot(dagger, 0, checkpoint_dir)]
    histories = [{"round": 0, "expert_labels": 160, "bc": evaluate(legacy, bc),
                 "dagger": evaluate(legacy, dagger), "broad_bc": evaluate(legacy, broad)}]
    rounds = []
    for round_index in range(1, 4):
        # D2: execute the frozen current policy; do not fit in the rollout loop.
        starts = rng.uniform(-1., .75, 24)
        events, paths = [], []
        for episode, start in enumerate(starts):
            states, success, final_error = legacy.rollout(dagger, start)
            path = []
            for t, x in enumerate(states):
                action = dagger.act(x)
                row = {"visit_id": f"r{round_index}_e{episode}_t{t}",
                       "episode": episode, "t": t, "state": float(x),
                       "executed_by": "learner", "executed_action": action,
                       "next_state": float(x + action)}
                events.append(row)
                path.append(row)
            paths.append({"start": float(start), "success": success,
                          "final_error": final_error, "steps": path})
        # D3: choose a label budget from visits; queries do not execute the expert.
        indices = rng.choice(len(events), 160, replace=len(events) < 160)
        chosen = np.array([events[int(i)]["state"] for i in indices])
        queries = []
        for j, i in enumerate(indices):
            visit = events[int(i)]
            queries.append({**visit, "sample_id": f"r{round_index}_label{j}",
                            "round": round_index, "expert_label": float(legacy.teacher(visit["state"]))})
        worst = max(queries, key=lambda r: abs(r["executed_action"] - r["expert_label"]))
        probe = one_label_gradient_probe(dagger, worst["state"], worst["expert_label"])
        before_params = [p.copy() for p in dagger.p]
        # D4: concatenate old and new labels; repeats retain their sample weight.
        dx.extend(chosen)
        data_rows.extend(queries)
        bx.extend(rng.uniform(-.9, -.5, 160))
        wx.extend(rng.uniform(-1., 1.3, 160))
        assert len(dx) == len(data_rows) == 160 * (round_index + 1)
        assert len({r["sample_id"] for r in data_rows}) == len(data_rows)
        # D5: the original 350 minibatch Adam updates per method, same call order.
        b_loss = bc.fit(bx, legacy.teacher(np.array(bx)), rng)
        stored_labels = np.array([row["expert_label"] for row in data_rows])
        d_loss = dagger.fit(dx, stored_labels, rng)
        w_loss = broad.fit(wx, legacy.teacher(np.array(wx)), rng)
        checkpoint = snapshot(dagger, round_index, checkpoint_dir)
        checkpoints.append(checkpoint)
        history = {"round": round_index, "expert_labels": len(dx),
                   "bc": evaluate(legacy, bc), "dagger": evaluate(legacy, dagger),
                   "broad_bc": evaluate(legacy, broad), "bc_training_mse": b_loss,
                   "dagger_training_mse": d_loss, "broad_bc_training_mse": w_loss}
        histories.append(history)
        # D6: compare the next fitted policy on the same probe start as well.
        next_states, next_success, next_error = legacy.rollout(dagger, .4)
        rounds.append({"round": round_index, "collector_policy": round_index - 1,
                       "beta": 0., "starts": starts.tolist(), "rollouts": paths,
                       "visited": distribution([r["state"] for r in events]),
                       "queried": distribution(chosen), "queries": queries,
                       "dataset_size": len(dx), "dataset_round_counts":
                           {str(i): sum(r["round"] == i for r in data_rows) for i in range(round_index + 1)},
                       "most_disagreed_query": worst, "single_label_probe": probe,
                       "parameter_l2_changes": [float(np.linalg.norm(p - q)) for p, q in zip(dagger.p, before_params)],
                       "post_update_checkpoint": checkpoint,
                       "post_update_fixed_start": {"start": .4, "states": next_states,
                                                  "success": next_success, "final_error": next_error}})
    # Recompute the original seed locally; no unpublished result file is needed.
    saved = legacy.run(3)["history"][:4]
    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                compare(a[k], b[k])
        else:
            assert np.isclose(a, b, rtol=1e-9, atol=1e-12), (a, b)
    for actual, expected in zip(histories, saved):
        compare(actual, expected)
    result = {"numpy": np.__version__, "seed": 3, "legacy_sha256": LEGACY_SHA256,
              "scope": "three traced original rounds; no new five-seed benchmark",
              "initial_training_mse": initial_loss, "initial_dataset": data_rows[:160],
              "checkpoints": checkpoints, "history": histories, "rounds": rounds,
              "original_first_four_rows_match": True}
    target = result_dir / "dagger_round_trace.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"original_first_four_rows_match": True,
                      "dataset_sizes": [r["expert_labels"] for r in histories],
                      "success": [r["dagger"]["success"] for r in histories],
                      "final_error": [r["dagger"]["final_error"] for r in histories]}))


if __name__ == "__main__":
    main()
