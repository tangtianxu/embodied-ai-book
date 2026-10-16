"""Align absolute joint targets to physical control steps before weighting them."""
from dataclasses import dataclass
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Chunk:
    start_step: int
    targets: tuple
    observation_step: int
    semantics: str = "joint_target_rad"


def select(chunks, current_step, decay=math.log(2)):
    # T1: a chunk's local index is the physical step minus its own starting step.
    candidates = []
    for chunk in chunks:
        index = current_step-chunk.start_step
        # Explicit coverage metadata retains valid targets whose numerical value is zero.
        if 0 <= index < len(chunk.targets):
            if chunk.observation_step > current_step:
                raise ValueError("A future observation is not available at this step.")
            if chunk.semantics != "joint_target_rad":
                raise ValueError("Cannot combine different command meanings or units.")
            candidates.append((chunk, index, float(chunk.targets[index])))
    # T2: an empty set is not the same as a command whose correct target is zero.
    if not candidates:
        raise LookupError("No prediction covers this physical control step.")
    # T3: original ACT orders candidates from oldest to newest; positive decay favors older ones.
    candidates.sort(key=lambda item: item[0].start_step)
    raw = [math.exp(-decay*i) for i in range(len(candidates))]
    weights = [w/sum(raw) for w in raw]
    # T4: only combine targets for the same physical step and with the same semantics.
    command = sum(w*item[2] for w,item in zip(weights,candidates))
    return {"current_step": current_step, "command_rad": command,
            "candidates": [{"start_step": c.start_step, "local_index": index,
                             "observation_step": c.observation_step,
                             "age_control_steps": current_step-c.observation_step,
                             "target_rad": action, "weight": weight}
                            for (c,index,action),weight in zip(candidates,weights)]}


def run():
    chunks = [
        Chunk(10,(.10,.12,.15),10),
        Chunk(11,(.13,.17,.21),11),
        Chunk(13,(.00,.05,.09),13),
    ]
    trace = [select(chunks,t) for t in (12,13,14)]
    assert math.isclose(trace[0]["command_rad"], .15*2/3+.17/3)
    assert math.isclose(trace[1]["command_rad"], .21*2/3)
    assert math.isclose(trace[2]["command_rad"], .05)
    assert trace[1]["candidates"][1]["target_rad"] == 0.
    try:
        select(chunks,16)
    except LookupError:
        missing = "LookupError, caller must decide a task-specific next action"
    else:
        raise AssertionError("An uncovered step must not be silently treated as zero.")
    try:
        select([Chunk(10,(.1,.2),10,"delta_joint_rad")],10)
    except ValueError:
        mismatch = "rejected"
    else:
        raise AssertionError("Different command semantics were silently combined.")
    try:
        select([Chunk(10,(.1,.2),11)],10)
    except ValueError:
        future = "rejected"
    else:
        raise AssertionError("A future observation was used.")
    endpoints = [(-.2,.1),(.2,.1)]
    midpoint = [(endpoints[0][j]+endpoints[1][j])/2 for j in (0,1)]
    inside = lambda point: -.05 <= point[0] <= .05 and 0. <= point[1] <= .2
    assert not any(inside(point) for point in endpoints) and inside(midpoint)
    result = {"scope": "scalar absolute joint-target indexing; no robot execution or ACT training",
              "decay": math.log(2), "original_code_decay": .01,
              "weight_order": "oldest to newest", "trace": trace,
              "uncovered_step_16": missing, "different_semantics": mismatch,
              "future_observation": future,
              "route_conflict": {"targets_xy_m": endpoints,
                                  "equal_weight_mean": midpoint,
                                  "closed_obstacle": {"x":[-.05,.05],"y":[0.,.2]},
                                  "mean_inside_obstacle": inside(midpoint)}}
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results/temporal_ensemble.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),
                                                      encoding="utf-8")
    print(json.dumps({"commands":[row["command_rad"] for row in trace],
                      "zero_target_retained":True,"uncovered_step":missing},ensure_ascii=False))


if __name__ == "__main__":
    run()
