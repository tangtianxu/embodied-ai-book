"""Three actual vector-env modes, two envs and three batched calls per mode."""
from pathlib import Path
import json
import gymnasium as gym
import numpy as np


class BoundaryEnv(gym.Env):
    metadata = {}

    def __init__(self, env_id, end_step, terminal):
        self.env_id, self.end_step, self.terminal = env_id, end_step, terminal
        self.episode = -1
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 100, (3,), dtype=np.int64)

    def observation(self):
        return np.array([self.env_id, self.episode, self.t], dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.t = 0
        return self.observation(), {}

    def step(self, action):
        self.t += 1
        ended = self.t >= self.end_step
        return self.observation(), 1., bool(ended and self.terminal), bool(ended and not self.terminal), {}


def trace_mode(mode):
    envs = gym.vector.SyncVectorEnv([
        lambda: BoundaryEnv(0, 2, True), lambda: BoundaryEnv(1, 3, False)], autoreset_mode=mode)
    observation, _ = envs.reset(seed=10)
    previous_ended = np.zeros(2, dtype=bool)
    records = []
    try:
        for call in range(1, 4):
            before = observation.copy()
            returned, reward, term, trunc, infos = envs.step(np.zeros(2, dtype=np.int64))
            ended = term | trunc
            actual_next = returned.copy()
            valid_transition = np.ones(2, dtype=bool)
            if mode == gym.vector.AutoresetMode.SAME_STEP:
                # Returned observations are for the policy's next call; final_obs is for this transition.
                for i in np.flatnonzero(ended):
                    assert infos["_final_obs"][i]
                    actual_next[i] = infos["final_obs"][i]
            elif mode == gym.vector.AutoresetMode.NEXT_STEP:
                # A reset-only slot did not execute the submitted action.
                valid_transition = ~previous_ended
            row = {"call": call, "before": before.tolist(), "returned": returned.tolist(),
                   "reward": reward.tolist(), "terminated": term.tolist(), "truncated": trunc.tolist(),
                   "transition_next": actual_next.tolist(), "store_transition": valid_transition.tolist()}
            observation = returned.copy()
            if mode == gym.vector.AutoresetMode.DISABLED and np.any(ended):
                observation, _ = envs.reset(options={"reset_mask": ended})
                row["after_masked_reset"] = observation.tolist()
            previous_ended = ended
            records.append(row)
        if mode == gym.vector.AutoresetMode.SAME_STEP:
            assert records[1]["returned"][0] == [0, 1, 0]
            assert records[1]["transition_next"][0] == [0, 0, 2]
            assert records[2]["transition_next"][1] == [1, 0, 3]
        if mode == gym.vector.AutoresetMode.NEXT_STEP:
            assert records[2]["store_transition"] == [False, True]
            assert records[2]["returned"][0] == [0, 1, 0]
        return records
    finally:
        envs.close()


if __name__ == "__main__":
    result = {"gymnasium": gym.__version__, "observation_fields": ["env_id", "episode_id", "step"],
              "modes": {mode.name: trace_mode(mode) for mode in gym.vector.AutoresetMode},
              "truncation_target_example": {"reward": 1, "gamma": .9, "V_final": 3,
                "V_reset": 10, "correct_target": 1 + .9 * 3,
                "wrong_reset_target": 1 + .9 * 10, "wrong_done_target": 1}}
    path = Path(__file__).resolve().parent / "results/vector_boundary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"modes_run": list(result["modes"]), "batch_calls_each": 3,
                      "correct_truncation_target": result["truncation_target_example"]["correct_target"]}))
