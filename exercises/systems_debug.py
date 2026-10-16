"""Small CPU references for data contracts and distributed-loss arithmetic."""
import numpy as np


def causal_indices(sensor_times, query_times, max_age):
    t, q = np.asarray(sensor_times, float), np.asarray(query_times, float)
    if np.any(np.diff(t) < 0):
        raise ValueError('sensor timestamps must be sorted')
    # 步骤1：找最近非未来采集时刻
    index = np.searchsorted(t, q, side='right') - 1
    # 步骤2：构造安全索引并检查观测年龄
    safe = np.maximum(index, 0)
    valid = (index >= 0) & (q - t[safe] <= max_age)
    return np.where(valid, index, -1)


def ddp_loss_scaling(local_sums, local_counts):
    sums, counts = np.asarray(local_sums, float), np.asarray(local_counts, int)
    total = counts.sum()
    if total <= 0:
        raise ValueError('global supervision is empty')
    # DDP averages gradients over W ranks: each differentiable sum gets W/N.
    scaled = len(sums) * sums / total
    return scaled, float(sums.sum()/total)


class ReplayBuffer:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity, self.data, self.next = capacity, [], 0

    def add(self, transition):
        # A transition contains its real final next observation, not reset obs.
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.next] = transition
        self.next = (self.next + 1) % self.capacity

    def sample(self, batch, rng):
        if not self.data:
            raise ValueError('cannot sample an empty buffer')
        return [self.data[i] for i in rng.integers(0, len(self.data), batch)]
