"""Minimal geometric algorithms with explicit simplified assumptions."""
import numpy as np


def inverse_transform(t):
    r, p = t[:3, :3], t[:3, 3]
    out = np.eye(4)
    # 步骤1：旋转转置，平移乘负旋转转置
    out[:3, :3], out[:3, 3] = r.T, -r.T @ p
    return out


def fk2(q, lengths=(1.0, 1.0)):
    a, b = q
    l1, l2 = lengths
    return np.array([l1*np.cos(a)+l2*np.cos(a+b), l1*np.sin(a)+l2*np.sin(a+b)])


def jacobian2(q, lengths=(1.0, 1.0)):
    a, b = q
    l1, l2 = lengths
    return np.array([[-l1*np.sin(a)-l2*np.sin(a+b), -l2*np.sin(a+b)],
                     [l1*np.cos(a)+l2*np.cos(a+b), l2*np.cos(a+b)]])


def dls_ik(target, initial, damping=.05, iterations=200, step_limit=.15):
    q = np.array(initial, dtype=float, copy=True)
    for _ in range(iterations):
        # 步骤1：计算末端误差
        error = np.asarray(target) - fk2(q)
        if np.linalg.norm(error) < 1e-7:
            break
        # 步骤2：计算局部雅可比
        j = jacobian2(q)
        # 步骤3：求阻尼最小二乘修正
        dq = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(2), error)
        # 步骤4：限制步长，再更新关节
        dq *= min(1, step_limit / max(np.linalg.norm(dq), 1e-12))
        q += dq
    return q, float(np.linalg.norm(np.asarray(target) - fk2(q)))


def fit_line_svd(points):
    points = np.asarray(points, float)
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    return normal, -normal @ center


def ransac_line(points, threshold=.03, iterations=300, seed=4):
    points = np.asarray(points, float)
    if len(points) < 2:
        raise ValueError('at least two points are required')
    rng, best = np.random.default_rng(seed), np.zeros(len(points), bool)
    for _ in range(iterations):
        # 步骤1：抽取两个不同样本
        i, j = rng.choice(len(points), 2, replace=False)
        d = points[j] - points[i]
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            continue
        # 步骤2：建立法向量形式的候选直线
        n = np.array([-d[1], d[0]]) / norm
        # 步骤3：按垂直距离寻找内点
        mask = np.abs(points @ n - points[i] @ n) < threshold
        # 步骤4：保留支持最多的候选
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 2:
        raise ValueError('no nondegenerate consensus found')
    # 步骤5：用最佳内点重新拟合
    n, offset = fit_line_svd(points[best])
    return n, offset, best


def segment_free(a, b, rectangles):
    """Analytic slab test: collision includes touching a closed rectangle."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    for low, high in rectangles:
        entry, exit_ = 0., 1.
        for axis in range(2):
            if abs(d[axis]) < 1e-12:
                if a[axis] < low[axis] or a[axis] > high[axis]:
                    entry, exit_ = 1., 0.
                    break
            else:
                t1, t2 = (low[axis]-a[axis])/d[axis], (high[axis]-a[axis])/d[axis]
                entry, exit_ = max(entry, min(t1,t2)), min(exit_, max(t1,t2))
        if entry <= exit_:
            return False
    return True


def rrt(start, goal, rectangles, step=.08, iterations=4000, seed=7):
    """Point robot in [0,1]^2; no optimality or robot-body collision claim."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if not segment_free(start, start, rectangles) or not segment_free(goal, goal, rectangles):
        return None
    rng, nodes, parents = np.random.default_rng(seed), [start], [-1]
    for _ in range(iterations):
        # 步骤1：采样候选目标
        sample = goal if rng.random() < .15 else rng.random(2)
        # 步骤2：找最近树节点
        index = int(np.argmin(np.linalg.norm(np.array(nodes) - sample, axis=1)))
        d = sample - nodes[index]
        # 步骤3：向候选推进有限距离
        candidate = nodes[index] + d * min(1, step / max(np.linalg.norm(d), 1e-12))
        # 步骤4：检查完整连接边
        if not segment_free(nodes[index], candidate, rectangles):
            continue
        # 步骤5：保存合法节点与父节点
        nodes.append(candidate); parents.append(index)
        if np.linalg.norm(candidate-goal) <= step and segment_free(candidate, goal, rectangles):
            nodes.append(goal); parents.append(len(nodes)-2)
            path, index = [], len(nodes)-1
            # 步骤6：沿父节点回溯完整路径
            while index >= 0:
                path.append(nodes[index]); index = parents[index]
            return np.array(path[::-1])
    return None
