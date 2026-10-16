"""Chapter 8.3: inspect a complete geometric search, not only its final path.

Run: python practice/rrt_trace.py --out results
Python 3.10+; the algorithm and examples use only the standard library.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import heapq
import json
import math
from pathlib import Path
import random
from typing import Iterable


Point = tuple[float, float]
GEOMETRY_TOL = 1e-12


def point(value) -> Point:
    if len(value) != 2 or not all(math.isfinite(float(x)) for x in value):
        raise ValueError('a point must contain two finite coordinates')
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class Box:
    name: str
    low: Point
    high: Point

    def __post_init__(self):
        lo, hi = point(self.low), point(self.high)
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError('box lower bounds must not exceed upper bounds')


@dataclass(frozen=True)
class Scene:
    boxes: tuple[Box, ...]
    low: Point = (0.0, 0.0)
    high: Point = (1.0, 1.0)

    def contains(self, q: Point) -> bool:
        return all(a <= x <= b for a, x, b in zip(self.low, q, self.high))


def collision_interval(a: Point, b: Point, box: Box):
    """Return the closed parameter interval where a+t*(b-a) hits the box.

    Each axis contributes an interval; their intersection with [0, 1] is
    the part of the whole segment inside the closed rectangle. A tiny
    outward padding keeps roundoff at an intended boundary conservative.
    """
    entry, exit_ = 0.0, 1.0
    for start, end, low, high in zip(a, b, box.low, box.high):
        low, high = low - GEOMETRY_TOL, high + GEOMETRY_TOL
        delta = end - start
        if delta == 0.0:
            if start < low or start > high:
                return None
        else:
            t1, t2 = (low - start) / delta, (high - start) / delta
            entry = max(entry, min(t1, t2))
            exit_ = min(exit_, max(t1, t2))
            if entry > exit_:
                return None
    return entry, exit_


def edge_collisions(scene: Scene, a: Point, b: Point):
    hits = []
    for box in scene.boxes:
        interval = collision_interval(a, b, box)
        if interval is not None:
            hits.append({'box': box.name, 'interval': interval})
    return hits


def edge_free(scene: Scene, a: Point, b: Point) -> bool:
    # The domain is a convex rectangle, so in-domain endpoints keep its segment in it.
    return scene.contains(a) and scene.contains(b) and not edge_collisions(scene, a, b)


@dataclass
class SearchResult:
    status: str
    nodes: list[Point]
    parents: list[int]
    events: list[dict]
    path_indices: list[int] | None

    @property
    def path(self):
        return None if self.path_indices is None else [self.nodes[i] for i in self.path_indices]

    def to_dict(self):
        result = asdict(self)
        result['path'] = self.path
        result['attempts'] = len(self.events)
        result['path_length'] = None if self.path is None else sum(
            math.dist(a, b) for a, b in zip(self.path, self.path[1:]))
        return result


def backtrack(parents: list[int], goal_index: int) -> list[int]:
    path = []
    index = goal_index
    while index != -1:
        path.append(index)
        index = parents[index]
    return path[::-1]


def run_rrt(scene: Scene, start, goal, *, step=0.2, max_iterations=4000,
            samples: Iterable[Point] | None = None, seed=7, goal_bias=0.15):
    start, goal = point(start), point(goal)
    if not math.isfinite(step) or step <= 0:
        raise ValueError('step must be finite and positive')
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError('max_iterations must be a nonnegative integer')
    if not 0 <= goal_bias <= 1:
        raise ValueError('goal_bias must be between 0 and 1')
    if not edge_free(scene, start, start):
        return SearchResult('invalid_start', [], [], [], None)
    if not edge_free(scene, goal, goal):
        return SearchResult('invalid_goal', [], [], [], None)

    nodes, parents, events = [start], [-1], []
    if start == goal:
        return SearchResult('solved', nodes, parents, events, [0])
    rng = random.Random(seed)
    sample_iter = None if samples is None else iter(samples)

    for iteration in range(1, max_iterations + 1):
        # 步骤1：从配置域采样；固定序列用于复算，随机模式偶尔采目标。
        if sample_iter is not None:
            try:
                sample = point(next(sample_iter))
            except StopIteration:
                return SearchResult('samples_exhausted', nodes, parents, events, None)
        else:
            sample = goal if rng.random() < goal_bias else tuple(
                rng.uniform(a, b) for a, b in zip(scene.low, scene.high))
        if not scene.contains(sample):
            raise ValueError('sample is outside the declared sampling domain')

        # 步骤2：找最近已有节点；距离相等时取较小节点编号。
        nearest = min(range(len(nodes)), key=lambda i: (math.dist(nodes[i], sample), i))
        origin = nodes[nearest]
        distance = math.dist(origin, sample)
        event = dict(iteration=iteration, sample=sample, nearest=nearest,
                     origin=origin, distance=distance, accepted=False,
                     new_index=None, goal_connected=False)
        if distance == 0.0:
            event.update(candidate=origin, reason='duplicate_sample', hits=[],
                         tree_size=len(nodes), goal_distance=math.dist(origin, goal))
            events.append(event)
            continue

        # 步骤3：只沿方向走 min(step, distance)，不越过近处样本。
        scale = min(1.0, step / distance)
        candidate = tuple(a + scale * (b - a) for a, b in zip(origin, sample))
        hits = edge_collisions(scene, origin, candidate)
        event.update(candidate=candidate, hits=hits,
                     goal_distance=math.dist(candidate, goal))

        # 步骤4：检查整段；碰撞时不改变节点表与父节点表。
        if hits:
            event.update(reason='collision', tree_size=len(nodes))
            events.append(event)
            continue

        # 步骤5：合法候选入树；节点表中的相邻编号不一定互为父子。
        nodes.append(candidate)
        parents.append(nearest)
        new_index = len(nodes) - 1
        event.update(accepted=True, reason='accepted', new_index=new_index)

        # 步骤6：进入目标连接范围后，还要检查候选到目标的完整边。
        if candidate == goal:
            goal_index = new_index
        elif math.dist(candidate, goal) <= step and edge_free(scene, candidate, goal):
            nodes.append(goal)
            parents.append(new_index)
            goal_index = len(nodes) - 1
        else:
            goal_index = None
        event.update(tree_size=len(nodes), goal_connected=goal_index is not None)
        events.append(event)
        if goal_index is not None:
            # 步骤7：从目标沿父指针回溯，再反转为起点到目标。
            return SearchResult('solved', nodes, parents, events,
                                backtrack(parents, goal_index))
    return SearchResult('budget_exhausted', nodes, parents, events, None)


DEMO_SCENE = Scene((Box('小障碍', (0.06, 0.06), (0.09, 0.09)),
                    Box('挡板', (0.38, 0.0), (0.62, 0.60))))
DEMO_START, DEMO_GOAL = (0.0, 0.0), (0.9, 0.2)
DEMO_SAMPLES = [(1.0, 1.0), (1.0, 0.0), (1.0, 0.0),
                (0.2, 1.0), (0.2, 1.0), (1.0, 0.4),
                (0.2, 1.0), (0.2, 1.0), (0.1, 0.2),
                (1.0, 0.8), (1.0, 0.8), (1.0, 0.8),
                DEMO_GOAL, DEMO_GOAL, DEMO_GOAL]


def fixed_demo():
    return run_rrt(DEMO_SCENE, DEMO_START, DEMO_GOAL, step=0.2,
                   samples=DEMO_SAMPLES, max_iterations=len(DEMO_SAMPLES))


def dijkstra(graph: list[list[tuple[int, float]]], start: int, goal: int):
    distances, parents = {start: 0.0}, {}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return path[::-1], distance
        for other, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances.get(other, math.inf):
                distances[other], parents[other] = candidate, node
                heapq.heappush(queue, (candidate, other))
    return None, math.inf


def build_roadmap(scene: Scene, points: list[Point], radius: float):
    if any(not edge_free(scene, q, q) for q in points):
        raise ValueError('roadmap nodes must be free configurations')
    graph = [[] for _ in points]
    attempts = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = math.dist(points[i], points[j])
            if distance <= radius:
                free = edge_free(scene, points[i], points[j])
                attempts.append(dict(nodes=[i, j], distance=distance, accepted=free))
                if free:
                    graph[i].append((j, distance))
                    graph[j].append((i, distance))
    return graph, attempts


def roadmap_query(scene: Scene, points, graph, start: Point, goal: Point):
    """For this small example attach each endpoint to its closest legal node.

    More general PRMs can try several neighbors. Failure in this finite graph
    is not a proof that the continuous planning problem has no solution.
    """
    if not edge_free(scene, start, start) or not edge_free(scene, goal, goal):
        return dict(status='invalid_query', path=None)
    augmented = [list(edges) for edges in graph] + [[], []]
    all_points = list(points) + [start, goal]
    connections = []
    for endpoint in (len(points), len(points) + 1):
        order = sorted(range(len(points)),
                       key=lambda i: (math.dist(all_points[endpoint], points[i]), i))
        neighbor = next((i for i in order if edge_free(scene, all_points[endpoint], points[i])), None)
        if neighbor is None:
            return dict(status='no_connection', path=None)
        distance = math.dist(all_points[endpoint], points[neighbor])
        augmented[endpoint].append((neighbor, distance))
        augmented[neighbor].append((endpoint, distance))
        connections.append([endpoint, neighbor])
    indices, cost = dijkstra(augmented, len(points), len(points) + 1)
    return dict(status='solved' if indices is not None else 'no_graph_path',
                connections=connections, path_indices=indices,
                path=None if indices is None else [all_points[i] for i in indices],
                cost=cost if indices is not None else None)


def prm_demo():
    points = [(0.2, 0.0), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)]
    graph, attempts = build_roadmap(DEMO_SCENE, points, radius=0.85)
    before = json.dumps(graph)
    queries = [roadmap_query(DEMO_SCENE, points, graph, DEMO_START, DEMO_GOAL),
               roadmap_query(DEMO_SCENE, points, graph, (0.1, 0.8), (0.9, 0.8))]
    assert json.dumps(graph) == before  # 查询接入临时图，不改共享路线图。
    return dict(points=points, radius=0.85, attempts=attempts, graph=graph, queries=queries)


def rrtstar_demo(*, propagate_costs=True, include_outer_descendant=False):
    """A local parent-selection/rewiring example, not a full RRT* planner.

    Disabling propagation deliberately demonstrates the error discussed in
    exercise 8.3-3; it is not an alternative correct implementation.
    """
    points = [(0.0, 0.0), (0.0, 3.0), (3.0, 3.0), (3.0, 0.0), (4.0, 0.0)]
    parents, costs = [-1, 0, 1, 2, 3], [0.0, 3.0, 6.0, 9.0, 10.0]
    if include_outer_descendant:
        points.append((4.5, 0.0))  # E是D的子节点，但在N的邻域之外。
        parents.append(4)
        costs.append(10.5)
    original = dict(points=list(points), parents=list(parents), costs=list(costs))
    new_point, radius = (2.0, 0.0), 2.1
    neighbors = [i for i, q in enumerate(points) if math.dist(q, new_point) <= radius]
    candidates = [dict(parent=i, edge_cost=math.dist(points[i], new_point),
                       total_cost=costs[i] + math.dist(points[i], new_point)) for i in neighbors]
    # 所有例示连接都在无障碍平面；按累计成本选父节点，非按距离选。
    parent = min(candidates, key=lambda c: (c['total_cost'], c['parent']))
    points.append(new_point)
    parents.append(parent['parent'])
    costs.append(parent['total_cost'])
    new_index = len(points) - 1
    changes, snapshots = [], []

    def update_descendants(index):
        for child, old_parent in enumerate(parents):
            if old_parent == index:
                old_cost = costs[child]
                costs[child] = costs[index] + math.dist(points[index], points[child])
                changes.append(dict(kind='descendant', node=child,
                                    old_cost=old_cost, new_cost=costs[child]))
                update_descendants(child)

    # 为使表格可复算，本例按节点编号检查邻居；相等成本不改父节点。
    for i in neighbors:
        candidate = costs[new_index] + math.dist(points[new_index], points[i])
        if candidate < costs[i]:
            old_parent, old_cost = parents[i], costs[i]
            parents[i], costs[i] = new_index, candidate
            changes.append(dict(kind='rewire', node=i, old_parent=old_parent,
                                new_parent=new_index, old_cost=old_cost, new_cost=candidate))
            if propagate_costs:
                update_descendants(i)
        snapshots.append(dict(checked_node=i, parents=list(parents), costs=list(costs)))
    return dict(original=original, new_point=new_point, radius=radius,
                candidates=candidates, points=points, parents=parents, costs=costs,
                changes=changes, snapshots=snapshots, new_index=new_index,
                propagation_enabled=propagate_costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = fixed_demo()
    payload = result.to_dict()
    payload['scene'] = asdict(DEMO_SCENE)
    payload['start'], payload['goal'], payload['step'] = DEMO_START, DEMO_GOAL, 0.2
    for name, data in [('rrt_trace', payload), ('prm_queries', prm_demo()),
                       ('rrtstar_rewire', rrtstar_demo()),
                       ('rrtstar_missing_propagation', {
                           'without_outer_node': rrtstar_demo(propagate_costs=False),
                           'with_outer_node': rrtstar_demo(propagate_costs=False, include_outer_descendant=True),
                           'correct_with_outer_node': rrtstar_demo(include_outer_descendant=True)})]:
        (args.out / (name + '.json')).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.out / 'rrt_trace.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['iteration', 'sample', 'nearest', 'candidate', 'reason',
                  'new_index', 'tree_size', 'goal_connected']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: e[key] for key in fields} for e in result.events)
    print(json.dumps({key: payload[key] for key in ['status', 'attempts', 'path_indices', 'path_length']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
