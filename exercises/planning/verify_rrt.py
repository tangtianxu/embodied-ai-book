"""Verify geometry with an independent exact-arithmetic intersection oracle."""
from fractions import Fraction
import json
import math
from pathlib import Path
import random

from rrt_trace import (Box, Scene, DEMO_GOAL, DEMO_SAMPLES, DEMO_SCENE, DEMO_START,
                       build_roadmap, collision_interval, dijkstra, edge_free,
                       fixed_demo, prm_demo, rrtstar_demo, run_rrt)


def exact_point(p):
    return tuple(Fraction(str(x)) for x in p)


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a, b, c):
    return orient(a, b, c) == 0 and all(min(x, y) <= z <= max(x, y) for x, y, z in zip(a, b, c))


def segments_intersect(a, b, c, d):
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0 and o3 * o4 < 0) or any(
        (on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b)))


def exact_hit(a, b, box):
    # Independent formulation: endpoint containment OR intersection with four edges.
    a, b, low, high = map(exact_point, (a, b, box.low, box.high))
    def inside(q):
        return all(l <= x <= h for l, x, h in zip(low, q, high))
    corners = [low, (high[0], low[1]), high, (low[0], high[1])]
    return inside(a) or inside(b) or any(segments_intersect(a, b, corners[i], corners[(i + 1) % 4])
                                       for i in range(4))


def check_path(scene, result):
    assert result.status == 'solved' and result.path[0] == DEMO_START and result.path[-1] == DEMO_GOAL
    assert result.parents[0] == -1
    for child, parent in enumerate(result.parents[1:], 1):
        assert 0 <= parent < child
        a, b = result.nodes[parent], result.nodes[child]
        assert math.dist(a, b) <= 0.2 + 1e-12
        assert scene.contains(a) and scene.contains(b)
        assert all(not exact_hit(a, b, box) for box in scene.boxes)


def main():
    rng = random.Random(97)
    for _ in range(2000):
        # Binary-exact grid coordinates avoid attributing decimal roundoff to topology.
        a = tuple(rng.randrange(17) / 16 for _ in range(2))
        b = tuple(rng.randrange(17) / 16 for _ in range(2))
        x0, x1 = sorted(rng.randrange(17) / 16 for _ in range(2))
        y0, y1 = sorted(rng.randrange(17) / 16 for _ in range(2))
        box = Box('oracle_case', (x0, y0), (x1, y1))
        assert (collision_interval(a, b, box) is not None) == exact_hit(a, b, box)

    result = fixed_demo()
    assert len(result.events) == 15
    assert [e['iteration'] for e in result.events if not e['accepted']] == [1, 3, 6]
    assert [e['tree_size'] for e in result.events[:3]] == [1, 2, 2]
    assert result.path_indices == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
    assert result.nodes[6] == (0.1, 0.2) and result.parents[6] == 2 and 6 not in result.path_indices
    assert result.events[8]['candidate'] == DEMO_SAMPLES[8]  # Short step does not overshoot.
    check_path(DEMO_SCENE, result)
    interval = result.events[0]['hits'][0]['interval']
    assert math.isclose(interval[0], 0.4242640687119285) and math.isclose(interval[1], 0.6363961030678927)
    assert not edge_free(DEMO_SCENE, (0.2, 0.6), (0.4, 0.6))  # Closed top boundary.
    safe_branch = run_rrt(DEMO_SCENE, DEMO_START, DEMO_GOAL,
                          samples=DEMO_SAMPLES[:8] + [(0.3, 0.6)], max_iterations=9)
    boundary_branch = run_rrt(DEMO_SCENE, DEMO_START, DEMO_GOAL,
                              samples=DEMO_SAMPLES[:8] + [(1.0, 0.6)], max_iterations=9)
    assert safe_branch.events[-1]['nearest'] == 4 and safe_branch.events[-1]['accepted']
    assert boundary_branch.events[-1]['nearest'] == 4 and not boundary_branch.events[-1]['accepted']

    prefix = run_rrt(DEMO_SCENE, DEMO_START, DEMO_GOAL,
                     samples=DEMO_SAMPLES, max_iterations=14)
    assert prefix.status == 'budget_exhausted' and prefix.path is None
    short_input = run_rrt(DEMO_SCENE, DEMO_START, DEMO_GOAL,
                          samples=DEMO_SAMPLES[:3], max_iterations=100)
    assert short_input.status == 'samples_exhausted' and len(short_input.events) == 3
    empty = Scene(())
    duplicate = run_rrt(empty, (0, 0), (1, 1), samples=[(0, 0)], max_iterations=1)
    assert duplicate.events[0]['reason'] == 'duplicate_sample' and len(duplicate.nodes) == 1
    same = run_rrt(empty, (0, 0), (0, 0), max_iterations=0)
    assert same.status == 'solved' and same.path == [(0.0, 0.0)] and not same.events
    exact_goal = run_rrt(empty, (0, 0), (0.1, 0), samples=[(0.1, 0)], max_iterations=1)
    assert len(exact_goal.nodes) == 2
    assert run_rrt(DEMO_SCENE, DEMO_START, (0.5, 0.3)).status == 'invalid_goal'
    assert run_rrt(DEMO_SCENE, (-0.1, 0), DEMO_GOAL).status == 'invalid_start'
    for step in (0, -0.2, math.nan, math.inf):
        try:
            run_rrt(empty, (0, 0), (1, 1), step=step)
            raise AssertionError('invalid step accepted')
        except ValueError:
            pass

    star = rrtstar_demo()
    assert star['parents'][3] == 5 and star['parents'][4] == 3
    for i in range(len(star['points'])):
        cost, j, seen = 0.0, i, set()
        while star['parents'][j] != -1:
            assert j not in seen
            seen.add(j)
            parent = star['parents'][j]
            cost += math.dist(star['points'][j], star['points'][parent])
            j = parent
        assert math.isclose(cost, star['costs'][i])
    assert star['costs'][3:5] == [3.0, 4.0]
    missing = rrtstar_demo(propagate_costs=False)
    after_c = next(x for x in missing['snapshots'] if x['checked_node'] == 3)
    assert after_c['costs'][4] == 10.0
    assert missing['costs'][4] == 4.0 and missing['parents'][4] == missing['new_index']
    missing_outer = rrtstar_demo(propagate_costs=False, include_outer_descendant=True)
    correct_outer = rrtstar_demo(include_outer_descendant=True)
    assert missing_outer['costs'][5] == 10.5 and correct_outer['costs'][5] == 4.5

    prm = prm_demo()
    assert len(prm['attempts']) == 5 and sum(x['accepted'] for x in prm['attempts']) == 3
    assert [q['path_indices'] for q in prm['queries']] == [[4, 0, 1, 2, 3, 5], [4, 1, 2, 5]]
    assert math.isclose(prm['queries'][0]['cost'], 2.3)
    assert math.isclose(prm['queries'][1]['cost'], 0.8)
    for query in prm['queries']:
        for a, b in zip(query['path'], query['path'][1:]):
            assert all(not exact_hit(a, b, box) for box in DEMO_SCENE.boxes)
    changed = Scene((DEMO_SCENE.boxes[0], Box('更高挡板', (0.38, 0.0), (0.62, 0.9))))
    updated, _ = build_roadmap(changed, list(map(tuple, prm['points'])), 0.85)
    assert dijkstra(updated, 0, 3)[0] is None  # Old shared graph is no longer valid.

    report = {'status': 'passed', 'collision_oracle_cases': 2000,
              'fixed_trace_attempts': 15, 'rejected_attempts': [1, 3, 6],
              'tree_nodes_including_goal': len(result.nodes),
              'path_nodes': len(result.path_indices), 'unused_branch': 6,
              'path_length': result.to_dict()['path_length'],
              'rrtstar_costs_C_D': star['costs'][3:5],
              'missing_propagation_outer_cached_vs_correct': [missing_outer['costs'][5], correct_outer['costs'][5]],
              'prm_query_costs': [q['cost'] for q in prm['queries']],
              'scope': '二维点、闭矩形与局部重连；不是机械臂或真机验证。'}
    destination = Path(__file__).resolve().parent / 'results'
    destination.mkdir(exist_ok=True)
    (destination / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
