"""閉じたループのでたらめな場面(closed_loop.py)を、(rec_hold_keep, rec_back_guard) の 3 組で同じ場面に回して比べる。
    python fuzz_cl.py [場面の数] [種]  → out/fuzz_cl_<種>.json"""
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as CL          # noqa: E402
from closed_loop import A, C, N, PARAMS, T0   # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'


def scene_of(rng):
    dur = 16.0
    xf = rng.uniform(1.2, 5.0)
    sc_ = {'dur': dur, 'xf': xf, 'ai_from': 1.0}
    if rng.random() < 0.75:
        sc_['xr'] = xf - rng.uniform(0.25, 1.5)
        if rng.random() < 0.2:
            sc_['rear_soft'] = rng.uniform(0.3, 0.9)
    if rng.random() < 0.3:
        a = rng.uniform(4.0, 12.0)
        sc_['human'] = [(a, a + rng.uniform(0.3, 1.5))]
    k = rng.choice([60.0, 120.0, 180.0])
    if rng.random() < 0.3:
        b = rng.choice([40.0, 60.0, 100.0])
        sc_['model_desc'] = 'fwd %d, brake +%d near' % (k, b)
        sc_['model'] = lambda t, car, k=k, b=b, xf=xf: (C, N + b) if xf - car.x < 0.5 else (C, N - k)
    else:
        sc_['model_desc'] = 'fwd %d' % k
        sc_['model'] = (C, N - k)
    sc_['drops'] = []
    for _ in range(rng.randint(0, 5)):
        a = rng.uniform(1.0, dur - 1.0)
        sc_['drops'].append((a, a + rng.uniform(0.12, 0.8)))
    if rng.random() < 0.5:
        sc_['hit_drop'] = (rng.uniform(0.0, 0.06), rng.uniform(0.12, 0.6))
    sc_['veh_skip'] = []
    for _ in range(rng.randint(0, 3)):
        a = rng.uniform(1.0, dur - 1.0)
        sc_['veh_skip'].append((a, a + rng.uniform(0.17, 0.5)))
    return sc_


def params_of(rng):
    lb = rng.choice([0, 1])
    return dict(PARAMS, start_kick=rng.choice([0, 1]), launch_boost=lb, launch_no_kick=rng.choice([0, 1]) if lb else 0,
                tof_guard=rng.choice([1, 2, 3]), recovery=1, speed_ceiling=rng.choice([0, 1]), zone_caps=rng.choice([0, 1]),
                throttle_gain=round(rng.uniform(0.6, 1.2), 2), rev_us=round(rng.uniform(150, 400)),
                rev_s=round(rng.uniform(0.2, 0.8), 2))


def giveup_cleared(rec):
    """同じ AI の区間で、中止の行のあとに中止でない AI の行(中立止めの行は除く)"""
    out, in_g = [], False
    for t, r in rec.rows:
        if r['hold']:
            if not r['select']:
                in_g = False
            continue
        if not r['select']:
            in_g = False
            continue
        k = int(r['assist'])
        if k == A.ASSIST_GIVEUP:
            in_g = True
        elif in_g:
            out.append(round((t - T0) / 1e9, 3))
            in_g = False
    return out


def main(n=100, seed=1):
    rng = random.Random(seed)
    res = {'n': 0, 'flags': collections.Counter(), 'examples': collections.defaultdict(list), 'sum': collections.defaultdict(float)}
    for i in range(n):
        sc_ = scene_of(rng)
        p0 = params_of(rng)
        per = {}
        for keep, guard in ((0, 0), (0, 1), (1, 1)):
            p = dict(p0, rec_hold_keep=keep, rec_back_guard=guard)
            sim, rec, pca = CL.run(p, sc_)
            m = CL.metrics(sim, rec)
            m['giveup_cleared'] = giveup_cleared(rec)
            m['kick_rows'] = sum(1 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_START)
            per[(keep, guard)] = m
            for k_, v in m.items():
                if isinstance(v, float):
                    res['sum']['%s|%d%d' % (k_, keep, guard)] += v
        desc = {k: v for k, v in sc_.items() if k != 'model'}
        pp = {k: p0[k] for k in ('start_kick', 'launch_boost', 'launch_no_kick', 'tof_guard', 'speed_ceiling', 'zone_caps',
                                 'throttle_gain', 'rev_us', 'rev_s')}
        on, g, off = per[(1, 1)], per[(0, 1)], per[(0, 0)]

        def flag(name, extra):
            res['flags'][name] += 1
            if len(res['examples'][name]) < 6:
                res['examples'][name].append({'i': i, 'scene': desc, 'params': pp, 'extra': extra,
                                              'on': on, 'guard_only': g, 'off': off})
        if on.get('rear_push_s', 0) > 0.12:
            flag('on_rear_push', on.get('rear_push_s'))
        if on.get('rear_push_s', 0) > g.get('rear_push_s', 0) + 0.05:
            flag('on_rear_push_worse_than_guard_only', (on.get('rear_push_s'), g.get('rear_push_s')))
        if on.get('front_push_after_giveup_s', 0) > 0.1:
            flag('on_front_push_after_giveup', on.get('front_push_after_giveup_s'))
        if on['giveup_cleared']:
            flag('on_giveup_cleared', on['giveup_cleared'])
        if pp['launch_boost'] and pp['launch_no_kick'] and on['kick_rows']:
            flag('kick_with_no_kick', on['kick_rows'])
        nrear = lambda m: [h for h in m['hits'] if h[1] == 'rear']
        if len(nrear(on)) > len(nrear(g)):
            flag('on_more_rear_hits', (nrear(on), nrear(g)))
        if max([-h[2] for h in nrear(on)] or [0]) > max([-h[2] for h in nrear(g)] or [0]) + 0.3:
            flag('on_faster_rear_hit', (nrear(on), nrear(g)))
        if on.get('front_push_s', 0) > g.get('front_push_s', 0) + 0.5:
            flag('on_more_front_push', (on.get('front_push_s'), g.get('front_push_s')))
        res['n'] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    res['flags'] = dict(res['flags'])
    res['examples'] = dict(res['examples'])
    res['sum'] = {k: round(v, 3) for k, v in sorted(res['sum'].items())}
    (OUT / ('fuzz_cl_%d.json' % seed)).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps({'n': res['n'], 'flags': res['flags'], 'sum': res['sum']}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100, int(sys.argv[2]) if len(sys.argv) > 2 else 1)
