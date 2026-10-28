"""打ち消しの票 B(指摘 rs8-1-real-record-backing-missed): fix7/safety/replay_all.py と同じ開いたループ(Assist に記録の値を入れる)を、
(a) 場面ごとに Assist を作り直す形(fix7 の形)と (b) 1 本の記録の場面をつないで 1 つの Assist で流す形で回し、
fix7 の前(tests/fixtures/fix7_safety_before/assist.py)と今の assist.py の出す値が初めて違うフレームを数える。
data/raw は読むだけ。装置・GPU・API に触れない。
    python -I -B replay_concat.py → out/replay_concat.json
"""
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A  # noqa: E402

RAW = Path('/home/jetson/jetformer/data/raw')
FIXB = Path('/home/jetson/jetformer/tests/fixtures/fix7_safety_before/assist.py')
HERE = Path(__file__).resolve().parent
NOW_SHA = 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'


def load_bef():
    assert hashlib.sha256(FIXB.read_bytes()).hexdigest() == '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4'
    spec = importlib.util.spec_from_loader('jetformer._rc_assist_f7b', loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(compile(FIXB.read_text(encoding='utf-8'), str(FIXB), 'exec'), m.__dict__)
    return m


def cat(dirs, name, keys):
    parts = [np.load(d / (name + '.npz')) for d in dirs]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys if all(k in p for p in parts)}


def replay(dirs, logdir, B):
    z = cat(dirs, 'AI_CMD', ('timestamps_us', 'select', 'hold', 'dry_run', 'model_out', 'out', 'latency_ms'))
    v = cat(dirs, 'VEHICLE', ('timestamps_us', 'erpm'))
    im = cat(dirs, 'IMU', ('timestamps_us', 'accel'))
    L = json.loads((logdir / 'log.json').read_text(encoding='utf-8'))
    dr = L.get('drive') or {}
    p = dr.get('params') or {}
    ref = (float(dr.get('steer_center_us', 1441)), float(dr.get('throttle_neutral_us', 1496)))
    t = z['timestamps_us'].astype(np.int64)
    sel = z['select'].astype(bool)
    hold = z['hold'].astype(bool)
    dry = z['dry_run'].astype(bool) if 'dry_run' in z else np.zeros(len(t), bool)
    mo = z['model_out'].astype(float) if 'model_out' in z else z['out'].astype(float)
    lat = z['latency_ms'].astype(float) if 'latency_ms' in z else np.full(len(t), np.nan)
    vt = v['timestamps_us'].astype(np.int64)
    ve = v['erpm'].astype(float)
    it = im['timestamps_us'].astype(np.int64)
    acc = im['accel'].astype(float)
    hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
    a0, a1 = B.Assist(ref), A.Assist(ref)
    gate = A.EscMayReverse(ref[1])
    q = None
    out = {'frames': 0, 'rec_starts': [], 'diverge': None, 'n_diff': 0, 'events_now': collections.Counter(),
           'events_bef': collections.Counter(), 'backing_in_bwg': []}
    kw = dict(esc_on=p.get('esc_reverse', 1) == 1, rec_on=p.get('recovery', 1) == 1, rev_us=float(p.get('rev_us', 300)),
              rev_s=float(p.get('rev_s', 0.4)), kick_on=int(p.get('start_kick', 0) or 0) == 1, rec_guard=True)
    t00 = int(t[0])
    for i in range(len(t)):
        now_us = t[i] + (lat[i] * 1000 if np.isfinite(lat[i]) else 0)
        ns = int(now_us * 1000)
        if hold[i]:
            a0.launch_hold(); a0.reset()
            a1.launch_hold(); a1.reset()
            if not sel[i] or dry[i]:
                gate.unknown(ns)
            else:
                gate.write(ns, ref[1])
            continue
        jv = np.searchsorted(vt, t[i], side='right') - 1
        e = ve[jv] if jv >= 0 else 0.0
        ji = np.searchsorted(it, now_us, side='right')
        idx = np.arange(max(0, ji - 10), ji)
        idx = idx[it[idx] > t[i] - 150_000]
        ig = float(hg[idx].max()) if len(idx) else 0.0
        er = None
        if q is not None:
            er = gate.rev_s(q, ns) > 0.0
        q = ns
        ph0 = a1.rec.phase
        s_ = bool(sel[i]) and not dry[i]
        o0, k0 = a0.step((mo[i, 0], mo[i, 1]), e, s_, now_us / 1e6, impact_g=ig, **kw)
        o1, k1 = a1.step((mo[i, 0], mo[i, 1]), e, s_, now_us / 1e6, impact_g=ig, esc_rev=er, **kw)
        ev1, a1.rec.guard_event = a1.rec.guard_event, None
        ev0, a0.rec.guard_event = a0.rec.guard_event, None
        if ev1:
            out['events_now'][ev1[0]] += 1
        if ev0:
            out['events_bef'][ev0[0]] += 1
        if ph0 is None and a1.rec.phase == 'brake':
            out['rec_starts'].append(round((t[i] - t00) / 1e6, 3))
        if ph0 in ('brake', 'wait', 'gap') and e > A.BACKING_ERPM:
            out['backing_in_bwg'].append((round((t[i] - t00) / 1e6, 3), ph0, float(e)))
        out['frames'] += 1
        if s_:
            gate.write(ns + 1, o1[1])
        else:
            gate.unknown(ns + 1)
        if (o0 != o1 or k0 != k1):
            if out['diverge'] is None:
                out['diverge'] = {'s': round((t[i] - t00) / 1e6, 3), 'phase0': ph0, 'phase_now': a1.rec.phase,
                                  'phase_bef': a0.rec.phase, 'erpm': float(e), 'esc_rev': er, 'ev_now': ev1 and ev1[0],
                                  'ev_bef': ev0 and ev0[0], 'out_bef': o0, 'out_now': o1}
            break
    out['events_now'] = dict(out['events_now'])
    out['events_bef'] = dict(out['events_bef'])
    return out


def main():
    assert hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest() == NOW_SHA
    B = load_bef()
    res = {}
    for mode in ('per_scene', 'concat'):
        tot = {'units': 0, 'skipped': 0, 'frames': 0, 'rec_starts': 0, 'diverged': [], 'backing_in_bwg': [],
               'events_now': collections.Counter(), 'events_bef': collections.Counter()}
        for logdir in sorted(p for p in RAW.glob('2026*') if p.is_dir()):
            if not (logdir / 'log.json').exists():
                continue
            scenes = [d for d in sorted(logdir.glob('2026*')) if d.is_dir() and all((d / f).exists() for f in ('AI_CMD.npz', 'VEHICLE.npz', 'IMU.npz'))]
            if not scenes:
                continue
            units = [[d] for d in scenes] if mode == 'per_scene' else [scenes]
            for u in units:
                try:
                    r = replay(u, logdir, B)
                except Exception as ex:      # noqa: BLE001
                    tot['skipped'] += 1
                    continue
                tot['units'] += 1
                tot['frames'] += r['frames']
                tot['rec_starts'] += len(r['rec_starts'])
                tot['events_now'].update(r['events_now'])
                tot['events_bef'].update(r['events_bef'])
                name = u[0].name if mode == 'per_scene' else logdir.name
                for b in r['backing_in_bwg']:
                    tot['backing_in_bwg'].append((name,) + tuple(b))
                if r['diverge'] is not None:
                    tot['diverged'].append(dict(r['diverge'], unit=name))
        tot['events_now'] = dict(tot['events_now'])
        tot['events_bef'] = dict(tot['events_bef'])
        res[mode] = tot
        print(mode, json.dumps({k: v for k, v in tot.items()}, ensure_ascii=False, default=str), flush=True)
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'replay_concat.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
