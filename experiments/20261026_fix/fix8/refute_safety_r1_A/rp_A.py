"""票 A: 本物の記録(data/raw。読むだけ)を開いたループで Assist に流し、見張り ON(rec_guard True。中立止めの行は reset =
rec_hold_keep 0)の出す値を fix7 の前(tests/fixtures/fix7_safety_before/assist.py)と今(jetformer/assist.py + drive と同じ
EscMayReverse の esc_rev)で比べる。場面(scene)ごとに Assist を作り直す流し方('scene')と、1 本(log)の場面をつないで
1 つの Assist で流す流し方('concat')の両方。違いが出たあとも最後まで比べて、違うフレームの数を数える(開いたループ)。
車両の値は 2 つの取り方: 'cam' = 撮像時刻より前の最新(fix7 の replay_all と同じ)/ 'proc' = 処理の時刻(撮像 + latency)より前の最新。
    python -I -B rp_A.py → 標準出力・out/rp_A.json
"""
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/jetson/jetformer')
import rec_io  # noqa: E402
from jetformer import assist as A  # noqa: E402

FIXB = Path('/home/jetson/jetformer/tests/fixtures/fix7_safety_before/assist.py')
NOW_SHA = 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'


def load_bef():
    assert hashlib.sha256(FIXB.read_bytes()).hexdigest() == '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4'
    spec = importlib.util.spec_from_loader('jetformer._rpA_assist_f7b', loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(compile(FIXB.read_text(encoding='utf-8'), str(FIXB), 'exec'), m.__dict__)
    return m


def replay(cat, L, B, vmode):
    z, v, im = cat['AI_CMD'], cat['VEHICLE'], cat['IMU']
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
    kw = dict(esc_on=p.get('esc_reverse', 1) == 1, rec_on=p.get('recovery', 1) == 1, rev_us=float(p.get('rev_us', 300)),
              rev_s=float(p.get('rev_s', 0.4)), kick_on=int(p.get('start_kick', 0) or 0) == 1, rec_guard=True)
    res = {'frames': 0, 'diff': [], 'rec_starts': 0, 'backing_in_bwg_now': [], 'events_now': collections.Counter(),
           'events_bef': collections.Counter()}
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
        tv = t[i] if vmode == 'cam' else now_us
        jv = np.searchsorted(vt, tv, side='right') - 1
        e = ve[jv] if jv >= 0 else 0.0
        ji = np.searchsorted(it, now_us, side='right')
        idx = np.arange(max(0, ji - 10), ji)
        idx = idx[it[idx] > t[i] - 150_000]
        ig = float(hg[idx].max()) if len(idx) else 0.0
        er = None
        if q is not None:
            er = gate.rev_s(q, ns) > 0.0
        q = ns
        ph0n, ph0b = a1.rec.phase, a0.rec.phase
        s_ = bool(sel[i]) and not dry[i]
        o0, k0 = a0.step((mo[i, 0], mo[i, 1]), e, s_, now_us / 1e6, impact_g=ig, **kw)
        o1, k1 = a1.step((mo[i, 0], mo[i, 1]), e, s_, now_us / 1e6, impact_g=ig, esc_rev=er, **kw)
        ev1, a1.rec.guard_event = a1.rec.guard_event, None
        ev0, a0.rec.guard_event = a0.rec.guard_event, None
        if ev1:
            res['events_now'][ev1[0]] += 1
        if ev0:
            res['events_bef'][ev0[0]] += 1
        if ph0n is None and a1.rec.phase == 'brake':
            res['rec_starts'] += 1
        if ph0n in ('brake', 'wait', 'gap') and e > A.BACKING_ERPM:
            res['backing_in_bwg_now'].append((int(t[i]), ph0n, int(e)))
        res['frames'] += 1
        if s_:
            gate.write(ns + 1, o1[1])
        else:
            gate.unknown(ns + 1)
        if o0 != o1 or k0 != k1:
            res['diff'].append({'t_us': int(t[i]), 'phase_bef': [ph0b, a0.rec.phase], 'phase_now': [ph0n, a1.rec.phase],
                                'erpm': int(e), 'esc_rev': er, 'ev_now': ev1 and ev1[0], 'ev_bef': ev0 and ev0[0],
                                'out_bef': [round(float(o0[0]), 1), round(float(o0[1]), 1)],
                                'out_now': [round(float(o1[0]), 1), round(float(o1[1]), 1)], 'kind': [int(k0), int(k1)]})
    return res


def main():
    assert hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest() == NOW_SHA
    B = load_bef()
    logs = sorted(p.name for p in rec_io.RAW.iterdir() if p.is_dir() and (p / 'log.json').exists() and rec_io.scenes_of(p.name))
    out = {}
    for vmode in ('cam', 'proc'):
        for how in ('scene', 'concat'):
            key = '%s_%s' % (how, vmode)
            tot = {'logs': 0, 'units': 0, 'frames': 0, 'rec_starts': 0, 'diff_frames': 0, 'diff_units': [],
                   'backing_in_bwg_now': 0, 'backing_units': collections.Counter(), 'skipped': []}
            for log in logs:
                ss = rec_io.scenes_of(log)
                units = [[s] for s in ss] if how == 'scene' else [ss]
                tot['logs'] += 1
                for u in units:
                    try:
                        cat, L, t0 = rec_io.load_log(log, u)
                        r = replay(cat, L, B, vmode)
                    except Exception as ex:     # noqa: BLE001  空の記録・古い形
                        tot['skipped'].append([u[0].name, repr(ex)[:80]])
                        continue
                    tot['units'] += 1
                    tot['frames'] += r['frames']
                    tot['rec_starts'] += r['rec_starts']
                    tot['backing_in_bwg_now'] += len(r['backing_in_bwg_now'])
                    if r['backing_in_bwg_now']:
                        tot['backing_units'][u[0].name if how == 'scene' else log] += len(r['backing_in_bwg_now'])
                    if r['diff']:
                        t0s = rec_io.load_log(log)[2]
                        tot['diff_frames'] += len(r['diff'])
                        tot['diff_units'].append({'unit': u[0].name if how == 'scene' else log, 'n': len(r['diff']),
                                                  'frames': [dict(d, t_log_s=round((d['t_us'] - t0s) / 1e6, 3)) for d in r['diff']]})
            tot['backing_units'] = dict(tot['backing_units'])
            out[key] = tot
            print(key, {k: v for k, v in tot.items() if k not in ('diff_units', 'skipped', 'backing_units')},
                  'diff units', [(d['unit'], d['n']) for d in tot['diff_units']], 'backing units', tot['backing_units'],
                  'skipped', len(tot['skipped']), flush=True)
    for key in out:
        for d in out[key]['diff_units']:
            if d['unit'].startswith('20261024-160847'):
                print('==', key, d['unit'], d['n'], 'frames')
                for f in d['frames']:
                    print('   %.3f erpm %5d esc_rev %s bef %s %s ev %s | now %s %s ev %s' % (
                        f['t_log_s'], f['erpm'], f['esc_rev'], f['phase_bef'], f['out_bef'], f['ev_bef'], f['phase_now'],
                        f['out_now'], f['ev_now']))
    (HERE / 'out' / 'rp_A.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
