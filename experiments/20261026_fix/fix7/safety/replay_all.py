"""直す 7-1 回目・係 safety: 本物の記録(data/raw の AI_CMD がある 242 本。読むだけ)を、tests/test_recovery_back_guard の _replay と
同じ形(開いたループ: 記録の車両の値・IMU・モデルの出力をそのまま入れる)で Assist に流し、見張り ON(rec_back_guard = 1)の出す値を
  bef = 直す前の写し(tests/fixtures/fix7_safety_before の assist.py)
  now = 今の assist.py(drive と同じく、書いた値を EscMayReverse に入れて esc_rev を渡す)
で比べる。開いたループなので、出す値が初めて違ったフレームより後は比べない(そのあとの車は記録の出力で動いている)。数えるもの:
本数・フレーム・リカバリーを始めた数・初めて違ったフレームの数と、そのときの段・出来事(今の back_no_esc・early_none)・跳ねの eRPM。
出力 out/replay_all.json。装置・GPU・API には触れない
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


def load_bef():
    assert hashlib.sha256(FIXB.read_bytes()).hexdigest() == '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4'
    spec = importlib.util.spec_from_loader('jetformer._ra_assist_f7b', loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(compile(FIXB.read_text(encoding='utf-8'), str(FIXB), 'exec'), m.__dict__)
    return m


def replay(d, B):
    z = np.load(d / 'AI_CMD.npz')
    v = np.load(d / 'VEHICLE.npz')
    im = np.load(d / 'IMU.npz')
    L = json.loads((d.parent / 'log.json').read_text(encoding='utf-8'))
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
    out = {'frames': 0, 'rec_starts': 0, 'diverge': None, 'events_now': collections.Counter(), 'spikes_in_guard': 0}
    kw = dict(esc_on=p.get('esc_reverse', 1) == 1, rec_on=p.get('recovery', 1) == 1, rev_us=float(p.get('rev_us', 300)),
              rev_s=float(p.get('rev_s', 0.4)), kick_on=int(p.get('start_kick', 0) or 0) == 1, rec_guard=True)
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
        a0.rec.guard_event = None
        if ev1:
            out['events_now'][ev1[0]] += 1
        if ph0 is None and a1.rec.phase == 'brake':
            out['rec_starts'] += 1
        if ph0 in ('brake', 'wait', 'gap') and e > A.BACKING_ERPM:
            out['spikes_in_guard'] += 1
        out['frames'] += 1
        if s_:
            gate.write(ns + 1, o1[1])
        else:
            gate.unknown(ns + 1)
        if (o0 != o1 or k0 != k1) and out['diverge'] is None:
            out['diverge'] = {'i': i, 's': round((t[i] - t[0]) / 1e6, 3), 'phase0': ph0, 'phase_now': a1.rec.phase,
                              'phase_bef': a0.rec.phase, 'erpm': float(e), 'esc_rev': er, 'ev_now': ev1 and ev1[0],
                              'out_bef': o0, 'out_now': o1}
            break
    return out


def main():
    B = load_bef()
    res = {'files': 0, 'skipped': 0, 'frames': 0, 'rec_starts': 0, 'diverged': [], 'events_now': collections.Counter(),
           'frames_backing_in_brake_wait_gap': 0}
    for d in sorted(RAW.glob('2026*/*')):
        if not (d / 'AI_CMD.npz').exists() or not (d / 'VEHICLE.npz').exists() or not (d / 'IMU.npz').exists():
            continue
        if not (d.parent / 'log.json').exists():
            res['skipped'] += 1
            continue
        try:
            r = replay(d, B)
        except Exception as ex:      # noqa: BLE001  壊れた記録・古い形は数えて飛ばす
            res['skipped'] += 1
            print('skip', d.name, repr(ex)[:120])
            continue
        res['files'] += 1
        res['frames'] += r['frames']
        res['rec_starts'] += r['rec_starts']
        res['events_now'].update(r['events_now'])
        res['frames_backing_in_brake_wait_gap'] += r['spikes_in_guard']
        if r['diverge'] is not None:
            res['diverged'].append(dict(r['diverge'], file=d.name))
            print('diverge', d.name, r['diverge'])
    res['events_now'] = dict(res['events_now'])
    (HERE / 'out' / 'replay_all.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != 'diverged'}, ensure_ascii=False), 'diverged', len(res['diverged']))


if __name__ == '__main__':
    main()
