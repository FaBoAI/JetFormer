"""直す 2 回目 safety2-1: 10/23 20261023-205458-0002 の記録の入力を assist.Assist に通す(後ろの見張り OFF / ON)。
OFF は記録の out・assist と同じかを数え、ON はリカバリーの段・出力・eRPM を並べる。読むだけ(装置・API に触れない)。
入力の作り方は refute_safety2_1/replay_now_cap.py と同じ(撮像の時刻の eRPM・drive.py と同じ直近 0.15 秒の IMU)。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
from jetformer import assist as A   # noqa: E402

REC = 'data/raw/20261023-205458/20261023-205458-0002'


def load(rec=REC):
    D = ROOT / rec
    z = np.load(D / 'AI_CMD.npz'); v = np.load(D / 'VEHICLE.npz'); im = np.load(D / 'IMU.npz')
    L = json.loads((D.parent / 'log.json').read_text())
    return z, v, im, L


def replay(guard, rec=REC, lo=None, hi=None):
    z, v, im, L = load(rec)
    p = L['drive']['params']
    ref = (float(L['drive']['steer_center_us']), float(L['drive']['throttle_neutral_us']))
    t = z['timestamps_us'].astype(np.int64); sel = z['select'].astype(bool); hold = z['hold'].astype(bool)
    dry = z['dry_run'].astype(bool)
    mo = z['model_out'].astype(float); out = z['out'].astype(float); kind = z['assist'].astype(int)
    lat = z['latency_ms'].astype(float)
    vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
    it = im['timestamps_us'].astype(np.int64); acc = im['accel'].astype(float)
    hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
    a = A.Assist(ref)
    rows, mism, events = [], [], []
    lo = 0 if lo is None else lo
    hi = len(t) if hi is None else hi
    for i in range(lo, hi):
        if hold[i]:
            a.launch_hold(); a.reset(); continue
        now_us = t[i] + (lat[i] * 1000 if np.isfinite(lat[i]) else 0)
        jv = np.searchsorted(vt, t[i], side='right') - 1
        e = ve[jv] if jv >= 0 else 0.0
        ji = np.searchsorted(it, now_us, side='right')
        idx = np.arange(max(0, ji - 10), ji)
        idx = idx[it[idx] > t[i] - 150_000]
        ig = float(hg[idx].max()) if len(idx) else 0.0
        o, k = a.step((mo[i, 0], mo[i, 1]), e, bool(sel[i]) and not dry[i], now_us / 1e6,
                      esc_on=p.get('esc_reverse', 1) == 1, rec_on=p.get('recovery', 1) == 1, impact_g=ig,
                      rev_us=float(p.get('rev_us', 300)), rev_s=float(p.get('rev_s', 0.4)),
                      kick_on=int(p.get('start_kick', 0) or 0) == 1, rec_guard=guard)
        if a.rec.guard_event is not None:
            events.append((i, a.rec.guard_event)); a.rec.guard_event = None
        if abs(o[1] - out[i, 1]) >= 0.6 or abs(o[0] - out[i, 0]) >= 0.6 or k != kind[i]:
            mism.append(i)
        rows.append({'i': i, 'phase': a.rec.phase, 'thr': round(o[1], 1), 'rec_thr': round(out[i, 1], 1), 'kind': k,
                     'erpm': int(e), 'g': round(ig, 2), 'backed_s': round(a.rec.backed_s, 3)})
    return {'rows': rows, 'mismatch': mism, 'events': events, 'n': hi - lo, 'ref': ref,
            'rev_us': float(p.get('rev_us', 300)), 'rev_s': float(p.get('rev_s', 0.4))}


def rev_while_backing(res, lo=95, hi=135):
    """リカバリーの間に、後ろへ動いている(eRPM > 300)フレームで後退側(中立 + 40 µs より大きい)を出した数"""
    n = res['ref'][1]
    return sum(1 for r in res['rows'] if lo <= r['i'] < hi and r['kind'] == A.ASSIST_RECOVERY
               and r['erpm'] > A.BACKING_ERPM and r['thr'] > n + A.EscReverse.DEADBAND_US)


if __name__ == '__main__':
    off, on = replay(False), replay(True)
    print('OFF: 行', off['n'], '記録と違う行', len(off['mismatch']), off['mismatch'][:10])
    print('ON : 記録と違う行', len(on['mismatch']), on['mismatch'][:30], '見張り', on['events'])
    print('下がりながら後退側を出したフレーム OFF %d・ON %d' % (rev_while_backing(off), rev_while_backing(on)))
    for r_off, r_on in zip(off['rows'], on['rows']):
        if 98 <= r_off['i'] <= 140:
            print(r_off['i'], r_off['phase'], r_off['thr'], '|', r_on['phase'], r_on['thr'], r_on['backed_s'], '| eRPM', r_on['erpm'], 'g', r_on['g'])
