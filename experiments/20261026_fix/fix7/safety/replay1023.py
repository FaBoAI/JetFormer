"""直す 7-1 回目・係 safety: 本物の 10/23 の記録(data/raw/20261023-205458-0002、読むだけ)を、tests/test_recovery_back_guard の
_replay と同じ形で Assist に流し、drive と同じく書いた値を EscMayReverse に入れて esc_rev を渡す。見張り ON で back_early が今まで通り
起きるか(10/23 の形 = ESC が後退を待っていた)、出す値が esc_rev なし(今までと同じ)と比べてどう違うかを見る。出力は標準出力"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A  # noqa: E402

REC = Path('/home/jetson/jetformer/data/raw/20261023-205458/20261023-205458-0002')


def replay(use_track):
    z = np.load(REC / 'AI_CMD.npz'); v = np.load(REC / 'VEHICLE.npz'); im = np.load(REC / 'IMU.npz')
    L = json.loads((REC.parent / 'log.json').read_text(encoding='utf-8'))
    p = L['drive']['params']
    ref = (float(L['drive']['steer_center_us']), float(L['drive']['throttle_neutral_us']))
    t = z['timestamps_us'].astype(np.int64); sel = z['select'].astype(bool); hold = z['hold'].astype(bool)
    dry = z['dry_run'].astype(bool); mo = z['model_out'].astype(float); lat = z['latency_ms'].astype(float)
    vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
    it = im['timestamps_us'].astype(np.int64); acc = im['accel'].astype(float)
    hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
    a = A.Assist(ref)
    tr = A.EscMayReverse(ref[1])
    q = None
    res = []
    for i in range(len(t)):
        now_us = t[i] + (lat[i] * 1000 if np.isfinite(lat[i]) else 0)
        if hold[i]:
            a.launch_hold(); a.reset()
            if use_track:
                if not sel[i] or dry[i]:
                    tr.unknown(now_us * 1000)
                else:
                    tr.write(now_us * 1000, ref[1])
            res.append(None); continue
        jv = np.searchsorted(vt, t[i], side='right') - 1
        e = ve[jv] if jv >= 0 else 0.0
        ji = np.searchsorted(it, now_us, side='right')
        idx = np.arange(max(0, ji - 10), ji)
        idx = idx[it[idx] > t[i] - 150_000]
        ig = float(hg[idx].max()) if len(idx) else 0.0
        er = None
        if use_track:
            if q is not None:
                er = tr.rev_s(q, now_us * 1000) > 0.0
            q = now_us * 1000
        o, k = a.step((mo[i, 0], mo[i, 1]), e, bool(sel[i]) and not dry[i], now_us / 1e6,
                      esc_on=p.get('esc_reverse', 1) == 1, rec_on=p.get('recovery', 1) == 1, impact_g=ig,
                      rev_us=float(p.get('rev_us', 300)), rev_s=float(p.get('rev_s', 0.4)),
                      kick_on=int(p.get('start_kick', 0) or 0) == 1, rec_guard=True, esc_rev=er)
        if use_track:
            if sel[i] and not dry[i]:
                tr.write(now_us * 1000 + 1, o[1])
            else:
                tr.unknown(now_us * 1000 + 1)
        ev, a.rec.guard_event = a.rec.guard_event, None
        res.append({'t': (t[i] - t[0]) / 1e6, 'out': o, 'kind': k, 'erpm': e, 'phase': a.rec.phase, 'ev': ev, 'er': er,
                    'mo': mo[i, 1]})
    return res


def main():
    a, b = replay(False), replay(True)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if (x is None) != (y is None) or (x and (x['out'] != y['out'] or x['phase'] != y['phase']))]
    print('frames', len(a), 'diff', len(diff), diff[:5])
    print('events none :', [(round(x['t'], 2), x['ev']) for x in a if x and x['ev']])
    print('events track:', [(round(x['t'], 2), x['ev']) for x in b if x and x['ev']])
    i0 = next(i for i, x in enumerate(b) if x and x['phase'] == 'brake')
    for x in b[i0 - 12:i0 + 25]:
        if x:
            print('  %.3f mo=%.0f out=%.0f erpm=%.0f phase=%s esc_rev=%s ev=%s' % (x['t'], x['mo'], x['out'][1], x['erpm'],
                                                                             x['phase'], x['er'], x['ev']))


if __name__ == '__main__':
    main()
