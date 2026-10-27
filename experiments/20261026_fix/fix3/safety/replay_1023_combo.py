"""直す 3 回目 safety (3): 10/23 20261023-205458-0002 のリカバリー(行 103〜129)を assist.Assist に記録の値で通し(開いたループ)、
後ろの見張り(rec_back_guard)と中立止めでリカバリーを残す(rec_hold_keep)の 4 つの組で比べる。
入力の作り方は fix2/safety2/replay_1023_recovery.py・fix2/round2/replay_guard.py と同じ(撮像の時刻の eRPM・直近 0.15 秒の IMU)。
この記録には中立止めが無いので、「もし中立止めがあったら」(行を中立止めにする)も流す。読むだけ。結果 out/replay_1023_combo.json
(drive と同じく rec_hold_keep は rec_back_guard = 1 のときだけ効く。Assist だけで keep を効かせた版は out/replay_1023_combo_assist_keep_alone.json)

距離: 記録の eRPM(後ろ +)× 0.5e-3 m/s を積んだ「記録のまま下がった距離」。開いたループなので、ON で後退側を早くやめたあとの
車の動きは記録に無い。やめた時点(+ ESC の遅れ 0.1 秒)の距離・速さと、後ろの物(記録では 1.23 m で当たった)の手前で止まるのに
要る惰性の減速を出す。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
from jetformer import assist as A   # noqa: E402

REC = ROOT / 'data/raw/20261023-205458/20261023-205458-0002'
OUT = Path(__file__).resolve().parent / 'out' / 'replay_1023_combo.json'
KV = 0.5e-3


def load():
    z = np.load(REC / 'AI_CMD.npz'); v = np.load(REC / 'VEHICLE.npz'); im = np.load(REC / 'IMU.npz')
    L = json.loads((REC.parent / 'log.json').read_text(encoding='utf-8'))
    return z, v, im, L


def replay(guard, keep, holds=()):
    z, v, im, L = load()
    p = L['drive']['params']
    ref = (float(L['drive']['steer_center_us']), float(L['drive']['throttle_neutral_us']))
    t = z['timestamps_us'].astype(np.int64); sel = z['select'].astype(bool); hold = z['hold'].astype(bool).copy()
    for i in holds:
        hold[i] = True
    dry = z['dry_run'].astype(bool); mo = z['model_out'].astype(float); out = z['out'].astype(float)
    kind = z['assist'].astype(int); lat = z['latency_ms'].astype(float)
    vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
    it = im['timestamps_us'].astype(np.int64); acc = im['accel'].astype(float)
    hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
    a = A.Assist(ref)
    rows, mism, events = [], [], []
    for i in range(len(t)):
        if hold[i]:
            a.launch_hold()
            a.reset(keep_recovery=bool(keep) and bool(guard) and bool(sel[i]) and not dry[i])   # drive の hold と同じ(見張りが ON のときだけ残す)
            rows.append({'i': i, 'hold': True, 'phase': a.rec.phase, 'thr': ref[1], 'kind': 0, 'erpm': None})
            continue
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
                      kick_on=int(p.get('start_kick', 0) or 0) == 1, rec_guard=bool(guard))
        if a.rec.guard_event is not None:
            events.append([i] + list(a.rec.guard_event)); a.rec.guard_event = None
        if not hold[i] and (abs(o[1] - out[i, 1]) >= 0.6 or abs(o[0] - out[i, 0]) >= 0.6 or k != kind[i]):
            mism.append(i)
        rows.append({'i': i, 'hold': False, 'phase': a.rec.phase, 'thr': round(o[1], 1), 'kind': int(k), 'erpm': int(e),
                     'g': round(ig, 2)})
    return {'rows': rows, 'mismatch': mism, 'events': events, 'ref': ref, 't': t, 'vt': vt, 've': ve,
            'starts': a.rec.count}


def summary(res, lo=95, hi=140):
    n = res['ref'][1]
    rr = [r for r in res['rows'] if lo <= r['i'] < hi]
    rev = [r for r in rr if not r['hold'] and r['kind'] == A.ASSIST_RECOVERY and r['thr'] > n + A.EscReverse.DEADBAND_US]
    rev_back = [r for r in rev if r['erpm'] is not None and r['erpm'] > A.BACKING_ERPM]
    last_rev = max((r['i'] for r in rev), default=None)
    brake_starts = sum(1 for a, b in zip(rr, rr[1:]) if b['phase'] == 'brake' and a['phase'] != 'brake')
    t, vt, ve = res['t'], res['vt'], res['ve']
    t_back0 = t[106]                                 # 記録で後ろへ動き始めたフレーム(eRPM > 300 の最初)

    def x_at(t_us):                                  # 記録のまま下がった距離 [m](t_back0 から)
        m = (vt >= t_back0 - 40_000) & (vt <= t_us)
        e = np.clip(ve[m], 0, None)
        return float(np.sum(e[:-1] * KV * np.diff(vt[m]) / 1e6)) if m.sum() > 1 else 0.0

    def v_at(t_us):
        j = np.searchsorted(vt, t_us, side='right') - 1
        return float(max(ve[j], 0.0) * KV)
    out = {'reverse_side_frames': len(rev), 'reverse_side_while_backing_frames': len(rev_back),
           'reverse_side_while_backing_s': round(len(rev_back) / 30.0, 2), 'last_reverse_row': last_rev,
           'brake_phase_starts': brake_starts, 'guard_events': res['events'], 'mismatch_rows': len(res['mismatch'])}
    if last_rev is not None:
        t_stop = t[last_rev + 1] + 100_000           # 後退側をやめたフレーム + ESC の遅れ 0.1 秒(10/23 211828-0007 で見た)
        x, v = x_at(t_stop), v_at(t_stop)
        room = 1.23 - x
        out.update({'stop_row': last_rev + 1, 'x_back_at_stop_plus_0p1s_m': round(x, 2), 'v_back_at_stop_plus_0p1s_mps': round(v, 2),
                    'room_to_rear_obstacle_m': round(room, 2),
                    'coast_decel_needed_to_stop_before_obstacle_mps2': round(v * v / (2 * room), 2) if room > 0 else None})
    return out


def main():
    res = {'scene': str(REC.relative_to(ROOT)), 'note': '開いたループ。記録には中立止めが無い。距離は記録の eRPM のまま',
           'record_dist_back_to_hit_m': 1.23, 'combos': {}, 'what_if_holds': {}}
    for guard in (0, 1):
        for keep in (0, 1):
            r = replay(guard, keep)
            res['combos']['guard%d_keep%d' % (guard, keep)] = summary(r)
    # もし中立止めがあったら: 'brake' の段で下がり始めた直後(行 108・109)・後退の途中(行 114・115)・後退の終わり近く(行 121・122)
    for name, holds in (('hold_108_109', (108, 109)), ('hold_114_115', (114, 115)), ('hold_121_122', (121, 122))):
        for guard in (0, 1):
            for keep in (0, 1):
                r = replay(guard, keep, holds)
                res['what_if_holds']['%s_guard%d_keep%d' % (name, guard, keep)] = summary(r)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=int))
    for k, v in res['combos'].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != 'guard_events'}, v['guard_events'])
    for k, v in res['what_if_holds'].items():
        print(k, {kk: vv for kk, vv in v.items() if kk not in ('guard_events', 'mismatch_rows')})


if __name__ == '__main__':
    main()
