"""オフライン強化学習の観点でのデータ在庫調査(読み取りのみ)。
各 raw ログについて: 人/AIの時間、AI走行中の介入(交代)、リカバリー、衝突(水平2.5G)、周回、ToFガードの判定、
報酬計算に使える信号(eRPM・ジャイロ)の有効率を数える。
"""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer.laps import LapCounter

RAW = '/home/jetson/jetformer/data/raw'
G = 9.80665

def onsets(x):
    x = np.asarray(x, bool)
    return int(np.sum(x[1:] & ~x[:-1]) + (1 if len(x) and x[0] else 0))

def impacts(t, acc):
    # 水平(x,y)の加速度、重力を除く: z軸を重力側と仮定。水平成分のノルムから静的バイアスを引く
    h = np.hypot(acc[:, 0], acc[:, 1])
    h = h - np.median(h)
    ev = []
    last = -1e9
    for ti, hi in zip(t, h / G):
        if hi >= 2.5 and ti - last > 0.5:
            ev.append(ti); last = ti
    return ev

rows = []
for log in sorted(os.listdir(RAW)):
    lj = json.load(open(os.path.join(RAW, log, 'log.json')))
    desc = lj.get('description', ''); tags = lj.get('tags', [])
    rec = dict(log=log, desc=desc, ai='ai-drive' in tags, dur=0, human_s=0, ai_s=0, moving_s=0,
               human_moving_s=0, ai_moving_s=0, takeovers=0, rec=0, giveup=0, esc=0, imp=0, imp_ai=0, laps=0,
               clean=0, lap_times=[], guard_slow_s=0, guard_brake_s=0, ttc_lt05_s=0, scenes=0, ai_frames=0,
               reboots=0, mode=lj.get('mode_id'))
    lc = LapCounter()
    for sc in sorted(glob.glob(os.path.join(RAW, log, log + '-*'))):
        try:
            V = np.load(os.path.join(sc, 'VEHICLE.npz')); I = np.load(os.path.join(sc, 'IMU.npz'))
        except Exception:
            continue
        rec['scenes'] += 1
        vt = V['timestamps_us'] / 1e6; sel = V['select'].astype(bool); erpm = V['erpm'].astype(float)
        dt = np.diff(vt, append=vt[-1]); dt = np.clip(dt, 0, 0.1)
        mv = np.abs(erpm) > 200
        rec['dur'] += float(dt.sum())
        rec['ai_s'] += float(dt[sel].sum()); rec['human_s'] += float(dt[~sel].sum())
        rec['moving_s'] += float(dt[mv].sum())
        rec['ai_moving_s'] += float(dt[mv & sel].sum()); rec['human_moving_s'] += float(dt[mv & ~sel].sum())
        rec['takeovers'] += int(np.sum(sel[:-1] & ~sel[1:]))
        bt = V['board_t_us']; rec['reboots'] += int(np.sum(np.diff(bt) < -1e6))
        it = I['timestamps_us'] / 1e6; acc = I['accel']; gz = I['gyro'][:, 2]
        ev = impacts(it, acc); rec['imp'] += len(ev)
        si = np.searchsorted(vt, ev).clip(0, len(vt) - 1)
        rec['imp_ai'] += int(np.sum(sel[si]))
        # gyro unit: deg/s in raw (task note)
        ve = np.interp(it, vt, erpm); vs = sel[np.searchsorted(vt, it).clip(0, len(vt) - 1)]
        hg = np.hypot(acc[:, 0], acc[:, 1]); hg = (hg - np.median(hg)) / G
        for k in range(len(it)):
            lap = lc.add(it[k], float(gz[k]), ve[k], ai=bool(vs[k]), impact_g=float(hg[k]))
        p = os.path.join(sc, 'AI_CMD.npz')
        if os.path.exists(p):
            A = np.load(p)
            at = A['timestamps_us'] / 1e6; adt = np.clip(np.diff(at, append=at[-1]), 0, 0.1)
            asel = A['select'].astype(bool)
            rec['ai_frames'] += int(asel.sum())
            if 'assist' in A.files:
                a = A['assist']
                rec['rec'] += onsets(a == 2); rec['giveup'] += onsets(a == 3); rec['esc'] += onsets(a == 1)
            if 'guard' in A.files:
                g = A['guard']
                rec['guard_slow_s'] += float(adt[(g == 1) & asel].sum()); rec['guard_brake_s'] += float(adt[(g == 2) & asel].sum())
                ttc = A['ttc']; rec['ttc_lt05_s'] += float(adt[(ttc < 0.5) & asel].sum())
    rec['laps'] = len(lc.laps); rec['clean'] = sum(1 for x in lc.laps if x['clean'])
    rec['lap_times'] = [x['time_s'] for x in lc.laps if x['clean']]
    rows.append(rec)

tot = {}
print('%-16s %-22s %4s %6s %6s %6s %5s %4s %4s %4s %4s %5s %5s' % ('log', 'desc', 'AI?', 'dur', 'hum_mv', 'ai_mv', 'take', 'rec', 'imp', 'impA', 'laps', 'clean', 'medL'))
for r in rows:
    med = np.median(r['lap_times']) if r['lap_times'] else float('nan')
    print('%-16s %-22s %4s %6.0f %6.0f %6.0f %5d %4d %4d %4d %4d %5d %5.2f' % (r['log'], r['desc'][:22], 'AI' if r['ai'] else '', r['dur'], r['human_moving_s'], r['ai_moving_s'], r['takeovers'], r['rec'], r['imp'], r['imp_ai'], r['laps'], r['clean'], med))
keys = ['dur', 'human_s', 'ai_s', 'moving_s', 'human_moving_s', 'ai_moving_s', 'takeovers', 'rec', 'giveup', 'imp', 'imp_ai', 'laps', 'clean', 'ai_frames', 'reboots', 'guard_slow_s', 'guard_brake_s', 'ttc_lt05_s']
for k in keys:
    tot[k] = sum(r[k] for r in rows)
print('TOTAL', {k: round(v, 1) for k, v in tot.items()})
aiR = [r for r in rows if r['ai']]
print('AI-run logs', len(aiR), {k: round(sum(r[k] for r in aiR), 1) for k in keys})
huR = [r for r in rows if not r['ai']]
print('human logs', len(huR), {k: round(sum(r[k] for r in huR), 1) for k in keys})
json.dump(rows, open(os.path.join(os.path.dirname(__file__), 'inventory.json'), 'w'), ensure_ascii=False, indent=1)
