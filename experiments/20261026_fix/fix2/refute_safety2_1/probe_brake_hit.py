"""反証 safety2-1: 偽の入力で今の assist.Assist を動かす(装置は開かない)。ESC と車はとても簡単な模型:
- ready = 後退を待つ状態(直前がブレーキ → 中立)。ready で止まっている・下がっている時に後退側(>1538)→ 0.08 秒後から下がる
- 中立の帯(1478〜1538)では ready のまま。前進側(<1478)で ready は消え、後ろへ動いていればすぐ止まる(後退→前進はすぐ)
- 後ろへの加速は 10/23 の記録に近い 4400 eRPM/秒(最初 1400)、中立では 800 eRPM/秒で減る
- 後ろの壁 wall_m(m)。当たったら 5G(drive.py と同じく直近 0.15 秒の最大 = 0.15 秒だけ見える)、そのあと動けない
ケース: R_wall = 直前がブレーキ・後ろ 0.05 m に壁 / R_free = 直前がブレーキ・壁なし / F_free = 直前が前進(対照)・壁なし
     / F_wall = 直前が前進・後ろ 0.05 m に壁(対照)"""
import json, sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A
KV = 0.5e-3
ref = (1441.0, 1496.0)

def run(ready0, wall_m):
    a = A.Assist(ref); dt = 1 / 30; t = 0.0
    for _ in range(30):                       # 走って arm
        a.step((1441, 1400), -3000, True, t); t += dt
    for _ in range(20):                       # 中立の帯で止まりかけ(止まる)
        a.step((1441, 1515), -100, True, t); t += dt
    ready, braked, erpm, x, t_rev, t_hit, hit_t = ready0, False, 0.0, 0.0, None, None, None
    t_rec, out = None, []
    for _ in range(200):
        g = 5.0 if (hit_t is not None and t - hit_t <= 0.15) else 0.0
        o, kind = a.step((1441, 1515), erpm, True, t, impact_g=g)
        thr = o[1]
        if t_rec is None and kind == A.ASSIST_RECOVERY:
            t_rec = t
        if t_rec is not None:
            out.append({'el': round(t - t_rec, 3), 'phase': a.rec.phase, 'thr': round(thr), 'erpm': int(erpm),
                        'x_back_m': round(x, 3), 'g': g, 'backed_s': round(a.rec.backed_s, 3), 'retries': a.rec.retries})
        # 次のフレームまでの車
        if thr > 1538:
            if ready and erpm >= -150:
                if t_rev is None:
                    t_rev = t
                if t - t_rev >= 0.08 and hit_t is None:
                    erpm = 1400.0 if erpm < 1400 else min(erpm + 4400 * dt, 6000)
            else:
                braked = True                             # 前進の直後 = ブレーキ(このあと中立で ready)
        elif thr < 1478:
            ready, braked, t_rev = False, False, None
            erpm = -1500.0 if hit_t is None or erpm <= 0 else 0.0
        else:
            t_rev = None
            if braked:
                ready, braked = True, False
            erpm = max(erpm - 800 * dt, 0.0) if erpm > 0 else erpm
        if erpm > 0 and hit_t is None:
            x += erpm * KV * dt
            if wall_m is not None and x >= wall_m:
                hit_t, erpm = t + dt, 0.0
        if hit_t is not None and erpm > 0:
            erpm = 0.0
        t += dt
        if t_rec is not None and a.rec.phase is None:
            break
    rev_back = sum(1 for r in out if r['thr'] > 1538 and r['erpm'] > 300) / 30
    push = sum(1 for r in out if r['thr'] > 1538 and hit_t is not None and r['el'] >= hit_t - t_rec) / 30
    esc = next((r['el'] for r in out if r['phase'] == 'escape'), None)
    return {'ready0': ready0, 'wall_m': wall_m, 'hit_el': None if hit_t is None else round(hit_t - t_rec, 3),
            'rev_side_while_backing_s': round(rev_back, 2), 'push_after_hit_s': round(push, 2),
            'max_back_erpm': max(r['erpm'] for r in out), 'back_m_at_coast': next((r['x_back_m'] for r in out if r['phase'] in ('coast', 'escape')), None),
            'escape_el': esc, 'rows': out}

res = {k: run(*v) for k, v in {'R_wall': (True, 0.05), 'R_free': (True, None), 'F_free': (False, None), 'F_wall': (False, 0.05)}.items()}
for k, r in res.items():
    print(k, json.dumps({kk: vv for kk, vv in r.items() if kk != 'rows'}, ensure_ascii=False))
for k in ('R_wall',):
    for row in res[k]['rows']:
        print(' ', row)
json.dump(res, open('/home/jetson/jetformer/experiments/20261026_fix/fix2/refute_safety2_1/out/probe_brake_hit.json', 'w'), ensure_ascii=False, indent=1)
