"""X2c: 同じ当たり(前の壁・約 1.8 m/s・6.5G)のあとカメラが L 秒来ない。壁の位置を 1 cm ずつずらして、当たりとカメラの
フレームの位相だけを変える(keep = 1・見張り 1)。明けてからの結果が「中止」か「自動で後退」かが位相で分かれるか"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl3                                    # noqa: E402
from cl3 import A, C, N, PARAMS, T0           # noqa: E402

res = []
for L in (0.6, 1.0, 2.0):
    for j in range(8):
        xf = 2.00 + 0.01 * j
        sc_ = {'dur': 8.0, 'xf': xf, 'model': (C, N - 120.0), 'drops': [], 'hit_drop': (0.0, L)}
        p = dict(PARAMS, recovery=1, start_kick=1, rec_back_guard=1, rec_hold_keep=1)
        sim, rec, pca, ph = cl3.run(p, sc_)
        t_hit = sim.car.hits[0][0]
        back = t_hit + L
        proc = [(t - T0) / 1e9 for t, r in rec.rows if not r['hold']]
        last_before = max(t for t in proc if t <= t_hit + 0.001)
        first_after_hit = min(t for t in proc if t > t_hit + 0.001)
        rec_rev = [(t - T0) / 1e9 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_RECOVERY and r['out'][1] > N + 40]
        gu = any(int(r['assist']) == A.ASSIST_GIVEUP for t, r in rec.rows if not r['hold'])
        xs = [r[1] for r in sim.log if r[0] >= back]
        res.append({'L': L, 'xf': round(xf, 2), 't_hit': t_hit, 'last_frame_before_hit_ms': round((t_hit - last_before) * 1000, 1),
                    'first_frame_after_hit_ms': round((first_after_hit - t_hit) * 1000, 1),
                    'result': 'giveup' if gu else ('reverse %.2f s after camera' % (rec_rev[0] - back) if rec_rev else 'none'),
                    'backed_m': round(xf - min(xs), 3),
                    'ev': [(e[1]['what'], e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard']})
for r in res:
    print(r)
(Path(__file__).resolve().parent / 'out' / 'x2c_phase.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
shutil.rmtree(cl3.TMP, ignore_errors=True)
