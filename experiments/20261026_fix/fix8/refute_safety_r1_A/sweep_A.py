"""票 A: 10/23 の形(前進 m µs で走り、壁 xf の db 手前からブレーキ側 → 止まって ESC が後退を待つ → リカバリー)を、
ESC の模型 × 場面で両方 ON・両方 OFF(一部は見張りだけ)に流す。後ろの物は置かずに走り、最初のリカバリーの始めの位置から
d [m] 後ろに物があったら最初に当たる速さを出す(当たる前の動きは物があってもなくても同じ。check_obstacles で照らす)。
    python -I -B sweep_A.py → out/sweep_A.json
"""
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cl_A  # noqa: E402

NOW = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e',
       'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}
DS = [round(0.1 + 0.05 * k, 3) for k in range(39)]        # 0.1〜2.0 m


def scenes():
    for m, (xf, db), bu, vdt in itertools.product((120.0, 200.0), ((2.4, 0.45), (3.0, 0.6)), (200.0, 100.0), (0.01, 0.02)):
        yield {'m': m, 'xf': xf, 'db': db, 'brake_us': bu, 'veh_dt': vdt}


def one(esc, sw, sc):
    sim, rec, pca, ph = cl_A.run(cl_A.ESC[esc], sw, m=sc['m'], xf=sc['xf'], db=sc['db'], veh_dt=sc['veh_dt'],
                                 brake_us=sc['brake_us'], dur=6.5)
    t0, t1 = cl_A.first_rec(ph)
    if t0 is None:
        return {'rec': None}
    prof, x0, back, vpk = cl_A.hit_profile(sim, t0, t1 if t1 is not None else 99.0, DS)
    pre = [(round(r[1], 6), round(r[2], 6), r[3]) for r in sim.log if r[0] < t0 - 1e-9]
    ev = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    phs, pp = [], None
    for tt, p0, p1, *_ in ph:
        if t0 - 0.01 <= tt <= (t1 if t1 is not None else 99) and p1 != pp:
            phs.append((tt, p1))
            pp = p1
    return {'rec': [t0, t1], 'x0': round(x0, 4), 'back': back, 'vpk': vpk, 'hit': {str(k): v for k, v in prof.items()},
            'pre_sha': hashlib.sha256(json.dumps(pre).encode()).hexdigest()[:16], 'events': ev, 'phases': phs[:8],
            'front_hits': [h for h in sim.car.hits if h[1] == 'front']}


def main():
    for fn, h in NOW.items():
        assert hashlib.sha256((cl_A.ROOT / 'jetformer' / fn).read_bytes()).hexdigest() == h, fn
    res = []
    t_s = time.time()
    try:
        for esc in sorted(cl_A.ESC):
            for sc in scenes():
                row = {'esc': esc, **sc}
                sws = ('on', 'off', 'guard') if esc in ('fit', 'human', 'cl') else ('on', 'off')
                for sw in sws:
                    row[sw] = one(esc, sw, sc)
                res.append(row)
            print(esc, len(res), '%.0fs' % (time.time() - t_s), flush=True)
    finally:
        cl_A.cleanup()
    (HERE / 'out' / 'sweep_A.json').write_text(json.dumps({'ds': DS, 'esc': cl_A.ESC, 'rows': res}, ensure_ascii=False))


if __name__ == '__main__':
    main()
