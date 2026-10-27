"""残したリカバリーの 'reverse' の始め(下がり始めを見る前)に中立止め → 明けたとき止まっていると「下がらない」と見て
やり直し(+50 µs・下がった時間 0 から)。後ろの物を 1.45 m(中立止めなしのリカバリーでは届かない)に置いて比べる。
カメラが来ない場合と、車両の値が来ない場合"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as CL          # noqa: E402
from closed_loop import C, N, PARAMS   # noqa: E402

XF = 2.0


def one(keep, guard, drop=None, vskip=None, xr=XF - 1.45):
    sc_ = {'dur': 7.0, 'xf': XF, 'xr': xr, 'model': (C, N - 120.0), 'drops': [drop] if drop else [],
           'veh_skip': [vskip] if vskip else []}
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=guard, rec_hold_keep=keep)
    sim, rec, pca = CL.run(p, sc_)
    ph = [(t, p0, p1) for t, p0, p1, hg, ig in CL.PH if p0 != p1]
    t_first_esc = next((t for t, p0, p1 in ph if p1 == 'escape'), 99)
    xs = [(t, x, v) for t, x, v, *_ in sim.log if 2.0 <= t <= t_first_esc + 0.5]
    rear = [h for h in sim.car.hits if h[1] == 'rear']
    # 最初のリカバリーで後退側を書いた µs の最大
    revmax = max((th for t, s, th in pca.w if 2.0 <= t <= t_first_esc), default=None)
    return {'back_m': round(XF - min(x for _, x, _ in xs), 3), 'v_back_max': round(min(v for _, _, v in xs), 2),
            'rear_hits': rear, 'rev_us_max_written': revmax, 'phases': ph[:8],
            'guard_events': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']}


def main():
    res = {}
    for keep in (0, 1):
        res['nohold keep=%d' % keep] = one(keep, 1)
    tr = 2.52       # 'reverse' に入る時刻(probe_hold_backing.py の中立止めなし)
    for o in (0.03, 0.06, 0.1, 0.14):
        for L in (0.3, 0.45, 0.6, 1.0):
            for keep in (0, 1):
                res['cam o=%.2f L=%.2f keep=%d' % (o, L, keep)] = one(keep, 1, drop=(tr + o, tr + o + L))
    for o in (0.06, 0.1):
        for L in (0.3, 0.5):
            for keep in (0, 1):
                res['veh o=%.2f L=%.2f keep=%d' % (o, L, keep)] = one(keep, 1, vskip=(tr + o, tr + o + L))
    for k, v in res.items():
        print(k, {kk: v[kk] for kk in ('back_m', 'v_back_max', 'rear_hits', 'rev_us_max_written', 'guard_events')})
    out = Path(__file__).resolve().parent / 'out'
    out.mkdir(exist_ok=True)
    (out / 'probe_hold_retry.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
