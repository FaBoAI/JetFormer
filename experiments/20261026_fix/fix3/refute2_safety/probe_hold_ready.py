"""ESC が後退を待っている(モデルが止まる前にブレーキ側)ときのリカバリー('brake' がそのまま後退になる。10/23 の形)に、
'brake' の間の中立止めをはさむ。後ろの物 xr を置き、下がった距離・後ろの当たりを (keep, guard) で比べる"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as CL          # noqa: E402
from closed_loop import C, N, PARAMS   # noqa: E402

XF = 2.0


def model(t, car):
    return (C, N + 60.0) if XF - car.x < 0.5 else (C, N - 120.0)


def one(keep, guard, drop=None, back=None):
    sc_ = {'dur': 7.0, 'xf': XF, 'xr': None if back is None else XF - back, 'model': model, 'drops': [drop] if drop else []}
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=guard, rec_hold_keep=keep)
    sim, rec, pca = CL.run(p, sc_)
    ph = [(t, p0, p1) for t, p0, p1, hg, ig in CL.PH if p0 != p1]
    t_first_esc = next((t for t, p0, p1 in ph if p1 == 'escape'), 99)
    xs = [(t, x, v) for t, x, v, *_ in sim.log if 1.0 <= t <= t_first_esc + 0.5]
    x_stop = max(x for _, x, _ in xs)
    return {'back_m': round(x_stop - min(x for _, x, _ in xs), 3), 'v_back_max': round(min(v for _, _, v in xs), 2),
            'rear_hits': [h for h in sim.car.hits if h[1] == 'rear'], 'phases': ph[:8],
            'guard_events': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard'], 'x_stop': round(x_stop, 3)}


def main():
    res = {}
    for keep, guard in ((0, 0), (0, 1), (1, 1)):
        res['nohold keep=%d guard=%d' % (keep, guard)] = one(keep, guard)
    b = res['nohold keep=1 guard=1']['phases'][0][0]      # 'brake' の始まり
    print('brake at', b, res['nohold keep=1 guard=1']['phases'])
    for o in (0.05, 0.1, 0.15):
        for L in (0.3, 0.5, 0.8):
            for keep in (0, 1):
                res['brake-hold o=%.2f L=%.1f keep=%d' % (o, L, keep)] = one(keep, 1, drop=(b + o, b + o + L))
    for k, v in res.items():
        print(k, {kk: v[kk] for kk in ('back_m', 'v_back_max', 'rear_hits', 'guard_events')})
    out = Path(__file__).resolve().parent / 'out'
    (out / 'probe_hold_ready.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
