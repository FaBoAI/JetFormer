"""X10: 'reverse'・'coast' の途中(後ろへ速く動いている)に長い中立止め → 明けて「中止」(中立)のまま惰性で下がるか。
keep=1(見張り 1)と keep=0・中立止めなしで、明けたときの後ろ向きの速さ・明けてから下がった距離・後ろの物への当たり"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl3                                    # noqa: E402
from cl3 import A, C, N, PARAMS, T0           # noqa: E402

XF = 2.0


def one(keep, drop, xr_back=None, rev_us=300.0, rev_s=0.4):
    sc_ = {'dur': 8.0, 'xf': XF, 'model': (C, N - 120.0), 'drops': [drop] if drop else []}
    if xr_back is not None:
        sc_['xr'] = XF - xr_back
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=keep, rev_us=rev_us, rev_s=rev_s)
    sim, rec, pca, ph = cl3.run(p, sc_)
    end = drop[1] if drop else None
    out = {'hits': [h for h in sim.car.hits if h[1] == 'rear'],
           'ev': [(e[1]['what'], e[1].get('phase'), e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard'],
           'phases': cl3.phases(ph)[:8]}
    if end is not None:
        at = next(r for r in sim.log if r[0] >= end)
        x_end = at[1]
        # 明けてから 1.5 秒の間の最も後ろ
        xs = [r[1] for r in sim.log if end <= r[0] <= end + 1.5]
        out.update(v_at_camera_back=round(at[2], 2), x_at_camera_back=round(x_end, 3),
                   backed_after_camera_m=round(x_end - min(xs), 3),
                   first_write_after=[(round(t - end, 3), th) for t, s, th in pca.w if t >= end][:2])
    out['min_x'] = round(min(r[1] for r in sim.log if r[0] > 1.6), 3)
    return out


def main():
    res = {}
    base = one(1, None)
    res['nohold'] = base
    tr = next(t for t, p0, p1 in base['phases'] if p1 == 'reverse')
    for o in (0.2, 0.3, 0.4, 0.5):
        for L in (0.55, 0.7, 1.0):
            for keep in (0, 1):
                res['rev+%.2f L=%.2f keep=%d' % (o, L, keep)] = one(keep, (tr + o, tr + o + L))
    for k, v in res.items():
        print(k, {kk: v.get(kk) for kk in ('v_at_camera_back', 'backed_after_camera_m', 'min_x', 'hits', 'ev', 'first_write_after')})
    (Path(__file__).resolve().parent / 'out' / 'x10_giveup_coast.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    shutil.rmtree(cl3.TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
