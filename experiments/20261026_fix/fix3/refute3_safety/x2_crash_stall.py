"""X2: 前の壁に当たった直後にカメラが止まる(長い中立止め)。keep=1(見張り 1)と keep=0 で、明けてから後退側を書くまでの秒など"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl3                                    # noqa: E402
from cl3 import C, N, PARAMS                  # noqa: E402


def one(keep, guard, hit_drop=None, dur=9.0, model=(C, N - 120.0), xf=2.0, xr=None, sk=0):
    sc_ = {'dur': dur, 'xf': xf, 'model': model, 'drops': []}
    if xr is not None:
        sc_['xr'] = xr
    if hit_drop:
        sc_['hit_drop'] = hit_drop
    p = dict(PARAMS, recovery=1, start_kick=sk, rec_back_guard=guard, rec_hold_keep=keep)
    sim, rec, pca, ph = cl3.run(p, sc_)
    hits = sim.car.hits
    t_hit = hits[0][0] if hits else None
    end = None
    if hit_drop and t_hit is not None:
        end = t_hit + hit_drop[0] + hit_drop[1]
    ref = end if end is not None else t_hit
    w = [(round(t - ref, 3), th) for t, s, th in pca.w if ref is not None and t >= ref]
    first_rev = next((t for t, th in w if th > N + 40), None)
    push = 0.0
    for (t0, x, v, thr, con, ro, sl), (t1, *_r) in zip(sim.log, sim.log[1:]):
        if ref is not None and t0 >= ref and con == 'front' and thr < N - 20:
            push += t1 - t0
    giveup = sum(1 for _, r in rec.rows if not r['hold'] and int(r['assist']) == 3)
    ev = [(e[1]['what'], e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard']
    tr = [(round(t - (ref or 0), 3), p0, p1) for t, p0, p1 in cl3.phases(ph)][:8]
    return {'t_hit': t_hit, 'cam_back': end, 'first_rev_after_s': first_rev, 'front_push_after_s': round(push, 3),
            'giveup_rows': giveup, 'guard_events': ev, 'phases_from_ref': tr,
            'hits': hits[:4], 'max_back_m': round(xf - min(r[1] for r in sim.log if ref is None or r[0] >= ref), 3)}


def main():
    res = {}
    for keep in (0, 1):
        res['nodrop keep=%d' % keep] = one(keep, 1)
        for L in (0.3, 0.5, 0.8, 1.2, 2.0, 3.0):
            res['hitdrop 0.0+%.1f keep=%d' % (L, keep)] = one(keep, 1, hit_drop=(0.0, L))
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False, default=str))
    out = Path(__file__).resolve().parent / 'out'
    out.mkdir(exist_ok=True)
    (out / 'x2_crash_stall.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    shutil.rmtree(cl3.TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
