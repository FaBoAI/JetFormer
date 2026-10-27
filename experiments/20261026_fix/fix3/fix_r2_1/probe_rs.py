"""直す 2 回目・1 組目(r2-r2-rs-1・rs-2)の確かめ: 反証の閉じたループ(refute2_safety/closed_loop.py。本物の drive._loop と
偽物の ESC・車)で、反証の場面を今のコードで流し直す。装置・GPU・API には触れない。一時の場所は fix3/tmp_fixr21(終わって消す)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix3/fix_r2_1/probe_rs.py [a|b|long|open|all]
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX3 = HERE.parent
for d in ('refute2_safety', 'refute_r2r2rs2'):
    sys.path.insert(0, str(FIX3 / d))
import closed_loop as CL          # noqa: E402
CL.TMP = FIX3 / 'tmp_fixr21'
import probe_hold_retry as PR     # noqa: E402
import probe_hold_ready as PY     # noqa: E402
import probe as PL                # noqa: E402  (refute_r2r2rs2/probe.py)
from closed_loop import C, N      # noqa: E402

KA = ('back_m', 'v_back_max', 'rear_hits', 'rev_us_max_written', 'guard_events')
KB = ('back_m', 'v_back_max', 'rear_hits', 'guard_events')


def scen_a():
    res = {'nohold keep=%d' % k: PR.one(k, 1) for k in (0, 1)}
    res['nohold keep=1 no-rear'] = PR.one(1, 1, xr=None)
    tr = 2.52
    for o in (0.03, 0.06, 0.1, 0.14):
        for L in (0.3, 0.45, 0.6, 1.0):
            for keep in (0, 1):
                res['cam o=%.2f L=%.2f keep=%d' % (o, L, keep)] = PR.one(keep, 1, drop=(tr + o, tr + o + L))
            res['cam o=%.2f L=%.2f keep=1 no-rear' % (o, L)] = PR.one(1, 1, drop=(tr + o, tr + o + L), xr=None)
    for o in (0.06, 0.1):
        for L in (0.3, 0.5):
            for keep in (0, 1):
                res['veh o=%.2f L=%.2f keep=%d' % (o, L, keep)] = PR.one(keep, 1, vskip=(tr + o, tr + o + L))
            res['veh o=%.2f L=%.2f keep=1 no-rear' % (o, L)] = PR.one(1, 1, vskip=(tr + o, tr + o + L), xr=None)
    for k, v in res.items():
        print('(a)', k, {kk: v[kk] for kk in KA})
    return res


def scen_b():
    res = {}
    for keep, guard in ((0, 0), (0, 1), (1, 1)):
        res['nohold keep=%d guard=%d' % (keep, guard)] = PY.one(keep, guard)
    b = res['nohold keep=1 guard=1']['phases'][0][0]
    for o in (0.05, 0.1, 0.15):
        for L in (0.3, 0.5, 0.8):
            for keep in (0, 1):
                res['brake-hold o=%.2f L=%.1f keep=%d' % (o, L, keep)] = PY.one(keep, 1, drop=(b + o, b + o + L))
    for back in (1.2, 1.3, 1.45):
        res['nohold guard=1 rear=%.2f' % back] = PY.one(1, 1, back=back)
        for o, L in ((0.1, 0.3), (0.1, 0.8), (0.15, 0.5)):
            res['brake-hold o=%.2f L=%.1f keep=1 rear=%.2f' % (o, L, back)] = PY.one(1, 1, drop=(b + o, b + o + L), back=back)
    for k, v in res.items():
        print('(b)', k, {kk: v[kk] for kk in KB})
    return res


def scen_long():
    res = {}
    sim, rec, pca, ph = PL.run(1)
    times = {}
    for t, p0, p1, hg, ig in ph:
        if p0 != p1 and p1 not in times:
            times[p1] = t
    t_slow = next(t for t, x, v, *_ in sim.log if t > 1.2 and abs(v) < 0.01 and x > 1.9)
    starts = {'stuck(no phase)+0.05': t_slow + 0.05, 'brake+0.05': times['brake'] + 0.05,
              'gap+0.05': times['gap'] + 0.05, 'reverse+0.05': times['reverse'] + 0.05}
    for name, t0 in starts.items():
        for L in (0.2, 0.4, 0.5, 1.0, 2.0, 5.0):
            for keep in (0, 1):
                end = t0 + L
                sim, rec, pca, ph = PL.run(keep, (t0, end), dur=end + 3.0)
                r = PL.summarize(sim, pca, ph, end)
                r['guard_events'] = [(e[1]['what'], e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard']
                r['giveup_rows'] = sum(1 for _, row in rec.rows if not row['hold'] and int(row['assist']) == 3)
                res['%s L=%.1f keep=%d' % (name, L, keep)] = r
                print('(long)', name, 'L=%.1f keep=%d' % (L, keep),
                      {k: r[k] for k in ('first_rev_cmd_s', 'back_m', 'max_thr_2s', 'guard_events', 'giveup_rows')})
    return res


def scen_open():
    import probe_open as PO      # noqa: E402
    res = {}
    sim0, *_ = PO.run(0, lambda t, car: (C, N) if 1.6 <= t < 4.0 else (C, N - 120.0), dur=4.0)
    t_in = next(t for t, x, v, *_ in sim0.log if t > 1.6 and abs(v / CL.KV) < 300)
    t_go = t_in + 0.3

    def model(t, car):
        return (C, N) if 1.6 <= t < t_go else (C, N - 120.0)
    for L in (0.0, 0.3, 0.5, 1.0, 2.0, 5.0):
        for keep in (0, 1):
            t0 = t_in + 0.1
            drop = (t0, t0 + L) if L > 0 else None
            end = t0 + L
            sim, rec, pca, ph = PO.run(keep, model, drop, dur=end + 4.0)
            r = PL.summarize(sim, pca, ph, end)
            r['recoveries'] = sum(1 for t, p0, p1, hg, ig in ph if p0 is None and p1 == 'brake')
            r['giveup_rows'] = sum(1 for _, row in rec.rows if not row['hold'] and int(row['assist']) == 3)
            r['guard_events'] = [(e[1]['what'], e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard']
            res['open slow+hold L=%.1f keep=%d' % (L, keep)] = r
            print('(open) L=%.1f keep=%d' % (L, keep), {k: r[k] for k in ('first_rev_cmd_s', 'back_m', 'recoveries',
                                                                          'giveup_rows', 'guard_events')})
    return res


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out = {}
    try:
        for name, fn in (('a', scen_a), ('b', scen_b), ('long', scen_long), ('open', scen_open)):
            if which in (name, 'all'):
                out[name] = fn()
    finally:
        shutil.rmtree(CL.TMP, ignore_errors=True)
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('probe_rs_%s.json' % which)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                           encoding='utf-8')


if __name__ == '__main__':
    main()
