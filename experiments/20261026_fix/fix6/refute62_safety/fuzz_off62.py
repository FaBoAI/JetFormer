"""反証 6-2・安全: スイッチ OFF(rec_hold_keep・rec_back_guard のどちらかが 0)と、ON でも中立止めの無い場面で、今の
drive.py・assist.py(fix6 のあと)が直す前の写し(tests/fixtures/fix6_safety_before = fix5 のあと)とビット単位で同じか。
場面・スイッチは fix4/review_safety/fuzz_closed.py の rand_scene・rand_params(読み込むだけ。launch_boost・launch_no_kick・
start_kick・tof_guard・速さの天井・区間の上限・rev_s・rev_us は でたらめ)。'nohold' は ON のままカメラ・車両の値の止まりを抜く。

    python -I fuzz_off62.py [本数] [種]
"""
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402
from c62 import R, CL    # noqa: E402
import fuzz_closed as FC  # noqa: E402


def same_run(a, b):
    (sa, ra, pa, _), (sb, rb, pb, _) = a, b
    if pa.w != pb.w:
        i = next((i for i, (x, y) in enumerate(zip(pa.w, pb.w)) if x != y), min(len(pa.w), len(pb.w)))
        return False, ('pca', i, pa.w[i:i + 1], pb.w[i:i + 1])
    ok, why = R.same_rows(ra.rows, rb.rows)
    if not ok:
        return False, why
    ok, why = R.same_events(ra.events, rb.events)
    if not ok:
        return False, ('events', str(why)[:300])
    return True, None


def main(n=120, seed=6201):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix6_safety_before')
    out = {'seed': seed, 'runs': 0, 'same': collections.Counter(), 'diff': [], 'combos': collections.Counter()}
    orig = CL.Sim
    CL.Sim = FC.FSim
    try:
        for i in range(n):
            sc = FC.rand_scene(rng)
            p = FC.rand_params(rng)
            mode = rng.choice(['k1g0', 'k1g0', 'k0g1', 'k0g0', 'nohold'])
            if mode == 'nohold':
                sc = dict(sc, drops=[], veh_skip=[], hit_drop=None)
            else:
                keep, guard = {'k1g0': (1, 0), 'k0g1': (0, 1), 'k0g0': (0, 0)}[mode]
                p.update(rec_hold_keep=keep, rec_back_guard=guard)
            combo = '%s_lb%d_lnk%d_sk%d_tg%d' % (mode, p['launch_boost'], p['launch_no_kick'], p['start_kick'], p['tof_guard'])
            out['combos'][mode] += 1
            a = R.run_closed(p, sc)
            b = R.run_closed(p, sc, mod=BEF[0], amod=BEF[1])
            ok, why = same_run(a, b)
            nh = sum(1 for t, r in a[1].rows if r['hold'])
            out['runs'] += 1
            out['same'][('same' if ok else 'diff') + '_' + mode] += 1
            if not ok:
                out['diff'].append({'run': i, 'combo': combo, 'holds': nh, 'why': str(why)[:400]})
            if (i + 1) % 20 == 0:
                print(i + 1, dict(out['same']), flush=True)
    finally:
        CL.Sim = orig
        K.cleanup()
    out['same'] = dict(out['same'])
    out['combos'] = dict(out['combos'])
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('fuzz_off62_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(out, ensure_ascii=False)[:2000])


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
