"""反証 6-1・安全: スイッチ OFF(rec_hold_keep・rec_back_guard のどちらかが 0)で、今の drive.py・assist.py が直す前の写し
(tests/fixtures/fix5_safety_before)とビット単位で同じか。fix5 の fuzz_fix5 は keep 0・見張り 1 と 4 つ全部 0 を見たので、
ここでは keep 1・見張り 0 を主に、keep 0・見張り 0 と keep 0・見張り 1 も混ぜる(launch_boost・launch_no_kick・start_kick・
tof_guard・速さの天井などは でたらめ)。場面は fix4/review_safety/fuzz_closed.py の rand_scene(読み込むだけ)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix6/refute_safety/fuzz_off.py [本数] [種]
出力: out/fuzz_off_<種>.json
"""
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6_common as K          # noqa: E402
from r6_common import R, CL    # noqa: E402
import fuzz_closed as FC       # noqa: E402


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


def main(n=120, seed=6101):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix5_safety_before')
    out = {'seed': seed, 'runs': 0, 'same': collections.Counter(), 'diff': [], 'combos': collections.Counter()}
    orig = CL.Sim
    CL.Sim = FC.FSim
    try:
        for i in range(n):
            sc = FC.rand_scene(rng)
            p = FC.rand_params(rng)
            keep, guard = rng.choice([(1, 0), (1, 0), (0, 1), (0, 0)])
            p.update(rec_hold_keep=keep, rec_back_guard=guard)
            combo = 'keep%d_guard%d_lb%d_lnk%d_sk%d' % (keep, guard, p['launch_boost'], p['launch_no_kick'],
                                                        p['start_kick'])
            out['combos']['keep%d_guard%d' % (keep, guard)] += 1
            a = R.run_closed(p, sc)
            b = R.run_closed(p, sc, mod=BEF[0], amod=BEF[1])
            ok, why = same_run(a, b)
            out['runs'] += 1
            out['same']['same' if ok else 'diff'] += 1
            if not ok:
                out['diff'].append({'run': i, 'combo': combo, 'why': str(why)[:400]})
            if (i + 1) % 20 == 0:
                print(i + 1, dict(out['same']), flush=True)
    finally:
        CL.Sim = orig
        R.cleanup()
    out['same'] = dict(out['same'])
    out['combos'] = dict(out['combos'])
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('fuzz_off_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(out, ensure_ascii=False)[:2000])


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
