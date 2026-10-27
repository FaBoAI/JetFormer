"""反証 7-1 回目・safety: 11/3 の回 1〜4 の形(rev_s 0.4・rev_us 300・start_kick 1・ToF ガード 1・lookahead 3・
launch_boost 0/1・天井 0/1・自動合わせ 0/1・左 4.1/4.2・ゲイン 0.85〜1.0)で、同じでたらめな場面(fix4/review_safety/fuzz_closed の
rand_scene を読み込むだけ)を
  on   = 今の版・rec_back_guard 1・rec_hold_keep 1(ユーザーの (c))
  off  = 今の版・両方 0(既定)
  onc  = on と同じ。ただしカメラ・車両の値の止まり(drops・veh_skip・hit_drop)を抜く(中立止めの扱いのせいかを見る)
  offb = 直す前の写し(f6b = fix5 のあと)・両方 0 → off とビット単位で同じか
  off4 / off4b = 4 つのスイッチ全部 0(launch_boost も 0)の今と直す前 → ビット単位で同じか
で回し、1 本ずつ数を残す(analyze_onoff.py で比べる)。装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix7/review_safety_r1/fuzz_onoff.py [本数] [種]
出力: out/fuzz_onoff_<種>.json
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402

sys.path.insert(0, str(rv7.FIXD / 'fix4' / 'review_safety'))
import fuzz_closed as FC  # noqa: E402


def rand_params(rng):
    ceil = rng.choice([0, 1, 1])
    p = rv7.base_params(launch_boost=rng.choice([0, 1]), speed_ceiling=ceil,
                        speed_auto=rng.choice([0, 1]) if ceil else 0, ceil_left_mps=rng.choice([4.1, 4.2]),
                        throttle_gain=rng.choice([0.85, 0.9, 0.95, 1.0]))
    return p


def clean(sc):
    s = dict(sc, drops=[], veh_skip=[])
    s.pop('hit_drop', None)
    return s


def summ(res):
    sim, rec, pca, ph = res
    m = rv7.metrics(sim, rec, pca)
    m['starts'] = [t for t, p0, p1, *_ in ph if p0 is None and p1 is not None]
    m['holds_ai'] = sum(1 for _, r in rec.rows if r['hold'] and r['select'])
    m['kept'] = [(t, p0, p1, hr, es, bs) for t, p0, p1, hs, hr, es, bs, e in ph if hs and p0 is not None]
    return m


def main(n=300, seed=7101):
    rv7.check_now()
    rng = random.Random(seed)
    out = {'seed': seed, 'runs': []}
    ident = {'off_same': 0, 'off_diff': [], 'off4_same': 0, 'off4_diff': [], 'off4_r2_same': 0, 'off4_r2_diff': []}
    try:
        for i in range(n):
            sc = FC.rand_scene(rng)
            p = rand_params(rng)
            r_on = rv7.run('now', rv7.ON(p), sc)
            r_off = rv7.run('now', rv7.OFF(p), sc)
            r_onc = rv7.run('now', rv7.ON(p), clean(sc))
            r_offb = rv7.run('f6b', rv7.OFF(p), sc)
            ok, why = rv7.same_run(r_off, r_offb)
            if ok:
                ident['off_same'] += 1
            else:
                ident['off_diff'].append((i, str(why)))
            p4 = dict(rv7.OFF(p), launch_boost=0, launch_no_kick=0)
            r4 = rv7.run('now', p4, sc)
            r4b = rv7.run('f6b', p4, sc)
            r4c = rv7.run('f6r2b', p4, sc)
            ok4, why4 = rv7.same_run(r4, r4b)
            ok5, why5 = rv7.same_run(r4, r4c)
            if ok4:
                ident['off4_same'] += 1
            else:
                ident['off4_diff'].append((i, str(why4)))
            if ok5:
                ident['off4_r2_same'] += 1
            else:
                ident['off4_r2_diff'].append((i, str(why5)))
            out['runs'].append({'i': i, 'params': {k: p[k] for k in ('launch_boost', 'speed_ceiling', 'speed_auto',
                                                                      'ceil_left_mps', 'throttle_gain')},
                                'scene': {k: v for k, v in sc.items() if k != 'model'},
                                'on': summ(r_on), 'off': summ(r_off), 'onc': summ(r_onc)})
            if (i + 1) % 25 == 0:
                print(i + 1, {k: (v if isinstance(v, int) else len(v)) for k, v in ident.items()}, flush=True)
    finally:
        rv7.cleanup()
    out['identity'] = ident
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('fuzz_onoff_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print(json.dumps(ident, ensure_ascii=False, default=str)[:2000])


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
