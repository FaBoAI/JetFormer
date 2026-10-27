"""r1-state3-1: ±0.01 のでたらめな押し方 40 通りで、今の app.js と直す前の写しが「0.90 + 押した数 × 0.01」にならない数を数える
(tests/test_ui_fix3_r1.py::test_state3_1_random_interleavings と同じ場面。出力は out/count_taps.json)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix3/fix_r1_1/count_taps.py
"""
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
import test_ui_fix3_r1 as U          # noqa: E402
from jetformer.web import bench      # noqa: E402

TMP = HERE.parent / 'tmp_fix_r1_1_taps'


def main():
    c = bench.load_candidates()
    di = [i for i, d in enumerate(c['days']) if d['date'] == '2026-11-03'][0]
    c = dict(c, today='2026-11-03')
    TMP.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261026)
    res = {'n': 0, 'bad_new': 0, 'bad_old': 0, 'old_examples': []}
    for i in range(40):
        steps, want = U._random_taps(rng)
        aor = rng.random() < 0.5
        for old in (False, True):
            st, _ = U.run_sim(TMP, c, steps, server=U.server_params(throttle_gain=0.9), old=old, name='r%d_%d.json' % (i, old),
                              defer_posts=True, apply_on_resolve=aor)
            s = U.starts(st)[0]
            got = (U.reads(st, 'gain')[0], U.reads(st, 'server')[0]['throttle_gain'], s['sent']['throttle_gain'])
            ok = got == (want, want, want)
            if old and not ok:
                res['bad_old'] += 1
                if len(res['old_examples']) < 5:
                    res['old_examples'].append({'i': i, 'want': want, 'got': got, 'apply_on_resolve': aor})
            if not old and not ok:
                res['bad_new'] += 1
        res['n'] += 1
    (HERE / 'out' / 'count_taps.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(res)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
