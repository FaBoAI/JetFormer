"""h12 の確かめ: 場面 rev(前の壁 2.0 m・前進 120 µs)を ESC の模型 3 つで ON / OFF / 写し f7b の OFF に流し、時間・段の移り・
記録の照らしを出す。
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261026_fix/fix8/review_safety_r2/smoke12.py
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS82_TAG', 'smoke')
sys.path.insert(0, str(HERE))
import h12  # noqa: E402

N, C = h12.N, h12.C


def main():
    h12.h11.check_now()
    p = h12.base_params()
    try:
        for esc in ('fit', 'cl', 'human'):
            sc = {'dur': 9.0, 'xf': 2.0, 'model': (C, N - 120.0), 'esc18': esc, 'veh_dt': 0.01}
            out = {}
            for name, ver, prm in (('on', 'now', h12.ON(p)), ('off', 'now', h12.OFF(p)), ('off_f7b', 'f7b', h12.OFF(p))):
                t0 = time.time()
                r = h12.run(prm, dict(sc), ver=ver)
                m = h12.h11.metrics(r, 1.0)
                out[name] = r
                trans = [(x[0], x[1], x[2]) for x in r[3] if x[1] != x[2]]
                print(esc, name, round(time.time() - t0, 2), 's', json.dumps({k: m[k] for k in (
                    'front_push_s', 'rear_n', 'recoveries', 'giveup_t', 'zero_push', 'guards', 'min_x')}, ensure_ascii=False),
                    'rows_bad', h12.check_rows(r)[:3], flush=True)
                print('   trans', trans[:14])
            print('  off==off_f7b', h12.h11.same_run(out['off'], out['off_f7b']), 'on==off', h12.h11.same_run(out['on'], out['off'])[0])
    finally:
        h12.cleanup()


if __name__ == '__main__':
    main()
