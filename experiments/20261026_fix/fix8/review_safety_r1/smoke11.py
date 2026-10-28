"""h11 の確かめ: 場面 rev(前の壁 2.0 m・モデル前進 120 µs)を ON / OFF / 写し f7b の OFF で流し、時間と数を出す。
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261026_fix/fix8/review_safety_r1/smoke11.py
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS8_TAG', 'smoke')
sys.path.insert(0, str(HERE))
import h11  # noqa: E402

N, C = h11.N, h11.C


def main():
    h11.check_now()
    p = h11.base_params()
    sc = {'dur': 9.0, 'xf': 2.0, 'model': (C, N - 120.0)}
    try:
        out = {}
        for name, ver, prm in (('on', 'now', h11.ON(p)), ('off', 'now', h11.OFF(p)), ('off_f7b', 'f7b', h11.OFF(p)),
                               ('on_r1', 'r1', h11.ON(p))):
            t0 = time.time()
            r = h11.run(ver, prm, sc)
            m = h11.metrics(r, 1.0)
            out[name] = (r, m)
            print(name, round(time.time() - t0, 2), 's', json.dumps({k: m[k] for k in (
                'front_push_s', 'rear_n', 'recoveries', 'giveup_t', 'zero_push', 'guards')}, ensure_ascii=False),
                [(s['t0'], s.get('back'), s.get('esc_push')) for s in m['segs']], flush=True)
        print('off==off_f7b', h11.same_run(out['off'][0], out['off_f7b'][0]))
        print('on==on_r1', h11.same_run(out['on'][0], out['on_r1'][0]))
    finally:
        h11.cleanup()


if __name__ == '__main__':
    main()
