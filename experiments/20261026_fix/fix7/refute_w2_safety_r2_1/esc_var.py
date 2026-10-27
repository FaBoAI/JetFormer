"""反証(w2-safety-rv7-safety-r2-1): 指摘の入力を、偽物の ESC が下がれるまでの中立の秒を変えて流す(直さない・装置に触れない)。
本物の ESC は人の操作で中立 0.05 秒で後退に入れた(10/10 の #5・#9)。0.033 秒の中立止めで下がれるなら、指摘の形にならないか。"""
import json, os, sys
from pathlib import Path
os.environ['RV8_TAG'] = 'refute_w2s1'
RV = Path('/home/jetson/jetformer/experiments/20261026_fix/fix7/review_safety_r2')
sys.path.insert(0, str(RV))
import h8  # noqa
from g8 import first_rec, scene  # noqa
from repro8 import stats  # noqa
N = h8.N
OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(exist_ok=True)


def neutral_runs(pca, t0, t1):
    w = [(t, th) for t, s, th in pca.w if t0 <= t <= t1]
    runs = []
    for (ta, tha), (tb, _) in zip(w, w[1:]):
        if abs(tha - N) <= 20:
            if runs and abs(runs[-1][1] - ta) < 1e-9:
                runs[-1][1] = tb
            else:
                runs.append([ta, tb])
    return [(round(a, 3), round(b - a, 3)) for a, b in runs]


def main():
    p = h8.base_params()
    drop = [(2.52, 2.62)]
    gl = [(2.68, 1322), (2.88, 1322)]
    out = {}
    try:
        for esc in (None, (0.1, 0.08), (0.05, 0.08), (0.04, 0.08), (0.03, 0.08), (0.02, 0.08)):
            for sw, prm in (('on', h8.ON(p)), ('off', h8.OFF(p))):
                sc = dict(scene('rev', dur=9.0, esc=esc), drops=drop, glitch=gl)
                res = h8.run('now', prm, sc)
                st = stats(res)
                st['neutral_runs_2.5_3.0'] = neutral_runs(res[2], 2.5, 3.0)
                k = '%s_%s' % ('orig' if esc is None else '%g' % esc[0], sw)
                out[k] = st
                print(k, json.dumps(st, ensure_ascii=False, default=str), flush=True)
    finally:
        h8.cleanup()
    (OUT / 'esc_var.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
