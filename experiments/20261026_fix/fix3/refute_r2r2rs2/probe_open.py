"""反証(r2-r2-rs-2)の 2: 壁に当たっていない(モデルが一度ゆるめてゆっくりになっただけ)ところで中立止め。
明けた最初のフレームで stuck と見なして後退するか(slow_since が中立止めの間も残る)。前の壁は 20 m 先・後ろの壁なし。"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute2_safety'))
import closed_loop as CL          # noqa: E402
from closed_loop import C, N, PARAMS   # noqa: E402
from probe import summarize      # noqa: E402

CL.TMP = HERE.parent / 'tmp_r2r2rs2'


def run(keep, model, drop=None, dur=8.0):
    sc_ = {'dur': dur, 'xf': 20.0, 'model': model, 'drops': [drop] if drop else []}
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=keep)
    sim, rec, pca = CL.run(p, sc_)
    return sim, rec, pca, list(CL.PH)


def main():
    res = {}
    # まず中立を長く(1.6〜4.0 秒)出して、遅い帯(|eRPM| < 300)に入る時刻を探す
    sim0, *_ = run(0, lambda t, car: (C, N) if 1.6 <= t < 4.0 else (C, N - 120.0), dur=4.0)
    t_in = next(t for t, x, v, *_ in sim0.log if t > 1.6 and abs(v / CL.KV) < 300)
    t_go = t_in + 0.3                   # 遅い帯に 0.3 秒(< 0.8 秒)いたあと、モデルは前進に戻る
    print('slow band at', t_in, 'model forward again at', t_go)

    def model(t, car):
        return (C, N) if 1.6 <= t < t_go else (C, N - 120.0)
    for L in (0.0, 0.5, 1.0, 2.0, 5.0):
        for keep in (0, 1):
            t0 = t_in + 0.1
            drop = (t0, t0 + L) if L > 0 else None
            end = t0 + L
            sim, rec, pca, ph = run(keep, model, drop, dur=end + 4.0)
            r = summarize(sim, pca, ph, end)
            r['recoveries'] = sum(1 for t, p0, p1, hg, ig in ph if p0 is None and p1 == 'brake')
            r['min_v_after'] = round(min(v for t, x, v, *_ in sim.log if t >= end), 2)
            key = 'open slow+hold L=%.1f keep=%d' % (L, keep)
            res[key] = r
            print(key, {k: r[k] for k in ('first_rev_cmd_s', 'first_backing_s', 'back_m', 'max_thr_2s', 'recoveries',
                                          'min_v_after')}, r['phases_after'][:3])
    (HERE / 'out_open.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    shutil.rmtree(CL.TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
