"""残したリカバリーの途中で長くカメラが来ない(2〜5 秒。Argus の止まりなど)。明けたあと、すぐ後退を出すか。
段: 'brake' の始め・'gap'・'reverse' の始め。後ろの壁なし。(keep, guard) = (0, 1)・(1, 1)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as CL          # noqa: E402
from closed_loop import C, N, PARAMS   # noqa: E402


def one(keep, drop, dur):
    sc_ = {'dur': dur, 'xf': 2.0, 'model': (C, N - 120.0), 'drops': [drop]}
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=keep)
    sim, rec, pca = CL.run(p, sc_)
    end = drop[1]
    after = [(round(t - end, 3), th) for t, s, th in pca.w if t >= end]
    first_rev = next((t for t, th in after if th > N + 40), None)
    rev_max = max((th for t, th in after if t < 1.5), default=None)
    xs = [(t, x) for t, x, *_ in sim.log if t >= end]
    back_after = round(max(x for _, x in xs) - min(x for _, x in xs[:200]), 3)
    return {'first_reverse_after_hold_s': first_rev, 'max_thr_us_1.5s_after': rev_max, 'back_m_2s_after': back_after,
            'phases': [(t, p0, p1) for t, p0, p1, hg, ig in CL.PH if p0 != p1 and t >= end - 0.01][:5]}


def main():
    res = {}
    # 中立止めなしの段の時刻(probe_hold_backing.py): brake 2.09・gap 2.35・reverse 2.52
    for name, t0 in (('brake+0.05', 2.14), ('gap+0.05', 2.40), ('reverse+0.05', 2.57)):
        for L in (2.0, 5.0):
            for keep in (0, 1):
                res['%s L=%.0f keep=%d' % (name, L, keep)] = one(keep, (t0, t0 + L), t0 + L + 3.0)
    for k, v in res.items():
        print(k, v)
    out = Path(__file__).resolve().parent / 'out'
    (out / 'probe_long_hold.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()


def stuck_before_start():
    """リカバリーが始まる前(壁に当たって止まった直後・段なし)から長く来ない場合"""
    res = {}
    for t0 in (1.6, 1.9):
        for L in (2.0, 5.0):
            for keep in (0, 1):
                res['stuck(no phase)+%.1f L=%.0f keep=%d' % (t0, L, keep)] = one(keep, (t0, t0 + L), t0 + L + 3.0)
    for k, v in res.items():
        print(k, v)
    return res
