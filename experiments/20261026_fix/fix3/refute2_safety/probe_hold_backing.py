"""残したリカバリーの後退の途中に中立止め(カメラが来ない)をはさむと、下がる距離はどう変わるか(閉じたループ・後ろの壁なし)。
場面: 人側 1 秒 → AI 側。モデルは弱い前進(中立 − 120)。前の壁 2.0 m に当たって止まる → リカバリー。
'reverse' に入った時刻 + o 秒から L 秒カメラが来ない。下がった距離(最も後ろ)・後ろ向きの最高の速さを比べる"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as CL          # noqa: E402
from closed_loop import C, N, PARAMS   # noqa: E402


def first_reverse(ph):
    return next((t for t, p0, p1, hg, ig in ph if p1 == 'reverse'), None)


def run_case(keep, guard, drop=None, rev_s=0.4, rev_us=300.0, extra=None):
    sc_ = {'dur': 9.0, 'xf': 2.0, 'model': (C, N - 120.0), 'drops': [drop] if drop else []}
    if extra:
        sc_.update(extra)
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=guard, rec_hold_keep=keep, rev_s=rev_s, rev_us=rev_us)
    sim, rec, pca = CL.run(p, sc_)
    ph = list(CL.PH)
    # 最初のリカバリーの区間: 'brake' の始まり → 次に None/giveup
    t0 = next((t for t, p0, p1, hg, ig in ph if p0 is None and p1 is not None), None)
    t1 = next((t for t, p0, p1, hg, ig in ph if t0 is not None and t > t0 and p1 is None), None)
    xs = [(t, x, v) for t, x, v, *_ in sim.log if t0 is not None and t0 <= t <= (t1 or 99) + 0.6]
    x0 = xs[0][1] if xs else None
    back = round(x0 - min(x for _, x, _ in xs), 3) if xs else None
    vmin = round(min(v for _, _, v in xs), 2) if xs else None
    return {'rec_start': t0, 'rec_end': t1, 'back_m': back, 'v_back_max': vmin,
            'phases': [(t, p0, p1) for t, p0, p1, hg, ig in ph if p0 != p1][:14],
            'reverse_at': first_reverse(ph), 'guard_events': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']}


def main():
    res = {}
    base = run_case(1, 1)
    tr = base['reverse_at']
    res['nohold keep1 guard1'] = base
    res['nohold keep0 guard1'] = run_case(0, 1)
    for o in (0.1, 0.2, 0.3):
        for L in (0.3, 0.6, 1.0):
            for keep in (0, 1):
                r = run_case(keep, 1, drop=(tr + o, tr + o + L))
                res['drop o=%.1f L=%.1f keep=%d guard=1' % (o, L, keep)] = r
    for k, v in res.items():
        print(k, {kk: v[kk] for kk in ('rec_start', 'rec_end', 'back_m', 'v_back_max', 'guard_events')})
    out = Path(__file__).resolve().parent / 'out'
    out.mkdir(exist_ok=True)
    (out / 'probe_hold_backing.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
