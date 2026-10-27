"""直す 7-2 回目・係 safety: 指摘の入力 1 つ(w2-safety-rv7-safety-r2-1)を、今・直す前(r1)・案 (a)・OFF などで流す。
場面 rev・回 1 の形・元の偽物の ESC・50 Hz。'brake' 2.52 秒から、2.52〜2.62 秒にカメラの止まり(hold が約 0.03 秒中立を書く)、
止まっている車の eRPM の跳ね 1322 が 2.68 秒・2.88 秒。装置に触れない・GPU なし・API なし。
    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix7/safety_r2/smoke9.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV9_TAG', 'smoke9')
sys.path.insert(0, str(HERE))
import h9  # noqa: E402
from g9 import first_rec, scene  # noqa: E402

N = h9.N


def stats(res):
    sim, rec, pca, ph = res
    m = first_rec(res)
    tb, te = m['t_brake'], m['t_escape']
    ws = [(t, th) for t, s, th in pca.w if tb <= t <= (te if te is not None else tb + 3.0)]
    rev_side = sum(t1 - t0 for (t0, th), (t1, _) in zip(ws, ws[1:]) if th > N + 40)
    return {'back_m': m['back'], 'escape_push_s': m['esc_push'], 'rev_side_written_s': round(rev_side, 2),
            't_escape': te, 'guard_events': [(e[1]['what'], e[1]['phase'], e[1]['backed_s']) for e in rec.events
                                             if e[0] == 'recovery_guard'],
            'fake_esc_first_reverse_t': next((t for t, r, c in sim.car.revlog if r), None),
            'front_push_s': m['front_push'], 'rear': m['rear']}


def main():
    p = h9.base_params()
    base = scene('rev', dur=9.0)
    drop = [(2.52, 2.62)]
    gl = [(2.68, 1322), (2.88, 1322)]
    sc = dict(base, drops=drop, glitch=gl)
    runs = {
        'now_on': ('now', h9.ON(p), sc, None),
        'r1_on': ('r1', h9.ON(p), sc, None),
        'f7b_on': ('f7b', h9.ON(p), sc, None),
        'a_on': ('r1', h9.ON(p), sc, {'READY': 0.08}),
        'ab_on': ('now', h9.ON(p), sc, {'READY': 0.08}),
        'now_off': ('now', h9.OFF(p), sc, None),
        'r1_off': ('r1', h9.OFF(p), sc, None),
        'now_guard_only': ('now', dict(h9.ON(p), rec_hold_keep=0), sc, None),
        'now_on_one': ('now', h9.ON(p), dict(base, drops=drop, glitch=gl[:1]), None),
        'now_on_nohold': ('now', h9.ON(p), dict(base, glitch=gl), None),
        'now_on_noglitch': ('now', h9.ON(p), dict(base, drops=drop), None),
        'now_on_esc002': ('now', h9.ON(p), dict(sc, esc=(0.02, 0.08)), None),
        'r1_on_esc002': ('r1', h9.ON(p), dict(sc, esc=(0.02, 0.08)), None),
        'a_on_esc002': ('r1', h9.ON(p), dict(sc, esc=(0.02, 0.08)), {'READY': 0.08}),
        'now_on_esc005': ('now', h9.ON(p), dict(sc, esc=(0.05, 0.08)), None),
        'r1_on_esc005': ('r1', h9.ON(p), dict(sc, esc=(0.05, 0.08)), None),
    }
    out = {'sha_now': h9.check_now()}
    try:
        for k, (name, prm, s, patch) in runs.items():
            res = h9.run(name, prm, s, patch=patch)
            out[k] = stats(res)
            print(k, json.dumps(out[k], ensure_ascii=False, default=str), flush=True)
            if k == 'now_on':
                out['now_on_trace'] = [x for x in res[3] if 2.45 <= x[0] <= 4.0]
    finally:
        h9.cleanup()
    (HERE / 'out' / 'smoke9.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
