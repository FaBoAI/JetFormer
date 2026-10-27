"""反証 7-2 回目・safety: 指摘の入力 1 つ(直さない・装置に触れない・GPU なし・API なし)。
場面 rev(前の壁 2.0 m・モデルは前進 120 µs)・回 1 の形(h8.base_params: start_kick 1・lookahead 3・ToF ガード 1・rev_s 0.4・
rev_us 300)・元の偽物の ESC(中立 0.1 秒で下がれる)・車両の値 50 Hz。
入力: 最初のリカバリーの 'brake'(2.52 秒から)の 2.52〜2.62 秒にカメラのフレームが来ない(= 中立止め。hold が 2.625〜2.658 秒に
約 0.03 秒だけ中立を書く。偽物の ESC はまだ下がれない)+ 止まっている車の eRPM の跳ね 2 つ(2.68 秒・2.88 秒に 1322)。
比べ: on = 今・両方 1(11/3 の形)/ off = 今・両方 0 / f7b_on = fix7 の直す前・両方 1 / on_nohold = 今・両方 1・同じ跳ねで止まりなし /
on_one = 今・両方 1・同じ止まりで跳ね 1 つ目だけ
    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix7/review_safety_r2/repro8.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV8_TAG', 'repro8')
sys.path.insert(0, str(HERE))
import h8  # noqa: E402
from g8 import first_rec, scene  # noqa: E402

N = h8.N


def stats(res):
    sim, rec, pca, ph = res
    m = first_rec(res)
    tb = m['t_brake']
    te = m['t_escape']
    # 'brake' の始めから escape までに後退側(中立 + 40 より後ろ)を書いた秒(フレームの書き込みの間で数える)
    ws = [(t, th) for t, s, th in pca.w if tb <= t <= (te if te is not None else tb + 3.0)]
    rev_side = sum(t1 - t0 for (t0, th), (t1, _) in zip(ws, ws[1:]) if th > N + 40)
    return {'back_m': m['back'], 'escape_push_s': m['esc_push'], 'rev_side_written_s': round(rev_side, 2),
            't_escape': te, 'guard_events': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard'],
            'fake_esc_first_reverse_t': next((t for t, r, c in sim.car.revlog if r), None)}


def main():
    p = h8.base_params()
    base = scene('rev', dur=9.0)
    drop = [(2.52, 2.62)]
    gl = [(2.68, 1322), (2.88, 1322)]
    runs = {
        'on': ('now', h8.ON(p), dict(base, drops=drop, glitch=gl)),
        'off': ('now', h8.OFF(p), dict(base, drops=drop, glitch=gl)),
        'f7b_on': ('f7b', h8.ON(p), dict(base, drops=drop, glitch=gl)),
        'on_nohold': ('now', h8.ON(p), dict(base, glitch=gl)),
        'on_one': ('now', h8.ON(p), dict(base, drops=drop, glitch=gl[:1])),
    }
    out = {}
    try:
        for k, (name, prm, sc) in runs.items():
            out[k] = stats(h8.run(name, prm, sc))
            print(k, json.dumps(out[k], ensure_ascii=False), flush=True)
    finally:
        h8.cleanup()
    (HERE / 'out' / 'repro8.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
