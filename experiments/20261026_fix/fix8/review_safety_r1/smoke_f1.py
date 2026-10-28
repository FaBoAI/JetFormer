"""指摘 1 の入力そのもの: 10/23 の形(前進 120 µs で走り、壁 2.4 m の 0.45 m 手前からブレーキ側 中立 + 200 → 止まって ESC が
後退を待つ → リカバリー)・回 1 の形・人に寄せた偽物の ESC(中立 0.12 秒・下がり始め 0.10 秒)・車両の値 100 Hz。
後ろの物を壁から 0.7 m・1.4 m に置き、今の ON(両方 1)・見張りだけ(keep 0)・OFF・fix7 の前(f7b)の ON を並べる。
    python -I -B smoke_f1.py → 標準出力・out/smoke_f1.json
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS8_TAG', 'smoke_f1')
sys.path.insert(0, str(HERE))
import h11  # noqa: E402
import exp11  # noqa: E402

N = h11.N


def main():
    h11.check_now()
    out = {}
    try:
        for esc in ('human', 'orig'):
            for rear in (0.7, 1.4, None):
                sd = {'kind': 'rd', 'form': 'kai1', 'xf': 2.4, 'model': {'kind': 'brakewall', 'm': 120.0, 'db': 0.45},
                      'esc': esc, 'rear': rear, 'veh_dt': 0.01, 'dur': 5.5}
                p = exp11.params_of(sd)
                for name, ver, prm in (('now_on', 'now', h11.ON(p)), ('now_guard_only', 'now', dict(h11.ON(p), rec_hold_keep=0)),
                                       ('now_off', 'now', h11.OFF(p)), ('f7b_on', 'f7b', h11.ON(p))):
                    res = h11.run(ver, prm, exp11.build(sd))
                    m = h11.metrics(res, 1.0)
                    sg = (m['segs'] or [{}])[0]
                    w = [(round(t, 3), th) for t, s, th in res[2].w if sg.get('t0', 0) - 0.01 <= t <= (sg.get('t1') or 9)]
                    seq, prev = [], None
                    for t, th in w:
                        if th != prev:
                            seq.append((t, th))
                            prev = th
                    row = {'rear_hits': m['rear'], 'back_m': sg.get('back'), 'peak_back_mps': sg.get('vmin'),
                           'guards': m['guards'][:3], 'writes': seq[:8], 'x0': sg.get('x0')}
                    out['%s_rear%s_%s' % (esc, rear, name)] = row
                    print(esc, rear, name, json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        h11.cleanup()
    (HERE / 'out' / 'smoke_f1.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
