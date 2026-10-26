#!/usr/bin/env python3
"""反証 V-replay: 段 drive が挙げた直し方の案ごとに、No.9 の失う時間と、出っ張りへの効き(9 件・2 回目の SW の通過)を同じ物差しで(2026-10-25)。
仮想の閉ループ(むだ時間 0.05 秒・記録の跳ねを入力に足す)。実装の値は変えていない(引数で試しただけ)。読むだけ。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_cache  # noqa: E402
import rr_ceiling as RC  # noqa: E402
import rr_common as C  # noqa: E402

ALTS = [('実装(4.5 / 左 4.0)', 4.0, {}),
        ('掛け始め 0・先読みなし', 4.0, dict(engage_e=0.0, pred_s=0.0)),
        ('左 4.2', 4.2, {}),
        ('左 4.2 + 掛け始め 0・先読みなし', 4.2, dict(engage_e=0.0, pred_s=0.0)),
        ('左 4.1', 4.1, {})]


def main():
    D = rr_cache.load()['runs']
    B = {log: RC.run_b(d) for log, d in D.items()}
    OL = {}
    for log, d in D.items():
        OL[(log, 'impl')] = C.open_loop(d, C.P('impl'))
    evs = [e for e in RC.event_list(D) if e['kind'] == 'SWX']
    out = []
    for name, cl, kw in ALTS:
        p = C.P('impl', 4.5, cl, **kw)
        rows = RC.lap_losses(D[RC.NO9], p, B[RC.NO9], dead=0.05)
        eff = [RC.event_effect(D, OL, e, B, p, 0.05) for e in evs]
        eff = [x for x in eff if x]
        sw = [x for x in RC.sw_pass_expectation(D, B, p, 0.05) if x['session'] == '10/24 2回目']
        r = {'name': name, 'no9_loss': float(np.mean([x['loss'] for x in rows])), 'no9_changed': float(np.mean([x['changed_frac'] for x in rows])),
             'swx_eff': sum(x['effective_real'] for x in eff), 'swx_n': len(eff),
             'swx_vmax': [round(x['vmax_v'], 2) for x in eff],
             'sw2_expected': float(sum(RC.sw_rate(x['vmax_v']) for x in sw)), 'sw2_expected_rec': float(sum(RC.sw_rate(x['vmax_rec']) for x in sw)),
             'sw2_n': len(sw)}
        out.append(r)
        print(r)
    C.jdump(out, os.path.join(C.HERE, 'rr_alt.json'))


if __name__ == '__main__':
    main()
