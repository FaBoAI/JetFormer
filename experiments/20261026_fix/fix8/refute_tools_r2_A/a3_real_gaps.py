"""票 A・指摘 2: 本物の 10/10・10/23・10/24(読むだけ)の R1 に届いた発進の、発進 → R1 の起点の AI 側の処理したフレーム
(中立止めの行でない)の間を、今の ⑤ の launch_stall と同じ並びで数える(launch_stall を包んで間の並びを写すだけ。出力は消す場所へ)。
drive の中立の見積もり Σ max(0, 間 − 0.1)・0.1 秒より長い間の数・いちばん長い間。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/refute_tools_r2_A/a3_real_gaps.py
  → out/a3_real_gaps.txt
"""
import contextlib
import io

import numpy as np

import common_a as C


def main():
    C.check_sha()
    m = C.mod('now')
    orig = m.launch_stall
    rec = []

    def wrap(z, i0, i1, t0, t_end, over=None):
        if over is None:
            t = np.asarray(z['t_s'], np.float64)[i0:i1 + 1]
            h = np.asarray(z['hold'], bool)[i0:i1 + 1] if 'hold' in z else np.zeros(len(t), bool)
            tp = t[~h]
            tp = tp[(tp > t0) & (tp < t_end)]
            dd = np.diff(np.concatenate([[t0], tp, [t_end]]))
            rec.append({'t0': round(float(t0), 2), 'to_r1': round(float(t_end - t0), 3), 'max': round(float(dd.max()), 3),
                        'neutral': round(float(np.clip(dd - 0.1, 0, None).sum()), 3), 'n_over_0p1': int((dd > 0.1 + 1e-6).sum()),
                        'n_over_0p15': int((dd > 0.15).sum()), 'holds': int(h.sum())})
        return orig(z, i0, i1, t0, t_end, over)
    m.launch_stall = wrap
    C.TMP.mkdir(parents=True, exist_ok=True)
    L = ['# 本物の R1 に届いた発進の、発進 → R1 のフレームの間(今の ⑤ の launch_stall と同じ並び)']
    try:
        for day in ('20261010', '20261023', '20261024'):
            rec.clear()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m.main(['--day', day, '--out', str(C.TMP / ('real_%s.json' % day)), '--ignore-gain'])
            if not rec:
                L.append('%s: R1 に届いた発進 0' % day)
                continue
            neu = sorted(r['neutral'] for r in rec)
            mx = sorted(r['max'] for r in rec)
            L.append('%s: R1 に届いた発進 %d・いちばん長い間 %.3f 秒・中立の見積もりの合計の最大 %.3f 秒(上から %s)・0.1 秒より長い間が 2 つ以上の発進 %d・'
                     '0.15 秒より長い間のある発進 %d' % (day, len(rec), mx[-1], neu[-1], neu[-3:][::-1],
                                                    sum(1 for r in rec if r['n_over_0p1'] >= 2), sum(1 for r in rec if r['n_over_0p15'])))
            for r in sorted(rec, key=lambda x: -x['neutral'])[:3]:
                L.append('    %s' % r)
    finally:
        C.cleanup()
    C.write('a3_real_gaps.txt', L)


if __name__ == '__main__':
    main()
