#!/usr/bin/env python3
"""モデルの 10 フレーム予測を、時刻をそろえて重ねる(ACT の temporal ensembling と同じ考え方)と、
出力の揺れ・左右の切り返し・遅れがどう変わるかをオフラインで数える(読み取りのみ、CPU)。

    nice -n 15 python ensemble.py > ../ensemble.json

今の出力: 先読み la のとき raw_t = pred_t[la-1]。
重ね合わせ: e_t = Σ_j w_j pred_{t-j}[la-1+j] / Σ w_j(j = 0..J、pred は 10 フレーム分なので la-1+j ≤ 9)、w_j = exp(-m j)。
どれも「時刻 t+la フレームの操作」を指す予測なので、平均しても遅れは原理的に増えない(古い予測ほど古い画像に基づく点は残る)。
指標: 1 フレームあたりの変化 |Δ| の中央値・p95、左右の切り返し回数/秒(中心 ±50µs をまたぐ)、raw との差、
相互相関で見た raw に対する遅れ(フレーム)。AI 側・連続フレーム(間隔 < 50 ms)だけ。ゲイン前の µs。
"""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load  # noqa: E402
LOGS = {'20261010-172910': 'ai_run_006', '20261010-173445': 'ai_run_007', '20261010-204208': 'ai_run_010'}
C = (1443.0, 1498.0)
def flips(x, c, t):
    s = np.where(x > c + 50, 1, np.where(x < c - 50, -1, 0))
    nz = s[s != 0]
    return float(np.sum(nz[1:] != nz[:-1])) / max(t, 1e-6)
res = {'definitions': __doc__, 'logs': {}}
for log, name in LOGS.items():
    _, d = load(log, ('AI_CMD',))
    c = d['AI_CMD']
    ts = c['timestamps_us'] / 1e6
    ok = c['select'] & ~c['hold']
    pred, la = c['pred'], c['lookahead'].astype(int)
    out = {}
    for J, m in ((0, 0.0), (2, 0.0), (3, 0.0), (3, 0.5), (5, 0.5), (7, 0.5)):
        for ch, nm in ((0, 'steer'), (1, 'thr')):
            e = np.full(len(ts), np.nan)
            for i in range(len(ts)):
                num = den = 0.0
                for j in range(J + 1):
                    k = la[i] - 1 + j
                    if i - j < 0 or k > 9 or not ok[i - j] or ts[i] - ts[i - j] > (j + 0.5) * 0.05 or la[i - j] != la[i]:
                        break
                    w = np.exp(-m * j)
                    num += w * pred[i - j, k, ch]
                    den += w
                e[i] = num / den if den else np.nan
            raw = pred[np.arange(len(ts)), la - 1, ch]
            good = ok & np.isfinite(e)
            cont = good[1:] & good[:-1] & (np.diff(ts) < 0.05)
            de = np.abs(np.diff(e))[cont]
            dr = np.abs(np.diff(raw))[cont]
            T = float(np.sum(np.diff(ts)[cont]))
            # 遅れ: raw に対する相互相関の最大(±5 フレーム)
            x = raw[good] - np.mean(raw[good]); y = e[good] - np.mean(e[good])
            lags = range(-5, 6)
            cc = [np.sum(x[max(0, -L):len(x) - max(0, L)] * y[max(0, L):len(y) - max(0, -L)]) for L in lags]
            lag = list(lags)[int(np.argmax(cc))]
            out['J%d_m%.1f|%s' % (J, m, nm)] = {
                'jitter_med_us': round(float(np.median(de)), 1), 'jitter_p95_us': round(float(np.percentile(de, 95)), 1),
                'raw_jitter_med_us': round(float(np.median(dr)), 1), 'raw_jitter_p95_us': round(float(np.percentile(dr, 95)), 1),
                'flips_per_s': round(flips(e[good], C[ch], T), 2), 'raw_flips_per_s': round(flips(raw[good], C[ch], T), 2),
                'mean_abs_diff_vs_raw_us': round(float(np.mean(np.abs(e[good] - raw[good]))), 1), 'lag_frames_vs_raw': lag,
                'ai_s': round(T, 1)}
    res['logs'][name] = out
json.dump(res, sys.stdout, ensure_ascii=False, indent=1)
