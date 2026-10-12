"""予測の重ね合わせ(temporal ensembling。numpy だけ。どの方策にも使える)

毎フレームの予測 [10,2](1〜10 フレーム先)のうち、過去のフレームの予測で「今から s フレーム先」に当たる行も混ぜて滑らかにする。
走行タブの「予測の重ね合わせ」(ensemble_j: 何フレーム前の予測まで混ぜるか、0 = オフ。ensemble_m: 古いほど軽くする強さ)。
**既定はオフ。方策の比較の判定には使わない**(参考の行だけ)。

ACT の元の式(古いほど重い、m = 0.01)は使わない(30fps の車で反応が遅れる。design_research.md §2)。
ここでは j = 0(一番新しい予測)が一番重い: w_j = exp(-m·j)。
"""

import collections

import numpy as np


class TemporalEnsembler(object):
    def __init__(self, horizon=10, fps=30):
        self.horizon = int(horizon)
        self.fps = float(fps)
        self.buf = collections.deque()          # (撮像時刻 ns, 予測 [horizon,2] µs float64)

    def reset(self):
        self.buf.clear()

    def push(self, t_ns, pred_us, j_max, m):
        """pred_us [10,2](µs)を時刻つきで貯め、出力の行 s = Σ_j w_j·p_{t-j}[s+j] / Σ w_j(s + j ≤ 9)。
        j = round((t_now - t_past)·fps)(フレームが抜けても時刻で数える)、w_j = exp(-m·j)(j = 0 が一番新しい = 一番重い)。
        j_max = 0 なら pred_us の写しをそのまま返す。j > j_max の古い予測は捨てる。戻り値 [10,2] float32"""
        pred = np.asarray(pred_us, np.float64)
        j_max = max(int(j_max), 0)
        m = max(float(m), 0.0)
        self.buf.append((int(t_ns), pred.copy()))
        keep = collections.deque()
        num = np.zeros_like(pred)
        den = np.zeros((len(pred), 1))
        for t_past, p in self.buf:
            j = int(round((int(t_ns) - t_past) * self.fps / 1e9))
            if j < 0 or j > j_max or j >= len(pred):
                continue                        # 未来の時刻(時計の巻き戻り)・古すぎるものは捨てる
            keep.append((t_past, p))
            w = float(np.exp(-m * j))
            num[:len(pred) - j] += w * p[j:]
            den[:len(pred) - j] += w
        self.buf = keep
        if j_max == 0:
            return np.asarray(pred_us, np.float32).copy()
        return (num / np.maximum(den, 1e-12)).astype(np.float32)
