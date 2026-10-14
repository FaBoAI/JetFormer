"""学習から外すフレームの決まり(Python の参照実装。2026-10-13、sim_rl_modes)。

JetFormer の取り込み(統合の段)が使う形。rl/runtime/perturb.mjs の maskFromLearn と同じ結果(試験 tests/test_masking.py)。
protocol.json definitions.mask_rule: フレーム j の操作が学習しないもの(frames.jsonl の action_info.jf.l = 0)なら、
それを正解(観測 k の正解は k+1〜k+10 の操作)に含む観測 k を外す。乱しの区間 [ps, pe) なら k ∈ [ps − 10, pe − 2]。
外した観測は future_valid[k] = False・mode[k] = 0(未設定 = 学習しない)。vehicle_valid は変えない。
"""
from __future__ import annotations

import numpy as np

HORIZON = 10


def mask_from_learn(learn, horizon: int = HORIZON) -> np.ndarray:
    """learn[j] = 1(学習に使う)/ 0(使わない)→ 外す観測の印 mask[k](True = 外す)。"""
    learn = np.asarray(learn).astype(bool)
    n = learn.shape[0]
    mask = np.zeros(n, bool)
    for j in np.nonzero(~learn)[0]:
        mask[max(0, j - horizon):j] = True
    return mask


def apply_label_mask(future_valid, mode, learn, horizon: int = HORIZON):
    """future_valid・mode(フレームごと)に印を反映した写しと、外した数を返す。最後の 10 フレームの False は元のまま。"""
    fv = np.asarray(future_valid, bool).copy()
    md = np.asarray(mode).copy()
    m = mask_from_learn(learn, horizon)
    newly = m & fv
    fv[m] = False
    md[m] = 0
    return fv, md, int(m.sum()), int(newly.sum())


def learn_from_frames_jsonl(lines) -> np.ndarray:
    """frames.jsonl の行(dict)→ learn の列。**安全側(fail closed)**: action_info.jf が無い行、jf.l が 0/1 でない行は 0(学習しない)。

    2026-10-14(recover r2、確認 recover r1 の指摘)で直した。r1 は「jf.l が無い行は 1(学習に使う)」で、ブラウザが info を落とすと
    乱した操作が正解に入りえた。JetFormer の取り込みは jetformer/sim/label_mask.py が同じ決まり(+ frame の番号で並べる・無いフレームも 0)で行う。
    強化学習モードでないシーン(jf がどこにも無い)には使わない(そのシーンは外さない)。
    """
    out = []
    for row in lines:
        info = row.get("action_info") if isinstance(row, dict) else None
        jf = info.get("jf") if isinstance(info, dict) else None
        l = jf.get("l") if isinstance(jf, dict) else None
        out.append(1 if (l == 1 and not isinstance(l, bool)) else 0)
    return np.asarray(out, np.int8)
