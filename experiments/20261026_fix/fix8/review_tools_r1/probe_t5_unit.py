"""反証 8-1 回目・係 tools・台本 7(読むだけ・速い): standing_3lap.abort_resumes(今の版)だけで境目を突く。
0.5 秒(中立止め)・0.2 秒(カメラの途切れ)・0.5 m(続き)・assist の段の変わり目(区間の最後の 2 フレームの assist)・止まっているか。
時刻は 1/30 秒のフレーム、車両は 0.01 秒。区間 1 = 0〜5 秒(AI)、人側 5〜6 秒、区間 2 = 6 秒〜(AI。走り直し)、人側 8〜9 秒、区間 3 = 9 秒〜。
出力 out/probe_t5_unit.txt

  jf-heavy --lock none --mem 1500M -- python -B probe_t5_unit.py
"""
import numpy as np

import common as C
from common import t8

s3 = t8.mods('new')[0]


def base():
    t = np.round(np.arange(0, 20, 1 / 30.0), 6)
    ai = (t < 5) | ((t >= 6) & (t < 8)) | (t >= 9)
    a = np.zeros(len(t), np.int8)
    h = np.zeros(len(t), bool)
    vt = np.round(np.arange(0, 20, 0.01), 6)
    ve = np.where(vt < 4.0, -2000.0, 0.0)                 # 4 秒までは走り、そのあと止まる
    ve = np.where((vt >= 6.2) & (vt < 8.0), -2000.0, ve)   # 走り直し
    ve = np.where(vt >= 9.2, -2000.0, ve)
    Df = np.concatenate([[0.0], np.cumsum(np.maximum(-ve[1:], 0) * 0.5e-3 * 0.01)])
    return t, ai, a, h, vt, ve, Df


def mark(t, ai, a, h, vt, ve, Df):
    r = s3.abort_resumes(t, ai, a, h, None, vt, Df, ve)
    return {round(float(t[i]), 2): (v.get('kind'), v.get('carried')) for i, v in sorted(r.items())}


def main():
    lines = []
    t, ai, a, h, vt, ve, Df = base()
    last = int(np.nonzero(t < 5)[0][-1])
    cases = []
    # assist の段の変わり目(区間 1 の最後の 2 フレーム)
    for pair in ((2, 2), (2, 3), (3, 2), (2, 0), (2, 1), (2, 4), (5, 2), (0, 2), (2, 5)):
        a2 = a.copy()
        a2[(t >= 3.5) & (t < 5)] = 2
        a2[last - 1], a2[last] = pair
        cases.append(('最後の 2 フレーム assist %s' % (pair,), a2, h))
    # 中立止め(区間 1 の終わり)。止まっている(4 秒から 0)。リカバリーあり / なし
    for hs in (0.4, 0.5, 0.6):
        for rec in (True, False):
            a2, h2 = a.copy(), h.copy()
            if rec:
                a2[(t >= 3.5) & (t < 5)] = 2
            m = (t >= 5 - hs - 1e-6) & (t < 5)
            h2[m] = True
            a2[m] = 0
            cases.append(('終わりの中立止め %.1f 秒・リカバリー %s' % (hs, rec), a2, h2))
    for name, a2, h2 in cases:
        lines.append('%s → 印 %s' % (name, mark(t, ai, a2, h2, vt, ve, Df)))
    # 中立止め 0.6 秒だが終わりで動いている(後ろへ 1000 eRPM。リカバリーの後退の途中)
    a2, h2 = a.copy(), h.copy()
    a2[(t >= 3.5) & (t < 5)] = 2
    m = (t >= 4.4) & (t < 5)
    h2[m], a2[m] = True, 0
    ve2 = np.where((vt >= 4.4) & (vt < 5.0), 1000.0, ve)
    lines.append('終わりの中立止め 0.6 秒・後ろへ動いたまま・リカバリー True → 印 %s' % mark(t, ai, a2, h2, vt, ve2, Df))
    # 続き(0.5 m): 区間 2 の前進が 0.45 / 0.55 m で人側
    for d in (0.45, 0.55):
        a2 = a.copy()
        a2[(t >= 3.5) & (t < 5)] = 2
        ve3 = np.where((vt >= 6.2) & (vt < 8.0), 0.0, ve)
        k = (vt >= 6.2) & (vt < 6.2 + d / (2000 * 0.5e-3))
        ve3 = np.where(k, -2000.0, ve3)
        Df3 = np.concatenate([[0.0], np.cumsum(np.maximum(-ve3[1:], 0) * 0.5e-3 * 0.01)])
        lines.append('走り直しが %.2f m で人側 → 印 %s' % (d, mark(t, ai, a2, h, vt, ve3, Df3)))
    C.write('probe_t5_unit.txt', lines)


if __name__ == '__main__':
    main()
