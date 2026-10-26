#!/usr/bin/env python3
"""段 Desk(2026-10-23・A10)の机上のベンチの追加(numpy だけ。torch を読まない。test_desk_lib2.py で試す)

10/21 の desk_lib.py(experiments/20261021_bench_prep/desk/)をそのまま使い、A10 の 3 本と新しい欄を足すだけ。
  - 候補の追加(家族 label_r1。名前に "-slow-"): SLOW1・SLOW-SEC0・SLOW-SEC1(protocol_amendments.md A10)
  - nonzone_thr_diff_us(A10): 保留の 4,670 フレーム(primary)のうち、v1.1 の区間の判定が none か unknown のフレームで、
    予測の 1 行目(index 0)のスロットル − 人のスロットル(正解の index 0)の平均 [µs]。正 = 区間の外でも人より緩める
    区間の判定は make_bench_tables.py と同じ(zones2.py)。
  - A10 の C1 の決め方(決定的): c1_rule_a10()
"""
import math
import sys
from pathlib import Path

import numpy as np

OLD_DESK = Path('/home/jetson/jetformer/experiments/20261021_bench_prep/desk')
if str(OLD_DESK) not in sys.path:
    sys.path.insert(0, str(OLD_DESK))
import desk_lib as D  # noqa: E402

NONZONE = ('none', 'unknown')
NONZONE_ROW = 0                    # A10: 予測の 1 行目
A10_R1_MIN_US = 20.0               # A10 1: r1_app_thr_diff_us ≥ +20.0
A10_POOL = ('SLOW', 'SLOW1', 'SLOW-SEC0', 'SLOW-SEC1')

NEW = [
    ('SLOW-SEC0', '遅く入る型・区間だけ 種0', 'bench_20261022-slow-sec-s0', 'label_r1'),
    ('SLOW1', '遅く入る型 種1', 'bench_20261021-slow-s1', 'label_r1'),
    ('SLOW-SEC1', '遅く入る型・区間だけ 種1', 'bench_20261022-slow-sec-s1', 'label_r1'),
]
NEW_IDS = tuple(c[0] for c in NEW)
IDS2 = ('B', 'SLOW') + NEW_IDS     # この段で表に出す 5 本(B・SLOW は 10/21 の予測を読み直す)


def register():
    """10/21 の desk_lib の表(CANDIDATES・IDS・RUN_OF・LABEL_OF・FAMILY_OF)に 3 本を足す(このプロセスの中だけ)"""
    for cid, label, run, fam in NEW:
        if cid in D.RUN_OF:
            continue
        D.CANDIDATES.append((cid, label, run, fam))
        D.IDS.append(cid)
        D.RUN_OF[cid] = run
        D.LABEL_OF[cid] = label
        D.FAMILY_OF[cid] = fam
    return D


def nonzone_mask(primary, zone):
    """primary かつ 区間の判定が none か unknown"""
    z = np.asarray(zone).astype(str)
    return np.asarray(primary, bool) & np.isin(z, NONZONE)


def nonzone_thr_diff(pred, y, primary, zone, row=NONZONE_ROW):
    """A10 の nonzone_thr_diff_us(row=None なら 10 行の平均。参考)"""
    return D.thr_diff(pred, y, nonzone_mask(primary, zone), row)


def zone_thr_diff(pred, y, primary, zone, row=NONZONE_ROW):
    """参考: 区間ごとの 予測 − 人 のスロットルの平均 [µs] と数"""
    z = np.asarray(zone).astype(str)
    out = {}
    for name in ('none', 'unknown', 'R1_app', 'R1_turn', 'GM', 'DE'):
        m = np.asarray(primary, bool) & (z == name)
        out[name] = {'n': int(m.sum()), 'thr_diff_us': D.thr_diff(pred, y, m, row) if m.any() else None}
    return out


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def c1_rule_a10(rows, pool=A10_POOL, r1_min=A10_R1_MIN_US):
    """A10 の C1 の枠の決め方。rows = {id: 行(gates_pass・r1_app_thr_diff_us・nonzone_thr_diff_us)}
    1. 対象 = pool のうち gates_pass が true かつ r1_app_thr_diff_us ≥ r1_min(欄が無い・null は対象外)
    2. 対象が空なら SLOW(protocol の step2 どおり)
    3. nonzone_thr_diff_us を小数第 1 位に四捨五入して小さい順 → r1_app_thr_diff_us が大きい順 → id の文字の順
    → {'pick', 'targets'(並べた順), 'excluded': {id: 理由}, 'empty_fallback'}"""
    targets, excluded = [], {}
    for cid in pool:
        r = rows.get(cid)
        if r is None:
            excluded[cid] = 'no_row'
            continue
        why = []
        if r.get('gates_pass') is not True:
            why.append('gates_pass')
        r1 = _num(r.get('r1_app_thr_diff_us'))
        if r1 is None:
            why.append('missing:r1_app_thr_diff_us')
        elif not r1 >= r1_min:
            why.append('r1_app_thr_diff_us<%.1f' % r1_min)
        nz = _num(r.get('nonzone_thr_diff_us'))
        if nz is None:
            why.append('missing:nonzone_thr_diff_us')
        if why:
            excluded[cid] = ','.join(why)
            continue
        targets.append((cid, nz, r1))
    if not targets:
        return {'pick': 'SLOW', 'targets': [], 'excluded': excluded, 'empty_fallback': True}

    def key(t):
        cid, nz, r1 = t
        return (_round1(nz), -r1, cid)
    targets.sort(key=key)
    return {'pick': targets[0][0],
            'targets': [{'id': c, 'nonzone_thr_diff_us_r1': _round1(nz), 'nonzone_thr_diff_us': nz, 'r1_app_thr_diff_us': r1}
                        for c, nz, r1 in targets],
            'excluded': excluded, 'empty_fallback': False}


def _round1(x):
    """小数第 1 位に四捨五入(0.5 は 0 から遠い方へ。Python の round の偶数丸めを使わない)"""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(float(x))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
