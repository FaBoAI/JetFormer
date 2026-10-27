#!/usr/bin/env python3
"""車の確認(2026-10-25、speed_control_20261025/spec.md 5.2): 1 本の記録から、車の左右の曲がり・まっすぐ進む舵・
直線の最高速・周・電圧・R1 の板の衝撃を出し、11/3 の「各回の最初の 1 本」の判定をする。

    python3 tools/car_check.py 20261103-xxxxxx          # 1 本
    python3 tools/car_check.py --latest                 # 今日のいちばん新しい AI の記録(dry run でない)
        (それより新しい記録の log.json が読めない・AI_CMD が無いときは「いちばん新しい記録 X は読めません(…)」と出して
         終わりの番号 3。1 本前を黙って出さない。2026-10-27 fix3 直す 3 回目 r3-r3-rec-1)
    python3 tools/car_check.py LOG --json out.json      # JSON の書き先(既定は data/reports/car_check/<LOG>.json)
    python3 tools/car_check.py LOG --no-json            # 画面に出すだけ

読むだけ(numpy。GPU・画像なし)。生ログ・ラベルは変えない。ラベルのファイルが無ければ label_events.label_log を
その場で呼ぶ(書き出さない)。周・通過・衝突・電圧・天井と自動合わせの記録は tools/section_scorecard.score_log と同じ
(偽の R1 の起点は外す。label_events.reject_false_anchors)。

出すもの(定義は 10/24 の台本と同じ。数は cond_metrics.json・steer_offset.json・r1_bump.json と合う):
  左右の旋回の効きの比  experiments/20261024_report/cond/cond_metrics.py の turn_gain_left ÷ turn_gain_right。
        フレーム = AI・dry run でない・中立止めなし・補助なし(直前 0.5 秒もずっと)。舵 = out − 中心、直前 0.2 秒の平均の
        |舵| ≥ 150 µs・速さ > 1.5 m/s・AI の衝突の前後 0.5 秒を除く。κ = gz(止まっているときの中央値を引き、IMU の
        5 サンプルの平均)[rad/s] ÷ max(速さ, 0.1)。κ ÷ 舵 × 1e4 の左・右それぞれの中央値
  まっすぐ進む舵 s0     col/steer_offset.py。AI・速さ > 1.5・衝突の前後 1 秒を除く・|0.1 秒前の舵| < 450 µs のフレームで、
        κ を 0.1 秒前の舵に直線で当てはめ、κ = 0 の舵 [µs](+ = 左に切ってやっとまっすぐ = 車が右へ流れる)
  直線の最高速          AI 中(前進 > 300 eRPM)の前進 eRPM の 90% 点 → m/s(× 0.5e-3)。HS だけの 90% 点も
  周                    R1 の起点 → 次の起点(5〜16 秒・ずっと AI)。周の時間・周の速さ p90(周の中のフレームの速さの 90% 点)
  電圧                  VEHICLE.npz の vbat_v(走り始め・終わりの 2 秒の中央値・1 秒ごとの中央値の最小)→ 無ければ説明欄
  R1 の板の衝撃          fix/r1_bump.py。AI の R1 の起点ごとに、起点 → +1.2 秒の 3 軸 |‖a‖/G − 1| の最大。中央・最大・2.5G 以上の数
  参考                  DE の衝突/通過(成績表の通過の数え方)・強い衝突の数・天井と自動合わせの記録の要約
  衝突                  1 本の AI の衝突(label の決まり。立て直し中・リカバリー中も入れる)の最大 G・8G 以上の数・場所ごとの数
                        (R1・GM・DE・SW・HS・LH・?)と強い衝突(4G 以上)の時刻・場所・G(2026-10-26 fix2 R2-rec-1。手順書 7 章の
                        「8G 以上 1 回」「同じ場所で 2 本続けて」「左 4.2 で SW の出口・LH」を 1 本ごとに決めるため。判定の番号には入れない)
  自動合わせの電圧の源    天井・自動合わせの行に「電圧の源 基板 / 説明欄 / なし」(AI_CMD の sa_src。手順書 6 章)
  人側にした直後の衝突    AI の区間の終わりから AFTER_AI_S(0.5 秒)以内の人側の衝突(ラベルの衝突 = suspect でないもの)も「衝突」の
                        行と 8G 以上の数・場所に入れ、「人側にした直後」の印と AI の終わりからの秒を付ける(2026-10-27 fix3 直す 2 回目
                        r2-rec2-4。人が当たる直前に CH3 を人側にした 9G が「衝突 0 回」と出て、手順書 7 章の「8G 以上 1 回」
                        「同じ場所で 2 本続けて」が読めなかった)。JSON の collisions に after_ai(あるときだけ)。判定の番号は変えない。
                        そういう衝突の無い本は、文・JSON とも前と同じ

判定(qualify_plan 3.2・3.3・3.7、ユーザーの了承の値)。終わりの番号: 0 合格・1 直す/注意・2 不合格・3 足りない
  合格   左右の比 ≥ 0.90 かつ s0 が −10〜+10 µs
  直す   比 0.85〜0.90、または s0 が +10〜+15 µs → steer_trim_us を +15〜+20 µs(左へ)にして 1 本走り、もう一度
  不合格 比 < 0.85、または s0 > +15 µs → 本番の設定に進まない(後輪・舵のリンクを見る)
  注意   s0 < −10 µs(左へ流れる向き。10/24 の No.9 は −17.8 µs で問題なし)
  板     R1 の 3 軸の中央 ≥ 4.5G、または 1 回でも ≥ 7.0G → 止めて後輪・取り付けを見る。次はゲイン −0.05(不合格に数える)。
         「1 回でも ≥ 7.0G」は R1 の通過が 1〜2 でも不合格(2026-10-26 fix3 直す 1 回目 r1-rec3-2。中央は通過 3 以上のときだけ)
  電圧   走り始めが 7.70 V 未満 → パックを替える(注意)
  速さ   周の p90 の中央値 → qualify_plan 3.2 節の表のゲインの直し(番号には入れない)
  足りない 左右のどちらかが 30 フレーム未満・s0 が 200 フレーム未満・R1 の通過が 3 未満(1 回でも ≥ 7.0G なら不合格)
  試験の選択肢 log.json の rec_back_guard・rec_hold_keep・launch_no_kick が 1 なら、文に「試験の選択肢: …」の行(判定には入れない。
         2026-10-26 fix3 直す 1 回目 r1-rec3-5。どれも 0 か無ければ行を出さない = 前と同じ)
  番号: 不合格が 1 つでもあれば 2、なければ足りないものがあれば 3、なければ直す・注意があれば 1、なければ 0
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jetformer import config  # noqa: E402
import label_events as le  # noqa: E402
import section_scorecard as ss  # noqa: E402

SCHEMA = 'jf-car-check/v1'
G = le.G
KV = le.KV

# ---- 判定の値(ユーザーの了承のとおり。変えるときは spec 9 章で聞く)
RATIO_PASS = 0.90
RATIO_FAIL = 0.85
S0_LO, S0_HI = -10.0, 10.0
S0_FAIL = 15.0
BOARD_MEDIAN_G = 4.5
BOARD_MAX_G = 7.0
VBAT_MIN_V = 7.70
MIN_TURN_FRAMES = 30
MIN_S0_FRAMES = 200
MIN_R1_PASSES = 3
SW_FAST_MPS = 4.3
# qualify_plan 3.2 節: 周の速さ p90 の中央値 → ゲインの直し([下限, 上限) m/s、P1 の直し、P2(天井あり)の直し)
SPEED_TABLE = [(-np.inf, 3.75, '+0.05', '+0.05', '遅い回'), (3.75, 3.90, '+0.03', '+0.03', ''),
               (3.90, 4.10, 'そのまま', 'そのまま', 'No.9 の水準'), (4.10, 4.25, '−0.03', 'そのまま', 'P2 は天井がある'),
               (4.25, np.inf, '−0.05', '−0.05', '')]

# ---- 定義(10/24 の台本と同じ)
W_CMD = 15                 # 0.5 秒(30 fps): 補助なしが直前 0.5 秒ずっと
W_ST = 6                   # 0.2 秒: 舵の直前の平均
TURN_MIN_US = 150.0
MIN_V = 1.5
TURN_NEAR_COLL_S = 0.5
GZ_SMOOTH = 5              # IMU のサンプル
S0_LAG = 3                 # 0.1 秒
S0_MAX_US = 450.0
S0_COLL_FRAMES = 61        # 衝突の前後 1 秒(30 fps の 61 フレームの窓)
MOVING_ERPM = 300.0
LOC_HS, LOC_SW = 6, 5      # label_events.LOC_DETAIL の番号
BUMP_W = (0.0, 1.2)
BUMP_THR = 2.5
BUMP_MIN_SAMPLES = 5
CODES = {0: '合格', 1: '直す・注意', 2: '不合格', 3: '足りない'}
AFTER_AI_S = 0.5           # 人側にした直後の衝突: AI の区間の終わりからこれだけ(r2-rec2-4。本物の 10/23・10/24 の 50 本で、この中の
                           # 人側の衝突は 7 件。4G 以上 4 件は 0.16〜0.35 秒あと、8G 以上 2 件は 0.34・0.35 秒あと)


def _r(x, n=3):
    return ss._r(x, n)


def conv_same(x, w):
    """長さ w の窓の和を、x と同じ長さで(np.convolve の 'same' と同じ値。x が w より短くても長さは x のまま)"""
    x = np.asarray(x, np.float64)
    if not len(x):
        return x
    full = np.convolve(x, np.ones(w))
    i0 = (w - 1) // 2
    return full[i0:i0 + len(x)]


def trailing_mean(x, w):
    """直前 w 個(今を含む)の平均(cond_metrics.trailing と同じ)"""
    x = np.asarray(x, np.float64)
    n = len(x)
    c = np.cumsum(np.r_[0.0, x])
    i0 = np.maximum(np.arange(n) - w + 1, 0)
    return (c[np.arange(n) + 1] - c[i0]) / (np.arange(n) + 1 - i0)


def trailing_all(m, w):
    """直前 w 個(今を含む)がずっと True(cond_metrics.trailing_all と同じ)"""
    m = np.asarray(m, bool)
    c = np.cumsum(np.r_[0, (~m).astype(int)])
    i0 = np.maximum(np.arange(len(m)) - w + 1, 0)
    return (c[np.arange(len(m)) + 1] - c[i0]) == 0


# ---------------------------------------------------------------- 指標
def turn_gains(lab, it, gz, vt, verpm, center):
    """左右の旋回の効き(cond_metrics.py の turn_gain_left・turn_gain_right)。返り値 dict"""
    t = lab['t_s'].astype(np.float64)
    n = len(t)
    ai = lab['ai'].astype(bool) & ~lab['dry_run'].astype(bool)
    assist = lab['assist'].astype(int)
    drv = ai & ~lab['hold'].astype(bool) & ((assist == 0) | (assist == -1))
    v = lab['speed_mps'].astype(np.float64)
    steer = lab['out'][:, 0].astype(np.float64) - center
    ve = np.interp(it, vt, verpm) if len(vt) else np.zeros(len(it))
    still = (np.abs(ve) < 50) & (np.abs(gz) < 5)
    bias = float(np.median(gz[still])) if still.sum() > 50 else -0.09
    gzs = conv_same(gz - bias, GZ_SMOOTH) / GZ_SMOOTH
    gz_f = np.interp(t, it, gzs)
    drvw = trailing_all(drv, W_CMD)
    near_c = np.zeros(n, bool)
    for c in t[lab['collision'].astype(bool) & ai]:
        near_c |= (t >= c - TURN_NEAR_COLL_S) & (t <= c + TURN_NEAR_COLL_S)
    st_avg = trailing_mean(steer, W_ST)
    kappa = np.radians(gz_f) / np.maximum(v, 0.1)
    with np.errstate(invalid='ignore'):
        ok = drvw & (np.abs(st_avg) >= TURN_MIN_US) & (v > MIN_V) & ~near_c
    kl = kappa[ok & (st_avg > 0)] / st_avg[ok & (st_avg > 0)] * 1e4
    kr = kappa[ok & (st_avg < 0)] / st_avg[ok & (st_avg < 0)] * 1e4
    left = float(np.median(kl)) if len(kl) else None
    right = float(np.median(kr)) if len(kr) else None
    ratio = left / right if left is not None and right else None
    return {'left': _r(left, 3), 'right': _r(right, 3), 'n_left': int(len(kl)), 'n_right': int(len(kr)),
            'ratio': _r(ratio, 3), 'gyro_bias_dps': _r(bias, 3)}


def steer_offset(lab, it, gz, center):
    """まっすぐ進む舵 s0(col/steer_offset.py)。返り値 dict"""
    t = lab['t_s'].astype(np.float64)
    g = conv_same(np.interp(t, it, gz), GZ_SMOOTH) / GZ_SMOOTH
    st = lab['out'][:, 0].astype(np.float64) - center
    stl = np.r_[np.full(S0_LAG, np.nan), st[:-S0_LAG]] if len(st) > S0_LAG else np.full(len(st), np.nan)
    v = lab['speed_mps'].astype(np.float64)
    bad = conv_same(lab['collision'].astype(float), S0_COLL_FRAMES) > 0
    with np.errstate(invalid='ignore'):
        m = lab['ai'].astype(bool) & (v > MIN_V) & np.isfinite(stl) & ~bad & (np.abs(stl) < S0_MAX_US)
    out = {'s0_us': None, 'slope_per_100us': None, 'n': int(m.sum())}
    if m.sum() < MIN_S0_FRAMES:
        return out
    k = np.deg2rad(g[m]) / v[m]
    x = stl[m]
    c, b = np.linalg.lstsq(np.c_[x, np.ones(len(x))], k, rcond=None)[0]
    if c == 0:
        return out
    out.update(s0_us=_r(-b / c, 1), slope_per_100us=_r(c * 100, 4))
    return out


def top_speed(lab, loc_detail):
    """直線の最高速(cond_metrics の erpm_p90_ai・erpm_p90_HS)"""
    ai = lab['ai'].astype(bool) & ~lab['dry_run'].astype(bool)
    e = -lab['erpm'].astype(np.float64)
    with np.errstate(invalid='ignore'):
        mv = ai & (e > MOVING_ERPM)
    p = lambda m: float(np.percentile(e[m], 90)) if m.any() else None      # noqa: E731
    e_all, e_hs = p(mv), p(mv & (loc_detail == LOC_HS))
    return {'erpm_p90': _r(e_all, 0), 'erpm_p90_HS': _r(e_hs, 0), 'mps_p90': _r(e_all * KV, 2) if e_all else None,
            'mps_p90_HS': _r(e_hs * KV, 2) if e_hs else None}


def lap_speeds(laps, lab, loc_detail):
    """周ごとの速さ p90 と SW の最高(周の中のフレームの速さ)"""
    t = lab['t_s'].astype(np.float64)
    v = lab['speed_mps'].astype(np.float64)
    rows = []
    for lp in laps:
        a, b = lp['t_start'], lp['t_start'] + lp['lap_s']
        m = (t >= a) & (t < b) & np.isfinite(v)
        sw = m & (loc_detail == LOC_SW)
        rows.append({'t_start': lp['t_start'], 'lap_s': lp['lap_s'], 'clean': lp.get('clean'),
                     'p90_mps': _r(float(np.percentile(v[m], 90)), 3) if m.any() else None,
                     'sw_max_mps': _r(float(v[sw].max()), 2) if sw.any() else None,
                     'dpsi_deg': lp.get('dpsi_deg'), 'dist_m': lp.get('dist_m')})
    med = lambda k: _r(float(np.median([x[k] for x in rows if x[k] is not None])), 3) \
        if any(x[k] is not None for x in rows) else None      # noqa: E731
    lt = [x['lap_s'] for x in rows]
    return {'n': len(rows), 'lap_median_s': _r(float(np.median(lt)), 3) if lt else None, 'lap_best_s': _r(min(lt), 3) if lt else None,
            'p90_median_mps': med('p90_mps'), 'sw_max_median_mps': med('sw_max_mps'), 'rows': rows}


def r1_bumps(passes, it, accel):
    """R1 の板の衝撃(fix/r1_bump.py): AI の通過ごとに 起点 → +1.2 秒の 3 軸 |‖a‖/G − 1| の最大"""
    a3 = np.abs(np.linalg.norm(np.asarray(accel, np.float64), axis=1) / G - 1.0)
    rows = []
    for ps in passes:
        if not ps.get('ai'):
            continue
        ta = ps['ta_s']
        m = (it >= ta + BUMP_W[0]) & (it <= ta + BUMP_W[1])
        if m.sum() < BUMP_MIN_SAMPLES:
            continue
        i = np.nonzero(m)[0][int(np.argmax(a3[m]))]
        rows.append({'ta_s': _r(ta, 2), 'entry_erpm': ps.get('entry_erpm'), 'a3_max': _r(float(a3[i]), 2),
                     't_peak_after': _r(float(it[i] - ta), 2)})
    g = [x['a3_max'] for x in rows]
    return {'passes': len(rows), 'a3_median': _r(float(np.median(g)), 2) if g else None, 'a3_max': _r(max(g), 2) if g else None,
            'n_ge_2p5': sum(1 for x in g if x >= BUMP_THR), 'rows': rows}


# ---------------------------------------------------------------- 判定
def speed_advice(p90, sw_max=None, ceiling_on=False):
    """qualify_plan 3.2 節の表: 周の速さ p90 の中央値(と SW の最高)→ ゲインの直し"""
    if p90 is None:
        return {'p90': None, 'p1': None, 'p2': None, 'text': '周が無いので速さの直しは出せません'}
    row = next(r for r in SPEED_TABLE if r[0] <= p90 < r[1])
    p1, p2, note = row[2], row[3], row[4]
    sw_fast = sw_max is not None and sw_max >= SW_FAST_MPS
    if sw_fast:
        p1 = p2 = '−0.05'
        note = 'SW の最高 %.2f m/s ≥ %.1f' % (sw_max, SW_FAST_MPS)
    if p1 == p2:
        text = 'ゲイン %s' % p1 if p1 != 'そのまま' else 'ゲインはそのまま'
    else:
        text = 'P1(天井なし)は ゲイン %s、P2(天井あり)は %s' % (p1, p2)
        if ceiling_on:
            text += '(この記録は天井 ON → %s)' % p2
    return {'p90': p90, 'sw_max': sw_max, 'p1': p1, 'p2': p2, 'note': note,
            'text': '周の速さ p90 の中央値 %.2f m/s → %s%s' % (p90, text, '(%s)' % note if note else '')}


def judge(m):
    """指標 → 判定。m: {'turn': {...}, 's0': {...}, 'r1_bump': {...}, 'vbat': {...}, 'laps': {...}, 'speed_ctrl': {...}}。
    返り値 {'code', 'label', 'items': [{'item', 'status', 'text'}], 'lines': [...]}。status: ok・fix・caution・fail・insufficient・na・info"""
    items = []
    tr = m.get('turn') or {}
    ratio = tr.get('ratio')
    if tr.get('n_left', 0) < MIN_TURN_FRAMES or tr.get('n_right', 0) < MIN_TURN_FRAMES or ratio is None:
        items.append(('turn', 'insufficient', '左右の旋回の効き: 判定に足りません(左 %d・右 %d フレーム、%d 以上要る)' % (
            tr.get('n_left', 0), tr.get('n_right', 0), MIN_TURN_FRAMES)))
    elif ratio >= RATIO_PASS:
        items.append(('turn', 'ok', '左右の旋回の効きの比 %.3f(≥ %.2f)' % (ratio, RATIO_PASS)))
    elif ratio >= RATIO_FAIL:
        items.append(('turn', 'fix', '左右の旋回の効きの比 %.3f(%.2f〜%.2f)→ 車を見る。steer_trim_us を +15〜+20 µs(左へ)にして 1 本走り、もう一度(トリムはブロックを選んでも残る。走行開始の確認の文の「トリム:」の行で見る)' % (
            ratio, RATIO_FAIL, RATIO_PASS)))
    else:
        items.append(('turn', 'fail', '左右の旋回の効きの比 %.3f(< %.2f)→ 本番の設定に進まない(後輪・舵のリンクを見る)' % (ratio, RATIO_FAIL)))
    so = m.get('s0') or {}
    s0 = so.get('s0_us')
    if s0 is None or so.get('n', 0) < MIN_S0_FRAMES:
        items.append(('s0', 'insufficient', 'まっすぐ進む舵 s0: 判定に足りません(%d フレーム、%d 以上要る)' % (so.get('n', 0), MIN_S0_FRAMES)))
    elif S0_LO <= s0 <= S0_HI:
        items.append(('s0', 'ok', 'まっすぐ進む舵 s0 %+.1f µs(%+.0f〜%+.0f)' % (s0, S0_LO, S0_HI)))
    elif S0_HI < s0 <= S0_FAIL:
        items.append(('s0', 'fix', 'まっすぐ進む舵 s0 %+.1f µs(+%.0f〜+%.0f)→ steer_trim_us を +15〜+20 µs(左へ)にして 1 本走り、もう一度(トリムはブロックを選んでも残る。走行開始の確認の文の「トリム:」の行で見る)' % (
            s0, S0_HI, S0_FAIL)))
    elif s0 > S0_FAIL:
        items.append(('s0', 'fail', 'まっすぐ進む舵 s0 %+.1f µs(> +%.0f)→ 本番の設定に進まない(後輪・舵のリンクを見る)' % (s0, S0_FAIL)))
    else:
        items.append(('s0', 'caution', 'まっすぐ進む舵 s0 %+.1f µs(< %+.0f): 左へ流れる向き(10/24 の No.9 は −17.8 µs で問題なし)' % (s0, S0_LO)))
    rb = m.get('r1_bump') or {}
    if 1 <= rb.get('passes', 0) < MIN_R1_PASSES and rb.get('a3_max') is not None and rb['a3_max'] >= BOARD_MAX_G:
        # 2026-10-26 fix3 直す 1 回目 r1-rec3-2: 「1 回でも ≥ 7.0G」は通過の数によらず不合格(止める条件。回 3 の止まって 1〜2 周の
        # 本は通過が 1〜2 で、前は「足りない(もう 1 本)」と出た)。中央 ≥ 4.5G の判定は今まで通り通過 3 以上のときだけ
        items.append(('r1_bump', 'fail', 'R1 の板の衝撃 最大 %.2fG(1 回でも ≥ %.1fG。AI の R1 の通過 %d。中央の判定は %d 通過から)→ 止めて後輪・取り付けを見る。次はゲイン −0.05' % (
            rb['a3_max'], BOARD_MAX_G, rb['passes'], MIN_R1_PASSES)))
    elif rb.get('passes', 0) < MIN_R1_PASSES:
        items.append(('r1_bump', 'insufficient', 'R1 の板の衝撃: 判定に足りません(AI の R1 の通過 %d、%d 以上要る)' % (rb.get('passes', 0), MIN_R1_PASSES)))
    elif rb['a3_median'] >= BOARD_MEDIAN_G or rb['a3_max'] >= BOARD_MAX_G:
        items.append(('r1_bump', 'fail', 'R1 の板の衝撃 中央 %.2fG・最大 %.2fG(中央 ≥ %.1fG か 最大 ≥ %.1fG)→ 止めて後輪・取り付けを見る。次はゲイン −0.05' % (
            rb['a3_median'], rb['a3_max'], BOARD_MEDIAN_G, BOARD_MAX_G)))
    else:
        items.append(('r1_bump', 'ok', 'R1 の板の衝撃 中央 %.2fG・最大 %.2fG(%d 通過)' % (rb['a3_median'], rb['a3_max'], rb['passes'])))
    vb = m.get('vbat') or {}
    v0 = vb.get('start_v')
    if v0 is None:
        items.append(('vbat', 'na', '電圧: 記録・説明欄に無い(LiPo チェッカーで測って説明欄に書く)'))
    elif v0 < VBAT_MIN_V:
        items.append(('vbat', 'caution', '走り始めの電圧 %.2f V(< %.2f V)→ パックを替える' % (v0, VBAT_MIN_V)))
    else:
        items.append(('vbat', 'ok', '走り始めの電圧 %.2f V(%s)' % (v0, vb.get('source'))))
    lp = m.get('laps') or {}
    sc_on = bool((m.get('speed_ctrl') or {}).get('speed_ceiling'))
    adv = speed_advice(lp.get('p90_median_mps'), lp.get('sw_max_median_mps'), sc_on)
    items.append(('speed', 'info', adv['text']))
    st = [s for _, s, _ in items]
    code = 2 if 'fail' in st else (3 if 'insufficient' in st else (1 if ('fix' in st or 'caution' in st) else 0))
    head = {0: '本番の設定へ進めます', 1: '直す・注意があります(下の行)', 2: '本番の設定に進まない(下の行)',
            3: '判定に足りません(もう 1 本)'}[code]
    return {'code': code, 'label': CODES[code], 'headline': head, 'speed_advice': adv,
            'items': [{'item': k, 'status': s, 'text': x} for k, s, x in items]}


# ---------------------------------------------------------------- 1 本
def check_log(log, labels_dir=None, raw_dir=None, labels=None):
    """1 本の記録の車の確認。返り値 dict(JSON にできる)"""
    t_start = time.time()
    labels_dir = Path(labels_dir or config.DATA_DIR / 'labels')
    raw_dir = Path(raw_dir or config.RAW_DIR)
    info, lab, src = labels if labels is not None else ss._labels(log, labels_dir, raw_dir)
    res = {'schema': SCHEMA, 'log': log, 'created_at': datetime.now().isoformat(timespec='seconds'),
           'labels': src, 'description': info.get('description', ''), 'no': ss.parse_description(info.get('description', '')).get('ai_no'),
           'run': info.get('run'), 'params': info.get('params_at_start')}
    if lab is None or 'events' not in info:
        res['error'] = info.get('error', 'ラベルがありません')
        res['judge'] = {'code': 3, 'label': CODES[3], 'headline': '判定に足りません(記録が読めない)', 'items': []}
        return res
    L = le.load_log(log, raw_dir)
    ex = {}
    sc = ss.score_log(log, labels_dir, raw_dir, labels=(info, lab, src), loaded=L, extras=ex)
    if sc.get('error'):
        res['error'] = sc['error']
        res['judge'] = {'code': 3, 'label': CODES[3], 'headline': '判定に足りません(%s)' % sc['error'], 'items': []}
        return res
    V, I = L['VEHICLE'], L['IMU']
    t0 = info['t0_us']
    vt = (np.asarray(V['timestamps_us'], np.int64) - t0) / 1e6
    verpm = V['erpm'].astype(np.float64)
    it = (np.asarray(I['timestamps_us'], np.int64) - t0) / 1e6
    gz = I['gyro'][:, 2].astype(np.float64)
    center = float((info.get('ref_us') or {}).get('steer_center') or 1500.0)
    loc_detail = lab['loc_detail'].astype(int)
    if sc.get('loc_relabeled'):
        loc_detail = le.locate(lab['t_s'].astype(np.float64), ex['anchors'], ex['gm_exit'], ex['it'], ex['hd'], ex['gzs']).astype(int)
    res['ai_s'] = sc.get('ai_s')
    res['duration_s'] = sc.get('duration_s')
    res['steer_center_us'] = center
    res['turn'] = turn_gains(lab, it, gz, vt, verpm, center)
    res['s0'] = steer_offset(lab, it, gz, center)
    res['top_speed'] = top_speed(lab, loc_detail)
    res['laps'] = lap_speeds(sc.get('laps') or [], lab, loc_detail)
    res['r1_bump'] = r1_bumps(sc.get('passes') or [], it, I['accel'])
    res['vbat'] = sc.get('vbat')
    ps = [p for p in sc.get('passes') or [] if p.get('ai')]
    de_n = [p for p in ps if p.get('gm_exit')]
    res['reference'] = {'r1_passes_ai': len(ps), 'de_collisions': sum(1 for p in de_n if p.get('de_collision')),
                        'de_passes': len(de_n), 'collisions_ai': (sc.get('collisions') or {}).get('label', {}).get('main'),
                        'strong_collisions_ai': (sc.get('collisions') or {}).get('label', {}).get('strong'),
                        'r1_anchors_rejected': sc.get('r1_anchors_rejected') or []}
    coll = sc.get('collision_events') or []
    after = after_ai_collisions(ex.get('ev') or info.get('events') or {}, lab)
    if after:                   # 2026-10-27 fix3 r2-rec2-4: 人側にした直後の衝突も(あるときだけ。無ければ前と同じ)
        coll = sorted(list(coll) + after, key=lambda c: float(c['t_s']))
    res['collisions'] = ss.collision_summary(coll)
    if after:
        res['collisions']['after_ai'] = [[_r(float(c['t_s']), 2), _r(float(c['g']), 2), ss.place_of(c), c['after_ai_s']]
                                         for c in after]
    res['speed_ctrl'] = sc.get('speed_ctrl')
    res['judge'] = judge(res)
    res['seconds'] = _r(time.time() - t_start, 2)
    return res


def after_ai_collisions(ev, lab, after_s=AFTER_AI_S):
    """AI の区間の終わりから after_s 秒以内の人側の衝突(ラベルの events.collisions。suspect は別の一覧なので入らない)。
    section_scorecard.score_log の collision_events と同じ形 + class 'after_ai'・after_ai_s(AI の最後のフレームからの秒)。
    人が当たる直前に CH3 を人側にした衝突(惰性で当たる。車への衝撃は AI 側と同じ)を落とさないため(r2-rec2-4)"""
    t = np.asarray(lab['t_s'], np.float64)
    ai = np.asarray(lab['ai'], bool)
    if 'dry_run' in lab:
        ai = ai & ~np.asarray(lab['dry_run'], bool)
    ta = t[ai]
    out = []
    if not len(ta):
        return out
    for c in (ev or {}).get('collisions', []):
        if c.get('ai'):
            continue
        tc = float(c['t_s'])
        j = int(np.searchsorted(ta, tc, side='right')) - 1
        if j < 0:
            continue
        dt = tc - float(ta[j])
        if not (0.0 < dt <= after_s):
            continue
        out.append({'t_s': c['t_s'], 'loc': c.get('loc'), 'loc_detail': c.get('loc_detail'), 'g': c['g'],
                    'strong': bool(c.get('strong')), 'outcome': c.get('outcome'), 'class': 'after_ai',
                    'speed_before_mps': c.get('speed_before_mps'), 'after_ai_s': _r(dt, 3)})
    return out


def latest_log(raw_dir=None, day=None):
    """その日(既定は今日)のいちばん新しい AI の記録(dry run でない・AI_CMD がある)"""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    day = (day or datetime.now().strftime('%Y%m%d')).replace('-', '')
    for d in sorted(raw_dir.glob(day + '-*'), reverse=True):
        try:
            meta = json.loads((d / 'log.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        drv = meta.get('drive') or {}
        if not drv or drv.get('dry_run'):
            continue
        if any((s / 'AI_CMD.npz').exists() for s in d.iterdir() if s.is_dir()):
            return d.name
    return None


def newer_unreadable(raw_dir=None, day=None, than=None):
    """その日の記録のうち、latest_log の選んだ記録 than(無ければ None = その日の全部)より新しく、log.json が読めない
    (0 バイト・壊れた・無い)か、AI の記録(dry run でない)なのに AI_CMD が無いもの [(名前, 理由)](新しい順)。
    2026-10-27 fix3 直す 3 回目 r3-r3-rec-1: 電源断の本物の形(10/23 の 20261023-204825 は log.json も 0 バイト)の本を latest_log が
    黙って飛ばし、--latest は 1 本前の記録を出した(落ちた本の 9G が出ず、手順書 7 章の「8G 以上 1 回」が読めない)。
    dry run・人の収集(drive なし)は前と同じく数えない"""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    day = (day or datetime.now().strftime('%Y%m%d')).replace('-', '')
    out = []
    for d in sorted(raw_dir.glob(day + '-*'), reverse=True):
        if than is not None and d.name <= than:
            break
        if not d.is_dir():
            continue
        f = d / 'log.json'
        try:
            if f.stat().st_size == 0:
                out.append((d.name, 'log.json が 0 バイト'))
                continue
            meta = json.loads(f.read_text(encoding='utf-8'))
        except FileNotFoundError:
            out.append((d.name, 'log.json が無い'))
            continue
        except OSError as e:
            out.append((d.name, 'log.json が読めない(%s)' % e.__class__.__name__))
            continue
        except ValueError:
            out.append((d.name, 'log.json が壊れている'))
            continue
        drv = (meta.get('drive') if isinstance(meta, dict) else None) or {}
        if not drv or drv.get('dry_run'):
            continue
        if not any((s / 'AI_CMD.npz').exists() for s in d.iterdir() if s.is_dir()):
            out.append((d.name, 'AI の記録なのに AI_CMD が無い'))
    return out


# 試験の選択肢(drive.py。どれも既定 0・起動のたびに 0)。2026-10-26 fix3 直す 1 回目 r1-rec3-5
OPTION_NAMES = (('rec_back_guard', 'リカバリーの後ろの見張り'), ('rec_hold_keep', '中立止めのあともリカバリーを残す'),
                ('launch_no_kick', '発進の前進を出さない区間は発進の補助も出さない'))


def options_on(params):
    """log.json の params で 1 の試験の選択肢の名前(無い・0・読めない値は OFF)"""
    out = []
    for k, name in OPTION_NAMES:
        try:
            if int(float((params or {}).get(k))) == 1:
                out.append(name)
        except (TypeError, ValueError):
            continue
    return out


def to_text(res):
    """画面に出す文"""
    j = res.get('judge') or {}
    L = ['車の確認: %s(No.%s)' % (res['log'], res.get('no') or '?'),
         '判定: %s — %s(番号 %s)' % (j.get('label'), j.get('headline'), j.get('code'))]
    if res.get('error'):
        L.append('読めません: %s' % res['error'])
        return '\n'.join(L)
    for x in j.get('items', []):
        mark = {'ok': '  ', 'info': '  ', 'na': '  ', 'fix': '! ', 'caution': '! ', 'fail': '× ', 'insufficient': '? '}[x['status']]
        L.append(mark + x['text'])
    tr, so, ts, lp, rb = res['turn'], res['s0'], res['top_speed'], res['laps'], res['r1_bump']
    L += ['', '数(定義は cond_metrics.py・steer_offset.py・r1_bump.py と同じ):',
          '  左右の旋回の効き 左 %s・右 %s(比 %s、%d・%d フレーム)' % (tr['left'], tr['right'], tr['ratio'], tr['n_left'], tr['n_right']),
          '  まっすぐ進む舵 s0 %s µs(%d フレーム)' % (so['s0_us'], so['n']),
          '  直線の最高速 p90 %s m/s(HS %s m/s)' % (ts['mps_p90'], ts['mps_p90_HS']),
          '  周 %d 本: 中央 %s 秒・ベスト %s 秒・周の速さ p90 の中央 %s m/s・SW の最高の中央 %s m/s' % (
              lp['n'], lp['lap_median_s'], lp['lap_best_s'], lp['p90_median_mps'], lp['sw_max_median_mps']),
          '  R1 の板の衝撃(3 軸): 中央 %s G・最大 %s G・2.5G 以上 %d / %d 通過' % (rb['a3_median'], rb['a3_max'], rb['n_ge_2p5'], rb['passes'])]
    vb = res.get('vbat') or {}
    L.append('  電圧: 始め %s・終わり %s・最小 %s V(%s)' % (vb.get('start_v'), vb.get('end_v'), vb.get('min_v'), vb.get('source') or 'なし'))
    rf = res['reference']
    L.append('  参考: DE の衝突 %d / %d 通過・AI の衝突 %s(強い %s)・外した R1 の起点 %d' % (
        rf['de_collisions'], rf['de_passes'], rf['collisions_ai'], rf['strong_collisions_ai'], len(rf['r1_anchors_rejected'])))
    cs = res.get('collisions')
    if cs is not None:          # 1 本ごとの最大 G・8G 以上・場所(手順書 7 章の止める条件)
        txt = ss.collision_text(cs)
        aa = cs.get('after_ai')
        if aa:                  # 2026-10-27 fix3 r2-rec2-4: 人側にした直後の衝突(あるときだけ。無ければ前と同じ文)
            txt = txt.replace('・after_ai', '・人側にした直後') + '。うち人側にした直後(AI の区間の終わりから %.1f 秒以内)%d: %s' % (
                AFTER_AI_S, len(aa), '、'.join('%.1f 秒 %s %.2fG(AI の終わりから %.2f 秒)' % (x[0], x[2], x[1], x[3]) for x in aa))
        L.append('%s 衝突(AI・立て直し中とリカバリー中%s): %s' % ('!' if cs.get('n_ge8') else ' ',
                                                       '・人側にした直後も' if aa else 'も', txt))
    sp = res.get('speed_ctrl') or {}
    if sp.get('ceil_recorded') or sp.get('auto_recorded'):
        c, a, v, f = ss.ctrl_cells({'speed_ctrl': sp, 'vbat': vb})
        L.append('  天井: %s / 自動合わせ: %s / 印: %s' % (c, a, f))
    opts = options_on(res.get('params'))
    if opts:                    # 2026-10-26 fix3 直す 1 回目 r1-rec3-5(どれも OFF なら行を出さない)
        L.append('  試験の選択肢(log.json の走行開始の値): %s が ON' % '・'.join(opts))
    L.append('  (%s 秒)' % res.get('seconds'))
    return '\n'.join(L)


def _json_default(o):
    return ss._json_default(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('log', nargs='?', help='記録の名前(YYYYMMDD-HHMMSS)')
    ap.add_argument('--latest', action='store_true', help='今日のいちばん新しい AI の記録(dry run でない)')
    ap.add_argument('--day', default=None, help='--latest で見る日(既定は今日)')
    ap.add_argument('--raw', default=str(config.RAW_DIR))
    ap.add_argument('--labels', default=str(config.DATA_DIR / 'labels'))
    ap.add_argument('--json', default=None, help='JSON の書き先(既定 data/reports/car_check/<LOG>.json)')
    ap.add_argument('--no-json', action='store_true', help='JSON を書かない')
    args = ap.parse_args(argv)
    log = args.log
    if args.latest:
        log = latest_log(args.raw, args.day)
        bad = newer_unreadable(args.raw, args.day, log)
        if bad:                         # r3-r3-rec-1: 読めない新しい本があれば、1 本前を黙って出さない
            name, why = bad[0]
            print('いちばん新しい記録 %s は読めません(%s。電源断・jetformer-web の落ちで書かれなかった本)。この本の car_check は'
                  'できません: 衝突・8G 以上を記録で確かめられない → 動画・紙で確かめる(手順書 7 章の「いつでも」の行)。%s%s' % (
                      name, why, ('ほかにも読めない新しい記録: %s。' % '・'.join('%s(%s)' % x for x in bad[1:])) if bad[1:] else '',
                      ('1 本前の読める記録 %s を見るときは名前で: python3 tools/car_check.py %s' % (log, log)) if log else
                      'その日の読める AI の記録はほかにありません'), file=sys.stderr)
            return 3
        if not log:
            print('%s の AI の記録(dry run でない)がありません' % (args.day or '今日'), file=sys.stderr)
            return 3
    if not log:
        ap.error('記録の名前か --latest を指定してください')
    if not (Path(args.raw) / log / 'log.json').exists():
        print('記録がありません: %s' % log, file=sys.stderr)
        return 3
    res = check_log(log, args.labels, args.raw)
    print(to_text(res))
    if not args.no_json:
        out = Path(args.json) if args.json else config.DATA_DIR / 'reports' / 'car_check' / (log + '.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + '.tmp')
        tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_json_default), encoding='utf-8')
        os.replace(tmp, out)
        print('書き出し:', out)
    return int(res['judge']['code'])


if __name__ == '__main__':
    sys.exit(main())
