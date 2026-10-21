#!/usr/bin/env python3
"""ラベル付け(ロードマップ S8-1): 生ログから「出来事の一覧」と「フレームごとのラベル」を作る

    python tools/label_events.py                      # すべてのログ → data/labels/<ログ>.json・<ログ>.npz・summary.json
    python tools/label_events.py 20261010-173445      # 指定したログだけ(summary.json はそのログの行だけ更新)
    python tools/label_events.py --sheets             # 検証の食い違いのコンタクトシートも作る(data/labels/validation/)

読み取りだけで、生ログは変えない。GPU は使わない(numpy だけ。--sheets のときだけ cv2 で画像を読む)。
summary.json には、目視で確かめた衝突の一覧(collision_map.md)・009/010/101 の衝突の場所・interventions.json の
交代の分類との照合結果も入れる。

出来事(<ログ>.json の events):
  collisions   衝突。水平の加速度(重力・静止時のずれを除く)が 2.5G 以上。0.6秒以内は1件にまとめる。
               前後 50ms に 1G を超えるサンプルが1つしか無いものはセンサーの異常として collision_suspects へ。
               止まったか(stop)・走り続けたか(glance: 0.5秒後も eRPM < −1500)、当たった向き、場所(下記)を付ける
  bumps        水平は 2.5G 未満だが、重力を引いた加速度の大きさが 2.5G 以上の衝撃(上下が主)。R1 のレールの切れ目
               (黒いスポンジ)や緑マットの入口で、走り続けたまま起きるものが多い。人のきれいな周にもあるので、
               既定では衝突に数えない(--bumps-as-collisions で報酬に入れる)
  takeovers    人の交代(AI → 人)。research/offline_rl_src/interventions.py と同じ規則で分類する
                 after_collision: 交代の前 3秒以内に衝突 / stuck: 前 1秒ずっと |eRPM| < 300
                 preemptive: 前 0.3秒に |eRPM| > 3000(走行中の先回り)/ operational: それ以外(終了・置き直しなど)
               基板の再起動の直後(1秒以内)の交代は、衝突が無ければ operational(cause='reboot')にする
               failure = after_collision・stuck・preemptive(ロードマップ 4.3 の「失敗による交代」)
  handbacks    人 → AI に戻した時刻
  recoveries   自動リカバリー(assist=2)の区間と結果(returned / giveup / takeover / log_end)
  esc_assists  後退の受け付け(assist=1)の区間 / giveups: リカバリー中止(assist=3)の区間
  guard_warnings ToFガードの警告(guard≠0)の区間。その後 1秒以内に衝突したか
  holds        中立で止めた区間(AI_CMD.hold)
  board_reboots コントローラ基板の再起動(VEHICLE.board_t_us が戻った)と、その前の車両データの途切れ
  laps         周回(jetformer/laps.py の LapCounter。衝撃は上と同じ水平の加速度を渡す)

場所の印(loc): R1 / GM / DE / LH / HS / other(奥の左スイーパー SW)/ unknown。loc_detail はもっと細かい区分
  (GM_in / GM_out / SW)。ジャイロの向きだけで決める(data/reports/ai_runs_20261010/collision_map.md 1.3 と
  overnight_20261011/runs/src/course.py の規則): 1周に1回だけある大きな右ヘアピンの入口を R1 の起点にし、
  起点から 1秒以内 = R1、R1 の後に向きが −25° まで戻った時刻(緑マット出口)の手前 = GM、±0.3秒 = GM(出口)、
  その 2秒後まで = DE、その後は起点からの向き(<150° = SW、<215° = HS、それ以上 = LH)。次の起点の 1.2秒前から = LH。
  起点が見つからない区間(止まっている・人が置き直した・15秒以上起点が無い)は unknown。

フレームごと(<ログ>.npz): AI_CMD のフレーム(30fps)。AI_CMD が無い人のログはカメラのフレーム。
  ai           そのフレームで車を操作していたのが AI か(CH3。基板の再起動直後の無効な値と 0.1秒未満の揺れは除く)
  valid_action 行動ラベルを学習に使ってよいか。AI のフレーム: hold・assist≠0・ガードが効いた・dry_run・基板の再起動の
               前後 0.5秒・交代/戻しの前後 0.3秒・車両データが古い・ゲイン 0 を除く。人のフレーム: スティックが受信中で、
               再起動・交代の前後でないこと。外した理由は mask_bits(MASK_BITS のビット和)
  progress_m   進み [m] = −eRPM × 0.5e-3 × Δt(100Hz の車両データを積分。後退は負)。progress_fwd_m は max(0, ·)
  reward       r = progress_fwd_m − 1.0·collision − 2.0·failure_takeover − 0.5·recovery_start + penalty_carried
               (ロードマップ 4.3、m)。衝突はその時刻を含むフレーム、失敗による交代は交代直前の AI のフレーム、
               リカバリーは始まったフレームに付ける
  penalty_carried 区間の外(基板の再起動の前後 0.5秒・hold など)のフレームに付いた衝突・失敗による交代の罰を、直前の
               AI の区間の最後のフレームへ移した量(移した先は負、元のフレームは正。合計は 0)。強い衝突で基板が
               再起動すると、衝突と交代がマスクに入り、罰が失われていた(155820・181504)。移す条件は carry_penalties
  segment_id   リターンを計算する区間(同じ操作者で、マスクや 0.2秒を超える途切れが無い連続フレーム)。-1 は区間外
  ret          γ=0.99 のモンテカルロのリターン。区間の終わりで打ち切る(その先を 0 とする)。
               打ち切りを後で扱えるように segment_end・end_code(END_CODES)・time_to_end_s・steps_to_end を残す
  action_label 行動ラベル(ゲイン前の µs)。AI: out をそのフレームのゲイン・トリムで割り戻す(drive.apply_gains の逆。
               速度のゲインは前進側だけ。2026-10-10 18:3x より前の走行はブレーキ・後退側にも掛かっていたので、
               ログごとに raw・out から確かめて使い分ける → gain_rule)。人: スティックの値(VEHICLE.pwm)
  action_exec  実際に車に出た値(AI: out、人: スティック)。action_sat: out が上下限(1000/2000µs)に張り付いた
  label_err    割り戻した値 − raw(assist=0・hold なし・ガードなし・張り付きなしのフレームだけ。検算用)
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer import config  # noqa: E402
from jetformer.laps import LapCounter  # noqa: E402

VERSION = 1
G = 9.80665
KV = 0.5e-3                     # m/s / eRPM(eRPM は負 = 前進。65mm タイヤ)
FPS = 30.0
MIN_US, MAX_US = float(config.DRIVE['min_us']), float(config.DRIVE['max_us'])
SENSOR_MAX_AGE_S = config.DRIVE['max_sensor_age_ms'] / 1000.0   # 走行と同じ: これより古い車両データは無効
GAP_S = 0.2                     # これより間の空いたフレームは区間を切る(ロードマップ 4.3)

# 衝突
IMPACT_G = 2.5
IMPACT_MERGE_S = 0.6
SPIKE_WINDOW_S, SPIKE_NEIGHBOR_G = 0.05, 1.0
STRONG_G = 4.0
CLIP_G = 7.8                    # 加速度センサーは ±8G で頭打ち
GLANCE_ERPM = -1500             # 0.5秒後もこれより速く前へ走っていれば「かすった」(glance)
NEAR_COLLISION_S = (1.0, 2.0)   # bc_mask で外す範囲(学習と同じ 1秒前〜2秒後)

# 交代の分類(interventions.py と同じ)
TK_COLLISION_S = 3.0
TK_STUCK_S, TK_STUCK_ERPM = 1.0, 300
TK_FAST_S, TK_FAST_ERPM = 0.3, 3000
TK_REBOOT_S = 1.0
SELECT_MIN_S = 0.1              # これより短い CH3 の切り替わりは揺れ(基板の再起動直後に 10〜30ms だけ人側になる)

# マスク
HANDOVER_MASK_S = 0.3
REBOOT_MASK_S = 0.5

# 報酬
REWARD_W = {'collision': 1.0, 'failure_takeover': 2.0, 'recovery_start': 0.5}
GAMMA = 0.99
CARRY_S = REBOOT_MASK_S + 0.1   # 区間の外に落ちた衝突の罰を、直前の区間の最後のフレームへ移せる間隔(再起動の前のマスク + 余裕)

TAKEOVER_CLASSES = ['operational', 'after_collision', 'stuck', 'preemptive']
FAILURE_CLASSES = ('after_collision', 'stuck', 'preemptive')
END_CODES = {0: 'none', 1: 'failure_takeover', 2: 'operational_takeover', 3: 'handback', 4: 'log_end', 5: 'truncated'}
MASK_BITS = {'hold': 1, 'assist': 2, 'guard': 4, 'dry_run': 8, 'vehicle': 16, 'reboot': 32, 'handover': 64,
             'no_label': 128}
LOC_NAMES = ['unknown', 'R1', 'GM', 'DE', 'other', 'HS', 'LH']
LOC_DETAIL = ['?', 'R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH']
_DETAIL_TO_LOC = np.array([0, 1, 2, 2, 3, 4, 5, 6], np.int8)
MAX_LAP_S = 15.0                # R1 の起点からこれより経ったら場所は分からない(止まった・置き直した)
GAIN_RULE_SINCE = datetime(2026, 10, 10, 18, 33)   # f198431(速度のゲインを前進側だけに)。証拠が無いときだけ使う

SENSORS = ('AI_CMD', 'VEHICLE', 'IMU', 'CAM_FRONT')


# ---------------------------------------------------------------- 読み込み
def _load_npz(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:      # 書きかけ・空のファイル(電源断)
        return e


def _concat(parts):
    """シーンごとの dict をつなぐ(全シーンにある配列だけ)。時刻順に並べ直す"""
    if not parts:
        return None
    keys = set.intersection(*[set(k for k, v in p.items() if np.ndim(v) >= 1 and len(v) == len(p['timestamps_us']))
                              for p in parts])
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    order = np.argsort(out['timestamps_us'], kind='stable')
    if np.any(order != np.arange(len(order))):
        out = {k: v[order] for k, v in out.items()}
    return out


def load_log(log, raw_dir=None):
    """ログの AI_CMD・VEHICLE・IMU・CAM_FRONT(時刻だけ)をつなぐ。壊れた・無いファイルは problems に残して飛ばす"""
    root = Path(raw_dir or config.RAW_DIR) / log
    with open(root / 'log.json', encoding='utf-8') as f:
        meta = json.load(f)
    scenes = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(log + '-'))
    parts = {n: [] for n in SENSORS}
    problems = []
    for si, s in enumerate(scenes):
        for n in SENSORS:
            p = root / s / (n + '.npz')
            if not p.exists():
                if n != 'AI_CMD':
                    problems.append('%s/%s: ありません' % (s, n))
                continue
            d = _load_npz(p)
            if isinstance(d, Exception) or 'timestamps_us' not in d:
                problems.append('%s/%s: 読めません(%s)' % (s, n, type(d).__name__ if isinstance(d, Exception) else '形式'))
                continue
            if not len(d['timestamps_us']):
                continue
            if n == 'CAM_FRONT':
                d = {'timestamps_us': d['timestamps_us']}
            d['scene'] = np.full(len(d['timestamps_us']), si, np.int16)
            parts[n].append(d)
    data = {n: _concat(parts[n]) for n in SENSORS}
    return {'log': log, 'meta': meta, 'scenes': scenes, 'problems': problems, **data}


# ---------------------------------------------------------------- 小道具
def runs(mask):
    """True が続く区間 [(i0, i1)](i1 を含む)"""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()))


def latest_index(src_t, query_t):
    """各 query_t に対し src_t <= query_t となる最新のインデックス(無ければ -1)"""
    return np.searchsorted(src_t, query_t, side='right') - 1


def clean_select(t_s, select, valid_ch3, min_s=SELECT_MIN_S):
    """CH3(人/AI)の値を整える。CH3 が無効なサンプル(基板の再起動直後・送信機の電源断)は直前の有効な値を保ち、
    min_s より短い切り替わりは揺れとして前の値に戻す"""
    s = np.asarray(select, bool).copy()
    v = np.asarray(valid_ch3, bool)
    if not len(s):
        return s
    if v.any():
        idx = np.maximum.accumulate(np.where(v, np.arange(len(s)), -1))
        s = np.where(idx >= 0, s[np.clip(idx, 0, None)], s[np.argmax(v)])
    while True:
        rr = runs(s) + [(a, b) for a, b in runs(~s)]
        short = [(a, b) for a, b in rr if 0 < a and b < len(s) - 1 and t_s[b + 1] - t_s[a] < min_s]
        if not short:
            return s
        a, b = min(short, key=lambda x: t_s[x[1] + 1] - t_s[x[0]])
        s[a:b + 1] = s[a - 1]


def horizontal_g(accel, still=None):
    """水平(x=右・y=前)の加速度 [G]。止まっているときの平均のずれ(取り付けの傾き)を引く"""
    a = np.asarray(accel, np.float64)
    bias = np.zeros(2)
    if still is not None and np.sum(still) >= 50:
        bias = np.median(a[still, :2], axis=0)
    return np.hypot(a[:, 0] - bias[0], a[:, 1] - bias[1]) / G, bias


def dynamic_g(accel, still=None):
    """重力(止まっているときの加速度の中央値)を引いた加速度の大きさ [G]。上下の衝撃も含む。
    collision_map の ||a|−1G| は水平の衝撃を小さく見積もる(水平 2.5G でも 1.7G)ので、こちらを使う"""
    a = np.asarray(accel, np.float64)
    bias = np.array([0.0, 0.0, G])
    if still is not None and np.sum(still) >= 50:
        bias = np.median(a[still], axis=0)
    return np.linalg.norm(a - bias, axis=1) / G


def detect_collisions(t_s, hg, thr=IMPACT_G, merge_s=IMPACT_MERGE_S, spike_s=SPIKE_WINDOW_S,
                      neighbor_g=SPIKE_NEIGHBOR_G):
    """衝突の候補: hg >= thr のサンプルを、最初のサンプルから merge_s 以内なら1件にまとめる。
    最初のサンプルとピークのどちらの前後 spike_s にも neighbor_g を超えるサンプルが他に無ければ suspect(1サンプルだけのノイズ)"""
    t_s = np.asarray(t_s, np.float64)
    hg = np.asarray(hg, np.float64)
    events = []
    for i in np.nonzero(hg >= thr)[0]:
        if events and t_s[i] - events[-1]['t'] <= merge_s:
            e = events[-1]
            e['n'] += 1
            if hg[i] > e['g']:
                e['g'], e['i_peak'] = float(hg[i]), int(i)
            continue
        events.append({'i': int(i), 'i_peak': int(i), 't': float(t_s[i]), 'g': float(hg[i]), 'n': 1})
    for e in events:
        alone = []
        for i in {e['i'], e['i_peak']}:
            near = (np.abs(t_s - t_s[i]) <= spike_s) & (hg > neighbor_g)
            alone.append(int(near.sum()) <= 1)
        e['suspect'] = bool(all(alone))
    return events


def board_reboots(vt_us, board_t_us):
    """コントローラ基板の再起動: board_t_us(基板の起動からの時刻)が戻ったところ。
    [(最後の正常なサンプルの時刻, 再起動後の最初のサンプルの時刻)] µs"""
    bt = np.asarray(board_t_us, np.int64)
    k = np.nonzero(np.diff(bt) < 0)[0]
    return [(int(vt_us[i]), int(vt_us[i + 1])) for i in k]


def classify_takeover(t, collision_t, vt, erpm, reboot_t=()):
    """交代(AI → 人)の分類。t・collision_t・vt・reboot_t は秒。戻り値 (分類, 詳細 dict)。
    interventions.py と同じ順に判定する: 衝突後 → スタック → 走行中 → それ以外(運用上)。
    基板の再起動の直後(衝突なし)は、止まった原因が電源なので運用上の交代にする"""
    collision_t = np.asarray(collision_t, np.float64)
    prev = collision_t[(collision_t <= t) & (collision_t > t - TK_COLLISION_S)]
    w = (vt > t - TK_STUCK_S) & (vt <= t)
    stopped = bool(np.all(np.abs(erpm[w]) < TK_STUCK_ERPM)) if w.any() else False
    w2 = (vt > t - TK_FAST_S) & (vt <= t)
    fast = float(np.max(np.abs(erpm[w2]))) if w2.any() else 0.0
    rb = [r for r in reboot_t if t - TK_REBOOT_S <= r <= t + 0.05]
    info = {'dt_since_collision_s': round(float(t - prev[-1]), 3) if len(prev) else None,
            'stopped_1s': stopped, 'erpm_abs_max_0.3s': int(fast), 'cause': None}
    if len(prev):
        return 'after_collision', info
    if rb:
        info['cause'] = 'reboot'
        return 'operational', info
    if stopped:
        return 'stuck', info
    if fast > TK_FAST_ERPM:
        return 'preemptive', info
    return 'operational', info


# ---------------------------------------------------------------- ゲイン
def apply_gains_vec(raw, gains, trim, ref, forward_only=True, lo=MIN_US, hi=MAX_US):
    """drive.apply_gains と同じ計算(配列版)。forward_only=False は 2026-10-10 18:3x より前の規則
    (速度のゲインをブレーキ・後退側にも掛けていた)"""
    raw = np.asarray(raw, np.float64)
    g = np.asarray(gains, np.float64)
    steer = ref[0] + np.asarray(trim, np.float64) + g[:, 0] * (raw[:, 0] - ref[0])
    dt = raw[:, 1] - ref[1]
    scaled = (dt < 0) if forward_only else np.ones(len(dt), bool)
    thr = ref[1] + np.where(scaled, g[:, 1] * dt, dt)
    return np.stack([np.clip(steer, lo, hi), np.clip(thr, lo, hi)], axis=1)


def invert_gains(out, gains, trim, ref, forward_only=True):
    """apply_gains の逆: 出した値 out → ゲイン前の値。ゲインが 0 のチャンネルは NaN。
    上下限に張り付いた値は、その限界に届く値に戻るだけ(本当はもっと切りたかった量は分からない)"""
    out = np.asarray(out, np.float64)
    g = np.asarray(gains, np.float64)
    trim = np.asarray(trim, np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        steer = np.where(g[:, 0] > 0, (out[:, 0] - ref[0] - trim) / g[:, 0] + ref[0], np.nan)
        dt = out[:, 1] - ref[1]
        scaled = (dt < 0) if forward_only else np.ones(len(dt), bool)
        thr = np.where(scaled, np.where(g[:, 1] > 0, ref[1] + dt / g[:, 1], np.nan), out[:, 1])
    return np.stack([steer, thr], axis=1)


def saturated(out, lo=MIN_US, hi=MAX_US, tol=0.5):
    out = np.asarray(out, np.float64)
    return (out <= lo + tol) | (out >= hi - tol)


def detect_gain_rule(raw, out, gains, trim, ref, usable, log_time=None):
    """速度のゲインがブレーキ・後退側にも掛かっていたか(古い規則)を、記録の raw と out から確かめる。
    ブレーキ側(raw > 中立 + 5µs)・速度のゲイン ≠ 1・張り付きなし のフレームで、どちらの規則が out に合うか数える。
    戻り値 {'rule': 'forward_only'|'both', 'source': 'evidence'|'irrelevant'|'time', 'n_both', 'n_forward'}"""
    raw = np.asarray(raw, np.float64)
    out = np.asarray(out, np.float64)
    g = np.asarray(gains, np.float64)
    ok = (np.asarray(usable, bool) & np.isfinite(raw).all(axis=1) & np.isfinite(out).all(axis=1)
          & ~saturated(out)[:, 1])
    m = ok & (raw[:, 1] - ref[1] > 5.0) & (np.abs(g[:, 1] - 1.0) > 0.05)
    pb = apply_gains_vec(raw, g, trim, ref, forward_only=False)[:, 1]
    pf = apply_gains_vec(raw, g, trim, ref, forward_only=True)[:, 1]
    eb = np.abs(out[:, 1] - pb) < 0.5
    ef = np.abs(out[:, 1] - pf) < 0.5
    nb, nf = int((m & eb & ~ef).sum()), int((m & ef & ~eb).sum())
    res = {'n_both': nb, 'n_forward': nf, 'n_checked': int(m.sum())}
    if nb + nf > 0:
        res.update(rule='both' if nb > nf else 'forward_only', source='evidence')
    elif not np.any(np.abs(g[ok, 1] - 1.0) > 0.05):
        res.update(rule='forward_only', source='irrelevant')     # 速度のゲインがずっと 1: どちらでも同じ
    else:
        old = log_time is not None and log_time < GAIN_RULE_SINCE
        res.update(rule='both' if old else 'forward_only', source='time')
    return res


# ---------------------------------------------------------------- 場所(ジャイロの向き)
def heading_deg(t_s, gz, still=None):
    """ジャイロ Z(deg/s、左 +)を積分した向き [deg]。止まっているときの中央値をずれとして引く。0.2秒を超える途切れは積分しない"""
    gz = np.asarray(gz, np.float64)
    bias = float(np.median(gz[still])) if still is not None and np.sum(still) > 50 else -0.09
    dt = np.diff(t_s, prepend=t_s[0])
    dt[(dt <= 0) | (dt > GAP_S)] = 0.0
    return np.cumsum((gz - bias) * dt), gz - bias


def _right_turns(t_s, gzs, thr=-170.0):
    """平滑したヨーが thr を下回る右旋回 [(開始, 終了, 向きの変化, 最小ヨー)]。前後は −60deg/s まで広げる"""
    m = gzs < thr
    out, i, n = [], 0, len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (m[j + 1] or (j + 8 < n and m[j + 1:j + 8].any())):
            j += 1
        a = i
        while a > 0 and gzs[a - 1] < -60:
            a -= 1
        b = j
        while b + 1 < n and gzs[b + 1] < -60:
            b += 1
        out.append((a, b, float(gzs[i:j + 1].min())))
        i = b + 1
    return out


def course_anchors(t_s, hd, gz_unbiased, vt_s, erpm):
    """R1 の起点(1周に1回だけある、左ヘアピンの直後の大きな右ヘアピンの入口)と、その後の緑マット出口の時刻 [秒]、
    平滑したヨーレート [deg/s](11サンプルの移動平均)"""
    gzs = np.convolve(gz_unbiased, np.ones(11) / 11, 'same') if len(t_s) >= 11 else np.asarray(gz_unbiased, float)
    if len(t_s) < 20:
        return np.zeros(0), np.zeros(0), gzs
    anchors = []
    for a, b, mn in _right_turns(t_s, gzs):
        ta = t_s[a]
        dh = hd[b] - hd[a]
        h_prev = hd[a] - np.interp(ta - 2.0, t_s, hd)
        w = (vt_s > ta - 0.5) & (vt_s < ta)
        if not w.any() or np.median(np.abs(erpm[w])) < 1500:
            continue
        if (dh < -90 or (dh < -40 and h_prev > 90)) and h_prev > 60:
            if anchors and ta - anchors[-1] < 5.0:
                continue
            anchors.append(float(ta))
    anchors = np.array(anchors)
    gm = np.full(len(anchors), np.inf)
    for k, a in enumerate(anchors):        # 止まって人が置き直したときも続きを追えるように、次の起点まで探す
        end = min(anchors[k + 1] if k + 1 < len(anchors) else np.inf, a + MAX_LAP_S)
        m = (t_s > a + 0.3) & (t_s < end)
        psi = hd[m] - np.interp(a, t_s, hd)
        if not len(psi) or psi.min() > -60:            # 右へ回っていない(R1 で止まった)
            continue
        i_min = int(np.argmin(psi))
        back = psi[i_min:] - psi[i_min]                # R1 のいちばん右から、左へ戻った量
        # 緑マット出口: 向きが −25° まで戻った、または左へ 60° 以上戻ってから左旋回が終わった(速い周は右向きのまま出る)
        j = np.nonzero((psi[i_min:] > -25) | ((back >= 60) & (gzs[m][i_min:] < 30)))[0]
        if len(j):
            gm[k] = float(t_s[m][i_min + j[0]])
    return anchors, gm, gzs


def locate(times, anchors, gm_exit, t_s, hd, gzs=None):
    """時刻ごとの場所の細かい区分(LOC_DETAIL の番号)。collision_map.md 1.3 / course.py の規則。

    直前の R1 の起点から追えないとき(緑マット出口が見つからない・15秒以上経った・起点がまだ無い)は:
      1. 次の起点が 12秒以内にあれば、そこまでの向きの変化で決める(1周で +360°: 次の起点の −150° までが LH、
         −180° までが HS、−335° までが SW、−410° までの直進が DE、左旋回中なら GM)
      2. それも無ければ、いちばん近い起点からの向き(1周 360° で割った余り)と、ヨーレートの形で決める
         (R1 の右旋回の後 4.5秒以内なら R1 / GM / DE、そうでなければ SW / HS / LH。迷うものは unknown)"""
    T = np.asarray(times, np.float64)
    det = np.zeros(len(T), np.int8)
    if not len(anchors) or not len(T):
        return det
    A = np.asarray(anchors)
    k = np.searchsorted(A, T + 0.05, side='right') - 1
    kc = np.clip(k, 0, len(A) - 1)
    has = k >= 0
    a = A[kc]
    tg = np.asarray(gm_exit)[kc]
    nk = k + 1
    t_next = np.where(nk < len(A), A[np.clip(nk, 0, len(A) - 1)], np.inf)
    dt = T - a
    h_now = np.interp(T, t_s, hd)
    psi = h_now - np.interp(a, t_s, hd)
    psi_n = h_now - np.interp(np.where(np.isfinite(t_next), t_next, T), t_s, hd)
    fin = np.isfinite(tg)
    lost = ~has | (dt > MAX_LAP_S) | (~fin & (dt >= 3.5))
    back = lost & (t_next - T <= 12.0)
    shape = lost & ~back
    # 形で決めるための値: いちばん近い起点からの向き(0〜360)、今のヨー、直近 4.5秒の右旋回
    near_a = A[np.argmin(np.abs(T[:, None] - A[None, :]), axis=1)]
    h_mod = (h_now - np.interp(near_a, t_s, hd)) % 360.0
    g_now = np.zeros(len(T))
    r_recent = np.zeros(len(T), bool)
    if gzs is not None and len(gzs) == len(t_s) and lost.any():
        g_now = np.interp(T - 0.3, t_s, gzs)          # 衝突そのもので回った分を避けて、少し前のヨーを見る
        for j in np.nonzero(shape)[0]:
            x, y = np.searchsorted(t_s, T[j] - 4.5), np.searchsorted(t_s, T[j], side='right')
            r_recent[j] = y > x and gzs[x:y].min() < -120
    around0 = (h_mod >= 330) | (h_mod < 40)
    rules = [
        (t_next - T < 1.2, 7),                       # 次の R1 の直前 = 黒カーテン前の左ヘアピン
        (back & (psi_n > -150), 7),
        (back & (psi_n > -180), 6),
        (back & (psi_n > -335), 5),
        (back & (psi_n > -410) & (g_now <= 60), 4),  # 速い周は緑マットを右向きのまま出る(次の起点の −370〜−400°)
        (back & (psi_n > -500) & (g_now > 60), 2),
        (back, 0),
        (shape & (g_now < -60) & ((h_mod >= 200) | (h_mod < 20)), 1),    # 右へ回っている = R1
        (shape & r_recent & (g_now > 60), 2),                             # R1 の後の左旋回 = GM
        (shape & r_recent & around0, 4),                                  # R1 の後でまっすぐ = DE
        (shape & ~r_recent & (h_mod >= 40) & (h_mod < 175), 5),
        (shape & ~r_recent & (h_mod >= 175) & (h_mod < 215) & (np.abs(g_now) < 60), 6),
        (shape & ~r_recent & (h_mod >= 215) & (h_mod < 330), 7),
        (shape & ~r_recent & (h_mod >= 200) & (g_now > 60), 7),           # R1 の前の左旋回 = LH
        (lost, 0),
        (dt < 1.0, 1),
        (~fin, 2),                                   # 起点から 3.5秒以内(緑マット出口がまだ無い)
        ((T < tg - 0.3) & (T >= tg - 3.0), 2),
        (T < tg - 0.3, 1),                           # 緑マット出口の 3秒より前 = まだ R1 のあたり(止まった・置き直した)
        (np.abs(T - tg) <= 0.3, 3),
        (T - tg <= 2.0, 4),
        (psi < 150, 5),
        (psi < 215, 6),
        (np.ones(len(T), bool), 7),
    ]
    done = np.zeros(len(T), bool)
    for cond, code in rules:
        m = cond & ~done
        det[m] = code
        done |= m
    return det


# ---------------------------------------------------------------- 区間とリターン
def make_segments(ai, ok, t_s, gap_s=GAP_S):
    """リターンを計算する区間の番号(ok でないフレームは -1)。操作者が変わる・ok でない・gap_s を超える途切れで切る"""
    ai = np.asarray(ai, bool)
    ok = np.asarray(ok, bool)
    n = len(ai)
    if not n:
        return np.zeros(0, np.int32)
    start = ok.copy()
    if n > 1:
        start[1:] = ok[1:] & (~ok[:-1] | (ai[1:] != ai[:-1]) | (np.diff(t_s) > gap_s))
    seg = np.cumsum(start) - 1
    return np.where(ok, seg, -1).astype(np.int32)


def mc_returns(r, seg, gamma=GAMMA):
    """区間ごとのモンテカルロのリターン(区間の終わりで打ち切り)。区間外は NaN"""
    r = np.asarray(r, np.float64)
    seg = np.asarray(seg)
    ret = np.full(len(r), np.nan)
    acc = 0.0
    for i in range(len(r) - 1, -1, -1):
        if seg[i] < 0:
            continue
        if i == len(r) - 1 or seg[i + 1] != seg[i]:
            acc = 0.0
        acc = r[i] + gamma * acc
        ret[i] = acc
    return ret


def carry_penalties(t, seg, ai, collision, failure_tk, max_s=CARRY_S, weights=REWARD_W):
    """区間の外のフレームに付いた衝突・失敗による交代の罰を、直前の AI の区間の最後のフレーム j へ移す。
    強い衝突で基板が再起動すると、衝突(と、その後の交代)が再起動の前後のマスクに入り、区間が「打ち切り」で終わって
    罰が失われる。そのままだと、いちばん強い衝突の手前が「良い」状態に見える。
    移す条件: j から罰のフレームまでずっと AI(交代・戻しをまたがない)で、
      衝突: j から max_s 以内(再起動の前のマスク 0.5秒の中)
      失敗による交代: j から max_s 以内か、同じ j へ衝突を移したとき(衝突 → 再起動 → 交代)
    それより離れたもの(例: 再起動で止まってから惰性で当たった)は、AI の操作の結果とは言えないので移さない。
    戻り値 (penalty_carried [n](移した先は −重み、元のフレームは +重み), 失敗で終わる区間の最後のフレーム [n] bool,
    移した一覧 [{'kind', 'from_frame', 'to_frame', 'dt_s'}])"""
    t = np.asarray(t, np.float64)
    seg = np.asarray(seg)
    ai = np.asarray(ai, bool)
    n = len(t)
    carried = np.zeros(n)
    fail_end = np.zeros(n, bool)
    moves = []
    in_seg = np.nonzero(seg >= 0)[0]
    if not len(in_seg):
        return carried, fail_end, moves
    coll_into = set()
    for kind, flags in (('collision', collision), ('failure_takeover', failure_tk)):
        w = weights[kind]
        for k in np.nonzero(np.asarray(flags, bool) & (seg < 0))[0]:
            p = int(np.searchsorted(in_seg, k)) - 1
            if p < 0:
                continue
            j = int(in_seg[p])                     # 直前の区間の最後のフレーム(j+1〜k は区間の外)
            if not ai[j:k + 1].all():
                continue
            dt = float(t[k] - t[j])
            if not (dt <= max_s or (kind == 'failure_takeover' and j in coll_into)):
                continue
            carried[k] += w
            carried[j] -= w
            if kind == 'collision':
                coll_into.add(j)
            else:
                fail_end[j] = True
            moves.append({'kind': kind, 'from_frame': int(k), 'to_frame': j, 'dt_s': round(dt, 3)})
    return carried, fail_end, moves


def cumulative_distance(vt_s, erpm, gap_s=GAP_S):
    """車両データ(100Hz)の eRPM を積分した距離 [m](符号つき、前進だけ)。gap_s を超える途切れは積分しない"""
    v = -np.asarray(erpm, np.float64) * KV
    if len(v) < 2:
        return np.zeros(len(v)), np.zeros(len(v))
    dt = np.diff(vt_s)
    ok = (dt > 0) & (dt <= gap_s)
    d = np.where(ok, 0.5 * (v[1:] + v[:-1]) * dt, 0.0)
    vf = np.maximum(v, 0.0)
    df = np.where(ok, 0.5 * (vf[1:] + vf[:-1]) * dt, 0.0)
    return np.concatenate([[0.0], np.cumsum(d)]), np.concatenate([[0.0], np.cumsum(df)])


# ---------------------------------------------------------------- 1本のログ
def _r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def _log_time(log):
    try:
        return datetime.strptime(log[:15], '%Y%m%d-%H%M%S')
    except ValueError:
        return None


def label_log(log, raw_dir=None, impact_g=IMPACT_G, gamma=GAMMA, signed_progress=False, bumps_as_collisions=False):
    """1本のログのラベルを作る。戻り値 (出来事などの dict, フレームごとの配列の dict)。フレームが無ければ (dict, None)。
    bumps_as_collisions: 上下の衝撃(bump)も衝突として報酬・マスクに入れる"""
    L = load_log(log, raw_dir)
    meta = L['meta']
    drive = meta.get('drive') or {}
    ref = (float(drive.get('steer_center_us') or 1500.0), float(drive.get('throttle_neutral_us') or 1500.0))
    A, V, I, C = L['AI_CMD'], L['VEHICLE'], L['IMU'], L['CAM_FRONT']
    info = {'log': log, 'version': VERSION, 'description': meta.get('description', ''), 'tags': meta.get('tags', []),
            'mode_id': meta.get('mode_id'), 'ai_log': A is not None, 'run': drive.get('run'),
            'params_at_start': drive.get('params'), 'ref_us': {'steer_center': ref[0], 'throttle_neutral': ref[1]},
            'scenes': len(L['scenes']), 'problems': L['problems']}
    if A is not None:
        t_us = A['timestamps_us'].astype(np.int64)
        info['frame_source'] = 'AI_CMD'
    elif C is not None:
        t_us = C['timestamps_us'].astype(np.int64)
        info['frame_source'] = 'CAM_FRONT'
    else:
        info['error'] = 'フレーム(AI_CMD・カメラ)がありません'
        return info, None
    n = len(t_us)
    t0 = int(t_us[0])
    sec = lambda x: (np.asarray(x, np.int64) - t0) / 1e6      # noqa: E731
    t = sec(t_us)
    info.update(t0_us=t0, frames=n, duration_s=round(float(t[-1]), 2),
                vehicle_t0_us=int(V['timestamps_us'][0]) if V is not None else None)
    frame_scene = A['scene'] if A is not None else C['scene']

    # ---- 車両データ: CH3(人/AI)・スティック・eRPM・基板の再起動
    if V is not None:
        vt = sec(V['timestamps_us'])
        valid = V['valid'].astype(bool)
        vsel = clean_select(vt, V['select'], valid[:, 2])
        verpm = V['erpm'].astype(np.float64)
        reboots = board_reboots(V['timestamps_us'], V['board_t_us']) if 'board_t_us' in V else []
        iv = latest_index(vt, t)
        ivc = np.clip(iv, 0, len(vt) - 1)
        fresh = (iv >= 0) & (t - vt[ivc] <= SENSOR_MAX_AGE_S)
        sel_ok = fresh & valid[ivc, 2]
        stick_ok = fresh & valid[ivc, 0] & valid[ivc, 1]
        human_pwm = np.where(stick_ok[:, None], V['pwm'][ivc, :2].astype(np.float64), np.nan)
        erpm_f = np.where(fresh, verpm[ivc], np.nan)
        D, Df = cumulative_distance(vt, verpm)
    else:
        vt = np.zeros(0)
        vsel = np.zeros(0, bool)
        verpm = np.zeros(0)
        reboots = []
        fresh = sel_ok = stick_ok = np.zeros(n, bool)
        human_pwm = np.full((n, 2), np.nan)
        erpm_f = np.full(n, np.nan)
        D = Df = np.zeros(0)
    reboot_s = [(float(sec(a)), float(sec(b))) for a, b in reboots]

    # ---- AI_CMD の項目(古いログに無いものは既定値)
    def a_get(k, default, dtype):
        if A is not None and k in A:
            return A[k].astype(dtype)
        return np.full((n,) + np.shape(default), default, dtype)
    has_assist = A is not None and 'assist' in A
    hold = a_get('hold', False, bool)
    assist = a_get('assist', -1, np.int8)
    guard = a_get('guard', -1, np.int8)
    guard_applied = a_get('guard_applied', False, bool)
    ttc = a_get('ttc', np.nan, np.float32)
    dry = a_get('dry_run', False, bool)
    ai_sel = a_get('select', False, bool)
    raw = a_get('raw', [np.nan, np.nan], np.float64)
    out = a_get('out', [np.nan, np.nan], np.float64)
    model_out = a_get('model_out', [np.nan, np.nan], np.float64)
    gains = a_get('gains', [1.0, 1.0], np.float64)
    trim = a_get('trim', 0.0, np.float64)
    lookahead = a_get('lookahead', 0, np.int16)

    # 操作者: 車両データが新しければ整えた CH3、無ければ AI_CMD の select(AI_CMD も無ければ人)
    if len(vt):
        ai = np.where(sel_ok, vsel[np.clip(latest_index(vt, t), 0, len(vt) - 1)], ai_sel)
    else:
        ai = ai_sel.copy()
    ai = ai.astype(bool)
    info['select_mismatch_frames'] = int(np.sum(sel_ok & (ai != ai_sel))) if A is not None else 0

    # ---- IMU: 衝突・場所・周回
    collisions, suspects, bumps, laps = [], [], [], []
    loc_frame = np.zeros(n, np.int8)
    anchors = np.zeros(0)
    if I is not None:
        it = sec(I['timestamps_us'])
        ie = np.interp(it, vt, verpm) if len(vt) else np.zeros(len(it))
        gz = I['gyro'][:, 2].astype(np.float64)
        still = (np.abs(ie) < 50) & (np.abs(gz) < 5)
        hg, acc_bias = horizontal_g(I['accel'], still)
        info['accel_bias_ms2'] = [round(float(x), 3) for x in acc_bias]
        hd, gzu = heading_deg(it, gz, still)
        anchors, gm_exit, gzs = course_anchors(it, hd, gzu, vt, verpm)
        loc_frame = locate(t, anchors, gm_exit, it, hd, gzs)
        info['r1_anchors'] = len(anchors)
        dyn = dynamic_g(I['accel'], still)

        def shock_event(e, kind):
            tc = float(it[e['i']])
            ip = e['i_peak']
            lat, lon = float(I['accel'][ip, 0] / G), float(I['accel'][ip, 1] / G)

            def vmean(a, b):
                w = (vt >= tc + a) & (vt < tc + b)
                return float(np.mean(verpm[w])) if w.any() else np.nan
            e05 = vmean(0.4, 0.6)
            k = int(np.clip(latest_index(t, tc), 0, n - 1))
            inside = t[0] <= tc <= t[-1] + GAP_S            # フレームのある時間の中か(外なら報酬に付けない)
            det = int(locate([tc], anchors, gm_exit, it, hd, gzs)[0])
            w = np.abs(it - tc) <= 0.1
            return {'t_s': round(tc, 3), 't_us': int(t0 + round(tc * 1e6)), 'frame': k if inside else None,
                    'kind': kind,
                    'g': round(e['g'], 2), 'horizontal_g': round(float(hg[w].max()), 2),
                    'dynamic_g': round(float(dyn[w].max()), 2),
                    'lon_g': round(lon, 2), 'lat_g': round(lat, 2),
                    'side': (('front' if lon < 0 else 'rear') if abs(lon) >= abs(lat)
                             else ('right' if lat < 0 else 'left')),
                    'strong': e['g'] >= STRONG_G, 'clipped': bool(np.max(np.abs(I['accel'][ip, :2])) / G >= CLIP_G),
                    'ai': bool(ai[k]) if inside else False,
                    'assist': int(assist[k]), 'erpm_before': _r(vmean(-0.5, -0.1), 0),
                    'erpm_after': _r(vmean(0.3, 0.8), 0),
                    'erpm_0.5s': _r(e05, 0), 'speed_before_mps': _r(-vmean(-0.5, -0.1) * KV, 2),
                    'outcome': 'glance' if np.isfinite(e05) and e05 < GLANCE_ERPM else 'stop',
                    'loc': LOC_NAMES[_DETAIL_TO_LOC[det]], 'loc_detail': LOC_DETAIL[det], 'samples': e['n']}
        for e in detect_collisions(it, hg, thr=impact_g):
            (suspects if e['suspect'] else collisions).append(shock_event(e, 'impact'))
        # 上下の衝撃(水平は 2.5G 未満): R1 の手前で仕切りの台の板に乗り上げた・軽く触れたなど。既定では報酬に入れない
        th = np.array([c['t_s'] for c in collisions + suspects])
        for e in detect_collisions(it, dyn, thr=impact_g):
            if e['suspect'] or (len(th) and np.min(np.abs(th - it[e['i']])) <= IMPACT_MERGE_S):
                continue
            bumps.append(shock_event(e, 'bump'))
        if bumps_as_collisions:
            collisions = sorted(collisions + bumps, key=lambda c: c['t_s'])
        # 周回(走行中と同じ LapCounter。リカバリー・hold もその周に数える)
        lc = LapCounter()
        isel = vsel[np.clip(latest_index(vt, it), 0, len(vt) - 1)] if len(vt) else np.zeros(len(it), bool)
        notes = sorted([(float(t[i0]), 'recovery') for i0, _ in runs(assist == 2)]
                       + [(float(t[i0]), 'hold') for i0, _ in runs(hold & ai)])
        ni = 0
        for j in range(len(it)):
            while ni < len(notes) and notes[ni][0] <= it[j]:
                lc.note(notes[ni][1])
                ni += 1
            lap = lc.add(float(it[j]), float(gz[j]), float(ie[j]), bool(isel[j]) if A is not None else False,
                         float(hg[j]))
            if lap:
                lap = dict(lap)
                lap['t_start'] = round(lap['t_end'] - lap['time_s'], 3)
                laps.append(lap)
        info['lap_summary'] = {k: v for k, v in lc.summary().items() if k != 'recent'}
    else:
        info['problems'].append('IMU がないので衝突・場所・周回は出せません')
    coll_t = np.array([c['t_s'] for c in collisions])

    # ---- 交代と戻し(整えた CH3 の切り替わり)
    takeovers, handbacks = [], []
    if len(vt):
        ch = np.nonzero(vsel[1:] != vsel[:-1])[0]
        last_tk = None
        for c in ch:
            tt = float(vt[c + 1])
            gap = float(vt[c + 1] - vt[c])
            if vsel[c] and not vsel[c + 1]:
                cls, det = classify_takeover(tt, coll_t, vt, verpm, [b for _, b in reboot_s])
                nxt = np.nonzero((vt > tt) & vsel)[0]
                t_back = float(vt[nxt[0]]) if len(nxt) else None
                k = int(latest_index(t, tt - 1e-6))
                kk = k
                while kk > max(k - 4, -1) and kk >= 0 and not ai[kk]:   # 交代直前の AI のフレーム
                    kk -= 1
                rec = bool(np.any((assist >= 2) & (t > tt - 3.0) & (t <= tt)))
                ev = {'t_s': round(tt, 3), 't_us': int(t0 + round(tt * 1e6)), 'class': cls,
                      'failure': cls in FAILURE_CLASSES, 'frame': kk if kk >= 0 and ai[kk] else None,
                      'gap_before_s': round(gap, 3), 'recovery_active_3s': rec,
                      'human_s': _r((t_back if t_back is not None else float(vt[-1])) - tt, 2),
                      'human_progress_m': _r(np.interp(t_back if t_back is not None else vt[-1], vt, D)
                                             - np.interp(tt, vt, D), 2),
                      'loc': LOC_NAMES[_DETAIL_TO_LOC[locate([tt], anchors, gm_exit, it, hd, gzs)[0]]]
                      if I is not None else 'unknown'}
                ev.update(det)
                takeovers.append(ev)
                last_tk = tt
            elif not vsel[c] and vsel[c + 1]:
                handbacks.append({'t_s': round(tt, 3), 't_us': int(t0 + round(tt * 1e6)),
                                  'frame': int(np.clip(np.searchsorted(t, tt), 0, n - 1)),
                                  'human_s': _r(tt - last_tk, 2) if last_tk is not None else None})
    handover_t = np.array([x['t_s'] for x in takeovers + handbacks])

    # ---- 補助・ガード・hold の区間
    def episodes(mask, extra=None):
        out_ = []
        for i0, i1 in runs(mask):
            e = {'t_s': round(float(t[i0]), 3), 'end_s': round(float(t[i1]), 3),
                 'duration_s': round(float(t[i1] - t[i0] + 1 / FPS), 3), 'frame': int(i0), 'frames': int(i1 - i0 + 1),
                 'ai': bool(ai[i0])}
            if extra:
                e.update(extra(i0, i1))
            out_.append(e)
        return out_

    def rec_extra(i0, i1):
        te = float(t[i1])
        nxt = i1 + 1
        if nxt >= n:
            res = 'log_end'
        elif assist[nxt] == 3 or (nxt + 1 < n and assist[nxt + 1] == 3):
            res = 'giveup'
        elif not ai[nxt] or np.any(~ai[i0:i1 + 1]):
            res = 'takeover'
        else:
            res = 'returned'
        prev = coll_t[(coll_t <= float(t[i0])) & (coll_t > float(t[i0]) - 2.0)]
        return {'result': res, 'after_collision': bool(len(prev)),
                'collision_during': bool(np.any((coll_t > float(t[i0])) & (coll_t <= te))),
                'next_recovery_within_5s': False}
    recoveries = episodes(assist == 2, rec_extra) if has_assist else []
    for a_, b_ in zip(recoveries[:-1], recoveries[1:]):
        a_['next_recovery_within_5s'] = b_['t_s'] - a_['end_s'] <= 5.0
    esc = episodes(assist == 1) if has_assist else []
    giveups = episodes(assist == 3) if has_assist else []

    def guard_extra(i0, i1):
        ts, te = float(t[i0]), float(t[i1])
        f = ttc[i0:i1 + 1]
        return {'max_code': int(guard[i0:i1 + 1].max()),
                'min_ttc_s': _r(np.nanmin(f), 3) if np.isfinite(f).any() else None,
                'applied': bool(guard_applied[i0:i1 + 1].any()),
                'collision_within_1s': bool(np.any((coll_t >= ts) & (coll_t <= te + 1.0)))}
    guards = episodes(guard > 0, guard_extra) if A is not None and 'guard' in A else []
    holds = episodes(hold) if A is not None else []

    # ---- フレームごとのマスク
    near = lambda times, w: (np.abs(t[:, None] - np.asarray(times)[None, :]) <= w).any(axis=1) \
        if len(times) else np.zeros(n, bool)      # noqa: E731
    reboot_win = np.zeros(n, bool)
    for a_, b_ in reboot_s:
        reboot_win |= (t >= a_ - REBOOT_MASK_S) & (t <= b_ + REBOOT_MASK_S)
    handover_win = near(handover_t, HANDOVER_MASK_S)

    # 行動ラベル
    rule = detect_gain_rule(raw, out, gains, trim, ref,
                            ~hold & ((assist == 0) | (assist == -1)) & ~guard_applied, _log_time(log)) \
        if A is not None else {'rule': None, 'source': 'no_ai_cmd'}
    fwd = rule['rule'] != 'both'
    label = np.where(ai[:, None], invert_gains(out, gains, trim, ref, fwd), human_pwm) if A is not None else human_pwm
    action_exec = np.where(ai[:, None], out, human_pwm)
    sat = saturated(out) & ai[:, None] if A is not None else np.zeros((n, 2), bool)
    check = (A is not None) & ~hold & ((assist == 0) | (assist == -1)) & ~guard_applied \
        & np.isfinite(raw).all(axis=1) & ~saturated(out).any(axis=1)
    ai_label_all = invert_gains(out, gains, trim, ref, fwd) if A is not None else np.full((n, 2), np.nan)
    label_err = np.where(check[:, None], ai_label_all - raw, np.nan)

    bits = np.zeros(n, np.int16)
    bits |= np.where(ai & hold, MASK_BITS['hold'], 0).astype(np.int16)
    bits |= np.where(ai & (assist > 0), MASK_BITS['assist'], 0).astype(np.int16)
    bits |= np.where(ai & guard_applied, MASK_BITS['guard'], 0).astype(np.int16)
    bits |= np.where(ai & dry, MASK_BITS['dry_run'], 0).astype(np.int16)
    veh_bad = np.where(ai, ~sel_ok, ~(sel_ok & stick_ok))     # 車両データが古い・CH3 やスティックが無効
    bits |= np.where(veh_bad, MASK_BITS['vehicle'], 0).astype(np.int16)
    bits |= np.where(reboot_win, MASK_BITS['reboot'], 0).astype(np.int16)
    bits |= np.where(handover_win, MASK_BITS['handover'], 0).astype(np.int16)
    bits |= np.where(~np.isfinite(label).all(axis=1), MASK_BITS['no_label'], 0).astype(np.int16)
    valid_action = bits == 0

    # ---- 進み・報酬・区間・リターン
    tn = np.empty(n)
    tn[:-1] = t[1:]
    tn[-1] = t[-1] + 1 / FPS
    gapped = tn - t > GAP_S
    tn = np.where(gapped, t + 1 / FPS, tn)
    dt = tn - t
    if len(vt):
        prog = np.interp(tn, vt, D) - np.interp(t, vt, D)
        prog_f = np.interp(tn, vt, Df) - np.interp(t, vt, Df)
    else:
        prog = prog_f = np.zeros(n)
    seg_ok = np.where(ai, sel_ok & ~hold & ~dry, sel_ok & stick_ok) & ~reboot_win
    collision = np.zeros(n, bool)
    collision_g = np.zeros(n, np.float32)
    bump = np.zeros(n, bool)
    for c in bumps:
        if c['frame'] is not None:
            bump[c['frame']] = True
    for c in collisions:
        if c['frame'] is not None:
            collision[c['frame']] = True
            collision_g[c['frame']] = c['g']
    failure_tk = np.zeros(n, bool)
    tk_class = np.full(n, -1, np.int8)
    for tk in takeovers:
        if tk['frame'] is not None:
            failure_tk[tk['frame']] |= tk['failure']
            tk_class[tk['frame']] = TAKEOVER_CLASSES.index(tk['class'])
    rec_start = np.zeros(n, bool)
    for r_ in recoveries:
        if r_['ai']:
            rec_start[r_['frame']] = True
    p_used = prog if signed_progress else prog_f
    seg = make_segments(ai, seg_ok, t)
    carried, fail_end, moves = carry_penalties(t, seg, ai, collision, failure_tk)
    reward = (p_used - REWARD_W['collision'] * collision - REWARD_W['failure_takeover'] * failure_tk
              - REWARD_W['recovery_start'] * rec_start + carried)
    ret = mc_returns(reward, seg, gamma)
    seg_end = np.zeros(n, bool)
    end_code = np.zeros(n, np.int8)
    time_to_end = np.full(n, np.nan)
    steps_to_end = np.full(n, -1, np.int32)
    seg_rows = []
    for s in np.unique(seg[seg >= 0]):
        idx = np.nonzero(seg == s)[0]
        i0, i1 = int(idx[0]), int(idx[-1])
        if fail_end[i1]:                    # 区間の外に落ちた失敗による交代を、この区間の終わりへ移した
            code = 1
        elif i1 == n - 1:
            code = 4
        elif seg[i1 + 1] < 0 or t[i1 + 1] - t[i1] > GAP_S:
            code = 5
        elif ai[i1] and not ai[i1 + 1]:
            code = 1 if failure_tk[i1] else 2
        else:
            code = 3
        seg_end[i1] = True
        end_code[idx] = code
        time_to_end[idx] = t[i1] - t[idx]
        steps_to_end[idx] = i1 - idx
        seg_rows.append((i0, i1, code))
    near_coll = np.zeros(n, bool)
    for c in collisions:
        near_coll |= (t >= c['t_s'] - NEAR_COLLISION_S[0]) & (t <= c['t_s'] + NEAR_COLLISION_S[1])
    lap_id = np.full(n, -1, np.int16)
    for j, lp in enumerate(laps):
        lap_id[(t >= lp['t_start']) & (t < lp['t_end'])] = j

    arrays = {
        't_us': t_us, 't_s': t, 'scene': frame_scene.astype(np.int16), 'ai': ai, 'ai_cmd_select': ai_sel,
        'hold': hold, 'assist': assist, 'guard': guard, 'guard_applied': guard_applied, 'ttc': ttc.astype(np.float32),
        'dry_run': dry, 'lookahead': lookahead, 'gains': gains.astype(np.float32), 'trim': trim.astype(np.float32),
        'vehicle_fresh': fresh, 'erpm': erpm_f.astype(np.float32), 'speed_mps': (-erpm_f * KV).astype(np.float32),
        'dt_s': dt.astype(np.float32), 'progress_m': prog.astype(np.float32),
        'progress_fwd_m': prog_f.astype(np.float32),
        'collision': collision, 'collision_g': collision_g, 'near_collision': near_coll, 'bump': bump,
        'failure_takeover': failure_tk, 'takeover_class': tk_class, 'recovery_start': rec_start,
        'reward': reward.astype(np.float32), 'penalty_carried': carried.astype(np.float32),
        'segment_id': seg, 'segment_end': seg_end, 'end_code': end_code,
        'time_to_end_s': time_to_end.astype(np.float32), 'steps_to_end': steps_to_end, 'ret': ret.astype(np.float32),
        'valid_action': valid_action, 'mask_bits': bits,
        'rl_mask': valid_action & (seg >= 0), 'bc_mask': valid_action & ~ai & ~near_coll,
        'action_exec': action_exec.astype(np.float32), 'action_label': label.astype(np.float32),
        'action_sat': sat, 'raw': raw.astype(np.float32), 'out': out.astype(np.float32),
        'model_out': model_out.astype(np.float32), 'human_pwm': human_pwm.astype(np.float32),
        'label_err': label_err.astype(np.float32), 'loc_code': _DETAIL_TO_LOC[loc_frame], 'loc_detail': loc_frame,
        'lap_id': lap_id,
    }

    # ---- 検算: 割り戻した値と raw / model_out と out
    def err_stats(mask):
        e = label_err[mask]
        if not len(e):
            return {'n': 0}
        a = np.abs(e)
        return {'n': int(len(e)), 'max_abs_us': [_r(a[:, 0].max(), 4), _r(a[:, 1].max(), 4)],
                'match_0.5us': [_r(np.mean(a[:, 0] <= 0.5), 4), _r(np.mean(a[:, 1] <= 0.5), 4)]}
    brake = np.isfinite(raw[:, 1]) & (raw[:, 1] > ref[1])
    rule['check_ai'] = err_stats(check & ai)
    rule['check_all'] = err_stats(check)
    rule['check_brake_side'] = err_stats(check & brake)
    rule['check_other_rule_brake_side'] = None
    if A is not None and (check & brake).any():           # 逆の規則で割り戻すと合わないことも確かめる
        other = invert_gains(out, gains, trim, ref, not fwd)
        m = check & brake & (np.abs(gains[:, 1] - 1.0) > 0.05)
        if m.any():
            rule['check_other_rule_brake_side'] = {
                'n': int(m.sum()), 'match_0.5us': _r(np.mean(np.abs(other[m, 1] - raw[m, 1]) <= 0.5), 4)}
    if A is not None and 'model_out' in A:
        m = ~hold & (assist == 0) & ~guard_applied & np.isfinite(model_out).all(axis=1)
        rule['out_equals_model_out'] = {'n': int(m.sum()), 'match_0.01us': _r(
            np.mean(np.abs(out[m] - model_out[m]).max(axis=1) <= 0.01), 4) if m.any() else None}
    rule['saturated_ai_frames'] = [int(sat[:, 0].sum()), int(sat[:, 1].sum())]

    # ---- まとめ
    def count(lst, key, vals):
        return {v: sum(1 for x in lst if x.get(key) == v) for v in vals}
    valid_ai = valid_action & ai
    n_moved = {k: sum(1 for m in moves if m['kind'] == k) for k in ('collision', 'failure_takeover')}
    for m in moves:
        m.update(t_from_s=round(float(t[m['from_frame']]), 3), t_to_s=round(float(t[m['to_frame']]), 3))
    ends = {END_CODES[c]: sum(1 for *_, cc in seg_rows if cc == c) for c in END_CODES if c}
    info.update({
        'gain_rule': rule,
        'counts': {
            'ai_frames': int(ai.sum()), 'human_frames': int((~ai).sum()),
            'valid_action_ai': int(valid_ai.sum()), 'valid_action_human': int((valid_action & ~ai).sum()),
            'rl_frames': int(arrays['rl_mask'].sum()), 'bc_frames': int(arrays['bc_mask'].sum()),
            'ai_s': _r(dt[ai].sum(), 1), 'human_s': _r(dt[~ai].sum(), 1),
            'ai_progress_m': _r(prog[ai & fresh].sum(), 1), 'human_progress_m': _r(prog[~ai & fresh].sum(), 1),
            'collisions': len(collisions), 'collisions_ai': sum(c['ai'] for c in collisions),
            'collisions_strong': sum(c['strong'] for c in collisions),
            'collisions_glance': sum(c['outcome'] == 'glance' for c in collisions),
            'collision_suspects': len(suspects),
            'bumps': len(bumps), 'bumps_ai': sum(c['ai'] for c in bumps),
            # 区間の外に落ちて、直前の区間へも移せなかった罰(報酬に入っていない)
            'collisions_on_masked_frames': int(np.sum(collision & (seg < 0))) - n_moved['collision'],
            'collisions_by_loc': {k: v for k, v in count([c for c in collisions if c['ai']], 'loc', LOC_NAMES).items()
                                  if v},
            'takeovers': len(takeovers), 'takeovers_by_class': count(takeovers, 'class', TAKEOVER_CLASSES),
            'failure_takeovers': sum(x['failure'] for x in takeovers), 'handbacks': len(handbacks),
            'failure_takeovers_on_masked_frames': sum(1 for x in takeovers if x['failure']
                                                      and (x['frame'] is None or seg[x['frame']] < 0))
            - n_moved['failure_takeover'],
            'penalties_carried': n_moved,
            'recoveries': len(recoveries), 'recovery_results': count(recoveries, 'result',
                                                                    ['returned', 'giveup', 'takeover', 'log_end']),
            'esc_assists': len(esc), 'giveups': len(giveups), 'guard_warnings': len(guards),
            'guard_warnings_ai': sum(g_['ai'] for g_ in guards), 'holds': len(holds),
            'holds_ai': sum(h['ai'] for h in holds), 'board_reboots': len(reboots),
            'laps': len(laps), 'clean_laps': sum(1 for x in laps if x.get('clean')),
            'segments': len(seg_rows), 'segment_ends': ends,
        },
        'mask_counts_ai': {k: int(np.sum(ai & ((bits & v) > 0))) for k, v in MASK_BITS.items()},
        'returns': {'gamma': gamma,
                    'ai_mean': _r(np.nanmean(ret[ai & (seg >= 0)]), 3) if (ai & (seg >= 0)).any() else None,
                    'human_mean': _r(np.nanmean(ret[~ai & (seg >= 0)]), 3) if (~ai & (seg >= 0)).any() else None,
                    'ai_frames_in_segments': int(np.sum(ai & (seg >= 0))),
                    'ai_frames_within_3.3s_of_truncation': int(np.sum(ai & (seg >= 0) & (end_code != 1)
                                                                      & (time_to_end < 3.3)))},
        'events': {
            'collisions': collisions, 'collision_suspects': suspects, 'bumps': bumps, 'takeovers': takeovers,
            'handbacks': handbacks,
            'recoveries': recoveries, 'esc_assists': esc, 'giveups': giveups, 'guard_warnings': guards,
            'holds': holds,
            'board_reboots': [{'t_last_s': round(a_, 3), 't_after_s': round(b_, 3), 'gap_s': round(b_ - a_, 3)}
                              for a_, b_ in reboot_s],
            'laps': laps,
            'carried_penalties': moves,
        },
        'settings': {'impact_g': impact_g, 'merge_s': IMPACT_MERGE_S, 'reward_weights': REWARD_W,
                     'carry_s': CARRY_S,
                     'bumps_as_collisions': bool(bumps_as_collisions),
                     'progress': 'signed' if signed_progress else 'forward_only', 'gamma': gamma,
                     'handover_mask_s': HANDOVER_MASK_S, 'reboot_mask_s': REBOOT_MASK_S, 'gap_s': GAP_S},
        'end_codes': END_CODES, 'mask_bits': MASK_BITS, 'loc_names': LOC_NAMES, 'loc_detail_names': LOC_DETAIL,
        'takeover_classes': TAKEOVER_CLASSES,
    })
    return info, arrays


# ---------------------------------------------------------------- 検証(目視で確かめた衝突の一覧と比べる)
COLLISION_MAP = config.DATA_DIR / 'reports' / 'ai_runs_20261010' / 'collision_map.md'
INTERVENTIONS = Path('/home/jetson/jetformer_reports/overnight_20261011/research/offline_rl_src/interventions.json')
MAP_RUNS = {'006': '20261010-172910', '007': '20261010-173445'}
# collision_map.md の時刻は「元のシーン 0001 の最初の AI_CMD」が 0。172910 は解析中にシーン 0001 が消されたので、
# 残っている最初の AI_CMD から 20.022 秒前が基準(cmap_src/ld.py)
MAP_T0_SHIFT_S = {'20261010-172910': 20.022, '20261010-173445': 0.0}
# collision_map の一覧に無いが、水平 2.5G 以上で検出した AI側の衝突(時刻は collision_map の基準)。
# 2026-10-11 にカメラの画像(−0.6〜+0.6秒)で確かめたもの。どれも壁・カーテンに軽く触れている(||a|−1G| では 2.5G 未満)
VISUAL_EXTRA_CONTACTS = {
    '20261010-172910': {34.23: 'LH: 黒カーテンに正面から触れる', 142.35: 'DE: 右の白い壁をこする'},
    '20261010-173445': {51.41: 'DE: 右の赤白の壁をこする', 150.69: 'DE: 右の白い壁をこする',
                        163.95: 'DE: 右の白い壁をこする', 216.27: 'DE: 右の白い壁をこする'},
}
# 009/010/101 の衝突の場所(画像で確かめたもの。overnight_20261011/runs/src/takeovers.py の LOC。t は最初の AI_CMD が 0)
VISUAL_LOC_0910 = {
    '20261010-203924': {15.16: 'LH', 19.23: 'LH', 23.98: 'GM_in', 45.80: 'HS_P', 61.56: 'DE', 65.72: 'HS_P'},
    '20261010-204208': {16.89: 'GM_out', 28.24: 'DE', 29.98: 'DE2', 36.42: 'GM_in', 38.49: 'DE', 40.59: 'DE2',
                        65.85: 'DE', 75.67: 'DE'},
    '20261010-204605': {12.51: 'LH', 25.81: 'DE', 27.58: 'DE2', 32.08: 'R1', 34.96: 'DE', 37.07: 'DE2', 42.48: 'LH'},
}


def map_loc(name):
    """collision_map / 目視の場所の名前 → LOC_NAMES"""
    s = name.replace('(軽)', '').replace('(手前・軽)', '').replace('(手前)', '').strip('*').strip()
    if s.startswith('R1'):
        return 'R1'
    if s.startswith('GM'):
        return 'GM'
    if s.startswith('DE'):
        return 'DE'          # DE2(DE の先から奥の道まで)も DE に含める
    if s.startswith('HS'):
        return 'HS'
    if s.startswith('LH'):
        return 'LH'
    if s.startswith('SW'):
        return 'other'
    return 'unknown'


def parse_collision_map(path=COLLISION_MAP):
    """collision_map.md 付録A(AI側の衝突一覧)。[{'log', 't', 'loc_name', 'g', 'recovery'}]"""
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = re.match(r'\|\s*(006|007)\s*\|\s*([\d.]+)\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|', line)
            if m:
                name = m.group(3)
                rows.append({'log': MAP_RUNS[m.group(1)], 't': float(m.group(2)), 'loc_name': name,
                             'g': float(m.group(4)), 'recovery': name == '回復'})
    return rows


def _map_t0_us(log, raw_dir=None):
    """collision_map の時刻の基準(µs)"""
    root = Path(raw_dir or config.RAW_DIR) / log
    for s in sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(log + '-')):
        d = _load_npz(root / s / 'AI_CMD.npz')
        if isinstance(d, dict) and len(d.get('timestamps_us', [])):
            return int(d['timestamps_us'][0]) - int(round(MAP_T0_SHIFT_S.get(log, 0.0) * 1e6))
    return None


def _match(ref_t, det_t, tol_s):
    """時刻の近い順に1対1で対応させる。戻り値 {ref の番号: det の番号}"""
    pairs = sorted((abs(d - r), i, j) for i, r in enumerate(ref_t) for j, d in enumerate(det_t) if abs(d - r) <= tol_s)
    used_r, used_d, out = set(), set(), {}
    for _, i, j in pairs:
        if i not in used_r and j not in used_d:
            out[i] = j
            used_r.add(i)
            used_d.add(j)
    return out


def validate_collisions(infos, raw_dir=None, tol_s=IMPACT_MERGE_S):
    """目視で確かめた衝突(collision_map.md 付録A: 006/007 の AI側 45件 + 回復中 3件)と、検出した衝突を比べる。
    collision_map は ||a|−1G| ≥ 2.5G で探したので、水平だけの軽い接触(||a|−1G| では 2.5G 未満)は一覧に無い。
    そうした検出は VISUAL_EXTRA_CONTACTS(このツールを作るときに画像で確かめた)に入っていれば正しい検出として数える"""
    if not COLLISION_MAP.exists():
        return {'error': '%s がありません' % COLLISION_MAP}
    ref = parse_collision_map()
    res = {'tolerance_s': tol_s, 'per_log': {}, 'unmatched_reference': [], 'extra_detected_ai': []}
    tot = {'ref': 0, 'ref_matched_impact': 0, 'ref_matched_impact_or_bump': 0, 'det_ai': 0, 'det_ai_in_reference': 0,
           'det_ai_visual_extra': 0, 'loc_checked': 0, 'loc_agree': 0}
    for log in sorted(set(r['log'] for r in ref)):
        info = infos.get(log)
        if not info or 'events' not in info:
            res['per_log'][log] = {'error': 'ラベルがありません'}
            continue
        t0 = _map_t0_us(log, raw_dir)
        ev = info['events']
        imp = [dict(c, t_map=(c['t_us'] - t0) / 1e6) for c in ev['collisions'] if c.get('kind', 'impact') == 'impact']
        bmp = [dict(c, t_map=(c['t_us'] - t0) / 1e6) for c in ev.get('bumps', [])]
        rr = [r for r in ref if r['log'] == log]
        lo, hi = min(r['t'] for r in rr) - 5.0, max(r['t'] for r in rr) + 5.0
        m_imp = _match([r['t'] for r in rr], [c['t_map'] for c in imp], tol_s)
        left = [i for i in range(len(rr)) if i not in m_imp]
        m_bmp = _match([rr[i]['t'] for i in left], [c['t_map'] for c in bmp], tol_s)
        m_bmp = {left[i]: j for i, j in m_bmp.items()}
        resid, n_imp, n_any, loc_c, loc_a = [], 0, 0, 0, 0
        for i, r in enumerate(rr):
            if i in m_imp:
                c = imp[m_imp[i]]
            elif i in m_bmp:
                c = bmp[m_bmp[i]]
            else:
                res['unmatched_reference'].append({'log': log, 't': r['t'], 'loc': r['loc_name'], 'g': r['g']})
                continue
            resid.append(c['t_map'] - r['t'])
            if r['recovery']:
                continue
            n_imp += i in m_imp
            n_any += 1
            loc_c += 1
            loc_a += c['loc'] == map_loc(r['loc_name'])
            if i in m_bmp:
                res.setdefault('reference_matched_by_bump', []).append(
                    {'log': log, 't': r['t'], 'loc': r['loc_name'], 'loc_detected': c['loc'], 'g_3d': r['g'],
                     'horizontal_g': c['horizontal_g'], 'dynamic_g': c['dynamic_g']})
        matched_det = set(m_imp.values())
        det_ai = [(j, c) for j, c in enumerate(imp) if c['ai'] and lo <= c['t_map'] <= hi]
        vis = VISUAL_EXTRA_CONTACTS.get(log, {})
        n_vis = 0
        for j, c in det_ai:
            if j in matched_det:
                continue
            note = next((v for k, v in vis.items() if abs(k - c['t_map']) <= 0.05), None)
            n_vis += note is not None
            res['extra_detected_ai'].append({'log': log, 't_map': round(c['t_map'], 2), 'g': c['g'], 'loc': c['loc'],
                                             'visual_check': note})
        n_ref = sum(1 for r in rr if not r['recovery'])
        n_in_ref = sum(1 for j, _ in det_ai if j in matched_det)
        res['per_log'][log] = {'reference': n_ref, 'matched_impact': n_imp, 'matched_impact_or_bump': n_any,
                               'reference_incl_recovery': len(rr), 'detected_ai': len(det_ai),
                               'detected_ai_in_reference': n_in_ref, 'detected_ai_visual_extra': n_vis,
                               'time_residual_s_median': _r(np.median(resid), 3) if resid else None,
                               'loc_agree': loc_a, 'loc_checked': loc_c}
        for k, v in (('ref', n_ref), ('ref_matched_impact', n_imp), ('ref_matched_impact_or_bump', n_any),
                     ('det_ai', len(det_ai)), ('det_ai_in_reference', n_in_ref), ('det_ai_visual_extra', n_vis),
                     ('loc_checked', loc_c), ('loc_agree', loc_a)):
            tot[k] += v
    res['totals'] = tot
    q = lambda a, b: _r(a / b, 4) if b else None      # noqa: E731
    res['recall_impact'] = q(tot['ref_matched_impact'], tot['ref'])
    res['recall_impact_or_bump'] = q(tot['ref_matched_impact_or_bump'], tot['ref'])
    res['precision_ai_vs_reference'] = q(tot['det_ai_in_reference'], tot['det_ai'])
    res['precision_ai_with_visual_check'] = q(tot['det_ai_in_reference'] + tot['det_ai_visual_extra'], tot['det_ai'])
    res['loc_agreement'] = q(tot['loc_agree'], tot['loc_checked'])
    res['target'] = 0.9
    res['pass_impact_only'] = bool((res['recall_impact'] or 0) >= 0.9
                                   and (res['precision_ai_with_visual_check'] or 0) >= 0.9)
    res['pass_impact_or_bump'] = bool((res['recall_impact_or_bump'] or 0) >= 0.9
                                      and (res['precision_ai_with_visual_check'] or 0) >= 0.9)
    return res


def validate_locations_0910(infos, tol_s=0.1):
    """009/010/101 の衝突の場所(画像で確かめたもの)と、ジャイロで決めた場所を比べる"""
    rows, agree = [], 0
    for log, d in VISUAL_LOC_0910.items():
        info = infos.get(log)
        if not info or 'events' not in info:
            continue
        for tt, name in d.items():
            c = min(info['events']['collisions'], key=lambda c: abs(c['t_s'] - tt), default=None)
            if c is None or abs(c['t_s'] - tt) > tol_s:
                rows.append({'log': log, 't': tt, 'visual': name, 'detected': None})
                continue
            ok = c['loc'] == map_loc(name) or (name == 'DE2' and c['loc'] == 'other')   # DE2 は DE の先〜奥の道
            agree += ok
            rows.append({'log': log, 't': tt, 'visual': name, 'loc': c['loc'], 'loc_detail': c['loc_detail'],
                         'agree': ok})
    return {'n': len(rows), 'agree': agree, 'agreement': _r(agree / len(rows), 4) if rows else None,
            'disagree': [r for r in rows if not r.get('agree')]}


def validate_takeovers(infos, path=INTERVENTIONS, tol_s=0.15):
    """interventions.py の分類(interventions.json)と比べる。基準の時刻はログの最初の VEHICLE サンプル"""
    if not Path(path).exists():
        return {'error': '%s がありません' % path}
    with open(path, encoding='utf-8') as f:
        ref = json.load(f)
    names = {'impact': 'after_collision', 'stuck': 'stuck', 'moving_fast': 'preemptive', 'slow/other': 'operational'}
    rows, agree, matched = [], 0, 0
    for r in ref:
        info = infos.get(r['log'])
        if not info or 'events' not in info:
            continue
        v0 = info.get('vehicle_t0_us')
        cand = [x for x in info['events']['takeovers'] if abs((x['t_us'] - v0) / 1e6 - r['t']) <= tol_s]
        if not cand:
            rows.append({'log': r['log'], 't': r['t'], 'ref': names[r['cls']], 'mine': None})
            continue
        matched += 1
        ok = cand[0]['class'] == names[r['cls']]
        agree += ok
        if not ok:
            rows.append({'log': r['log'], 't': r['t'], 'ref': names[r['cls']], 'mine': cand[0]['class'],
                         'cause': cand[0].get('cause')})
    mine = sum(info['counts']['takeovers'] for info in infos.values() if 'counts' in info)
    return {'reference': len(ref), 'matched': matched, 'class_agree': agree,
            'agreement': _r(agree / matched, 4) if matched else None, 'mine_total': mine, 'differences': rows}


def write_sheets(validation, out_dir, raw_dir=None):
    """検証の食い違いをカメラ画像で確かめるためのコンタクトシート(1行 = 1件、−0.6〜+0.6秒)"""
    import cv2
    offs = [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6]
    cache = {}

    def frames(log):
        if log not in cache:
            root = Path(raw_dir or config.RAW_DIR) / log
            ts, fs = [], []
            for s_ in sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(log + '-')):
                d = _load_npz(root / s_ / 'CAM_FRONT.npz')
                if isinstance(d, dict) and 'files' in d:
                    ts.append(d['timestamps_us'])
                    fs += [str(root / s_ / str(f)) for f in d['files']]
            cache[log] = (np.concatenate(ts) if ts else np.zeros(0, np.int64), fs)
        return cache[log]

    def sheet(items, path):
        rows = []
        for log, t_us, label in items:
            ct, cf = frames(log)
            if not len(ct):
                continue
            ims = []
            for o in offs:
                i = int(np.clip(np.searchsorted(ct, t_us + o * 1e6), 0, len(ct) - 1))
                im = cv2.imread(cf[i])
                im = cv2.resize(im if im is not None else np.zeros((224, 224, 3), np.uint8), (200, 200))
                txt = label if o == 0 else '%+.1f' % o
                for col, w in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.putText(im, txt, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, w)
                ims.append(im)
            rows.append(np.hstack(ims))
        if rows:
            cv2.imwrite(str(path), np.vstack(rows))
            return str(path)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    v = validation['collision_map']
    t0 = {log: _map_t0_us(log, raw_dir) for log in MAP_RUNS.values()}
    extras = [(e['log'], int(t0[e['log']] + e['t_map'] * 1e6),
               '%s %.2fs %.1fG %s' % (e['log'][-6:], e['t_map'], e['g'], e['loc']))
              for e in v.get('extra_detected_ai', [])]
    bumps = [(r['log'], int(t0[r['log']] + r['t'] * 1e6),
              '%s %.2fs h%.1fG %s' % (r['log'][-6:], r['t'], r['horizontal_g'], r['loc_detected']))
             for r in v.get('reference_matched_by_bump', [])]
    return {'extra_detected_ai': sheet(extras, out_dir / 'collision_extras_006_007.jpg'),
            'reference_matched_by_bump': sheet(bumps, out_dir / 'collision_bumps_006_007.jpg')}


# ---------------------------------------------------------------- 書き出し
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_json_default)
    os.replace(tmp, path)


def summarize(infos):
    rows = {}
    keys_sum = ['ai_frames', 'human_frames', 'valid_action_ai', 'valid_action_human', 'rl_frames', 'bc_frames',
                'collisions', 'collisions_ai', 'collisions_strong', 'collision_suspects', 'bumps', 'bumps_ai',
                'takeovers',
                'failure_takeovers', 'handbacks', 'recoveries', 'esc_assists', 'giveups', 'guard_warnings', 'holds',
                'board_reboots', 'laps', 'clean_laps']
    tot = {k: 0 for k in keys_sum}
    tk = {c: 0 for c in TAKEOVER_CLASSES}
    for log, info in sorted(infos.items()):
        if 'counts' not in info:
            rows[log] = {'error': info.get('error')}
            continue
        c = info['counts']
        gr = info['gain_rule']
        rows[log] = {'description': info['description'], 'ai_log': info['ai_log'], 'frames': info['frames'],
                     'duration_s': info['duration_s'], **{k: c[k] for k in keys_sum},
                     'ai_s': c['ai_s'], 'human_s': c['human_s'], 'ai_progress_m': c['ai_progress_m'],
                     'takeovers_by_class': c['takeovers_by_class'], 'collisions_by_loc': c['collisions_by_loc'],
                     'gain_rule': gr.get('rule'), 'gain_rule_source': gr.get('source'),
                     'label_match_ai': (gr.get('check_ai') or {}).get('match_0.5us'),
                     'label_checked_ai': (gr.get('check_ai') or {}).get('n', 0),
                     'problems': len(info['problems'])}
        for k in keys_sum:
            tot[k] += c[k]
        for k, v in c['takeovers_by_class'].items():
            tk[k] += v
    tot['takeovers_by_class'] = tk
    for k in ('collisions_on_masked_frames', 'failure_takeovers_on_masked_frames'):
        tot[k] = sum((i.get('counts') or {}).get(k, 0) for i in infos.values())
    tot['penalties_carried'] = {k: sum(((i.get('counts') or {}).get('penalties_carried') or {}).get(k, 0)
                                       for i in infos.values()) for k in ('collision', 'failure_takeover')}
    # 割り戻しの検算(全ログの AI フレーム)
    n = m0 = m1 = 0
    for info in infos.values():
        ch = (info.get('gain_rule') or {}).get('check_ai') or {}
        if ch.get('n'):
            n += ch['n']
            m0 += ch['n'] * ch['match_0.5us'][0]
            m1 += ch['n'] * ch['match_0.5us'][1]
    tot['label_check_ai'] = {'n': n, 'match_0.5us': [_r(m0 / n, 4), _r(m1 / n, 4)] if n else None}
    seg_ai = sum((i.get('returns') or {}).get('ai_frames_in_segments', 0) for i in infos.values())
    near_end = sum((i.get('returns') or {}).get('ai_frames_within_3.3s_of_truncation', 0) for i in infos.values())
    tot['ai_frames_in_segments'] = seg_ai
    tot['ai_frames_within_3.3s_of_truncation'] = near_end
    tot['ai_truncation_ratio'] = _r(near_end / seg_ai, 4) if seg_ai else None
    tot['gain_rule_by_log'] = {log: '%s/%s' % (i['gain_rule'].get('rule'), i['gain_rule'].get('source'))
                               for log, i in sorted(infos.items()) if i.get('ai_log') and i.get('gain_rule')}
    return rows, tot


NOTES = [
    '衝突(collisions)は水平の加速度 2.5G 以上(取り付けの傾きを引く)。0.6秒以内は1件、1サンプルだけのものは collision_suspects。',
    'bumps は水平が 2.5G 未満で、重力を引いた加速度の大きさが 2.5G 以上の衝撃(上下が主)。ほとんどが走り続けていて(glance)、'
    'R1(レールの切れ目の黒いスポンジ)と緑マットの入口に多く、人のきれいな周にもある。既定では報酬に入れない(--bumps-as-collisions)。',
    'collision_map の目視 45件のうち 6件は bumps(水平 1.0〜2.4G)。水平だけでは 39/45、bumps を含めると 45/45 が見つかる。',
    '一覧に無い水平の検出 6件は、カメラの画像で壁・カーテンに触れていることを確かめた(VISUAL_EXTRA_CONTACTS、validation/*.jpg)。',
    '行動ラベルは out をフレームごとのゲインで割り戻した値。速度のゲインの規則はログごとに raw と out から確かめた(gain_rule)。',
    '172910 の collision_map の時刻は、残っている最初の AI_CMD から 20.022秒前が 0(シーン 0001 が消されたため)。',
    'リターンは区間の終わりで打ち切り(その先 0)。end_code で、失敗による交代(1)と打ち切り(2〜5)を分けて扱える。',
    '強い衝突で基板が再起動すると、衝突・交代が再起動の前後のマスク(区間の外)に入る。その罰は、直前の AI の区間の最後の'
    'フレームへ移す(衝突は %.1f秒以内、交代はそれ以内か同じ区間へ衝突を移したとき。penalty_carried・carried_penalties)。'
    'totals の *_on_masked_frames は移せずに報酬に入っていない数。' % CARRY_S,
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('logs', nargs='*', help='ログ名(省略するとすべて)')
    ap.add_argument('--raw', default=str(config.RAW_DIR), help='生ログの場所')
    ap.add_argument('--out', default=str(config.DATA_DIR / 'labels'), help='書き出す場所')
    ap.add_argument('--impact-g', type=float, default=IMPACT_G, help='衝突とみなす水平の加速度 [G]')
    ap.add_argument('--gamma', type=float, default=GAMMA)
    ap.add_argument('--signed-progress', action='store_true', help='報酬の進みを符号つき(後退で負)にする')
    ap.add_argument('--sheets', action='store_true', help='検証の食い違いのコンタクトシートを <out>/validation/ に作る')
    ap.add_argument('--bumps-as-collisions', action='store_true',
                    help='上下の衝撃(水平 2.5G 未満、R1 の手前の仕切りの台など)も衝突として報酬に入れる')
    args = ap.parse_args()
    raw = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_logs = sorted(d.name for d in raw.iterdir() if (d / 'log.json').exists())
    logs = args.logs or all_logs
    infos = {}
    failed = []
    for log in logs:
        try:                # 1 本の壊れた記録で残りのラベルが作られないことが無いように(10/21 Fix-all、V-tools minor 7)
            info, arrays = label_log(log, raw, args.impact_g, args.gamma, args.signed_progress, args.bumps_as_collisions)
        except Exception as e:      # noqa: BLE001
            failed.append(log)
            print('%s ラベルを作れません(飛ばして続けます): %s: %s' % (log, type(e).__name__, e), file=sys.stderr)
            continue
        info['created_at'] = datetime.now().isoformat(timespec='seconds')
        _write_json(out_dir / (log + '.json'), info)
        if arrays is not None:
            tmp = out_dir / (log + '.tmp.npz')
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, out_dir / (log + '.npz'))
        infos[log] = info
        c = info.get('counts', {})
        gr = info.get('gain_rule', {})
        print('%s %-24s 衝突 %3s(AI %3s) 交代 %3s %s 戻し %3s リカバリー %2s 再起動 %2s 周 %3s 有効AI %5s 規則 %s/%s 検算 %s' % (
            log, info.get('description', '')[:24], c.get('collisions'), c.get('collisions_ai'), c.get('takeovers'),
            c.get('takeovers_by_class'), c.get('handbacks'), c.get('recoveries'), c.get('board_reboots'), c.get('laps'),
            c.get('valid_action_ai'), gr.get('rule'), gr.get('source'), (gr.get('check_ai') or {}).get('match_0.5us')))
    # 指定したログだけのときは、既にある他のログのラベルも読んで summary.json を作り直す
    for log in all_logs:
        p = out_dir / (log + '.json')
        if log not in infos and p.exists():
            with open(p, encoding='utf-8') as f:
                infos[log] = json.load(f)
    rows, tot = summarize(infos)
    validation = {'collision_map': validate_collisions(infos, raw), 'locations_009_101': validate_locations_0910(infos),
                  'interventions': validate_takeovers(infos)}
    summary = {'version': VERSION, 'created_at': datetime.now().isoformat(timespec='seconds'),
               'tool': 'tools/label_events.py', 'notes': NOTES, 'logs': rows, 'totals': tot, 'validation': validation,
               'settings': {'impact_g': args.impact_g, 'gamma': args.gamma, 'reward_weights': REWARD_W,
                            'bumps_as_collisions': args.bumps_as_collisions,
                            'progress': 'signed' if args.signed_progress else 'forward_only'}}
    if args.sheets:
        summary['validation']['sheets'] = write_sheets(validation, out_dir / 'validation', raw)
    _write_json(out_dir / 'summary.json', summary)
    v = validation['collision_map']
    print('衝突の検証(collision_map 目視 %s件): 水平だけで一致 %s / 上下の衝撃も入れて %s / 検出のうち正しい %s'
          '(一覧にあるもの %s)/ 場所の一致 %s' % (
              v.get('totals', {}).get('ref'), v.get('recall_impact'), v.get('recall_impact_or_bump'),
              v.get('precision_ai_with_visual_check'), v.get('precision_ai_vs_reference'), v.get('loc_agreement')))
    print('場所(009/010/101 の目視):', {k: validation['locations_009_101'].get(k) for k in ('n', 'agree', 'agreement')})
    print('交代の分類(interventions.json):', {k: validation['interventions'].get(k)
                                            for k in ('reference', 'matched', 'agreement')})
    print('割り戻しの検算(AI、assist=0・hold なし):', tot['label_check_ai'])
    print('書き出し:', out_dir)
    if failed:      # 作れなかったログがあれば終了コード 1(夜の 1 コマンドが気づくように)。ほかのログのラベルは書いてある
        print('ラベルを作れなかった記録 %d 本: %s' % (len(failed), ' '.join(failed)), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
