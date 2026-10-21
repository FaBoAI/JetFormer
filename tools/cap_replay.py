#!/usr/bin/env python3
"""区間の速度上限(安全層 v1.1、jetformer/safety_caps.py)を新しい記録で再生し、G1 の 6 項目を 1 コマンドで判定する(U4、2026-10-21)

    python tools/cap_replay.py --day 20261023                       # その日の記録 → data/reports/cap_replay_20261023/
    python tools/cap_replay.py 20261023-101500 20261023-103000 --out DIR
    python tools/cap_replay.py --day 20261010 --g1-logs 20261010-172910,...   # 10/10 の G1(g1_replay.json と同じ数)

experiments/20261016_sim_calibration/g1_replay.py(10/16 の G1)の規則と数字をそのまま使い、記録を選べるようにしたもの。
読み取りだけ(numpy、GPU なし)。data/raw・data/labels は変えない。drive.py の走行には関わらない。

G1 の 6 項目(action_plan_20261016/plan.md 注 2、fast_and_safe_plan.md 6 章。g1_replay の main_keys と同じ):
  1 R1_signal_ge_90pct          R1 で衝突した AI の通過のうち、起点の 0.1 秒以上前に R1_app の合図が出た割合 ≥ 90%
  2 R1_signal_strong_ge_90pct   強い衝突(4G 以上)の通過での合図 ≥ 90%(10/10 では 007 の R1 の板 = g1_replay の
                                R1_signal_007_board_ge_90pct。--strong-logs で数えるログを選ぶ。既定は G1 のログ全部)
  3 false_signal_le_1_per_min   R1 の起点の窓(−2.0〜+1.5 秒)の外での合図の立ち上がり ≤ 1 回/分(G1 のログそれぞれ。AI で動いている時間で割る)
  4 gyro_r1_detection           因果的なジャイロの位相の R1 検出 ≥ 95%(AI の通過)・誤検出 0(AI のログ)
  5 lap_loss_le_0.3             仮想の閉ループ(縦だけ、P_cal の惰性・こすれ・ブレーキ・駆動)での計算上のラップの損 ≤ 0.3 秒/周
  6 brake_invariants            1 m/s 未満でのブレーキ 0・R1 の旋回中(起点 → +0.5 秒)のブレーキ 0・0.5 秒以内のブレーキ ↔ 中立の往復 0
                                (影の再生と仮想の閉ループの両方)
  1〜4 は区間の判定だけで決まり、上限の案によらない。5・6 は案ごと。

区間の案(--variants。既定は 3zone,R1GM,DE3.3):
  3zone   R1 2.5・GM 2.2・DE 3.0 m/s(safety_caps.params_real_car() = v1.1 の既定)
  R1GM    R1 2.5・GM 2.2 だけ(DE は上限なし。位相が不確かなときの 3.0 は残す)
  DE3.3   R1 2.5・GM 2.2・DE 3.3 m/s
  R1_2.75 R1 2.75・GM 2.2・DE 3.0 m/s(10/16 の比べ)

記録の影の判定の欄(S1 が drive.py につないだあとの記録): AI_CMD に SHADOW_FIELDS のどれかの名前の欄があれば読み、
再生で出した区間・モードと、フレームごとにどれだけ一致するかを出す(--use-recorded で、合図の判定に記録の区間を使う)。
欄が無い記録は、再生だけで判定する(10/10 の記録はこちら)。欄の名前は bench_prep_20261021/build/S1.md に合わせて直すこと。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import label_events as le  # noqa: E402
from jetformer import config  # noqa: E402
from jetformer import safety_caps as sc  # noqa: E402

SCHEMA = 'jf-cap-replay/v1'
PCAL_DEFAULT = Path('/home/jetson/jetformer_reports/sim_calibration_20261016/profiles/P_cal.json')
KV = 0.5e-3
PRED_TH = 1150.0
SIG_WIN = 1.5          # 合図を探す: R1 の起点の 1.5 秒前まで
R1_WIN = (2.0, 1.5)    # 誤った合図の数え方: R1 の起点の 2.0 秒前〜1.5 秒後の外の合図の立ち上がり
TURN_NO_BRAKE = 0.5    # R1 の旋回中 = 起点 → +0.5 秒
MIN_LEAD = 0.1         # 合図として数えるのは、R1 の起点の 0.1 秒(3 フレーム)以上前に出たもの
LOC_NAMES = ['unknown', 'R1', 'GM', 'DE', 'other', 'HS', 'LH']
VER = 'v1.1'           # 区間の判定の版(safety_caps の既定。g1_replay の 'v1.1')
VARIANTS = {
    '3zone': lambda: sc.params_real_car(),
    'R1GM': lambda: sc.params_real_car(cap_DE=sc.INF),
    'DE3.3': lambda: sc.params_real_car(cap_DE=3.3),
    'R1_2.75': lambda: sc.params_r1_275(),
}
VARIANT_TEXT = {'3zone': 'R1 2.5・GM 2.2・DE 3.0', 'R1GM': 'R1 2.5・GM 2.2(DE なし)', 'DE3.3': 'R1 2.5・GM 2.2・DE 3.3',
                'R1_2.75': 'R1 2.75・GM 2.2・DE 3.0'}
# S1(drive.py の zone_caps の影の判定、bench_prep_20261021/build/S1.md)が AI_CMD に書く欄。先頭が S1 の名前
SHADOW_FIELDS = {
    'zone': ('cap_zone',),               # int8: safety_caps.ZONES の番号(0 none〜5 DE)、−1 = 計算していない・失敗
    'mode': ('cap_mode',),               # int8: −2 失敗・−1 動かしていない・0 そのまま・1 上限で切った・2 中立・3 ブレーキ
    'thr_out': ('cap_thr_us',),          # 上限ならこう出したスロットル [µs]
    'thr_in': ('cap_in_us',),            # 層への入力のスロットル(車に出す直前。OFF なら out と同じ)
    'active': ('cap_on',),               # そのフレームで zone_caps = 1 だったか
    'applied': ('cap_applied',),         # 実際に車への出力を変えたか
    'brake_start': ('cap_brake_start',),
}
CAP_MODES = ('pass', 'cap', 'neutral', 'brake')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def jdump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def d(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))

    def clean(o):
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(clean(json.loads(json.dumps(obj, default=d))), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def prev_idx(src, q):
    return np.clip(np.searchsorted(src, q, side='right') - 1, 0, len(src) - 1)


def load_pcal(path=PCAL_DEFAULT):
    p = json.loads(Path(path).read_text())
    lo = p['longitudinal']
    return {'c0': lo['c0'], 'k_s': lo['k_s'], 'tau_up': lo['tau_up'], 'a_cap': lo['a_cap'],
            'a_b200': {k: v['a_b_at_200us'] for k, v in lo['brake_scenarios'].items()}}


def _decode_zone(x):
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode()
    if isinstance(x, str):
        return x
    try:
        i = int(x)
    except (TypeError, ValueError):
        return 'off'
    return sc.ZONES[i] if 0 <= i < len(sc.ZONES) else 'off'      # −1 = 計算していない


def _decode_mode(x):
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode()
    if isinstance(x, str):
        return x
    try:
        i = int(x)
    except (TypeError, ValueError):
        return 'off'
    return CAP_MODES[i] if 0 <= i < len(CAP_MODES) else 'off'


def recorded_shadow(A, order):
    """AI_CMD の影の判定の欄(あれば)。返り値 {'zone': 配列, 'mode': 配列, ...} と使った欄の名前"""
    out, used = {}, {}
    if A is None:
        return out, used
    for k, names in SHADOW_FIELDS.items():
        for nm in names:
            if nm in A and len(A[nm]) == len(order):
                arr = A[nm][order]
                if k == 'zone':
                    arr = np.array([_decode_zone(x) for x in arr], dtype=object)
                elif k == 'mode':
                    arr = np.array([_decode_mode(x) for x in arr], dtype=object)
                out[k] = arr
                used[k] = nm
                break
    return out, used


# ------------------------------------------------------------------ 1 本のログ
def load_labels(log_name, labels_dir=None, raw_dir=None):
    """ラベル(data/labels/<ログ>.json・.npz)。無ければ tools/label_events.py の label_log でその場で作る(書き出さない)"""
    labels_dir = Path(labels_dir or config.DATA_DIR / 'labels')
    pj, pn = labels_dir / (log_name + '.json'), labels_dir / (log_name + '.npz')
    if pj.exists() and pn.exists():
        info = json.loads(pj.read_text(encoding='utf-8'))
        with np.load(pn) as z:
            lab = {k: z[k] for k in z.files}
        return info, lab, 'file'
    info, lab = le.label_log(log_name, raw_dir)
    if lab is None:
        raise ValueError('ラベルのファイルが無く、その場でも作れない(フレームが無い)')
    return info, lab, 'computed'


def prepare(log_name, labels_dir=None, raw_dir=None, use_recorded=False):
    info, lab, lab_src = load_labels(log_name, labels_dir, raw_dir)
    t0 = info['t0_us']
    L = le.load_log(log_name, raw_dir)
    I, V, A = L['IMU'], L['VEHICLE'], L['AI_CMD']
    it = (I['timestamps_us'].astype(np.int64) - t0) / 1e6
    gz = I['gyro'][:, 2].astype(np.float64)
    vt = (V['timestamps_us'].astype(np.int64) - t0) / 1e6
    erpm = V['erpm'].astype(np.float64)
    v_imu = -erpm[prev_idx(vt, it)] * KV
    # 後から見た R1 の起点(正解)
    still = (np.abs(np.interp(it, vt, erpm)) < 50) & (np.abs(gz) < 5)
    hd, gzu = le.heading_deg(it, gz, still)
    anchors, gm, gzs11 = le.course_anchors(it, hd, gzu, vt, erpm)
    out = {'log': log_name, 'info': info, 'lab': lab, 'it': it, 'gz': gz, 'vt': vt, 'erpm': erpm, 'anchors': np.asarray(anchors),
           'gm': np.asarray(gm), 'ver': {}, 'recorded_fields': {}, 'labels_source': lab_src, 'problems': list(L.get('problems') or [])}
    has_ai = A is not None and len(A.get('timestamps_us', []))
    rec = {}
    if has_ai:
        at = (A['timestamps_us'].astype(np.int64) - t0) / 1e6
        o_ = np.argsort(at, kind='stable')
        at = at[o_]
        sel = A['select'][o_].astype(bool) & ~A['hold'][o_].astype(bool) & ~A['dry_run'][o_].astype(bool)
        pred_min = A['pred'][o_][:, :, 0].min(axis=1)
        thr = A['out'][o_][:, 1].astype(np.float64)
        steer_us = A['out'][o_][:, 0].astype(np.float64)
        ki = prev_idx(it, at)
        kv = prev_idx(vt, at)
        v_at = -erpm[kv] * KV
        fresh = (at - vt[kv]) < 0.1
        kl = np.clip(np.searchsorted(lab['t_s'], at), 0, len(lab['t_s']) - 1)
        rec, used = recorded_shadow(A, o_)
        out['recorded_fields'] = used
        if 'thr_in' in rec:
            # S1: ON の走行では out は上限のあとの値なので、層への入力 cap_in_us で再生する(OFF なら out と同じ)
            tin = rec['thr_in'].astype(np.float64)
            thr = np.where(np.isfinite(tin), tin, thr)
        out.update({'at': at, 'sel': sel & fresh, 'pred_sig': (pred_min <= PRED_TH) & sel, 'v_at': v_at, 'thr': thr,
                    'steer_norm': np.array([sc.steer_us_to_norm(x) for x in steer_us]), 'loc_at': lab['loc_code'][kl], 'ki': ki,
                    'recorded': rec})
    # 因果的なジャイロの位相(v1.1)
    n = len(it)
    phase = sc.PhaseParams()
    st = sc.PhaseState()
    zone_g = np.empty(n, dtype=object)
    turning = np.zeros(n, bool)
    conf = np.zeros(n, bool)
    anchor_c = np.full(n, np.nan)
    for i in range(n):
        st, o = sc.phase_step(st, float(it[i]), float(gz[i]), float(v_imu[i]), phase)
        zone_g[i] = o.zone
        turning[i] = o.turning_right
        conf[i] = o.r1_confirmed
        if o.anchor_t is not None:
            anchor_c[i] = o.anchor_t
    dv = {'zone_g': zone_g, 'turning': turning, 'causal_anchors': np.unique(anchor_c[conf])}
    if has_ai:
        zg = zone_g[out['ki']]
        dv['gyro_app'] = np.array([z == 'R1_app' for z in zg])
        dv['zone'] = np.array([sc.combine_zone(z, bool(tr), bool(ps), v=float(v), pred_min_speed=sc.PRED_MIN_SPEED)
                               for z, tr, ps, v in zip(zg, turning[out['ki']], out['pred_sig'], out['v_at'])], dtype=object)
        dv['pred_eff'] = out['pred_sig'] & (out['v_at'] >= sc.PRED_MIN_SPEED)
        if 'zone' in rec:
            dv['zone_replayed'] = dv['zone']
            if use_recorded:
                dv['zone'] = rec['zone']
    out['ver'][VER] = dv
    del L
    return out



def shadow(d, ver, p):
    """影の再生(開ループ)。AI 側のフレームだけ。AI でない・0.5 秒を超える途切れで層の状態を戻す"""
    at, sel, v, thr, steer = d['at'], d['sel'], d['v_at'], d['thr'], d['steer_norm']
    zone = d['ver'][ver]['zone']
    n = len(at)
    mode = np.empty(n, dtype=object)
    thr_out = thr.copy()
    bstart = np.zeros(n, bool)
    st = sc.CapState()
    last = -1e9
    for i in range(n):
        if not sel[i]:
            mode[i] = 'off'
            st = sc.CapState()
            continue
        if at[i] - last > 0.5:
            st = sc.CapState()
        last = at[i]
        u = sc.thr_us_to_cmd(float(thr[i]))
        st, o = sc.cap_step(st, float(at[i]), zone[i], float(v[i]), u, p, steer=float(steer[i]))
        thr_out[i] = thr[i] if o.u >= u else max(thr[i], sc.cmd_to_thr_us(o.u))
        mode[i] = o.mode
        bstart[i] = o.brake_start
    return {'mode': mode, 'thr_out': thr_out, 'brake_start': bstart}


def r1_passes(d):
    """後から見た R1 の起点ごと: AI か・進入の速さ・R1 の衝突(offline_rl.course_passes と同じ規則)"""
    lab, info = d['lab'], d['info']
    cs = info['events']['collisions']
    vt, erpm = d['vt'], d['erpm']
    rows = []
    for j, ta in enumerate(d['anchors']):
        tg = d['gm'][j]
        w = (vt >= ta - 0.4) & (vt <= ta)
        entry = float(np.max(-erpm[w])) * KV if w.any() else float('nan')
        c_r1 = [c for c in cs if c['loc'] == 'R1' and ta - 0.5 <= c['t_s'] <= (tg - 0.3 if np.isfinite(tg) else ta + 3.5)]
        k = int(np.clip(np.searchsorted(lab['t_s'], ta), 0, len(lab['t_s']) - 1))
        rows.append({'ta': float(ta), 'tg': float(tg) if np.isfinite(tg) else None, 'ai': bool(lab['ai'][k]), 'entry_mps': entry,
                     'r1_collision': bool(c_r1), 'r1_strong': any(c['strong'] for c in c_r1),
                     'collision_g': max([c['g'] for c in c_r1], default=None)})
    return rows


def signal_of_pass(d, ver, ta, which='zone'):
    """R1 の起点 ta の手前 SIG_WIN 秒の合図(AI のフレーム)。返り値: (あり, 先行の秒, 合図の始まりの速さ, 始まりから起点までの距離)"""
    at = d['at']
    m = (at >= ta - SIG_WIN) & (at <= ta) & d['sel']
    dv = d['ver'][ver]
    if which == 'zone':
        s = np.array([z == 'R1_app' for z in dv['zone']]) & m
    elif which == 'gyro':
        s = dv['gyro_app'] & m
    else:
        s = dv['pred_eff'] & m
    if not s.any():
        return False, None, None, None
    i = int(np.nonzero(s)[0][0])
    lead = float(ta - at[i])
    if lead < MIN_LEAD:
        return False, lead, None, None
    vt, erpm = d['vt'], d['erpm']
    w = (vt >= at[i]) & (vt <= ta)
    dist = float(np.sum(-erpm[w][:-1] * KV * np.diff(vt[w]))) if w.sum() > 1 else 0.0
    return True, lead, float(d['v_at'][i]), dist


def false_signals(d, ver, all_anchors, sh=None):
    """R1 の起点の窓(−2.0〜+1.5 秒)の外での、合図(R1_app)の立ち上がり。AI 側・動いている(v > 0.5)フレーム。
    参考: その場所(ラベルの loc)が LH・R1 なら『R1 の手前まで来たが R1 を通らなかった』(衝突・交代)として別に数える。
    効いた合図 = その合図の間に、層が出力を変えた(sh を渡したとき)"""
    at, sel = d['at'], d['sel']
    dv = d['ver'][ver]
    moving = d['v_at'] > 0.5
    inwin = np.zeros(len(at), bool)
    for ta in all_anchors:
        inwin |= (at >= ta - R1_WIN[0]) & (at <= ta + R1_WIN[1])
    res = {}
    for name, sig in (('combined', np.array([z == 'R1_app' for z in dv['zone']])), ('gyro', dv['gyro_app']), ('pred', dv['pred_eff'])):
        on = sig & sel & moving
        onsets, times, approach, effective, effective_all = 0, [], 0, 0, 0
        prev = False
        i = 0
        while i < len(at):
            cur = bool(on[i])
            if cur and not prev and not inwin[i]:
                loc = LOC_NAMES[int(d['loc_at'][i])]
                j = i
                while j + 1 < len(at) and on[j + 1]:
                    j += 1
                eff = bool(sh is not None and name == 'combined' and any(sh['mode'][k] in ('cap', 'neutral', 'brake') for k in range(i, j + 1)))
                if loc in ('LH', 'R1'):
                    approach += 1
                else:
                    onsets += 1
                    effective += eff
                effective_all += eff
                times.append([round(float(at[i]), 2), loc, round(float(d['v_at'][i]), 2), eff])
            prev = cur
            i += 1
        res[name] = {'onsets_strict': onsets + approach, 'onsets_excl_LH_R1': onsets, 'approach_LH_R1': approach,
                     'effective_excl_LH_R1': effective if name == 'combined' else None,
                     'effective_strict': effective_all if name == 'combined' else None, 'events': times[:60]}
    dt = np.clip(np.diff(at, append=at[-1]), 0, 0.1)
    minutes = float(np.sum(dt * (sel & moving)) / 60.0)
    for k in ('combined', 'gyro', 'pred'):
        for key in ('onsets_strict', 'onsets_excl_LH_R1', 'effective_excl_LH_R1', 'effective_strict'):
            x = res[k][key]
            res[k][key.replace('onsets', 'per_min').replace('effective', 'effective_per_min')] = (x / minutes) if (x is not None and minutes > 0) else None
    res['ai_moving_min'] = minutes
    return res


def brake_checks(at, mode, bstart, v, anchors):
    """ブレーキの規則: 1 m/s 未満での始まり、R1 の旋回中(起点 → +0.5 秒)のブレーキのフレーム、0.5 秒以内の往復"""
    brake = np.array([m == 'brake' for m in mode])
    starts = np.nonzero(bstart)[0]
    low = int(np.sum(v[starts] < 1.0))
    in_turn = 0
    for ta in anchors:
        in_turn += int(np.sum(brake & (at >= ta) & (at <= ta + TURN_NO_BRAKE)))
    ends = [i for i in range(1, len(brake)) if brake[i - 1] and not brake[i]]
    rt = 0
    for e in ends:
        nxt = starts[starts >= e]
        if len(nxt) and at[nxt[0]] - at[e] < 0.5 - 1e-9:
            rt += 1
    durs = []
    for s_ in starts:
        j = s_
        while j + 1 < len(brake) and brake[j + 1]:
            j += 1
        durs.append(float(at[j] - at[s_]) + 1 / 30)
    return {'brake_starts': int(len(starts)), 'brake_frames': int(brake.sum()), 'brake_start_below_1mps': low,
            'brake_frames_in_r1_turn_0_0p5s': in_turn, 'brake_neutral_roundtrips_lt_0p5s': rt,
            'brake_duration_s_p50_max': [float(np.median(durs)), float(np.max(durs))] if durs else None}


# ------------------------------------------------------------------ 仮想の閉ループ(縦だけ)
def virtual_laps(d, ver, p, pcal, brake='nominal'):
    """AI の周(後から見た R1 の起点〜次の起点、全部 AI、5〜16 秒)を距離で刻んで、層 + P_cal の前後の式で速さ v' を出す。
    続いている周は v' を持ち越す(前の周の R1 の進入の上限が次の周の始まりに効く)。返り値: 周の表と、層の出力の規則の数"""
    vt, erpm = d['vt'], d['erpm']
    v_rec = np.maximum(-erpm * KV, 0.0)
    v_rec = np.convolve(v_rec, np.ones(5) / 5, mode='same')
    s_rec = np.concatenate([[0.0], np.cumsum(v_rec[:-1] * np.clip(np.diff(vt), 0, 0.2))])
    r = np.deg2rad(np.interp(vt, d['it'], d['gz']))
    kap = np.where(v_rec > 0.3, r / np.maximum(v_rec, 0.3), 0.0)
    ka = prev_idx(d['at'], vt)
    zone_v = d['ver'][ver]['zone'][ka]
    u_v = np.array([sc.thr_us_to_cmd(x) for x in d['thr'][ka]])
    steer_v = d['steer_norm'][ka]
    ai_v = d['sel'][ka] & ((vt - d['at'][ka]) < 0.1)
    anchors = d['anchors']
    lap_ok = []
    for k in range(len(anchors) - 1):
        a, b = anchors[k], anchors[k + 1]
        m = (vt >= a) & (vt < b)
        ok = 5.0 <= b - a <= 16.0 and m.sum() > 100 and ai_v[m].all() and not np.any(np.diff(vt[m]) > 0.2)
        lap_ok.append(ok)
    ab = pcal['a_b200'][brake]
    ds = 0.01
    laps = []
    ev = {'brake_starts': 0, 'brake_start_below_1mps': 0, 'brake_in_r1_turn_steps': 0, 'roundtrips_lt_0p5s': 0}
    k = 0
    while k < len(lap_ok):
        if not lap_ok[k]:
            k += 1
            continue
        k_end = k
        while k_end + 1 < len(lap_ok) and lap_ok[k_end + 1]:
            k_end += 1
        a, b = anchors[k], anchors[k_end + 1]
        i0 = int(np.searchsorted(vt, a)); i1 = int(np.searchsorted(vt, b))
        s0, s1 = s_rec[i0], s_rec[i1]
        grid = np.arange(s0, s1, ds)
        idx = np.clip(np.searchsorted(s_rec, grid, side='right') - 1, i0, i1 - 1)
        vp = v_rec[i0]
        vr_prev = v_rec[idx[0]]
        st = sc.CapState()
        loss = 0.0
        lap_bounds = [s_rec[int(np.searchsorted(vt, anchors[j]))] for j in range(k, k_end + 2)]
        lap_loss = [0.0] * (k_end + 1 - k)
        vps = np.empty(len(grid))
        last_brake_end = -1e9
        braking = False
        for g_i, s in enumerate(grid):
            i = idx[g_i]
            vr = v_rec[i]
            tau = vt[i] + loss
            if vr < 0.3:                       # 止まっている・ぶつかった: 記録のまま(時間の差は数えない)
                vp = vr
                vr_prev = vr
                vps[g_i] = vp
                continue
            st, o = sc.cap_step(st, float(tau), zone_v[i], float(vp), float(u_v[i]), p, steer=float(steer_v[i]))
            # 規則の数(仮想の時刻で)
            if o.brake_start:
                ev['brake_starts'] += 1
                ev['brake_start_below_1mps'] += int(vp < 1.0)
                if tau - last_brake_end < 0.5 - 1e-9:
                    ev['roundtrips_lt_0p5s'] += 1
            if o.mode == 'brake':
                braking = True
                if np.any((vt[i] >= anchors) & (vt[i] <= anchors + TURN_NO_BRAKE)):
                    ev['brake_in_r1_turn_steps'] += 1
            elif braking:
                braking = False
                last_brake_end = tau
            scrub = pcal['k_s'] * vp * vp * abs(kap[i])
            tgt = None
            if o.mode == 'brake':
                a_ = -(pcal['c0'] + scrub + ab)
            elif o.mode == 'neutral':
                a_ = -(pcal['c0'] + scrub)
            else:
                tgt = min(vr, o.cap_eff) if o.mode == 'cap' else vr
                if vp < tgt:
                    # 上限から戻る: 較正の駆動(τ_up・a_cap)か、その場所で記録の車が出した加速の大きい方
                    a_rec = (vr * vr - vr_prev * vr_prev) / (2 * ds)
                    a_ = max(min(pcal['a_cap'], (tgt - vp) / pcal['tau_up']), a_rec)
                else:
                    a_ = None
            if a_ is None:
                vn = tgt
            else:
                vn = math.sqrt(max(vp * vp + 2 * a_ * ds, 0.09))
                if tgt is not None:
                    vn = min(vn, tgt)
            vn = min(vn, vr)
            dl = ds * 2.0 / (vp + vn) - ds * 2.0 / (vr_prev + vr)
            loss += dl
            lap_j = int(np.searchsorted(lap_bounds, s, side='right') - 1)
            if 0 <= lap_j < len(lap_loss):
                lap_loss[lap_j] += dl
            vp = vn
            vr_prev = vr
            vps[g_i] = vp
        t_grid = vt[idx]
        for j in range(k, k_end + 1):
            row = {'log': d['log'], 'lap_start': float(anchors[j]), 'lap_s': float(anchors[j + 1] - anchors[j]), 'loss_s': lap_loss[j - k]}
            ta_next = anchors[j + 1]
            w = (t_grid >= ta_next - 0.4) & (t_grid <= ta_next)
            if w.any():
                row['r1_entry_rec'] = float(np.max(v_rec[idx][w]))
                row['r1_entry_capped'] = float(np.max(vps[w]))
            laps.append(row)
        k = k_end + 1
    return laps, ev




# ------------------------------------------------------------------ 本体
def run(preps, g1_logs, other_ai_logs=(), strong_logs=None, variants=('3zone', 'R1GM', 'DE3.3'), pcal=None,
        brakes=('nominal',)):
    """preps: {ログ: prepare の結果}。g1_logs: G1 を判定する AI のログ。other_ai_logs: G1 には入れないが AI のログとして
    検出の誤りを数えるもの。strong_logs: 項目 2 を数えるログ(既定 g1_logs)"""
    g1_logs = [lg for lg in g1_logs if lg in preps]
    strong_logs = [lg for lg in (strong_logs or g1_logs) if lg in preps]
    caps = {v: VARIANTS[v]() for v in variants}
    main_v = '3zone' if '3zone' in caps else variants[0]
    ver = VER
    out_logs = {}
    passes_all = []
    det = {'human': [0, 0, 0], 'ai': [0, 0, 0]}     # (後から見た起点, 因果で検出, 誤検出)
    for lg, d in preps.items():
        dv = d['ver'][ver]
        ca = dv['causal_anchors']
        matched = [bool(len(ca) and np.min(np.abs(ca - ta)) <= 0.3) for ta in d['anchors']]
        false_c = [round(float(c), 2) for c in ca if not len(d['anchors']) or np.min(np.abs(d['anchors'] - c)) > 0.3]
        rows = r1_passes(d)
        for r_, mt in zip(rows, matched):
            r_['causal_detected'] = mt
            det['ai' if r_['ai'] else 'human'][0] += 1
            det['ai' if r_['ai'] else 'human'][1] += mt
        is_ai_log = lg in g1_logs or lg in other_ai_logs
        det['ai' if is_ai_log else 'human'][2] += len(false_c)
        entry = {'role': 'g1' if lg in g1_logs else ('ai' if is_ai_log else 'other'), 'posthoc_r1': len(d['anchors']),
                 'causal_r1': len(ca), 'causal_matched': int(sum(matched)), 'causal_false': false_c,
                 'recorded_fields': d.get('recorded_fields', {})}
        if 'at' in d and d['sel'].any():
            for r_ in rows:
                if not r_['ai']:
                    continue
                for which in ('zone', 'gyro', 'pred'):
                    ok, lead, vsig, dist = signal_of_pass(d, ver, r_['ta'], which)
                    r_[f'sig_{which}'] = ok
                    r_[f'lead_{which}_s'] = lead
                    if which == 'zone':
                        r_['v_at_signal'] = vsig
                        r_['dist_signal_to_r1_m'] = dist
                for cap in (2.5, 2.75):
                    e, vs, dd = r_['entry_mps'], r_.get('v_at_signal'), r_.get('dist_signal_to_r1_m')
                    if e is not None and e > cap and vs is not None and dd and dd > 0.05:
                        r_[f'need_decel_{cap}'] = max(0.0, (max(vs, e) ** 2 - cap ** 2) / (2 * dd))
                    elif e is not None and e > cap:
                        r_[f'need_decel_{cap}'] = None      # 合図なし
            entry['shadow'], entry['virtual'], entry['virtual_rules'] = {}, {}, {}
            for vname, p in caps.items():
                sh = shadow(d, ver, p)
                chk = brake_checks(d['at'], sh['mode'], sh['brake_start'], d['v_at'], d['anchors'])
                modes = sh['mode'][d['sel']]
                chk['mode_share'] = {m: float(np.mean(modes == m)) for m in CAP_MODES}
                inwin = np.zeros(len(d['at']), bool)
                for ta in d['anchors']:
                    inwin |= (d['at'] >= ta - R1_WIN[0]) & (d['at'] <= ta + R1_WIN[1])
                chk['brake_starts_outside_r1_window'] = int(np.sum(sh['brake_start'] & ~inwin))
                chk['brake_starts_outside_r1_window_loc'] = [[round(float(d['at'][i]), 2), LOC_NAMES[int(d['loc_at'][i])]]
                                                              for i in np.nonzero(sh['brake_start'] & ~inwin)[0]]
                entry['shadow'][vname] = chk
                if vname == main_v:
                    entry['false_signals'] = false_signals(d, ver, d['anchors'], sh)
                    rec = d.get('recorded') or {}
                    agree = {}
                    m = d['sel']
                    if 'zone' in rec and m.any():
                        rz = d['ver'][ver].get('zone_replayed', d['ver'][ver]['zone'])
                        agree['zone'] = float(np.mean(rec['zone'][m] == rz[m]))
                    if 'mode' in rec and m.any():
                        agree['mode'] = float(np.mean(rec['mode'][m] == sh['mode'][m]))
                    if agree:
                        entry['recorded_vs_replay'] = agree
                    if 'mode' in rec:
                        entry['recorded'] = recorded_counts(d, rec)
                entry['virtual'][vname], entry['virtual_rules'][vname] = {}, {}
                for brake in brakes:
                    laps, ev = virtual_laps(d, ver, p, pcal, brake)
                    entry['virtual'][vname][brake] = laps
                    entry['virtual_rules'][vname][brake] = ev
        entry['passes'] = rows
        out_logs[lg] = entry
        passes_all += [dict(r_, log=lg) for r_ in rows]
        fs = entry.get('false_signals', {}).get('combined', {})
        log(lg, entry['role'], 'R1', len(d['anchors']), 'causal', len(ca), 'false', len(false_c),
            'fs/min strict %.2f' % (fs.get('per_min_strict') or 0) if fs else '')
    return {'logs': out_logs, 'summary': summarize(out_logs, passes_all, det, caps, g1_logs, strong_logs, brakes)}


def recorded_counts(d, rec):
    """記録の影の判定の数(AI のフレーム)。ON の走行(cap_on)では、実際に出たブレーキが R1 の旋回中に無いかを見る"""
    m = d['sel']
    mode = rec['mode']
    brake = np.array([x == 'brake' for x in mode])
    on = rec['active'].astype(bool) if 'active' in rec else np.zeros(len(mode), bool)
    zone = rec.get('zone')
    turn_zone = np.array([z == 'R1_turn' for z in zone]) if zone is not None else np.zeros(len(mode), bool)
    in_turn = np.zeros(len(mode), bool)
    for ta in d['anchors']:
        in_turn |= (d['at'] >= ta) & (d['at'] <= ta + TURN_NO_BRAKE)
    starts = np.nonzero(brake & ~np.concatenate([[False], brake[:-1]]))[0]
    return {'ai_frames': int(m.sum()), 'on_frames': int((on & m).sum()),
            'shadow_cap_frames': int(np.sum(m & np.array([x in ('cap', 'neutral', 'brake') for x in mode]))),
            'shadow_brake_starts': int(len(starts)),
            'applied_frames': int(np.sum(rec['applied'].astype(bool) & m)) if 'applied' in rec else None,
            'on_brake_frames_in_r1_turn_zone': int(np.sum(on & brake & turn_zone)),
            'on_brake_frames_in_r1_turn_0_0p5s': int(np.sum(on & brake & in_turn)),
            'shadow_brake_frames_in_r1_turn_0_0p5s': int(np.sum(brake & in_turn & m)),
            'not_run_frames': int(np.sum(np.array([str(x) == 'off' for x in mode]) & m))}


def _gate_rate(k, n):
    if not n:
        return {'value': '0/0', 'pass': None, 'note': '数える通過がありません(判定できない)'}
    return {'value': f'{k}/{n}', 'pass': bool(k / n >= 0.9)}


def summarize(logs, passes, det, caps, g1_logs, strong_logs, brakes=('nominal',)):
    S = {'g1_logs': list(g1_logs), 'strong_logs': list(strong_logs)}
    ai_p = [p for p in passes if p['ai'] and 'sig_zone' in p]
    g1p = [p for p in ai_p if p['log'] in g1_logs]
    stp = [p for p in ai_p if p['log'] in strong_logs]
    for name, rows in (('G1_logs', g1p), ('all_ai_logs', ai_p), ('strong_logs', stp)):
        col = [p for p in rows if p['r1_collision']]
        strong = [p for p in rows if p['r1_strong']]
        S.setdefault('r1_signal', {})[name] = {
            'passes': len(rows), 'collision_passes': len(col), 'strong_passes': len(strong),
            **{f'signal_{w}_on_collision': sum(p[f'sig_{w}'] for p in col) for w in ('zone', 'gyro', 'pred')},
            **{f'signal_{w}_on_strong': sum(p[f'sig_{w}'] for p in strong) for w in ('zone', 'gyro', 'pred')},
            'signal_zone_rate_collision': (sum(p['sig_zone'] for p in col) / len(col)) if col else None,
            'lead_s_on_collision': sorted(round(p['lead_zone_s'], 2) for p in col if p['lead_zone_s'] is not None),
            'missed_collision_passes': [{'log': p['log'], 'ta': round(p['ta'], 2), 'entry_mps': round(p['entry_mps'], 2),
                                         'strong': p['r1_strong'], 'lead_zone_s': p['lead_zone_s'], 'lead_gyro_s': p['lead_gyro_s'],
                                         'lead_pred_s': p['lead_pred_s']} for p in col if not p['sig_zone']],
            'signal_zone_all_passes': sum(p['sig_zone'] for p in rows),
            'fast_entry_passes_gt_2.5': sum(1 for p in rows if p['entry_mps'] is not None and p['entry_mps'] > 2.5),
            'signal_zone_on_fast_entry_gt_2.5': sum(p['sig_zone'] for p in rows if p['entry_mps'] is not None and p['entry_mps'] > 2.5),
        }
    fs = {}
    for lg in g1_logs:
        e = logs.get(lg, {})
        if 'false_signals' in e:
            f = e['false_signals']
            fs[lg] = {'ai_moving_min': f['ai_moving_min'],
                      **{f'{k}_{q}': f[k][q] for k in ('combined', 'gyro', 'pred') for q in ('per_min_strict', 'per_min_excl_LH_R1', 'onsets_strict', 'onsets_excl_LH_R1')},
                      'combined_effective_per_min_excl_LH_R1': f['combined']['effective_per_min_excl_LH_R1'],
                      'combined_effective_excl_LH_R1': f['combined']['effective_excl_LH_R1'],
                      'combined_effective_strict': f['combined']['effective_strict']}
    tot_min = sum(v['ai_moving_min'] for v in fs.values())
    pooled = {'ai_moving_min': tot_min,
              'combined_per_min_strict': sum(v['combined_onsets_strict'] for v in fs.values()) / tot_min if tot_min else None,
              'combined_per_min_excl_LH_R1': sum(v['combined_onsets_excl_LH_R1'] for v in fs.values()) / tot_min if tot_min else None,
              'onsets_strict': sum(v['combined_onsets_strict'] for v in fs.values()),
              'onsets_excl_LH_R1': sum(v['combined_onsets_excl_LH_R1'] for v in fs.values())}
    S['false_signals'] = fs
    S['false_signals_pooled'] = pooled
    S['gyro_r1_detection'] = {k: {'posthoc': v[0], 'detected': v[1], 'rate': v[1] / v[0] if v[0] else None, 'false': v[2]} for k, v in det.items()}
    S['gyro_r1_detection_g1_logs'] = {lg: {'posthoc': logs[lg]['posthoc_r1'], 'matched': logs[lg]['causal_matched'],
                                           'false': len(logs[lg]['causal_false'])} for lg in g1_logs if lg in logs}
    for vname, p in caps.items():
        cap = 2.75 if abs(p.cap_R1_app - 2.75) < 1e-6 else 2.5
        need = [q.get(f'need_decel_{cap}') for q in g1p if q['entry_mps'] is not None and q['entry_mps'] > cap]
        sh = {lg: logs[lg]['shadow'][vname] for lg in g1_logs if 'shadow' in logs.get(lg, {})}
        vr = {lg: logs[lg]['virtual_rules'][vname]['nominal'] for lg in g1_logs if 'virtual_rules' in logs.get(lg, {})}
        S.setdefault('brakes', {})[vname] = {
            'passes_entry_gt_cap': len(need), 'no_signal': sum(x is None for x in need),
            'need_decel_sorted': sorted(round(x, 2) for x in need if x is not None),
            'need_decel_le_2.8': sum(1 for x in need if x is not None and x <= 2.8),
            'shadow_brake_starts': {k: v['brake_starts'] for k, v in sh.items()},
            'shadow_brake_starts_outside_r1_window': {k: v['brake_starts_outside_r1_window'] for k, v in sh.items()},
            'shadow_invariants': {k: {'below_1mps': v['brake_start_below_1mps'], 'in_r1_turn': v['brake_frames_in_r1_turn_0_0p5s'],
                                      'roundtrips_lt_0p5s': v['brake_neutral_roundtrips_lt_0p5s']} for k, v in sh.items()},
            'shadow_mode_share': {k: v['mode_share'] for k, v in sh.items()},
            'virtual_rules_nominal': vr,
        }
        for brake in brakes:
            def agg(L):
                if not L:
                    return None
                x = np.array([l_['loss_s'] for l_ in L])
                ent = [(l_['r1_entry_rec'], l_['r1_entry_capped']) for l_ in L if 'r1_entry_rec' in l_]
                return {'laps': len(L), 'loss_mean_s': float(x.mean()), 'loss_median_s': float(np.median(x)),
                        'loss_p90_s': float(np.percentile(x, 90)), 'loss_max_s': float(x.max()),
                        'lap_s_median': float(np.median([l_['lap_s'] for l_ in L])),
                        'r1_entry_rec_gt_cap': sum(1 for a_, b_ in ent if a_ > cap),
                        'r1_entry_capped_gt_cap_plus_0.1': sum(1 for a_, b_ in ent if b_ > cap + 0.1),
                        'r1_entry_rec_median': float(np.median([a_ for a_, b_ in ent])) if ent else None,
                        'r1_entry_capped_median': float(np.median([b_ for a_, b_ in ent])) if ent else None,
                        'r1_entry_capped_max': max((b_ for a_, b_ in ent), default=None)}
            laps = [l_ for lg in g1_logs if 'virtual' in logs.get(lg, {}) for l_ in logs[lg]['virtual'][vname][brake]]
            S.setdefault('lap_loss', {}).setdefault(vname, {})[brake] = {
                'G1_logs': agg(laps),
                'by_log': {lg: agg(logs[lg]['virtual'][vname][brake]) for lg in g1_logs if 'virtual' in logs.get(lg, {})}}
    sig = S['r1_signal']['G1_logs']
    sst = S['r1_signal']['strong_logs']
    gd = S['gyro_r1_detection']['ai']
    gate = {}
    gate['R1_signal_ge_90pct'] = _gate_rate(sig['signal_zone_on_collision'], sig['collision_passes'])
    gate['R1_signal_strong_ge_90pct'] = _gate_rate(sst['signal_zone_on_strong'], sst['strong_passes'])
    per = {k: v['combined_per_min_strict'] for k, v in fs.items()}
    vals = [v for v in per.values() if v is not None]
    gate['false_signal_le_1_per_min'] = {'value': {k: round(v, 2) for k, v in per.items() if v is not None},
                                         'pass': bool(vals and max(vals) <= 1.0) if vals else None}
    gate['false_signal_pooled(参考)'] = {'value': {'strict': pooled['combined_per_min_strict'], 'excl_LH_R1': pooled['combined_per_min_excl_LH_R1'],
                                                  'ai_min': pooled['ai_moving_min']},
                                        'pass': bool(pooled['combined_per_min_strict'] is not None and pooled['combined_per_min_strict'] <= 1.0)}
    gate['gyro_r1_detection_ge_95pct_false_0'] = {'value': {'rate_ai': gd['rate'], 'false_ai_logs': gd['false']},
                                                  'pass': bool(gd['rate'] >= 0.95 and gd['false'] == 0) if gd['rate'] is not None else None}
    common = ['R1_signal_ge_90pct', 'R1_signal_strong_ge_90pct', 'false_signal_le_1_per_min', 'gyro_r1_detection_ge_95pct_false_0']
    for vname in caps:
        ll = S['lap_loss'][vname]['nominal']['G1_logs'] if 'nominal' in S['lap_loss'][vname] else None
        inv = S['brakes'][vname]['shadow_invariants']
        vr = S['brakes'][vname]['virtual_rules_nominal']
        gate[f'lap_loss_le_0.3_{vname}'] = {'value': round(ll['loss_mean_s'], 3) if ll else None,
                                            'pass': bool(ll['loss_mean_s'] <= 0.3) if ll else None}
        ok_sh = all(v['below_1mps'] == 0 and v['in_r1_turn'] == 0 and v['roundtrips_lt_0p5s'] == 0 for v in inv.values())
        ok_vr = all(v['brake_start_below_1mps'] == 0 and v['brake_in_r1_turn_steps'] == 0 and v['roundtrips_lt_0p5s'] == 0 for v in vr.values())
        gate[f'brake_invariants_{vname}'] = {'value': {'shadow': inv, 'virtual': vr}, 'pass': bool(ok_sh and ok_vr) if inv else None}
        nd = S['brakes'][vname]
        gate[f'need_decel_reported_{vname}'] = {'value': f"{nd['need_decel_le_2.8']}/{nd['passes_entry_gt_cap']} ≤ 2.8 m/s²(合図なし {nd['no_signal']})", 'pass': None}
        keys = common + [f'lap_loss_le_0.3_{vname}', f'brake_invariants_{vname}']
        gate[f'G1_pass_{vname}'] = all(gate[k]['pass'] is True for k in keys)
        gate[f'G1_failed_items_{vname}'] = [k for k in keys if gate[k]['pass'] is False]
        gate[f'G1_undecided_items_{vname}'] = [k for k in keys if gate[k]['pass'] is None]
        # protocol(day2.cap_plan_A)と action_plan 注 2 の G1 は 4 項目: R1 の合図 ≥ 90%・R1 以外の誤った合図 ≤ 1 回/分・
        # 旋回中のブレーキ 0・計算上のラップの損 ≤ 0.3 秒(V-tools 4 章 6)。A 案・B 案はこちらで決める。6 項目は参考に並べる
        tb_sh = [v['in_r1_turn'] for v in inv.values()]
        tb_vr = [v['brake_in_r1_turn_steps'] for v in vr.values()]
        gate[f'turn_brake_0_{vname}'] = {'value': {'shadow_frames': int(sum(tb_sh)), 'virtual_steps': int(sum(tb_vr))},
                                         'pass': bool(sum(tb_sh) == 0 and sum(tb_vr) == 0) if (inv or vr) else None}
        pkeys = ['R1_signal_ge_90pct', 'false_signal_le_1_per_min', f'turn_brake_0_{vname}', f'lap_loss_le_0.3_{vname}']
        gate[f'G1_protocol_pass_{vname}'] = all(gate[k]['pass'] is True for k in pkeys)
        gate[f'G1_protocol_failed_items_{vname}'] = [k for k in pkeys if gate[k]['pass'] is False]
        gate[f'G1_protocol_undecided_items_{vname}'] = [k for k in pkeys if gate[k]['pass'] is None]
    S['gate'] = gate
    S['gate_items'] = common + ['lap_loss_le_0.3', 'brake_invariants']
    S['gate_items_protocol'] = ['R1_signal_ge_90pct', 'false_signal_le_1_per_min', 'turn_brake_0', 'lap_loss_le_0.3']
    return S


# ------------------------------------------------------------------ Markdown
ITEM_TEXT = {
    'R1_signal_ge_90pct': '1 R1 で衝突した通過の合図 ≥ 90%',
    'R1_signal_strong_ge_90pct': '2 強い衝突の通過の合図 ≥ 90%',
    'false_signal_le_1_per_min': '3 R1 以外の誤った合図 ≤ 1 回/分(ログそれぞれ)',
    'gyro_r1_detection_ge_95pct_false_0': '4 ジャイロの R1 検出 ≥ 95%・誤検出 0',
    'lap_loss_le_0.3': '5 計算上のラップの損 ≤ 0.3 秒/周',
    'brake_invariants': '6 ブレーキの規則(1 m/s 未満・旋回中・往復 が 0)',
    'turn_brake_0': '旋回中のブレーキ 0(影・仮想とも、起点 → +0.5 秒)',
}


def _pf(x):
    return {True: '合格', False: '**不合格**', None: '判定できない'}[x]


def _val(v):
    if isinstance(v, dict):
        if 'shadow' in v:
            bad = {k: x for k, x in v['shadow'].items() if any(x.values())}
            return '違反なし' if not bad else '影 %s' % json.dumps(bad, ensure_ascii=False)
        return ', '.join('%s %s' % (k, ('%.2f' % x) if isinstance(x, float) else x) for k, x in v.items())
    if isinstance(v, float):
        return '%.2f' % v
    return str(v)


def to_markdown(res, title='区間の速度上限の再生(G1)'):
    S = res['summary']
    g = S['gate']
    vs = res['variants']
    L = ['# %s' % title, '',
         '- 作成: %s(tools/cap_replay.py、読み取りだけ。drive.py の走行には関わらない)' % res['created_at'],
         '- G1 のログ: %s' % ', '.join(S['g1_logs']),
         '- 区間の判定: safety_caps %s(ジャイロの位相 + モデルの予測ステアの合図)。記録の影の欄: %s' % (
             VER, ', '.join('%s=%s' % (k, v) for k, v in res.get('recorded_fields_seen', {}).items()) or 'なし(再生だけ)'),
         '', '## G1 の 6 項目', '',
         '| 項目 | ' + ' | '.join('%s(%s)' % (v, VARIANT_TEXT[v]) for v in vs) + ' |',
         '|---|' + '---|' * len(vs)]
    for k in S['gate_items']:
        cells = []
        for v in vs:
            kk = k if k in g else '%s_%s' % (k, v)
            x = g[kk]
            cells.append('%s(%s)' % (_pf(x['pass']), _val(x['value'])))
        L.append('| %s | %s |' % (ITEM_TEXT[k], ' | '.join(cells)))
    L.append('| **G1** | %s |' % ' | '.join(
        ('**合格**' if g['G1_pass_%s' % v] else '不合格・未定') + ('(落ちた: %s)' % ', '.join(g['G1_failed_items_%s' % v]) if g['G1_failed_items_%s' % v] else '')
        + ('(判定できない: %s)' % ', '.join(g['G1_undecided_items_%s' % v]) if g['G1_undecided_items_%s' % v] else '') for v in vs))
    if S.get('gate_items_protocol') and 'G1_protocol_pass_%s' % vs[0] in g:
        L += ['', '## G1 の判定(protocol の 4 項目。A 案・B 案はこちらで決める)', '',
              '- protocol.json day2.cap_plan_A と action_plan 注 2 の G1 = 1 R1 の合図 ≥ 90%・3 R1 以外の誤った合図 ≤ 1 回/分・'
              '旋回中のブレーキ 0・5 計算上のラップの損 ≤ 0.3 秒。上の 6 項目(g1_replay v1.1)は 2(強い衝突の合図)と 4(ジャイロの検出)を足したもので、参考。',
              '', '| 項目 | ' + ' | '.join(vs) + ' |', '|---|' + '---|' * len(vs)]
        for k in S['gate_items_protocol']:
            cells = []
            for v in vs:
                kk = k if k in g else '%s_%s' % (k, v)
                cells.append('%s(%s)' % (_pf(g[kk]['pass']), _val(g[kk]['value'])))
            L.append('| %s | %s |' % (ITEM_TEXT[k], ' | '.join(cells)))
        L.append('| **G1(4 項目)** | %s |' % ' | '.join(
            ('**合格**' if g['G1_protocol_pass_%s' % v] else '不合格・未定')
            + ('(落ちた: %s)' % ', '.join(g['G1_protocol_failed_items_%s' % v]) if g['G1_protocol_failed_items_%s' % v] else '')
            + ('(判定できない: %s)' % ', '.join(g['G1_protocol_undecided_items_%s' % v]) if g['G1_protocol_undecided_items_%s' % v] else '')
            for v in vs))
        L.append('| G1(6 項目、参考) | %s |' % ' | '.join('合格' if g['G1_pass_%s' % v] else '不合格・未定' for v in vs))
    L += ['', '## 補足', '',
          '- 合図(G1 のログ): 衝突した通過 %s・強い衝突 %s・R1 に 2.5 m/s を超えて入った通過 %d のうち合図 %d' % (
              g['R1_signal_ge_90pct']['value'], g['R1_signal_strong_ge_90pct']['value'],
              S['r1_signal']['G1_logs']['fast_entry_passes_gt_2.5'], S['r1_signal']['G1_logs']['signal_zone_on_fast_entry_gt_2.5']),
          '- 合図を逃した衝突の通過: %s' % ('; '.join('%s %.2f 秒(進入 %.2f m/s、合図の先行 %s 秒)' % (
              m['log'], m['ta'], m['entry_mps'], _val(m['lead_zone_s'])) for m in S['r1_signal']['G1_logs']['missed_collision_passes']) or 'なし'),
          '- 誤った合図(まとめ): %s 回/分(LH・R1 を除くと %s)' % (
              _val(S['false_signals_pooled']['combined_per_min_strict'] or 0.0), _val(S['false_signals_pooled']['combined_per_min_excl_LH_R1'] or 0.0)),
          '']
    L += ['| 案 | ラップの損 平均/中央値/最大 秒 | 周 | R1 進入(記録 → 上限つき)中央値 m/s | 2.5 を超える進入(記録 → 上限つき +0.1) | 必要な減速 ≤ 2.8 m/s² |',
          '|---|---|---|---|---|---|']
    for v in vs:
        ll = S['lap_loss'][v]['nominal']['G1_logs']
        nd = S['brakes'][v]
        if ll:
            L.append('| %s | %.3f / %.3f / %.3f | %d | %.2f → %.2f | %d → %d | %d/%d(合図なし %d) |' % (
                v, ll['loss_mean_s'], ll['loss_median_s'], ll['loss_max_s'], ll['laps'], ll['r1_entry_rec_median'] or 0,
                ll['r1_entry_capped_median'] or 0, ll['r1_entry_rec_gt_cap'], ll['r1_entry_capped_gt_cap_plus_0.1'],
                nd['need_decel_le_2.8'], nd['passes_entry_gt_cap'], nd['no_signal']))
        else:
            L.append('| %s | — | 0 | — | — | — |' % v)
    agree = {lg: e['recorded_vs_replay'] for lg, e in res['logs'].items() if e.get('recorded_vs_replay')}
    if agree:
        L += ['', '## 記録の影の判定と再生の一致(AI のフレーム)', '', '| ログ | 区間 | モード |', '|---|---|---|']
        for lg, a in agree.items():
            L.append('| %s | %s | %s |' % (lg, _val(a.get('zone')), _val(a.get('mode'))))
    recs = {lg: e['recorded'] for lg, e in res['logs'].items() if e.get('recorded')}
    if recs:
        L += ['', '## 記録の影の判定の数(AI_CMD の cap_*、AI のフレーム)', '',
              '| ログ | AI のフレーム | ON のフレーム | 上限が掛かったはず(影) | 影のブレーキの始まり | 実際に変えた | ON で R1 の旋回中のブレーキ(区間 R1_turn / 起点 +0.5 秒) |',
              '|---|---|---|---|---|---|---|']
        for lg, r in recs.items():
            L.append('| %s | %d | %d | %d | %d | %s | %d / %d |' % (lg, r['ai_frames'], r['on_frames'], r['shadow_cap_frames'], r['shadow_brake_starts'],
                                                               r['applied_frames'], r['on_brake_frames_in_r1_turn_zone'], r['on_brake_frames_in_r1_turn_0_0p5s']))
        L.append('- ON の走行で R1 の旋回中のブレーキが 1 つでもあれば、その走行は中止の決まり(protocol safety.stop_run)。')
    if res.get('errors') or res.get('problems') or res.get('computed_labels') or not S['g1_logs']:
        L += ['', '## 読めなかったログ・飛ばしたもの', '']
        if not S['g1_logs']:
            L.append('- **G1 のログが 0 本**(説明欄の cand= の AI の記録が無い、またはラベルが無い)。全部の項目が「判定できない」になる。')
        L += ['- %s: 読めなかった(%s)。G1 にも R1 の検出の誤りにも入れていない' % (lg, err) for lg, err in (res.get('errors') or {}).items()]
        L += ['- %s: 壊れた・無いファイルを飛ばした %d 件(%s)' % (lg, len(pr), '・'.join(pr[:5]) + (' ほか' if len(pr) > 5 else ''))
              for lg, pr in (res.get('problems') or {}).items()]
        if res.get('computed_labels'):
            L.append('- ラベルのファイルが無く、その場で作った: %s' % '・'.join(res['computed_labels']))
    L += ['', '- 限界: 開ループの再生(車の動きは記録のまま)と、縦だけの仮想の閉ループ。上限で速さが変わったあとの舵・道すじの変化は入らない。',
          '- ブレーキの効きは P_cal(10/16 の較正、仮のブレーキ表 16 回)の名目の値。']
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------ 呼び出し
def logs_of_cand(logs, cands, labels_dir=None, raw_dir=None, assign_path=None):
    """説明欄の cand= が cands のどれかで、AI のフレームのある記録"""
    import section_scorecard as ssc
    assign = json.loads(Path(assign_path).read_text(encoding='utf-8')) if assign_path else {}
    out = []
    for lg in logs:
        if lg in assign:
            desc = assign[lg]
        else:
            p = Path(labels_dir or config.DATA_DIR / 'labels') / (lg + '.json')
            q = Path(raw_dir or config.RAW_DIR) / lg / 'log.json'
            src = p if p.exists() else q
            try:
                desc = json.loads(src.read_text(encoding='utf-8')).get('description', '')
            except Exception:
                desc = ''
        if ssc.parse_description(desc).get('cand') in cands and has_ai_frames(lg, raw_dir):
            out.append(lg)
    return out


def has_ai_frames(log, raw_dir=None):
    root = Path(raw_dir or config.RAW_DIR) / log
    return any((s / 'AI_CMD.npz').exists() for s in root.iterdir() if s.is_dir())


def replay(logs, g1_logs=None, other_ai_logs=None, strong_logs=None, variants=('3zone', 'R1GM', 'DE3.3'),
           labels_dir=None, raw_dir=None, pcal_path=PCAL_DEFAULT, brakes=('nominal',), use_recorded=False):
    t_start = time.time()
    pcal = load_pcal(pcal_path)
    preps, errors = {}, {}
    for lg in logs:
        try:
            preps[lg] = prepare(lg, labels_dir, raw_dir, use_recorded)
        except Exception as e:  # 壊れたログ・ラベルの無いログ
            errors[lg] = repr(e)
    ai_logs = [lg for lg in preps if 'at' in preps[lg] and preps[lg]['sel'].any()]
    if g1_logs is None:
        g1_logs = ai_logs
    if other_ai_logs is None:                     # 既定: G1 に入れない AI の記録も、R1 の検出の誤りは AI として数える
        other_ai_logs = [lg for lg in ai_logs if lg not in g1_logs]
    res = {'schema': SCHEMA, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'), 'tool': 'tools/cap_replay.py',
           'layer': 'jetformer/safety_caps.py %s' % VER, 'variants': list(variants),
           'variant_params': {v: VARIANTS[v]()._asdict() for v in variants}, 'phase': sc.PhaseParams()._asdict(),
           'pred_min_speed': sc.PRED_MIN_SPEED, 'pcal_longitudinal': pcal, 'brakes': list(brakes),
           'logs_in': list(logs), 'errors': errors, 'use_recorded': bool(use_recorded),
           'recorded_fields_seen': {k: v for d in preps.values() for k, v in d.get('recorded_fields', {}).items()},
           'problems': {lg: d['problems'] for lg, d in preps.items() if d.get('problems')},
           'computed_labels': [lg for lg, d in preps.items() if d.get('labels_source') == 'computed'],
           'no_ai_frames': [lg for lg in preps if lg not in ai_logs],
           'definitions': __doc__}
    res.update(run(preps, g1_logs, other_ai_logs, strong_logs, variants, pcal, brakes))
    res['elapsed_s'] = round(time.time() - t_start, 1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('logs', nargs='*')
    ap.add_argument('--day', help='この日(YYYYMMDD)のログ全部(人のログも、R1 の検出の誤りを数えるのに使う)')
    ap.add_argument('--labels', default=str(config.DATA_DIR / 'labels'))
    ap.add_argument('--raw', default=str(config.RAW_DIR))
    ap.add_argument('--g1-logs', default=None, help='G1 を判定する AI のログ(カンマ区切り。既定は AI のフレームのあるログ全部)')
    ap.add_argument('--other-ai-logs', default=None, help='G1 に入れないが AI のログとして数えるもの(カンマ区切り。既定は AI のフレームのある残り全部)')
    ap.add_argument('--strong-logs', default=None, help='項目 2(強い衝突の合図)を数えるログ(既定は G1 のログ)')
    ap.add_argument('--variants', default='3zone,R1GM,DE3.3', help='区間の案(%s)' % ','.join(VARIANTS))
    ap.add_argument('--brakes', default='nominal', help='ブレーキの効きの想定(nominal,low,high)')
    ap.add_argument('--pcal', default=str(PCAL_DEFAULT))
    ap.add_argument('--use-recorded', action='store_true', help='記録の影の区間の欄があれば、それで合図を判定する')
    ap.add_argument('--cand', default=None, help='G1 を判定するのを、説明欄の cand= がこれ(カンマ区切り)の記録だけにする(protocol: B の記録)')
    ap.add_argument('--assign', default=None, help='{ログ: 説明欄の文} の JSON(--cand の判定で説明欄を上書き)')
    ap.add_argument('--out', default=None, help='書き出す場所(既定 data/reports/cap_replay_<日>)')
    args = ap.parse_args(argv)
    logs = list(args.logs)
    if args.day:
        day = args.day.replace('-', '')
        logs += sorted(d.name for d in Path(args.raw).glob(day + '-*') if (d / 'log.json').exists())
    logs = sorted(set(logs))
    if not logs:
        ap.error('ログがありません')
    sp = lambda s: [x for x in s.split(',') if x] if s else None      # noqa: E731
    if args.cand and not args.g1_logs:
        args.g1_logs = ','.join(logs_of_cand(logs, sp(args.cand), args.labels, args.raw, args.assign))
        if not args.g1_logs:
            print('説明欄の cand=%s の AI の記録がありません。G1 は判定できません' % args.cand)
            args.g1_logs = None
            g1_empty = True
        else:
            g1_empty = False
    else:
        g1_empty = False
    res = replay(logs, [] if g1_empty else sp(args.g1_logs), sp(args.other_ai_logs), sp(args.strong_logs), tuple(sp(args.variants)),
                 args.labels, args.raw, args.pcal, tuple(sp(args.brakes)), args.use_recorded)
    out = Path(args.out) if args.out else config.DATA_DIR / 'reports' / ('cap_replay_' + (args.day or logs[0][:8]).replace('-', ''))
    out.mkdir(parents=True, exist_ok=True)
    jdump(out / 'cap_replay.json', res)
    (out / 'cap_replay.md').write_text(to_markdown(res), encoding='utf-8')
    g = res['summary']['gate']
    for v in res['variants']:
        print('G1 %s: %s 落ちた %s 判定できない %s' % (v, g['G1_pass_' + v], g['G1_failed_items_' + v], g['G1_undecided_items_' + v]))
        if 'G1_protocol_pass_' + v in g:
            print('G1(protocol 4 項目)%s: %s 落ちた %s 判定できない %s' % (v, g['G1_protocol_pass_' + v], g['G1_protocol_failed_items_' + v],
                                                                 g['G1_protocol_undecided_items_' + v]))
    print('書き出し:', out / 'cap_replay.json', out / 'cap_replay.md', '(%.1f 秒)' % res['elapsed_s'])
    return res


if __name__ == '__main__':
    main()
