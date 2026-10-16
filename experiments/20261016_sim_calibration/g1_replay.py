#!/usr/bin/env python3
"""安全層 v1(jetformer/safety_caps.py)を 10/10 の実車の記録で再生し、机上のゲート G1 を判定する(2026-10-16)

    cd ~/jetformer/experiments/20261016_sim_calibration
    /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 python g1_replay.py

G1(summary_20261010/fast_and_safe_plan.md 6 章、action_plan_20261016/plan.md 注 2):
  - R1 の合図: R1 で衝突した通過の手前で合図が出ている ≥ 90%(007 の R1 の板への強い衝突は前の版で 8/8)
  - R1 以外での誤った合図 ≤ 1 回/分(AI の 5 本 006/007/009/010/101 すべて)
  - ジャイロの位相の R1 検出 ≥ 95%・誤検出 0
  - 上限 2.5 と 2.75 m/s の両方で、ブレーキの回数と必要な減速(≤ 2.8 m/s²)を出す
  - 計算上のラップの損 ≤ 0.3 秒/周(R1・GM・DE の上限を全部入れて)
  - 1 m/s 未満でのブレーキ 0 回、R1 の旋回中(起点 → +0.5 秒)のブレーキ 0 回、0.5 秒以内のブレーキ ↔ 中立の往復 0 回
  (DE 出口の整形と ToF ガード v2 は、この層とは別なので、ここでは判定しない)
再生のしかた:
  1. 影の再生(開ループ): 記録の AI のフレーム(30 Hz)ごとに、そのときの区間・速さ(eRPM)・モデルのスロットル(ゲインのあと)を
     層に入れ、出力を記録する。車の動きは記録のまま(層の出力は効かない)。合図・誤った合図・ブレーキの規則はこれで数える
  2. 仮想の閉ループ(縦だけ): AI の周(R1 の起点〜次の起点)を距離で刻み、層の出力を較正の物理 P_cal の前後の式
     (惰性 c0 + こすれ k_s|a_lat|、ブレーキ s_b(200 − db_b)、駆動 τ_up・a_cap)で速さ v' にする。走る道すじ・区間・
     モデルの指令は記録のまま(位置の関数)。v' は記録の速さを超えない。ラップの損 = Σ ds (1/v' − 1/v_記録)。
     R1 の進入の速さ(起点の手前 0.4 秒の最大)が層でどこまで下がるかも出す
区間は jetformer/safety_caps.py のジャイロの位相(因果的、IMU 100 Hz)と、モデルの予測ステアの合図(AI_CMD.pred ≤ 1150 µs)。
「正解」の R1 の起点と衝突は、tools/label_events.py の course_anchors(後から見た判定)と data/labels の events。
読み取りだけ。data/raw・data/labels は変えない。drive.py にはつながない。
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
import label_events as le  # noqa: E402
from jetformer import safety_caps as sc  # noqa: E402

REPORT = Path('/home/jetson/jetformer_reports/sim_calibration_20261016')
OUTD = REPORT / 'g1'
LABELS = JF / 'data' / 'labels'
G1_LOGS = {'20261010-172910': 'ai_run_006', '20261010-173445': 'ai_run_007', '20261010-203924': 'ai_run_009',
           '20261010-204208': 'ai_run_010', '20261010-204605': 'ai_run_101'}
OTHER_AI = {'20261010-171639': 'ai_run_003', '20261010-172505': 'ai_run_004', '20261010-181504': 'ai_run_008'}
KV = 0.5e-3
PRED_TH = 1150.0
SIG_WIN = 1.5          # 合図を探す: R1 の起点の 1.5 秒前まで
R1_WIN = (2.0, 1.5)    # 誤った合図の数え方: R1 の起点の 2.0 秒前〜1.5 秒後の外の合図の立ち上がり
TURN_NO_BRAKE = 0.5    # R1 の旋回中 = 起点 → +0.5 秒
MIN_LEAD = 0.1         # 合図として数えるのは、R1 の起点の 0.1 秒(3 フレーム)以上前に出たもの(それより遅い合図は層が何もできない)
# 版: v1 = 設計どおり(protocol の governor_v1_sim と同じ規則)、v1.1 = G1 の再生のあとの 1 回だけの直し(safety_caps.py の冒頭)
VERSIONS = {
    'v1': {'phase': sc.PHASE_V1, 'pred_min_speed': 0.0,
           'caps': {'R1_2.5': sc.params_v1(), 'R1_2.75': sc.params_v1(cap_R1_app=2.75, cap_R1_turn=2.75)}},
    'v1.1': {'phase': sc.PhaseParams(), 'pred_min_speed': sc.PRED_MIN_SPEED,
             'caps': {'R1_2.5': sc.params_real_car(), 'R1_2.75': sc.params_r1_275()}},
}
LOC_NAMES = ['unknown', 'R1', 'GM', 'DE', 'other', 'HS', 'LH']


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


def load_pcal():
    p = json.loads((REPORT / 'profiles' / 'P_cal.json').read_text())
    lo = p['longitudinal']
    return {'c0': lo['c0'], 'k_s': lo['k_s'], 'tau_up': lo['tau_up'], 'a_cap': lo['a_cap'],
            'a_b200': {k: v['a_b_at_200us'] for k, v in lo['brake_scenarios'].items()}}


# ------------------------------------------------------------------ 1 本のログ
def prepare(log_name):
    info = json.loads((LABELS / (log_name + '.json')).read_text(encoding='utf-8'))
    lab = np.load(LABELS / (log_name + '.npz'))
    t0 = info['t0_us']
    L = le.load_log(log_name)
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
           'gm': np.asarray(gm), 'ver': {}}
    has_ai = A is not None and len(A.get('timestamps_us', []))
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
        out.update({'at': at, 'sel': sel & fresh, 'pred_sig': (pred_min <= PRED_TH) & sel, 'v_at': v_at, 'thr': thr,
                    'steer_norm': np.array([sc.steer_us_to_norm(x) for x in steer_us]), 'loc_at': lab['loc_code'][kl], 'ki': ki})
    # 因果的なジャイロの位相(版ごと)
    n = len(it)
    for ver, cfg in VERSIONS.items():
        st = sc.PhaseState()
        zone_g = np.empty(n, dtype=object)
        turning = np.zeros(n, bool)
        conf = np.zeros(n, bool)
        anchor_c = np.full(n, np.nan)
        for i in range(n):
            st, o = sc.phase_step(st, float(it[i]), float(gz[i]), float(v_imu[i]), cfg['phase'])
            zone_g[i] = o.zone
            turning[i] = o.turning_right
            conf[i] = o.r1_confirmed
            if o.anchor_t is not None:
                anchor_c[i] = o.anchor_t
        dv = {'zone_g': zone_g, 'turning': turning, 'causal_anchors': np.unique(anchor_c[conf])}
        if has_ai:
            zg = zone_g[out['ki']]
            dv['gyro_app'] = np.array([z == 'R1_app' for z in zg])
            dv['zone'] = np.array([sc.combine_zone(z, bool(tr), bool(ps), v=float(v), pred_min_speed=cfg['pred_min_speed'])
                                   for z, tr, ps, v in zip(zg, turning[out['ki']], out['pred_sig'], out['v_at'])], dtype=object)
            dv['pred_eff'] = out['pred_sig'] & (out['v_at'] >= cfg['pred_min_speed'])
        out['ver'][ver] = dv
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
def run_version(ver, preps, pcal):
    caps = VERSIONS[ver]['caps']
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
        is_ai_log = lg in G1_LOGS or lg in OTHER_AI
        det['ai' if is_ai_log else 'human'][2] += len(false_c)
        entry = {'name': G1_LOGS.get(lg) or OTHER_AI.get(lg) or 'human', 'posthoc_r1': len(d['anchors']), 'causal_r1': len(ca),
                 'causal_matched': int(sum(matched)), 'causal_false': false_c}
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
                chk['mode_share'] = {m: float(np.mean(modes == m)) for m in ('pass', 'cap', 'neutral', 'brake')}
                inwin = np.zeros(len(d['at']), bool)
                for ta in d['anchors']:
                    inwin |= (d['at'] >= ta - R1_WIN[0]) & (d['at'] <= ta + R1_WIN[1])
                chk['brake_starts_outside_r1_window'] = int(np.sum(sh['brake_start'] & ~inwin))
                chk['brake_starts_outside_r1_window_loc'] = [[round(float(d['at'][i]), 2), LOC_NAMES[int(d['loc_at'][i])]]
                                                              for i in np.nonzero(sh['brake_start'] & ~inwin)[0]]
                entry['shadow'][vname] = chk
                if vname == 'R1_2.5':
                    entry['false_signals'] = false_signals(d, ver, d['anchors'], sh)
                entry['virtual'][vname], entry['virtual_rules'][vname] = {}, {}
                for brake in ('nominal', 'low', 'high'):
                    laps, ev = virtual_laps(d, ver, p, pcal, brake)
                    entry['virtual'][vname][brake] = laps
                    entry['virtual_rules'][vname][brake] = ev
        entry['passes'] = rows
        out_logs[lg] = entry
        passes_all += [dict(r_, log=lg) for r_ in rows]
        fs = entry.get('false_signals', {}).get('combined', {})
        log(ver, lg, entry['name'], 'R1', len(d['anchors']), 'causal', len(ca), 'false', len(false_c),
            'fs/min strict %.2f excl %.2f' % (fs.get('per_min_strict') or 0, fs.get('per_min_excl_LH_R1') or 0) if fs else '')
    return {'logs': out_logs, 'summary': summarize(out_logs, passes_all, det, caps)}


def summarize(logs, passes, det, caps):
    S = {}
    ai_p = [p for p in passes if p['ai'] and 'sig_zone' in p]
    g1p = [p for p in ai_p if p['log'] in G1_LOGS]
    for name, rows in (('G1_logs', g1p), ('all_ai_logs', ai_p), ('ai_run_007', [p for p in ai_p if p['log'] == '20261010-173445'])):
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
        }
    fs = {}
    for lg, name in G1_LOGS.items():
        e = logs.get(lg, {})
        if 'false_signals' in e:
            f = e['false_signals']
            fs[name] = {'ai_moving_min': f['ai_moving_min'],
                        **{f'{k}_{q}': f[k][q] for k in ('combined', 'gyro', 'pred') for q in ('per_min_strict', 'per_min_excl_LH_R1', 'onsets_strict', 'onsets_excl_LH_R1')},
                        'combined_effective_per_min_excl_LH_R1': f['combined']['effective_per_min_excl_LH_R1'],
                        'combined_effective_excl_LH_R1': f['combined']['effective_excl_LH_R1'],
                        'combined_effective_strict': f['combined']['effective_strict']}
    tot_min = sum(v['ai_moving_min'] for v in fs.values())
    fs['pooled_5_logs'] = {'ai_moving_min': tot_min,
                           'combined_per_min_strict': sum(v['combined_onsets_strict'] for v in fs.values()) / tot_min if tot_min else None,
                           'combined_per_min_excl_LH_R1': sum(v['combined_onsets_excl_LH_R1'] for v in fs.values()) / tot_min if tot_min else None,
                           'combined_effective_per_min_excl_LH_R1': sum(v['combined_effective_excl_LH_R1'] for v in fs.values()) / tot_min if tot_min else None,
                           'combined_effective_per_min_strict': sum(v['combined_effective_strict'] for v in fs.values()) / tot_min if tot_min else None,
                           'onsets_strict': sum(v['combined_onsets_strict'] for v in fs.values()),
                           'onsets_excl_LH_R1': sum(v['combined_onsets_excl_LH_R1'] for v in fs.values())}
    S['false_signals'] = fs
    S['gyro_r1_detection'] = {k: {'posthoc': v[0], 'detected': v[1], 'rate': v[1] / v[0] if v[0] else None, 'false': v[2]} for k, v in det.items()}
    S['gyro_r1_detection_g1_logs'] = {G1_LOGS[lg]: {'posthoc': logs[lg]['posthoc_r1'], 'matched': logs[lg]['causal_matched'],
                                                     'false': len(logs[lg]['causal_false'])} for lg in G1_LOGS if lg in logs}
    for vname in caps:
        cap = 2.5 if vname == 'R1_2.5' else 2.75
        need = [p.get(f'need_decel_{cap}') for p in g1p if p['entry_mps'] is not None and p['entry_mps'] > cap]
        sh = {G1_LOGS[lg]: logs[lg]['shadow'][vname] for lg in G1_LOGS if 'shadow' in logs.get(lg, {})}
        vr = {G1_LOGS[lg]: logs[lg]['virtual_rules'][vname]['nominal'] for lg in G1_LOGS if 'virtual_rules' in logs.get(lg, {})}
        S.setdefault('brakes', {})[vname] = {
            'passes_entry_gt_cap': len(need), 'no_signal': sum(x is None for x in need),
            'need_decel_sorted': sorted(round(x, 2) for x in need if x is not None),
            'need_decel_le_2.8': sum(1 for x in need if x is not None and x <= 2.8),
            'need_decel_le_2.0_nominal_brake': sum(1 for x in need if x is not None and x <= 2.0),
            'shadow_brake_starts': {k: v['brake_starts'] for k, v in sh.items()},
            'shadow_brake_starts_outside_r1_window': {k: v['brake_starts_outside_r1_window'] for k, v in sh.items()},
            'shadow_brake_starts_outside_r1_window_loc': {k: v['brake_starts_outside_r1_window_loc'] for k, v in sh.items()},
            'shadow_invariants': {k: {'below_1mps': v['brake_start_below_1mps'], 'in_r1_turn': v['brake_frames_in_r1_turn_0_0p5s'],
                                      'roundtrips_lt_0p5s': v['brake_neutral_roundtrips_lt_0p5s']} for k, v in sh.items()},
            'shadow_mode_share': {k: v['mode_share'] for k, v in sh.items()},
            'shadow_brake_duration_s_p50_max': {k: v['brake_duration_s_p50_max'] for k, v in sh.items()},
            'virtual_rules_nominal': vr,
        }
    for vname in caps:
        capv = 2.5 if vname == 'R1_2.5' else 2.75
        for brake in ('nominal', 'low', 'high'):
            def agg(L):
                if not L:
                    return None
                x = np.array([l['loss_s'] for l in L])
                ent = [(l['r1_entry_rec'], l['r1_entry_capped']) for l in L if 'r1_entry_rec' in l]
                return {'laps': len(L), 'loss_mean_s': float(x.mean()), 'loss_median_s': float(np.median(x)), 'loss_p90_s': float(np.percentile(x, 90)),
                        'loss_max_s': float(x.max()), 'lap_s_median': float(np.median([l['lap_s'] for l in L])),
                        'r1_entry_rec_gt_cap': sum(1 for a_, b_ in ent if a_ > capv), 'r1_entry_capped_gt_cap_plus_0.1': sum(1 for a_, b_ in ent if b_ > capv + 0.1),
                        'r1_entry_rec_ge_3.1': sum(1 for a_, b_ in ent if a_ >= 3.1), 'r1_entry_capped_ge_3.1': sum(1 for a_, b_ in ent if b_ >= 3.1),
                        'r1_entry_rec_median': float(np.median([a_ for a_, b_ in ent])) if ent else None,
                        'r1_entry_capped_median': float(np.median([b_ for a_, b_ in ent])) if ent else None,
                        'r1_entry_capped_max': max((b_ for a_, b_ in ent), default=None)}
            laps = [l for lg in G1_LOGS if 'virtual' in logs.get(lg, {}) for l in logs[lg]['virtual'][vname][brake]]
            laps_all = [l for lg, e in logs.items() if 'virtual' in e for l in e['virtual'][vname][brake]]
            S.setdefault('lap_loss', {}).setdefault(vname, {})[brake] = {
                'G1_logs': agg(laps), 'all_ai_logs': agg(laps_all),
                'by_log': {G1_LOGS[lg]: agg(logs[lg]['virtual'][vname][brake]) for lg in G1_LOGS if 'virtual' in logs.get(lg, {})}}
    sig = S['r1_signal']['G1_logs']
    s7 = S['r1_signal']['ai_run_007']
    gd = S['gyro_r1_detection']['ai']
    gate = {}
    gate['R1_signal_ge_90pct'] = {'value': f"{sig['signal_zone_on_collision']}/{sig['collision_passes']}",
                                  'pass': bool(sig['signal_zone_rate_collision'] is not None and sig['signal_zone_rate_collision'] >= 0.9)}
    gate['R1_signal_007_board_ge_90pct'] = {'value': f"{s7['signal_zone_on_strong']}/{s7['strong_passes']}",
                                            'pass': bool(s7['strong_passes'] and s7['signal_zone_on_strong'] / s7['strong_passes'] >= 0.9)}
    per5 = {k: v['combined_per_min_strict'] for k, v in fs.items() if k != 'pooled_5_logs'}
    gate['false_signal_le_1_per_min_all5'] = {'value': {k: round(v, 2) for k, v in per5.items() if v is not None},
                                              'pass': bool(per5 and max(v for v in per5.values() if v is not None) <= 1.0)}
    per5x = {k: v['combined_per_min_excl_LH_R1'] for k, v in fs.items() if k != 'pooled_5_logs'}
    pooled = fs['pooled_5_logs']
    gate['false_signal_pooled_5_logs(参考)'] = {'value': {'strict': pooled['combined_per_min_strict'], 'excl_LH_R1': pooled['combined_per_min_excl_LH_R1'],
                                                         'effective_strict': pooled['combined_effective_per_min_strict'], 'ai_min': pooled['ai_moving_min']},
                                               'pass': bool(pooled['combined_per_min_strict'] is not None and pooled['combined_per_min_strict'] <= 1.0)}
    gate['false_signal_le_1_per_min_all5_excl_LH_R1(参考)'] = {'value': {k: round(v, 2) for k, v in per5x.items() if v is not None},
                                                             'pass': bool(per5x and max(v for v in per5x.values() if v is not None) <= 1.0)}
    gate['gyro_r1_detection_ge_95pct_false_0'] = {'value': {'rate_ai': gd['rate'], 'false_ai_logs': gd['false']},
                                                  'pass': bool(gd['rate'] is not None and gd['rate'] >= 0.95 and gd['false'] == 0)}
    for vname in caps:
        ll = S['lap_loss'][vname]['nominal']['G1_logs']
        inv = S['brakes'][vname]['shadow_invariants']
        vr = S['brakes'][vname]['virtual_rules_nominal']
        gate[f'lap_loss_le_0.3_{vname}'] = {'value': round(ll['loss_mean_s'], 3) if ll else None, 'pass': bool(ll and ll['loss_mean_s'] <= 0.3)}
        ok_sh = all(v['below_1mps'] == 0 and v['in_r1_turn'] == 0 and v['roundtrips_lt_0p5s'] == 0 for v in inv.values())
        ok_vr = all(v['brake_start_below_1mps'] == 0 and v['brake_in_r1_turn_steps'] == 0 and v['roundtrips_lt_0p5s'] == 0 for v in vr.values())
        gate[f'brake_invariants_{vname}'] = {'value': {'shadow': inv, 'virtual': vr}, 'pass': bool(ok_sh and ok_vr)}
        nd = S['brakes'][vname]
        gate[f'need_decel_reported_{vname}'] = {'value': f"{nd['need_decel_le_2.8']}/{nd['passes_entry_gt_cap']} ≤ 2.8 m/s²(合図なし {nd['no_signal']})", 'pass': None}
    main_keys = ['R1_signal_ge_90pct', 'R1_signal_007_board_ge_90pct', 'false_signal_le_1_per_min_all5', 'gyro_r1_detection_ge_95pct_false_0',
                 'lap_loss_le_0.3_R1_2.5', 'brake_invariants_R1_2.5']
    gate['G1_pass_R1_2.5'] = all(gate[k]['pass'] for k in main_keys)
    gate['G1_pass_R1_2.75'] = all(gate[k.replace('R1_2.5', 'R1_2.75')]['pass'] for k in main_keys)
    gate['G1_failed_items_R1_2.5'] = [k for k in main_keys if not gate[k]['pass']]
    S['gate'] = gate
    return S


def main():
    t_start = time.time()
    pcal = load_pcal()
    logs = sorted(p.stem for p in LABELS.glob('2026*.json'))
    preps = {}
    errors = {}
    for lg in logs:
        if not (JF / 'data' / 'raw' / lg).exists():
            continue
        try:
            preps[lg] = prepare(lg)
        except Exception as e:  # 壊れたログ
            errors[lg] = repr(e)
    res = {'schema': 'jf-safety-g1-replay/v2', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'),
           'layer': 'jetformer/safety_caps.py(drive.py にはつないでいない)',
           'versions': {ver: {'phase': cfg['phase']._asdict(), 'pred_min_speed': cfg['pred_min_speed'],
                              'caps': {k: v._asdict() for k, v in cfg['caps'].items()}} for ver, cfg in VERSIONS.items()},
           'definitions': __doc__, 'errors': errors, 'pcal_longitudinal': pcal}
    for ver in VERSIONS:
        res[ver] = run_version(ver, preps, pcal)
    jdump(REPORT / 'g1_replay.json', res)
    log('done', round(time.time() - t_start, 1), 's')
    return res


if __name__ == '__main__':
    main()
