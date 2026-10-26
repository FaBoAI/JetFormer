#!/usr/bin/env python3
"""段 Cap4(2026-10-23): 4 つめの上限(位相が不確か = unknown のときの 3.0 m/s)を外すと、10/10 の AI 走行で何が変わるか。

読み取りだけ・CPU だけ(numpy)。tools/cap_replay.py・jetformer/safety_caps.py は変えない。
再生の側で、区間の案 'noU' = safety_caps.params_real_car(cap_unknown=INF) を cap_replay.VARIANTS に足して回す
(drive.py で外すなら ZoneCapConfig(cap=CapParams(cap_unknown=INF)) と同じ値)。

  (a) 3zone = v1.1 そのまま(R1 2.5・GM 2.2・DE 3.0・unknown 3.0)
  (b) noU   = unknown の上限を外した版(R1 2.5・GM 2.2・DE 3.0・unknown なし)

出すもの(--out の下):
  cap4_replay.json : G1 の 6 項目(と protocol の 4 項目)・フレームの割合と区間の内訳・unknown の理由ごとの時間・
                     ブレーキの回数・「上限が無ければ出した前進」との差・仮想の閉ループ(AI の区間全部)の時間の損・
                     unknown のまま R1 に入った通過・止める決まり(A2 3 周・2 周・元の 4 G の読み方)
  cap_replay.md    : cap_replay.to_markdown の表(2 案)

  jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I cap4_replay.py --out DIR
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
import cap_replay as cr  # noqa: E402
from jetformer import config  # noqa: E402
from jetformer import safety_caps as sc  # noqa: E402

# g1_replay.py(10/16)と同じ分け方
G1_LOGS = ['20261010-172910', '20261010-173445', '20261010-203924', '20261010-204208', '20261010-204605']
OTHER_AI = ['20261010-171639', '20261010-172505', '20261010-181504']
EARLY_AI = ['20261010-155820', '20261010-160306', '20261010-163247']     # 10/10 の最初の 3 本(V-protocol の 11 本に入る)
STRONG_LOGS = ['20261010-173445']                                        # 項目 2 = 007 の R1 の板(g1_replay と同じ)
NAMES = {'20261010-172910': '006', '20261010-173445': '007', '20261010-203924': '009', '20261010-204208': '010',
         '20261010-204605': '101', '20261010-171639': '003', '20261010-172505': '004', '20261010-181504': '008',
         '20261010-155820': 'try', '20261010-160306': '001', '20261010-163247': '002'}
VARIANTS = ('3zone', 'noU')
cr.VARIANTS['noU'] = lambda: sc.params_real_car(cap_unknown=sc.INF)
cr.VARIANT_TEXT['noU'] = 'R1 2.5・GM 2.2・DE 3.0・不確かなときの上限なし'
VER = cr.VER
KV = cr.KV
CAPPED = ('cap', 'neutral', 'brake')
CAUSES = ('start', 'lost16', 'stop', 'fallback', 'psi_range')
CAUSE_TEXT = {'start': '走り始め〜最初の R1', 'lost16': 'R1 から 16 秒超', 'stop': '1 秒以上の停止のあと',
              'fallback': 'R1 の手前で向きが 240 度を下回った(引き返し)', 'psi_range': '向きが範囲外'}
OTHER_LOC = {'other': 'そのほか', 'HS': 'そのほか', 'unknown': 'そのほか'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ------------------------------------------------------------------ unknown の理由
def phase_causes(d):
    """safety_caps.phase_step を記録の IMU で回し直し、各サンプルの区間(ジャイロ)と、unknown ならその理由を返す。
    prepare の zone_g と一致することを確かめる(違えば例外)"""
    it, gz, vt, erpm = d['it'], d['gz'], d['vt'], d['erpm']
    v_imu = -erpm[cr.prev_idx(vt, it)] * KV
    p = sc.PhaseParams()
    st = sc.PhaseState()
    n = len(it)
    zone = np.empty(n, dtype=object)
    cause = np.full(n, '', dtype=object)
    reason = 'start'
    for i in range(n):
        prev = st
        st, o = sc.phase_step(st, float(it[i]), float(gz[i]), float(v_imu[i]), p)
        if prev.anchor_t is not None and st.anchor_t is None:
            stopped = st.stop_t0 is not None and (float(it[i]) - st.stop_t0) >= p.stop_reset_s - 1e-12
            reason = 'stop' if stopped else 'fallback'
        zone[i] = o.zone
        if o.zone == 'unknown':
            if st.anchor_t is None:
                cause[i] = reason
            elif float(it[i]) - st.anchor_t > p.lost_s:
                cause[i] = 'lost16'
            else:
                cause[i] = 'psi_range'
    zg = d['ver'][VER]['zone_g']
    if not np.array_equal(zone, zg):
        raise RuntimeError('phase_causes: 区間が prepare と一致しない(%d サンプル)' % int(np.sum(zone != zg)))
    return zone, cause


# ------------------------------------------------------------------ フレームの数
def frame_stats(d, sh, cause_at):
    sel, at = d['sel'], d['at']
    dt = np.clip(np.diff(at, append=at[-1]), 0, 0.1)
    zone = d['ver'][VER]['zone']
    mode = sh['mode']
    capped = np.array([m in CAPPED for m in mode]) & sel
    thr_in, thr_out = d['thr'], sh['thr_out']
    dthr = np.where(sel, thr_out - thr_in, 0.0)                 # µs、正 = 前進を弱めた
    u_in = np.array([sc.thr_us_to_cmd(x) for x in thr_in])
    u_out = np.array([sc.thr_us_to_cmd(x) for x in thr_out])
    du = np.where(sel, u_in - u_out, 0.0)                       # 速さの単位の指令 [m/s]
    out = {'ai_frames': int(sel.sum()), 'ai_s': float(np.sum(dt[sel])), 'capped_frames': int(capped.sum()),
           'capped_s': float(np.sum(dt[capped])),
           'capped_share': float(capped.sum() / sel.sum()) if sel.sum() else None,
           'mode_frames': {m: int(np.sum(sel & np.array([x == m for x in mode]))) for m in sc.CAP_MODES},
           'dthr_us_mean_all_ai': float(dthr[sel].mean()) if sel.any() else None,
           'dthr_us_mean_capped': float(dthr[capped].mean()) if capped.any() else None,
           'du_mps_mean_all_ai': float(du[sel].mean()) if sel.any() else None,
           'du_mps_mean_capped': float(du[capped].mean()) if capped.any() else None,
           'thr_in_fwd_us_mean_ai': float((sc.THR_NEUTRAL_US - thr_in[sel]).mean()) if sel.any() else None}
    bz = {}
    for z in sc.ZONES:
        m = sel & np.array([x == z for x in zone])
        c = m & capped
        bz[z] = {'frames': int(m.sum()), 'ai_s': float(np.sum(dt[m])), 'capped_frames': int(c.sum()),
                 'capped_s': float(np.sum(dt[c])),
                 'dthr_us_mean_capped': float(dthr[c].mean()) if c.any() else None,
                 'du_mps_mean_capped': float(du[c].mean()) if c.any() else None,
                 'dthr_us_sum_x_s': float(np.sum(dthr[m] * dt[m]))}
    out['by_zone'] = bz
    unk = sel & np.array([x == 'unknown' for x in zone])
    out['unknown_by_cause'] = {c_: {'ai_s': float(np.sum(dt[unk & (cause_at == c_)])),
                                    'capped_s': float(np.sum(dt[unk & capped & (cause_at == c_)])),
                                    'capped_frames': int(np.sum(unk & capped & (cause_at == c_)))} for c_ in CAUSES}
    return out


def unknown_time(d, cause_at):
    """AI 側の時間のうち、ジャイロの位相が unknown だった時間(理由ごと)と、合図 (b) と合わせた最後の区間も unknown だった時間"""
    sel, at = d['sel'], d['at']
    dt = np.clip(np.diff(at, append=at[-1]), 0, 0.1)
    zg = d['ver'][VER]['zone_g'][d['ki']]
    zf = d['ver'][VER]['zone']
    g_unk = sel & np.array([x == 'unknown' for x in zg])
    f_unk = sel & np.array([x == 'unknown' for x in zf])
    res = {'ai_s': float(np.sum(dt[sel])), 'gyro_unknown_s': float(np.sum(dt[g_unk])), 'final_unknown_s': float(np.sum(dt[f_unk])),
           'gyro_unknown_by_cause_s': {c: float(np.sum(dt[g_unk & (cause_at == c)])) for c in CAUSES},
           'final_unknown_by_cause_s': {c: float(np.sum(dt[f_unk & (cause_at == c)])) for c in CAUSES}}
    # unknown の続いたかたまり(AI 側の時間で、0.5 秒以内の途切れはつなぐ)
    eps = []
    cur = None
    for i in np.nonzero(f_unk)[0]:
        if cur is not None and at[i] - cur['t1'] <= 0.5 and cause_at[i] == cur['cause']:
            cur['t1'] = float(at[i]); cur['s'] += float(dt[i])
        else:
            if cur is not None:
                eps.append(cur)
            cur = {'t0': float(at[i]), 't1': float(at[i]), 's': float(dt[i]), 'cause': cause_at[i]}
    if cur is not None:
        eps.append(cur)
    res['episodes'] = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()} for e in eps]
    res['episodes_n_ge_1s'] = sum(1 for e in eps if e['s'] >= 1.0)
    return res


# ------------------------------------------------------------------ 仮想の閉ループ(AI の区間全部)
def virtual_stretches(d, p, pcal, brake='nominal', min_len_s=1.0):
    """cap_replay.virtual_laps と同じ式で、周に切らず、AI が続いた区間全部(1 周目の手前・16 秒を超える周も含む)を回す。
    返り値: 時間の損の合計と区間ごとの内訳、R1 の進入の速さ(記録 → 上限つき)、ブレーキの規則の数"""
    vt, erpm, at = d['vt'], d['erpm'], d['at']
    v_rec = np.maximum(-erpm * KV, 0.0)
    v_rec = np.convolve(v_rec, np.ones(5) / 5, mode='same')
    s_rec = np.concatenate([[0.0], np.cumsum(v_rec[:-1] * np.clip(np.diff(vt), 0, 0.2))])
    r = np.deg2rad(np.interp(vt, d['it'], d['gz']))
    kap = np.where(v_rec > 0.3, r / np.maximum(v_rec, 0.3), 0.0)
    ka = cr.prev_idx(at, vt)
    zone_v = d['ver'][VER]['zone'][ka]
    u_v = np.array([sc.thr_us_to_cmd(x) for x in d['thr'][ka]])
    steer_v = d['steer_norm'][ka]
    ai_v = d['sel'][ka] & ((vt - at[ka]) < 0.1)
    anchors = d['anchors']
    ab = pcal['a_b200'][brake]
    ds = 0.01
    # AI が続いた区間
    stretches = []
    i = 0
    n = len(vt)
    while i < n:
        if not ai_v[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ai_v[j + 1] and vt[j + 1] - vt[j] <= 0.2:
            j += 1
        if vt[j] - vt[i] >= min_len_s:
            stretches.append((i, j + 1))
        i = j + 1
    ev = {'brake_starts': 0, 'brake_start_below_1mps': 0, 'brake_in_r1_turn_steps': 0, 'roundtrips_lt_0p5s': 0}
    loss_total = 0.0
    loss_zone = {z: 0.0 for z in sc.ZONES}
    entries = []
    rec_time = 0.0
    for (i0, i1) in stretches:
        grid = np.arange(s_rec[i0], s_rec[i1 - 1], ds)
        if len(grid) < 2:
            continue
        rec_time += float(vt[i1 - 1] - vt[i0])
        idx = np.clip(np.searchsorted(s_rec, grid, side='right') - 1, i0, i1 - 1)
        vp = v_rec[i0]
        vr_prev = v_rec[idx[0]]
        st = sc.CapState()
        loss = 0.0
        vps = np.empty(len(grid))
        last_brake_end = -1e9
        braking = False
        for g_i in range(len(grid)):
            k = idx[g_i]
            vr = v_rec[k]
            tau = vt[k] + loss
            if vr < 0.3:
                vp = vr
                vr_prev = vr
                vps[g_i] = vp
                continue
            st, o = sc.cap_step(st, float(tau), zone_v[k], float(vp), float(u_v[k]), p, steer=float(steer_v[k]))
            if o.brake_start:
                ev['brake_starts'] += 1
                ev['brake_start_below_1mps'] += int(vp < 1.0)
                if tau - last_brake_end < 0.5 - 1e-9:
                    ev['roundtrips_lt_0p5s'] += 1
            if o.mode == 'brake':
                braking = True
                if np.any((vt[k] >= anchors) & (vt[k] <= anchors + cr.TURN_NO_BRAKE)):
                    ev['brake_in_r1_turn_steps'] += 1
            elif braking:
                braking = False
                last_brake_end = tau
            scrub = pcal['k_s'] * vp * vp * abs(kap[k])
            tgt = None
            if o.mode == 'brake':
                a_ = -(pcal['c0'] + scrub + ab)
            elif o.mode == 'neutral':
                a_ = -(pcal['c0'] + scrub)
            else:
                tgt = min(vr, o.cap_eff) if o.mode == 'cap' else vr
                if vp < tgt:
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
            loss_zone[zone_v[k]] += dl
            vp = vn
            vr_prev = vr
            vps[g_i] = vp
        loss_total += loss
        t_grid = vt[idx]
        for ta in anchors:
            if t_grid[0] <= ta - 0.4 and ta <= t_grid[-1]:
                w = (t_grid >= ta - 0.4) & (t_grid <= ta)
                if w.any():
                    entries.append({'ta': float(ta), 'rec': float(np.max(v_rec[idx][w])), 'capped': float(np.max(vps[w]))})
    return {'stretches': len(stretches), 'stretch_s': rec_time, 'loss_s': loss_total, 'loss_by_zone_s': loss_zone,
            'r1_entries': entries, 'rules': ev}


# ------------------------------------------------------------------ R1 の通過(手前の位相)
def approach_phase(d, ta):
    """R1 の起点 ta の 0.5 秒前のジャイロの区間(unknown = 位相が分からないまま R1 に来た)"""
    k = int(cr.prev_idx(d['it'], np.array([ta - 0.5]))[0])
    return d['ver'][VER]['zone_g'][k]


# ------------------------------------------------------------------ 止める決まり(A2 と元の決まり)
def ai_seconds_until(lab, t):
    m = lab['ai'].astype(bool) & (lab['t_s'] < t)
    return float(np.sum(lab['dt_s'][m]))


def streak_fires(items, n, lap_of):
    """items: (場所, 時刻) の時刻順。同じ場所で n 周続いたら 1 回止める。止めたら数え直す(次の走行のつもり)"""
    fires = []
    streak = {}
    for loc, t in sorted(items, key=lambda x: x[1]):
        k = lap_of(t)
        if k is None:
            continue
        last, c = streak.get(loc, (None, 0))
        if last == k:
            continue
        c = c + 1 if (last is not None and last == k - 1) else 1
        streak[loc] = (k, c)
        if c >= n:
            fires.append({'t_s': round(float(t), 2), 'loc': loc, 'lap': int(k)})
            streak = {}
    return fires


def consecutive_strong_fires(cols):
    """元の文の読み方 (a): 続いた 2 件の AI の衝突が、どちらも 4 G 以上で同じ場所。止めたら数え直す"""
    fires = []
    prev = None
    for c in cols:
        if prev is not None and prev['strong'] and c['strong'] and prev['loc'] == c['loc']:
            fires.append({'t_s': round(float(c['t_s']), 2), 'loc': c['loc']})
            prev = None
            continue
        prev = c
    return fires


def stop_rules(lg, d, passes):
    info = d['info']
    lab = d['lab']
    ev = info['events']
    cols = sorted([c for c in ev['collisions'] if c.get('ai')], key=lambda c: c['t_s'])
    anchors_all = [float(x) for x in d['anchors']]
    anchors_ai = [r['ta'] for r in passes if r['ai']]

    def lap_all(t):           # 周 = R1 の入口を通るたびに 1 つ(人の通過も)。最初の R1 の前は 0 周目
        return int(sum(1 for x in anchors_all if t >= x - 0.5))

    def lap_ai(t):            # Fix-all の stop_rule_a2.py と同じ(AI の通過だけ・最初の AI の通過の前は数えない)
        k = None
        for i_, x in enumerate(anchors_ai):
            if t >= x - 0.5:
                k = i_
        return k
    tk = []
    mism = 0
    for x in ev['takeovers']:
        if x.get('class') != 'after_collision' or not x.get('failure'):
            continue
        dtc = x.get('dt_since_collision_s')
        cand = [c for c in ev['collisions'] if c['t_s'] <= x['t_s'] + 1e-6 and (dtc is None or x['t_s'] - c['t_s'] <= dtc + 0.1)]
        loc = cand[-1]['loc'] if cand else x.get('loc')
        mism += int(bool(cand) and cand[-1]['loc'] != x.get('loc'))
        tk.append((loc, float(x['t_s'])))
    merged = [(OTHER_LOC.get(l_, l_), t) for l_, t in tk]
    strong = [c for c in cols if c.get('strong')]
    out = {'ai_s': float(info['counts']['ai_s']), 'collisions_ai': len(cols), 'strong_ai': len(strong),
           'takeovers_after_collision': len(tk), 'takeover_loc_vs_collision_loc_mismatch': mism,
           'takeovers_by_loc': {k: sum(1 for l_, _ in tk if l_ == k) for k in sorted(set(l_ for l_, _ in tk))},
           'takeover_times': [[l_, round(t, 2), lap_all(t)] for l_, t in tk],
           'r1_passes_all': len(anchors_all), 'r1_passes_ai': len(anchors_ai)}
    rules = {
        'A2_3laps': streak_fires(merged, 3, lap_all),
        'A2_2laps(了承なしの読み方)': streak_fires(merged, 2, lap_all),
        'A2_3laps_ai_passes_only(stop_rule_a2 と同じ数え方)': streak_fires(merged, 3, lap_ai),
        'A2_2laps_ai_passes_only(stop_rule_a2 と同じ数え方)': streak_fires(merged, 2, lap_ai),
        'orig_a_consecutive_strong_same_loc': consecutive_strong_fires(cols),
        'orig_b_strong_same_loc_2laps': streak_fires([(c['loc'], c['t_s']) for c in strong], 2, lap_all),
    }
    for k, f in rules.items():
        for x in f:
            x['ai_s_at_stop'] = round(ai_seconds_until(lab, x['t_s']), 1)
    out['rules'] = {k: {'stops': len(f), 'first': f[0] if f else None, 'all': f} for k, f in rules.items()}
    return out


# ------------------------------------------------------------------ 本体
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--raw', default=str(config.RAW_DIR))
    ap.add_argument('--labels', default=str(config.DATA_DIR / 'labels'))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    logs = sorted(p.parent.name for p in Path(args.raw).glob('20261010-*/log.json'))
    log('ログ', len(logs))
    pcal = cr.load_pcal(cr.PCAL_DEFAULT)
    preps, errors = {}, {}
    for lg in logs:
        try:
            preps[lg] = cr.prepare(lg, args.labels, args.raw)
        except Exception as e:  # noqa: BLE001
            errors[lg] = repr(e)
    log('読めた', len(preps), '読めない', errors)
    res = cr.run(preps, G1_LOGS, OTHER_AI, STRONG_LOGS, VARIANTS, pcal, ('nominal',))
    full = {'schema': 'jf-cap4/v1', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'), 'tool': 'experiments/20261022_prep/cap4/cap4_replay.py',
            'layer': 'jetformer/safety_caps.py %s' % VER, 'variants': list(VARIANTS),
            'variant_params': {v: cr.VARIANTS[v]()._asdict() for v in VARIANTS}, 'g1_logs': G1_LOGS, 'other_ai_logs': OTHER_AI,
            'early_ai_logs': EARLY_AI, 'strong_logs': STRONG_LOGS, 'names': NAMES, 'errors': errors,
            'logs_in': logs, 'cause_text': CAUSE_TEXT}
    # cap_replay の表(2 案)
    md_res = dict(full, logs=res['logs'], summary=res['summary'], recorded_fields_seen={}, problems={}, computed_labels=[])
    (out / 'cap_replay.md').write_text(cr.to_markdown(md_res, '区間の速度上限の再生(10/10、v1.1 と 4 つめを外した版)'), encoding='utf-8')
    full['g1_gate'] = res['summary']['gate']
    full['g1_brakes'] = res['summary']['brakes']
    full['g1_lap_loss'] = res['summary']['lap_loss']
    full['g1_r1_signal'] = res['summary']['r1_signal']
    per = {}
    ai_logs = [lg for lg in preps if 'at' in preps[lg] and preps[lg]['sel'].any()]
    for lg in ai_logs:
        d = preps[lg]
        zone_imu, cause = phase_causes(d)
        cause_at = cause[d['ki']]
        e = {'name': NAMES.get(lg, lg), 'role': 'g1' if lg in G1_LOGS else ('other_ai' if lg in OTHER_AI else 'early_ai' if lg in EARLY_AI else 'ai'),
             'unknown_time': unknown_time(d, cause_at), 'frames': {}, 'virtual': {}, 'shadow_brakes': {}}
        for v in VARIANTS:
            p = cr.VARIANTS[v]()
            sh = cr.shadow(d, VER, p)
            e['frames'][v] = frame_stats(d, sh, cause_at)
            e['shadow_brakes'][v] = res['logs'][lg]['shadow'][v]
            e['virtual'][v] = virtual_stretches(d, p, pcal)
        # R1 の通過: 手前の位相が unknown か
        rows = []
        ent = {v: {round(x['ta'], 3): x for x in e['virtual'][v]['r1_entries']} for v in VARIANTS}
        for r_ in res['logs'][lg]['passes']:
            if not r_['ai']:
                continue
            ap_ = approach_phase(d, r_['ta'])
            row = {'ta': round(r_['ta'], 2), 'approach_gyro_zone': ap_, 'entry_mps': r_['entry_mps'], 'r1_collision': r_['r1_collision'],
                   'r1_strong': r_['r1_strong'], 'sig_zone': r_.get('sig_zone'), 'sig_gyro': r_.get('sig_gyro'), 'sig_pred': r_.get('sig_pred')}
            for v in VARIANTS:
                x = ent[v].get(round(r_['ta'], 3))
                row['virtual_entry_%s' % v] = x['capped'] if x else None
                row['virtual_entry_rec'] = x['rec'] if x else row.get('virtual_entry_rec')
            rows.append(row)
        e['r1_passes_ai'] = rows
        e['stop_rules'] = stop_rules(lg, d, res['logs'][lg]['passes'])
        per[lg] = e
        f3, fn = e['frames']['3zone'], e['frames']['noU']
        log(NAMES.get(lg, lg), 'AI %.1f s' % f3['ai_s'], 'capped 3zone %d noU %d / %d' % (f3['capped_frames'], fn['capped_frames'], f3['ai_frames']),
            'unknown final %.1f s' % e['unknown_time']['final_unknown_s'],
            'loss 3zone %.3f noU %.3f' % (e['virtual']['3zone']['loss_s'], e['virtual']['noU']['loss_s']))
    full['per_log'] = per

    # まとめ
    def agg(group):
        S = {}
        for v in VARIANTS:
            fr = [per[lg]['frames'][v] for lg in group if lg in per]
            n_ai = sum(x['ai_frames'] for x in fr)
            n_c = sum(x['capped_frames'] for x in fr)
            bz = {z: {'frames': sum(x['by_zone'][z]['frames'] for x in fr), 'capped_frames': sum(x['by_zone'][z]['capped_frames'] for x in fr),
                      'capped_s': sum(x['by_zone'][z]['capped_s'] for x in fr), 'ai_s': sum(x['by_zone'][z]['ai_s'] for x in fr),
                      'dthr_us_sum_x_s': sum(x['by_zone'][z]['dthr_us_sum_x_s'] for x in fr)} for z in sc.ZONES}
            ai_s = sum(x['ai_s'] for x in fr)
            vv = [per[lg]['virtual'][v] for lg in group if lg in per]
            S[v] = {'logs': len(fr), 'ai_frames': n_ai, 'ai_s': ai_s, 'capped_frames': n_c, 'capped_share': n_c / n_ai if n_ai else None,
                    'capped_s': sum(x['capped_s'] for x in fr),
                    'capped_frames_by_zone': {z: bz[z]['capped_frames'] for z in sc.ZONES},
                    'capped_share_of_capped_by_zone': {z: (bz[z]['capped_frames'] / n_c if n_c else None) for z in sc.ZONES},
                    'by_zone': bz,
                    'dthr_us_mean_all_ai': (sum(bz[z]['dthr_us_sum_x_s'] for z in sc.ZONES) / ai_s) if ai_s else None,
                    'mode_frames': {m: sum(x['mode_frames'][m] for x in fr) for m in sc.CAP_MODES},
                    'unknown_capped_by_cause_s': {c: sum(x['unknown_by_cause'][c]['capped_s'] for x in fr) for c in CAUSES},
                    'unknown_capped_s': sum(x['by_zone']['unknown']['capped_s'] for x in fr),
                    'shadow_brake_starts': sum(per[lg]['shadow_brakes'][v]['brake_starts'] for lg in group if lg in per),
                    'virtual_brake_starts': sum(x['rules']['brake_starts'] for x in vv),
                    'virtual_rules': {k: sum(x['rules'][k] for x in vv) for k in ('brake_start_below_1mps', 'brake_in_r1_turn_steps', 'roundtrips_lt_0p5s')},
                    'virtual_loss_s': sum(x['loss_s'] for x in vv), 'virtual_stretch_s': sum(x['stretch_s'] for x in vv),
                    'virtual_loss_by_zone_s': {z: sum(x['loss_by_zone_s'][z] for x in vv) for z in sc.ZONES}}
        ut = [per[lg]['unknown_time'] for lg in group if lg in per]
        S['unknown_time'] = {'runs': len(ut), 'ai_s': sum(x['ai_s'] for x in ut), 'gyro_unknown_s': sum(x['gyro_unknown_s'] for x in ut),
                             'final_unknown_s': sum(x['final_unknown_s'] for x in ut),
                             'final_unknown_by_cause_s': {c: sum(x['final_unknown_by_cause_s'][c] for x in ut) for c in CAUSES},
                             'gyro_unknown_by_cause_s': {c: sum(x['gyro_unknown_by_cause_s'][c] for x in ut) for c in CAUSES}}
        rows = [dict(r_, log=lg) for lg in group if lg in per for r_ in per[lg]['r1_passes_ai']]
        un = [r_ for r_ in rows if r_['approach_gyro_zone'] == 'unknown']
        S['r1_passes'] = {'ai_passes': len(rows), 'approach_unknown': len(un),
                          'approach_unknown_collision': sum(r_['r1_collision'] for r_ in un),
                          'approach_unknown_strong': sum(r_['r1_strong'] for r_ in un),
                          'approach_unknown_entry_gt_3.0': sum(1 for r_ in un if r_['entry_mps'] is not None and r_['entry_mps'] > 3.0),
                          'approach_unknown_entry_gt_2.5': sum(1 for r_ in un if r_['entry_mps'] is not None and r_['entry_mps'] > 2.5),
                          'approach_unknown_sig_pred': sum(bool(r_['sig_pred']) for r_ in un),
                          'approach_unknown_sig_zone': sum(bool(r_['sig_zone']) for r_ in un),
                          'approach_known_collision': sum(r_['r1_collision'] for r_ in rows if r_['approach_gyro_zone'] != 'unknown'),
                          'approach_known': sum(1 for r_ in rows if r_['approach_gyro_zone'] != 'unknown'),
                          'approach_unknown_rows': un}
        return S
    full['summary'] = {'G1_logs_5': agg(G1_LOGS), 'AI_logs_8': agg(G1_LOGS + OTHER_AI), 'AI_logs_11': agg(G1_LOGS + OTHER_AI + EARLY_AI)}
    full['elapsed_s'] = round(time.time() - t_start, 1)
    cr.jdump(out / 'cap4_replay.json', full)
    g = full['g1_gate']
    for v in VARIANTS:
        log('G1 6 項目 %s: %s 落ちた %s' % (v, g['G1_pass_' + v], g['G1_failed_items_' + v]))
        log('G1 4 項目 %s: %s 落ちた %s' % (v, g['G1_protocol_pass_' + v], g['G1_protocol_failed_items_' + v]))
    log('書き出し', out, '%.1f 秒' % full['elapsed_s'])


if __name__ == '__main__':
    main()
