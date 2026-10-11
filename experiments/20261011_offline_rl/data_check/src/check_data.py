"""ai_mix_20261011・intervention_val_20261011 の点検(オフライン強化学習の前提)。CPU だけ・読み取りだけ

    python check_data.py          # → ../data_check.json(節ごとに ../cache/<節>.json を残し、あれば読み直さない。--force で全部やり直す)

節:
  actions   AI の行動ラベル(out を評価時のゲイン D3 で割り戻した値)を、生の AI_CMD から独立に計算し直して照合する
  rewards   報酬・リターンを data/labels から計算し直して照合。source・split・場所ごとの分布。区間の終わりと交代・衝突
  collisions collision_map.md(006/007)との件数の照合
  locations 場所(loc_code)の数と、コース図の当てはめ(overnight_20261011/course/tracks_*.npz、AI のログ5本)との一致。
            R1・GM・DE の通過(R1 の起点ごと)と、R1 の進入速度・DE 出口のヨー(10/10 の診断の再現)
  splits    train / val の分け方(シーン・フレーム・ログ)と、介入の検証セットとの重なり
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import label_events as le  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTD = HERE.parent
CACHE = OUTD / 'cache'
DS = ROOT / 'data' / 'datasets' / 'ai_mix_20261011'
IV = ROOT / 'data' / 'datasets' / 'intervention_val_20261011'
LAB = ROOT / 'data' / 'labels'
RAW = ROOT / 'data' / 'raw'
TRACKS = Path('/home/jetson/jetformer_reports/overnight_20261011/course')
D3 = (1.3, 1.0)
GAMMA = 0.99
FORCE = '--force' in sys.argv

sys.path.insert(0, str(HERE))
import frames as fr  # noqa: E402


def r(x, n=3):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def stats(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n': 0}
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return {'n': int(len(x)), 'mean': r(x.mean(), 4), 'std': r(x.std(), 4), 'min': r(x.min(), 4),
            'p5': r(p[0], 4), 'p25': r(p[1], 4), 'median': r(p[2], 4), 'p75': r(p[3], 4), 'p95': r(p[4], 4),
            'max': r(x.max(), 4)}


def cached(name, fn):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / (name + '.json')
    if p.exists() and not FORCE:
        return json.loads(p.read_text())
    res = fn()
    tmp = str(p) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=le._json_default)
    os.replace(tmp, p)
    return res


_LAB = {}


def labels(log):
    if log not in _LAB:
        info = json.loads((LAB / (log + '.json')).read_text())
        with np.load(LAB / (log + '.npz')) as z:
            arr = {k: z[k] for k in z.files}
        _LAB[log] = (info, arr)
    return _LAB[log]


def raw_ai_cmd(log):
    parts = []
    for d in sorted((RAW / log).iterdir()):
        p = d / 'AI_CMD.npz'
        if d.is_dir() and p.exists():
            try:
                with np.load(p) as z:
                    if len(z['timestamps_us']):
                        parts.append({k: z[k] for k in ('timestamps_us', 'raw', 'out', 'gains', 'trim', 'hold', 'select')
                                      if k in z.files})
            except (EOFError, OSError, ValueError):       # 電源断で空・書きかけのファイル
                continue
    keys = set.intersection(*[set(p) for p in parts])
    a = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    o = np.argsort(a['timestamps_us'], kind='stable')
    return {k: v[o] for k, v in a.items()}


def F():
    return fr.build()


def usable(f):
    return f['future_valid'] & f['vehicle_valid'] & (f['mode'] != 0)


# ---------------------------------------------------------------- 1. 行動ラベル
def check_actions():
    f = F()
    ref_s, ref_t = 1443.0, 1498.0
    out = {'per_log': {}, 'rule': '独立に計算: steer = ref + (out − ref)/1.3、throttle = out(速度 1.0・前進側だけ)'}
    ai = f['source'] == 1
    tot = {'n': 0, 'label_eq': 0, 'fut0_eq': 0, 'fut0_n': 0, 'futk_eq': 0, 'futk_n': 0, 'prev_eq': 0, 'prev_n': 0}
    for log in sorted(set(f['log'][ai].tolist())):
        info, la = labels(log)
        rs = info['ref_us']
        ref = (rs['steer_center'], rs['throttle_neutral'])
        assert ref == (ref_s, ref_t), (log, ref)
        A = raw_ai_cmd(log)
        m = ai & (f['log'] == log)
        t = f['timestamps_us'][m]
        k = np.searchsorted(A['timestamps_us'], t)
        k = np.clip(k, 0, len(A['timestamps_us']) - 1)
        found = A['timestamps_us'][k] == t
        o = A['out'][k].astype(np.float64)
        exp = np.stack([ref[0] + (o[:, 0] - ref[0]) / D3[0], o[:, 1]], axis=1)    # 速度 1.0 なら前進側もそのまま
        lab = f['action_label'][m].astype(np.float64)
        eq = found & (np.abs(lab - exp) < 0.01).all(axis=1)
        fv = f['future_valid'][m]
        fut = f['fut'][m].astype(np.float64)
        f0 = fv & found
        f0_eq = f0 & (np.abs(fut[:, 0] - exp) < 0.01).all(axis=1)
        # k フレーム先: AI_CMD の i+k 行(途切れなし = 時刻が k フレーム分のとき)
        fk_n = fk_eq = 0
        for kk in range(1, 10):
            j = np.clip(k + kk, 0, len(A['timestamps_us']) - 1)
            dt = (A['timestamps_us'][j] - t) / 1e6
            cont = f0 & (np.abs(dt - kk / 30.0) < 0.5 / 30.0)
            oj = A['out'][j].astype(np.float64)
            ej = np.stack([ref[0] + (oj[:, 0] - ref[0]) / D3[0], oj[:, 1]], axis=1)
            fk_n += int(cont.sum())
            fk_eq += int((cont & (np.abs(fut[:, kk] - ej) < 0.01).all(axis=1)).sum())
        # 今の操作値 = 1フレーム前の行動ラベル
        jp = np.clip(k - 1, 0, None)
        op = A['out'][jp].astype(np.float64)
        ep = np.stack([ref[0] + (op[:, 0] - ref[0]) / D3[0], op[:, 1]], axis=1)
        vv = f['vehicle_valid'][m] & found & (k > 0) & ((t - A['timestamps_us'][jp]) <= 200000)
        # 前のフレームが人なら(交代の直後)スティックなので除く
        li = f['label_index'][m]
        prev_ai = la['ai'][np.clip(li - 1, 0, None)] & (li > 0)
        vv &= prev_ai
        p_eq = vv & (np.abs(f['vehicle'][m][:, :2] - ep) < 0.01).all(axis=1)
        # 記録したゲインが一定の区間で、raw との関係(割り戻した値 = ref + g/1.3·(raw − ref) + trim/1.3)
        g = A['gains'][k].astype(np.float64)
        tr = A['trim'][k].astype(np.float64)
        rw = A['raw'][k].astype(np.float64)
        sat = (o <= 1000.5) | (o >= 1999.5)
        ok_s = found & ~sat[:, 0] & np.isfinite(rw[:, 0]) & ~A['hold'][k]
        pred_s = ref[0] + (tr + g[:, 0] * (rw[:, 0] - ref[0])) / D3[0]
        combos = {}
        gk = np.round(g, 2)
        for gs, gt in sorted(set(map(tuple, gk[found].tolist()))):
            sel = found & (gk[:, 0] == gs) & (gk[:, 1] == gt)
            ss = sel & ok_s
            dev = np.abs(lab[ss, 0] - ref[0])
            rdev = np.abs(rw[ss, 0] - ref[0])
            big = rdev > 20
            combos['%.2f/%.2f' % (gs, gt)] = {
                'frames': int(sel.sum()), 'usable': int((sel & fv).sum()),
                'steer_match_raw_rel_0.5us': r(np.mean(np.abs(lab[ss, 0] - pred_s[ss]) < 0.5), 4) if ss.any() else None,
                'steer_amp_ratio_label_over_raw': r(np.median(dev[big] / rdev[big]), 3) if big.any() else None,
                'expected_ratio': r(gs / D3[0], 3)}
        # 一定のゲインの区間のうち一番長いもので、割り戻したラベルと raw を並べた例(5 行)
        out['per_log'][log] = {
            'ai_frames_in_dataset': int(m.sum()), 'found_in_ai_cmd': int(found.sum()),
            'label_equals_independent': int(eq.sum()),
            'future0_checked': int(f0.sum()), 'future0_equal': int(f0_eq.sum()),
            'futurek_checked': fk_n, 'futurek_equal': fk_eq,
            'vehicle_prev_checked': int(vv.sum()), 'vehicle_prev_equal': int(p_eq.sum()),
            'gain_combos': combos,
            'saturated_out_frames': [int(sat[found, 0].sum()), int(sat[found, 1].sum())],
            'recorded_rule': info['gain_rule']['rule'],
            'recorded_check_match': (info['gain_rule'].get('check_ai') or {}).get('match_0.5us'),
        }
        for kk_, v in (('n', int(m.sum())), ('label_eq', int(eq.sum())), ('fut0_eq', int(f0_eq.sum())),
                       ('fut0_n', int(f0.sum())), ('futk_eq', fk_eq), ('futk_n', fk_n), ('prev_eq', int(p_eq.sum())),
                       ('prev_n', int(vv.sum()))):
            tot[kk_] += v
    out['totals'] = tot
    # 人と AI のラベルの分布(使えるフレーム、ステアは中心からのずれの絶対値)
    u = usable(f)
    dist = {}
    for src, name in ((0, 'human'), (1, 'ai')):
        mm = u & (f['source'] == src)
        a0 = f['fut'][mm, 0].astype(np.float64)
        dist[name] = {'steer_abs_dev_from_1443': stats(np.abs(a0[:, 0] - ref_s)),
                      'throttle_minus_1498': stats(a0[:, 1] - ref_t),
                      'steer_at_limit_1000_or_2000_after_d3': r(np.mean(np.abs(a0[:, 0] - ref_s) >= (443 / 1.3) - 0.5), 4),
                      'throttle_brake_side_frac': r(np.mean(a0[:, 1] > ref_t + 5), 4)}
    out['label_distribution_usable_k0'] = dist
    return out


# ---------------------------------------------------------------- 2. 報酬・リターン
def mc(rw, seg, gamma=GAMMA):
    ret = np.full(len(rw), np.nan)
    acc = 0.0
    for i in range(len(rw) - 1, -1, -1):
        if seg[i] < 0:
            continue
        if i == len(rw) - 1 or seg[i + 1] != seg[i]:
            acc = 0.0
        acc = rw[i] + gamma * acc
        ret[i] = acc
    return ret


def check_rewards():
    f = F()
    res = {'recompute': {}, 'segments': {}, 'masked_events': []}
    # (a) data/labels の報酬・リターンを独立に計算し直す
    bad_r = bad_ret = n_all = 0
    for p in sorted(LAB.glob('2*.json')):
        log = p.stem
        info, a = labels(log)
        rw = (a['progress_fwd_m'].astype(np.float64) - 1.0 * a['collision'] - 2.0 * a['failure_takeover']
              - 0.5 * a['recovery_start'] + a.get('penalty_carried', np.zeros(len(a['reward']))))
        carried_from = {m['from_frame']: m for m in info['events'].get('carried_penalties', [])}
        bad_r += int((np.abs(rw - a['reward']) > 1e-4).sum())
        ret = mc(a['reward'].astype(np.float64), a['segment_id'])
        both = np.isfinite(ret) & np.isfinite(a['ret'])
        bad_ret += int((np.abs(ret[both] - a['ret'][both]) > 1e-3).sum()) + int((np.isfinite(ret) != np.isfinite(a['ret'])).sum())
        n_all += len(rw)
        # 交代と区間の終わり
        seg = a['segment_id']
        ev = info['events']
        rows = []
        for tk in ev['takeovers']:
            fr_ = tk['frame']
            row = {'log': log, 't_s': tk['t_s'], 'class': tk['class'], 'failure': tk['failure'], 'frame': fr_}
            if fr_ is not None and fr_ in carried_from and tk['failure']:
                j = carried_from[fr_]['to_frame']
                row['carried_to_frame'] = j
                row['carried_ok'] = bool(a['segment_end'][j]) and int(a['end_code'][j]) == 1
                res['masked_events'].append(row)
                continue
            if fr_ is None or seg[fr_] < 0:
                row['problem'] = '交代直前の AI のフレームが区間の外(報酬・終わりの印が付かない)'
                if fr_ is not None:
                    row['mask_bits'] = int(a['mask_bits'][fr_])
                    row['hold'] = bool(a['hold'][fr_])
                    # 直前の区間の終わり
                    prev = np.nonzero((seg[:fr_] >= 0))[0]
                    if len(prev):
                        j = prev[-1]
                        row['prev_segment_end_code'] = int(a['end_code'][j])
                        row['prev_segment_end_dt_s'] = r(a['t_s'][fr_] - a['t_s'][j], 3)
                res['masked_events'].append(row)
                continue
            exp_code = 1 if tk['failure'] else 2
            ok = bool(a['segment_end'][fr_]) and int(a['end_code'][fr_]) == exp_code
            if not ok:
                row['problem'] = '区間の終わりの印が合わない(segment_end=%s end_code=%d)' % (
                    bool(a['segment_end'][fr_]), int(a['end_code'][fr_]))
                res['masked_events'].append(row)
        for c in ev['collisions']:
            fr_ = c['frame']
            if fr_ is not None and fr_ in carried_from:
                res['masked_events'].append({'log': log, 't_s': c['t_s'], 'kind': 'collision', 'ai': c['ai'], 'g': c['g'],
                                             'loc': c['loc'], 'frame': fr_,
                                             'carried_to_frame': carried_from[fr_]['to_frame'],
                                             'carried_dt_s': carried_from[fr_]['dt_s']})
                continue
            if fr_ is None or seg[fr_] < 0:
                row = {'log': log, 't_s': c['t_s'], 'kind': 'collision', 'ai': c['ai'], 'g': c['g'], 'loc': c['loc'],
                       'frame': fr_, 'problem': '衝突のフレームが区間の外(罰が付かない)'}
                if fr_ is not None:
                    row['mask_bits'] = int(a['mask_bits'][fr_])
                    row['hold'] = bool(a['hold'][fr_])
                    row['reboot_near'] = [b for b in ev['board_reboots']
                                          if b['t_last_s'] - 1.0 <= c['t_s'] <= b['t_after_s'] + 1.0]
                res['masked_events'].append(row)
        for x in ev['recoveries']:
            if x['ai'] and seg[x['frame']] < 0:
                res['masked_events'].append({'log': log, 't_s': x['t_s'], 'kind': 'recovery_start',
                                             'problem': 'リカバリー開始が区間の外'})
        # 区間の終わり方
        ends = {}
        for s in np.unique(seg[seg >= 0]):
            i1 = np.nonzero(seg == s)[0][-1]
            code = le.END_CODES[int(a['end_code'][i1])]
            who = 'ai' if a['ai'][i1] else 'human'
            ends.setdefault(who, {}).setdefault(code, 0)
            ends[who][code] += 1
        res['segments'][log] = ends
    res['recompute'] = {'label_frames': n_all, 'reward_mismatch': bad_r, 'return_mismatch': bad_ret}

    # (b) データセット側: ラベルの行と同じ値か、使えるフレームで NaN が無いか
    u = usable(f)
    nan_ret_usable = int((u & ~np.isfinite(f['mc_return'])).sum())
    nan_ret_fv = int((f['future_valid'] & ~np.isfinite(f['mc_return'])).sum())
    nan_rw_fv = int((f['future_valid'] & ~np.isfinite(f['reward'])).sum())
    seg_neg_fv = int((f['future_valid'] & (f['segment_id'] < 0)).sum())
    mism = 0
    for log in sorted(set(f['log'].tolist())):
        info, a = labels(log)
        m = f['log'] == log
        li = f['label_index'][m]
        ok = li >= 0
        mism += int((a['t_us'][li[ok]] != f['timestamps_us'][m][ok]).sum())
        for k_d, k_l in (('reward', 'reward'), ('mc_return', 'ret'), ('segment_id', 'segment_id'), ('loc_code', 'loc_code'),
                         ('collision', 'collision'), ('failure_takeover', 'failure_takeover')):
            x, y = f[k_d][m][ok].astype(np.float64), a[k_l][li[ok]].astype(np.float64)
            mism += int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())
    res['dataset'] = {'frames': int(len(u)), 'no_label_frames': int((f['label_index'] < 0).sum()),
                      'mismatch_vs_labels': mism, 'usable': int(u.sum()),
                      'usable_mc_return_nan': nan_ret_usable, 'future_valid_mc_return_nan': nan_ret_fv,
                      'future_valid_reward_nan': nan_rw_fv, 'future_valid_outside_segment': seg_neg_fv}

    # (c) 分布(使えるフレーム): source × split、場所、ジャンル
    names = le.LOC_NAMES
    src_name = {0: 'human', 1: 'ai'}
    dist = {}
    for sp in ('train', 'val'):
        for src in (0, 1):
            mm = u & (f['split'] == sp) & (f['source'] == src)
            key = '%s/%s' % (sp, src_name[src])
            dist[key] = {'frames': int(mm.sum()), 'reward': stats(f['reward'][mm]), 'mc_return': stats(f['mc_return'][mm]),
                         'progress_m_per_frame': stats(f['progress_m'][mm]),
                         'collision_frames': int((mm & f['collision']).sum()),
                         'failure_takeover_frames': int((mm & f['failure_takeover']).sum()),
                         'near_collision_frac': r(np.mean(f['near_collision'][mm]), 4) if mm.any() else None,
                         'end_code_failure_frac': r(np.mean(f['end_code'][mm] == 1), 4) if mm.any() else None,
                         'within_3.3s_of_non_failure_end_frac': r(np.mean((f['end_code'][mm] != 1)
                                                                          & (f['time_to_end_s'][mm] < 3.3)), 4)
                         if mm.any() else None}
    res['by_split_source'] = dist
    by_loc = {}
    for src in (0, 1):
        for li_, nm in enumerate(names):
            mm = u & (f['source'] == src) & (f['loc_code'] == li_)
            if mm.any():
                by_loc['%s/%s' % (src_name[src], nm)] = {'frames': int(mm.sum()), 'reward_mean': r(f['reward'][mm].mean(), 4),
                                                         'mc_return': stats(f['mc_return'][mm]),
                                                         'collision_frames': int((mm & f['collision']).sum())}
    res['by_source_loc'] = by_loc
    by_mode = {}
    for src in (0, 1):
        for mo in sorted(set(f['mode'][u].tolist())):
            mm = u & (f['source'] == src) & (f['mode'] == mo)
            if mm.any():
                by_mode['%s/mode%d' % (src_name[src], mo)] = {'frames': int(mm.sum()), 'mc_return': stats(f['mc_return'][mm])}
    res['by_source_mode'] = by_mode
    # 衝突の罰は、全部のフレームと使えるフレームでいくつ入っているか(使えないフレームでもリターンには入る)
    res['collisions_in_dataset'] = {
        'all_frames': {'human': int((f['collision'] & (f['source'] == 0)).sum()),
                       'ai': int((f['collision'] & (f['source'] == 1)).sum())},
        'usable_frames': {'human': int((f['collision'] & u & (f['source'] == 0)).sum()),
                          'ai': int((f['collision'] & u & (f['source'] == 1)).sum())},
        'failure_takeover_all': int(f['failure_takeover'].sum()),
        'failure_takeover_usable': int((f['failure_takeover'] & u).sum())}
    # フレームの間隔(報酬は 1フレームごと。間隔が揃っているか)
    dts = []
    for log in sorted(set(f['log'].tolist())):
        info, a = labels(log)
        dts.append(np.diff(a['t_s'])[(np.diff(a['segment_id']) == 0) & (a['segment_id'][1:] >= 0)])
    d = np.concatenate(dts)
    res['frame_interval_in_segments_ms'] = {'median': r(np.median(d) * 1000, 2), 'p99': r(np.percentile(d, 99) * 1000, 2),
                                           'frac_over_50ms': r(np.mean(d > 0.05), 5)}
    return res


# ---------------------------------------------------------------- 3. collision_map との照合
def check_collisions():
    s = json.loads((LAB / 'summary.json').read_text())
    v = s['validation']['collision_map']
    out = {'validation_from_summary': {k: v.get(k) for k in ('totals', 'recall_impact', 'recall_impact_or_bump',
                                                             'precision_ai_vs_reference',
                                                             'precision_ai_with_visual_check', 'loc_agreement')},
           'per_log': v['per_log'], 'unmatched_reference': v['unmatched_reference'],
           'reference_matched_by_bump': v.get('reference_matched_by_bump'), 'extra_detected_ai': v['extra_detected_ai']}
    # 独立に数え直す: collision_map 付録A の行数と、ラベルの AI 側の衝突(水平 2.5G)
    ref = le.parse_collision_map()
    for log in ('20261010-172910', '20261010-173445'):
        info, a = labels(log)
        cs = info['events']['collisions']
        rr = [x for x in ref if x['log'] == log]
        out.setdefault('counts', {})[log] = {
            'collision_map_ai_rows': len(rr), 'collision_map_ai_rows_excl_recovery': sum(1 for x in rr if not x['recovery']),
            'labels_ai_collisions_2.5G': sum(1 for c in cs if c['ai']),
            'labels_ai_strong_4G': sum(1 for c in cs if c['ai'] and c['strong']),
            'labels_ai_by_loc': {k: sum(1 for c in cs if c['ai'] and c['loc'] == k) for k in le.LOC_NAMES
                                 if any(c['ai'] and c['loc'] == k for c in cs)},
            'collision_map_by_loc_excl_recovery': {k: sum(1 for x in rr if not x['recovery'] and le.map_loc(x['loc_name']) == k)
                                                   for k in le.LOC_NAMES
                                                   if any(not x['recovery'] and le.map_loc(x['loc_name']) == k for x in rr)},
            'labels_bumps_ai': sum(1 for c in info['events']['bumps'] if c['ai'])}
    return out


# ---------------------------------------------------------------- 4. 場所
def anchors_for(log):
    """label_events と同じ手順で R1 の起点・緑マット出口を求める(+ 生の IMU・車両)"""
    L = le.load_log(log)
    A, V, I = L['AI_CMD'], L['VEHICLE'], L['IMU']
    info, a = labels(log)
    t0 = info['t0_us']
    sec = lambda x: (np.asarray(x, np.int64) - t0) / 1e6      # noqa: E731
    vt = sec(V['timestamps_us'])
    verpm = V['erpm'].astype(np.float64)
    it = sec(I['timestamps_us'])
    ie = np.interp(it, vt, verpm)
    gz = I['gyro'][:, 2].astype(np.float64)
    still = (np.abs(ie) < 50) & (np.abs(gz) < 5)
    hd, gzu = le.heading_deg(it, gz, still)
    anchors, gm, gzs = le.course_anchors(it, hd, gzu, vt, verpm)
    return {'anchors': anchors, 'gm': gm, 'it': it, 'gzu': gzu, 'gzs': gzs, 'vt': vt, 'erpm': verpm}


def check_locations():
    f = F()
    u = usable(f)
    names = le.LOC_NAMES
    res = {}
    # (a) フレーム数: split × source × 場所(全フレーム・使えるフレーム)
    cov = {}
    for which, mask in (('all_labeled', f['label_index'] >= 0), ('usable', u)):
        for sp in ('train', 'val'):
            for src, sn in ((0, 'human'), (1, 'ai')):
                mm = mask & (f['split'] == sp) & (f['source'] == src)
                cov.setdefault(which, {})['%s/%s' % (sp, sn)] = {nm: int((mm & (f['loc_code'] == i)).sum())
                                                                for i, nm in enumerate(names)}
    res['frames_by_loc'] = cov
    # (b) 場所ごとのヨーレート(使えるフレーム、最新の gyro z、左 +)と速度
    yaw = {}
    for src, sn in ((0, 'human'), (1, 'ai')):
        for i, nm in enumerate(names):
            mm = u & (f['source'] == src) & (f['loc_code'] == i)
            if mm.any():
                yaw['%s/%s' % (sn, nm)] = {'yaw_dps_median': r(np.median(f['gz'][mm]), 1),
                                           'yaw_dps_p10_p90': [r(np.percentile(f['gz'][mm], 10), 1),
                                                               r(np.percentile(f['gz'][mm], 90), 1)],
                                           'speed_erpm_median': r(np.median(-f['vehicle'][mm, 3]), 0)}
    res['yaw_by_loc_usable'] = yaw
    # (c) コース図の当てはめ(位置)と比べる: AI のログ5本、ラベルのフレーム(AI_CMD)ごと
    sec_map = {'HS': 'HS', 'LH': 'LH', 'R1': 'R1', 'GM': 'GM', 'DE': 'DE', 'SW': 'other'}
    conf = {}
    per_log = {}
    for p in sorted(TRACKS.glob('tracks_*.npz')):
        log = p.stem.split('_', 1)[1]
        info, a = labels(log)
        with np.load(p) as z:
            tt, sec_, val, tai = z['t_us'], z['section'], z['valid'], z['ai']
            sn = [str(x) for x in z['section_names']]
        k = np.clip(np.searchsorted(tt, a['t_us']), 0, len(tt) - 1)
        hit = (tt[k] == a['t_us']) & val[k]
        gy = np.array([names[c] for c in a['loc_code']])
        tr = np.array([sec_map[sn[s]] for s in sec_[k]])
        m = hit & a['ai'] & (a['segment_id'] >= 0)
        known = m & (gy != 'unknown')
        per_log[log] = {'frames_compared': int(m.sum()), 'gyro_unknown': int((m & (gy == 'unknown')).sum()),
                        'agree_of_known': r(np.mean(gy[known] == tr[known]), 4) if known.any() else None}
        for g_, t_ in zip(gy[m], tr[m]):
            conf.setdefault(g_, {}).setdefault(t_, 0)
            conf[g_][t_] += 1
    res['vs_course_fit'] = {'per_log': per_log, 'confusion_gyro_rows_vs_fit_cols': conf,
                            'note': 'コース図の区間の境目は向きの規則と違う(R1 は起点から 1秒、GM 出口 ±0.3秒など)ので、'
                                    '境目の近くの食い違いは定義の違い'}
    # (d) 通過(R1 の起点ごと): 誰が走っていたか、split、R1 の進入 eRPM と衝突、DE 出口のヨー
    splits = json.loads((DS / 'splits.json').read_text())
    split_of = {s: k for k in ('train', 'val') for s in splits[k]}
    passes = []
    for p in sorted(LAB.glob('2*.json')):
        log = p.stem
        info, a = labels(log)
        if not info.get('r1_anchors'):
            continue
        X = anchors_for(log)
        cs = info['events']['collisions']
        sc_names = info.get('scenes') and sorted(d.name for d in (RAW / log).iterdir()
                                                 if d.is_dir() and d.name.startswith(log + '-'))
        for j, ta in enumerate(X['anchors']):
            tg = X['gm'][j]
            i = int(np.clip(np.searchsorted(a['t_s'], ta), 0, len(a['t_s']) - 1))
            scene = sc_names[int(a['scene'][i])]
            w = (X['vt'] >= ta - 0.4) & (X['vt'] <= ta)
            entry = float(np.max(-X['erpm'][w])) if w.any() else np.nan
            t_end = tg + 2.3 if np.isfinite(tg) else ta + 3.5
            c_r1 = [c for c in cs if c['loc'] == 'R1' and ta - 0.5 <= c['t_s'] <= (tg - 0.3 if np.isfinite(tg) else ta + 3.5)]
            c_gm = [c for c in cs if c['loc'] == 'GM' and ta <= c['t_s'] <= (tg + 0.3 if np.isfinite(tg) else ta + 3.5)]
            c_de = [c for c in cs if c['loc'] == 'DE' and np.isfinite(tg) and tg - 0.3 <= c['t_s'] <= t_end]
            de_yaw = np.nan
            if np.isfinite(tg):
                ww = (X['it'] >= tg + 0.2) & (X['it'] <= tg + 0.8)
                de_yaw = float(np.mean(X['gzu'][ww])) if ww.any() else np.nan
            ii = np.nonzero((a['t_s'] >= ta - 0.4) & (a['t_s'] <= ta))[0]
            passes.append({'log': log, 'scene': scene, 'split': split_of.get(scene), 't': r(ta, 2),
                           'ai': bool(a['ai'][i]), 'ai_log': bool(info['ai_log']),
                           'gm_exit_found': bool(np.isfinite(tg)),
                           'entry_erpm': r(entry, 0), 'r1_collision': bool(c_r1), 'r1_strong': any(c['strong'] for c in c_r1),
                           'gm_collision': bool(c_gm), 'de_collision': bool(c_de),
                           'de_strong': any(c['strong'] for c in c_de),
                           'de_yaw_0.2_0.8s': r(de_yaw, 1),
                           'usable_frames_r1_0.4s_before': int(len(ii))})
    res['passes'] = passes
    # 集計
    def agg(ps):
        o = {'passes': len(ps), 'r1_collision': sum(p['r1_collision'] for p in ps),
             'r1_strong': sum(p['r1_strong'] for p in ps),
             'gm_exit_found': sum(p['gm_exit_found'] for p in ps),
             'gm_collision': sum(p['gm_collision'] for p in ps), 'de_collision': sum(p['de_collision'] for p in ps)}
        bands = {}
        for lo, hi, nm in ((0, 5500, '<5500'), (5500, 6200, '5500-6200'), (6200, 1e9, '>=6200')):
            b = [p for p in ps if p['entry_erpm'] is not None and lo <= p['entry_erpm'] < hi]
            bands[nm] = '%d/%d strong(4G) , %d any(2.5G)' % (sum(p['r1_strong'] for p in b), len(b),
                                                           sum(p['r1_collision'] for p in b))
        o['r1_by_entry_erpm'] = bands
        y = [p['de_yaw_0.2_0.8s'] for p in ps if p['de_yaw_0.2_0.8s'] is not None]
        o['de_yaw_median_dps'] = r(np.median(y), 1) if y else None
        o['de_yaw_n'] = len(y)
        return o
    in_ds = [p for p in passes if p['split'] is not None]
    summ = {}
    for key, sel in (('all_logs_human', [p for p in passes if not p['ai']]),
                     ('all_logs_ai', [p for p in passes if p['ai']]),
                     ('ai_run_006_007', [p for p in passes if p['ai'] and p['log'] in ('20261010-172910', '20261010-173445')]),
                     ('dataset_train_human', [p for p in in_ds if p['split'] == 'train' and not p['ai']]),
                     ('dataset_val_human', [p for p in in_ds if p['split'] == 'val' and not p['ai']]),
                     ('dataset_train_ai', [p for p in in_ds if p['split'] == 'train' and p['ai']]),
                     ('dataset_val_ai', [p for p in in_ds if p['split'] == 'val' and p['ai']])):
        summ[key] = agg(sel)
    res['pass_summary'] = summ
    return res


# ---------------------------------------------------------------- 5. split
def check_splits():
    f = F()
    splits = json.loads((DS / 'splits.json').read_text())
    tr, va = set(splits['train']), set(splits['val'])
    res = {'scene_overlap': sorted(tr & va), 'train_scenes': len(tr), 'val_scenes': len(va)}
    # フレーム(時刻)の重なり
    ttr = set(f['timestamps_us'][f['split'] == 'train'].tolist())
    tva = set(f['timestamps_us'][f['split'] == 'val'].tolist())
    res['frame_timestamp_overlap'] = len(ttr & tva)
    # ログごと
    logs = {}
    for log in sorted(set(f['log'].tolist())):
        m = f['log'] == log
        sc = sorted(set(f['scene'][m].tolist()))
        logs[log] = {'train_scenes': [s[-4:] for s in sc if s in tr], 'val_scenes': [s[-4:] for s in sc if s in va]}
    res['per_log'] = logs
    res['logs_in_both'] = [k for k, v in logs.items() if v['train_scenes'] and v['val_scenes']]
    res['logs_val_only'] = [k for k, v in logs.items() if v['val_scenes'] and not v['train_scenes']]
    # 同じログの train と val のシーンの境目(続けて撮ったシーンなら、境目の前後は似たフレーム)
    u = usable(f)
    near = {}
    adj = []
    for log in res['logs_in_both']:
        m = f['log'] == log
        t = f['timestamps_us'][m] / 1e6
        sp = f['split'][m]
        sc = f['scene'][m]
        order = np.argsort(t)
        t, sp, sc, uu = t[order], sp[order], sc[order], u[m][order]
        ch = np.nonzero(sp[1:] != sp[:-1])[0]
        for c in ch:
            adj.append({'log': log, 'from': sc[c][-4:], 'to': sc[c + 1][-4:], 'gap_s': r(t[c + 1] - t[c], 2)})
        tv = t[sp == 'val']
        ttr_ = t[sp == 'train']
        for w in (1.0, 3.3, 10.0):
            if len(ttr_) and len(tv):
                d = np.min(np.abs(tv[:, None] - ttr_[None, ::5]), axis=1) if len(tv) * len(ttr_) < 4e7 else None
                if d is not None:
                    near.setdefault(str(w), 0)
                    near[str(w)] += int(((d <= w) & uu[sp == 'val']).sum())
    res['train_val_boundaries'] = adj
    res['val_usable_frames_within_s_of_train_frame_same_log'] = near
    # 人の検証シーンが clean_driving_20261011(モデルB の学習)と同じか
    cd = ROOT / 'data' / 'datasets' / 'clean_driving_20261011'
    cs = json.loads((cd / 'splits.json').read_text())
    hum_val = {s for s in va if s.rsplit('-', 1)[0] in set(f['log'][f['source'] == 0].tolist())}
    res['model_b_split_consistency'] = {
        'clean_val': len(cs['val']), 'clean_train': len(cs['train']),
        'ai_mix_val_human_scenes_equal_clean_val': sorted(set(cs['val'])) == sorted(s for s in va if s in set(cs['val']) | set(cs['train'])),
        'clean_val_in_ai_mix_train': sorted(set(cs['val']) & tr),
        'clean_train_in_ai_mix_val': sorted(set(cs['train']) & va),
        'clean_scenes_missing_in_ai_mix': sorted((set(cs['val']) | set(cs['train'])) - tr - va)}
    # 介入の検証セットとの重なり
    ivs = json.loads((IV / 'splits.json').read_text())['val']
    card = json.loads((IV / 'dataset_card.json').read_text())
    rows = {w['name']: w for w in card['intervention']['windows'] if w['included']}
    ov = []
    tmap = dict(zip(f['timestamps_us'].tolist(), f['split'].tolist()))
    umap = dict(zip(f['timestamps_us'][u].tolist(), [True] * int(u.sum())))
    for w in sorted(ivs):
        with np.load(IV / 'aligned' / (w + '.npz')) as z:
            t = z['timestamps_us']
            fv = z['future_valid']
            src = z['source']
        log = w.rsplit('-', 1)[0]
        sp = [tmap.get(int(x)) for x in t]
        lead_ai = src == 1
        ov.append({'window': w, 'class': rows[w]['class'], 'loc': rows[w]['loc'], 'scored': int(fv.sum()),
                   'log_split_in_ai_mix': ('val' if log in res['logs_val_only'] else 'train'),
                   'frames_in_ai_mix_train': int(sum(1 for x in sp if x == 'train')),
                   'frames_in_ai_mix_val': int(sum(1 for x in sp if x == 'val')),
                   'ai_lead_frames_usable_in_ai_mix_train': int(sum(1 for x, a_ in zip(t, lead_ai)
                                                                    if a_ and tmap.get(int(x)) == 'train' and umap.get(int(x)))),
                   'scored_frames_usable_in_ai_mix': int(sum(1 for x, v_ in zip(t, fv) if v_ and umap.get(int(x))))})
    res['intervention_overlap'] = ov
    res['intervention_summary'] = {
        'windows': len(ov), 'scored': sum(o['scored'] for o in ov),
        'windows_from_ai_mix_val_logs': sum(1 for o in ov if o['log_split_in_ai_mix'] == 'val'),
        'scored_from_ai_mix_val_logs': sum(o['scored'] for o in ov if o['log_split_in_ai_mix'] == 'val'),
        'scored_frames_usable_in_ai_mix': sum(o['scored_frames_usable_in_ai_mix'] for o in ov),
        'ai_lead_frames_trained_on': sum(o['ai_lead_frames_usable_in_ai_mix_train'] for o in ov)}
    return res


def check_intervention():
    card = json.loads((IV / 'dataset_card.json').read_text())
    iv = card['intervention']
    ivs = json.loads((IV / 'splits.json').read_text())['val']
    n_sc = 0
    src_sc = {}
    kinds = {}
    ok_label = 0
    thr_fwd = 0
    for w in ivs:
        with np.load(IV / 'aligned' / (w + '.npz')) as z:
            fv, src, kind, lv = z['future_valid'], z['source'], z['correction_kind'], z['label_valid']
            fut = z['future_actions']
            vv = z['vehicle_valid']
            ts = z['t_since_takeover_s']
            md = z['mode']
        n_sc += int(fv.sum())
        for s in src[fv]:
            src_sc[int(s)] = src_sc.get(int(s), 0) + 1
        for k in kind[fv]:
            kinds[int(k)] = kinds.get(int(k), 0) + 1
        ok_label += int((fv & lv & vv & (ts >= 0) & (md != 0)).sum())
        thr_fwd += int((fv & (fut[:, :, 1] < 1498 - 30).any(axis=1)).sum())
    return {'windows': len(ivs), 'scored': n_sc, 'scored_by_source': src_sc, 'scored_by_kind': kinds,
            'scored_with_label_valid_vehicle_valid_after_takeover_mode_set': ok_label,
            'scored_with_forward_throttle_command': thr_fwd,
            'counts': iv['counts'], 'baselines_all': iv['baselines']['all'],
            'baseline_compare_clean': iv['baselines'].get('compare_clean_driving_20261011')}


# ---------------------------------------------------------------- 6. 価値の正解の数・ログ単位の分割・品質
def collision_times_us(log):
    info, _ = labels(log)
    return np.array(sorted(c['t_us'] for c in info['events']['collisions']), np.int64)


def check_value_and_folds():
    f = F()
    u = usable(f)
    loc = check_locations_cached()
    passes = loc['passes']
    res = {}
    # (a) 「1秒以内に衝突」の陽性(使えるフレームのうち、直前 2秒以内に衝突が無いもの)
    t_next = np.full(len(u), np.inf)
    t_prev = np.full(len(u), np.inf)
    for log in sorted(set(f['log'].tolist())):
        m = f['log'] == log
        ct = collision_times_us(log)
        t = f['timestamps_us'][m]
        if len(ct):
            k = np.searchsorted(ct, t, side='right')
            nxt = np.where(k < len(ct), ct[np.clip(k, 0, len(ct) - 1)], np.iinfo(np.int64).max)
            prv = np.where(k > 0, ct[np.clip(k - 1, 0, len(ct) - 1)], np.iinfo(np.int64).min // 2)
            t_next[m] = (nxt - t) / 1e6
            t_prev[m] = (t - prv) / 1e6
    base = u & (t_prev > 2.0)
    pos = base & (t_next <= 1.0)
    vt = {}
    for sp in ('train', 'val'):
        for src, sn in ((0, 'human'), (1, 'ai')):
            mm = base & (f['split'] == sp) & (f['source'] == src)
            vt['%s/%s' % (sp, sn)] = {'frames': int(mm.sum()), 'positives_1s': int((pos & mm).sum()),
                                      'positive_rate': r(np.mean(pos[mm]), 4) if mm.any() else None,
                                      'collision_events': int(len(set(np.round((f['timestamps_us'][pos & mm]
                                                                                 + t_next[pos & mm] * 1e6) / 1e5).tolist())))}
    res['collision_within_1s'] = {'rule': '使えるフレーム(future_valid・vehicle_valid・ジャンルあり)で、直前 2秒以内に衝突が無いもの。'
                                          '陽性 = 1秒以内に衝突(水平 2.5G、人・AI どちらの衝突も)', 'by_split_source': vt}
    # (b) ログごとの数(ログ単位の分割を作るため)
    rows = {}
    for log in sorted(set(f['log'].tolist())):
        m = f['log'] == log
        info, _ = labels(log)
        ps = [p for p in passes if p['log'] == log and p['split'] is not None]
        rows[log] = {'ai_log': bool(info['ai_log']), 'description': info.get('description'),
                     'split': sorted(set(f['split'][m].tolist())),
                     'usable_human': int((u & m & (f['source'] == 0)).sum()), 'usable_ai': int((u & m & (f['source'] == 1)).sum()),
                     'positives_1s_ai': int((pos & m & (f['source'] == 1)).sum()),
                     'positives_1s_human': int((pos & m & (f['source'] == 0)).sum()),
                     'r1_passes_ai': sum(1 for p in ps if p['ai']), 'r1_passes_human': sum(1 for p in ps if not p['ai']),
                     'r1_strong_ai': sum(1 for p in ps if p['ai'] and p['r1_strong']),
                     'r1_strong_human': sum(1 for p in ps if not p['ai'] and p['r1_strong']),
                     'r1_any_ai': sum(1 for p in ps if p['ai'] and p['r1_collision']),
                     'de_collision_ai': sum(1 for p in ps if p['ai'] and p['de_collision']),
                     'collisions_ai': sum(1 for c in info['events']['collisions'] if c['ai']),
                     'failure_takeovers': sum(1 for x in info['events']['takeovers'] if x['failure'])}
    res['per_log'] = rows
    # (c) 提案: ログ単位の 5分割(AI のログは R1 の通過・衝突の数で釣り合わせ、人のログはフレーム数で)。検証用の
    #     AI のログ 172505・204208 は同じ分割(fold 0)に入れて、今の val と重ねる
    ai_logs = sorted([k for k, v in rows.items() if v['ai_log']], key=lambda k: -(rows[k]['r1_passes_ai'] * 10
                                                                                 + rows[k]['collisions_ai']))
    hu_logs = sorted([k for k, v in rows.items() if not v['ai_log']], key=lambda k: -rows[k]['usable_human'])
    K = 5
    folds = [[] for _ in range(K)]
    load = np.zeros((K, 3))
    for lg in ('20261010-172505', '20261010-204208'):
        if lg in rows:
            folds[0].append(lg)
            load[0] += [rows[lg]['r1_passes_ai'], rows[lg]['collisions_ai'], rows[lg]['usable_ai'] / 1000]
    for lg in ai_logs:
        if lg in folds[0]:
            continue
        v = np.array([rows[lg]['r1_passes_ai'], rows[lg]['collisions_ai'], rows[lg]['usable_ai'] / 1000])
        k = int(np.argmin(load[:, 2] + load[:, 1] / 10))
        folds[k].append(lg)
        load[k] += v
    hl = np.zeros(K)
    for lg in hu_logs:
        k = int(np.argmin(hl))
        folds[k].append(lg)
        hl[k] += rows[lg]['usable_human']
    res['log_folds_proposal'] = {
        'k': K, 'folds': folds,
        'per_fold': [{'logs': fl, 'usable_ai': sum(rows[x]['usable_ai'] for x in fl),
                      'usable_human': sum(rows[x]['usable_human'] for x in fl),
                      'r1_passes_ai': sum(rows[x]['r1_passes_ai'] for x in fl),
                      'r1_strong_ai': sum(rows[x]['r1_strong_ai'] for x in fl),
                      'collisions_ai': sum(rows[x]['collisions_ai'] for x in fl),
                      'positives_1s_ai': sum(rows[x]['positives_1s_ai'] for x in fl),
                      'positives_1s_human': sum(rows[x]['positives_1s_human'] for x in fl)} for fl in folds]}
    return res


def check_locations_cached():
    return cached('locations', check_locations)


def check_quality():
    f = F()
    u = usable(f)
    res = {}
    fut = f['fut']
    h = u & (f['source'] == 0)
    res['human_throttle_glitch'] = {
        'rule': '人の未来の操作のどれかが 900µs 未満(受信機の読み取りの異常。10ms のサンプル 1〜2個)',
        'usable_frames': int((h & (fut[:, :, 1] < 900).any(axis=1)).sum()),
        'usable_frames_val': int((h & (f['split'] == 'val') & (fut[:, :, 1] < 900).any(axis=1)).sum()),
        'min_us': r(fut[h][:, :, 1].min(), 0)}
    res['erpm_valid_false_on_ai_frames'] = int(((f['source'] == 1) & ~f['erpm_valid']).sum())
    # モデルB の中心・中立
    rb = ROOT / 'data' / 'runs' / 'clean_driving_20261011-nocmd-s3-20261011-024626'
    for nm in ('result.json', 'model_config.json'):
        p = rb / nm
        if p.exists():
            d = json.loads(p.read_text())
            res.setdefault('model_b', {})[nm] = {k: d.get(k) for k in ('steer_center_us', 'throttle_neutral_us',
                                                                      'steer_center', 'throttle_neutral',
                                                                      'trained_modes', 'command_input', 'dataset')
                                                 if k in d}
    # AI のフレームのジャンル
    res['modes_usable'] = {'%s/%d' % ('ai' if s_ else 'human', m): int((u & (f['source'] == s_) & (f['mode'] == m)).sum())
                           for s_ in (0, 1) for m in (4, 5)}
    # 罰を移したもの(label_events の修正)
    s = json.loads((LAB / 'summary.json').read_text())
    res['carried_penalties'] = {'totals': {k: s['totals'].get(k) for k in ('penalties_carried', 'collisions_on_masked_frames',
                                                                         'failure_takeovers_on_masked_frames')},
                                'events': {log: json.loads((LAB / (log + '.json')).read_text())['events'].get('carried_penalties')
                                           for log in ('20261010-155820', '20261010-181504')}}
    return res


def main():
    out = {}
    for name, fn in (('actions', check_actions), ('rewards', check_rewards), ('collisions', check_collisions),
                     ('locations', check_locations), ('splits', check_splits), ('intervention', check_intervention),
                     ('value_and_folds', check_value_and_folds), ('quality', check_quality)):
        print('...', name, flush=True)
        out[name] = cached(name, fn)
    p = OUTD / 'data_check.json'
    with open(str(p) + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=le._json_default)
    os.replace(str(p) + '.tmp', p)
    print('書き出し:', p)


if __name__ == '__main__':
    main()
