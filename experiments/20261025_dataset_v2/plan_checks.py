#!/usr/bin/env python3
"""確定版の計画(plan.md)のための数え直し(2026-10-25)。読むだけ。

批評(critique.md)の直しに要る数だけを出す。numpy・OpenCV(読み込みだけ)。torch・GPU は使わない。data/ は変えない。
inventory.py の 1 本の記録の処理(process_log)をそのまま使い、マスクを 3 つ足して数える
(inventory.py のファイルは変えない。読み込んだ関数の写しに足す。明るさの画像は読まない)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        python3 -I /home/jetson/jetformer/experiments/20261025_dataset_v2/plan_checks.py

出力: /home/jetson/jetformer_reports/dataset_v2_plan_20261025/plan_checks.json

数えるもの:
  1. 新しい人の走行(棚卸しの (a)・衝突の前後を除く・ジャンル 4・5・clean_driving に無い・10/10 の介入の窓の外)を
     「走りの区間」(前後 1 秒の前進 eRPM の中央値 >= 3,000)と「交代の直後 3 秒」に分けた数。直線のスロットルの中央値も
  2. 同じ分け方で、今の学習 41 シーン(clean_driving_20261011 の train)と封の回(10/24 1・3 回目)
  3. 止まった場面の上限(AI 側・dry run でない・|eRPM| < 400・衝突の前後でない)。前の ToF の条件の前の数
  4. 閾値の数(train_seed_spread.json から): 種 2・4 本で比べるときの標準誤差
"""
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path('/home/jetson/jetformer/experiments/20261025_dataset_v2')
REP = Path('/home/jetson/jetformer_reports/dataset_v2_plan_20261025')
OUT = REP / 'plan_checks.json'
RUN_ERPM = 3000.0       # 走りの区間: 前後 1 秒(61 フレーム)の前進 eRPM の中央値がこれ以上
RUN_K = 61
STOP_ERPM = 400.0       # 0.2 m/s
SEALED = ('10/24 1 回目(13:46〜)', '10/24 3 回目(18:31〜)')
TRAIN_SESS_2 = '10/24 2 回目(16:01〜)'


def load_inventory():
    spec = importlib.util.spec_from_file_location('inv_copy', str(HERE / 'inventory.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fast_scene_span(sdir):
    """明るさの画像を読まない scene_brightness(時刻と span だけ)"""
    try:
        with np.load(sdir / 'CAM_FRONT.npz', allow_pickle=False) as z:
            ts = z['timestamps_us'].astype(np.int64)
    except Exception:
        names = sorted(os.listdir(sdir / 'CAM_FRONT')) if (sdir / 'CAM_FRONT').is_dir() else []
        names = [x for x in names if x.endswith('.jpg') and x[:-4].isdigit()]
        ts = np.array([int(x[:-4]) for x in names], np.int64)
    idx = np.arange(0, len(ts), 3)
    span = (int(ts[0]), int(ts[-1])) if len(ts) else None
    nan = np.full(len(idx), np.nan)
    return (ts[idx] if len(ts) else ts), nan, nan.copy(), len(ts), span


EXTRA = '''
    # ---- plan_checks で足した数(ここから)
    _e = -erpm
    if n >= PC_RUN_K:
        from numpy.lib.stride_tricks import sliding_window_view as _swv
        _h = PC_RUN_K // 2
        _pad = np.concatenate([np.full(_h, np.nan), _e, np.full(_h, np.nan)])
        _med = np.nanmedian(_swv(_pad, PC_RUN_K), axis=1)
    else:
        _med = np.full(n, np.nanmedian(_e) if n else np.nan)
    _run = _med >= PC_RUN_ERPM
    _ivw = np.zeros(n, bool)
    for _w in PC_IV.get(log, []):
        _ivw |= (t >= _w) & (t < _w + 5.0)
    _neu = float(rec.get('human_throttle_neutral_us') or 1500.0)
    _u = _neu - lab['action_exec'][:, 1].astype(np.float64)
    _new = a_mask & ~near & ~base_tr & ~base_va & np.isin(mode, (4, 5)) & ~_ivw
    _base = a_mask & ~near & base_tr
    _val = a_mask & ~near & base_va
    _ivin = a_mask & ~near & ~base_tr & ~base_va & np.isin(mode, (4, 5)) & _ivw
    def _grp(m):
        st = m & straight
        return {'samples': samples(m, idx_in), 'run': samples(m & _run, idx_in), 'tk': samples(m & b_mask, idx_in),
                'run_or_tk': samples(m & (_run | b_mask), idx_in), 'neither': samples(m & ~_run & ~b_mask, idx_in),
                'mode4': samples(m & (mode == 4), idx_in), 'mode5': samples(m & (mode == 5), idx_in),
                'keep_mode4': samples(m & (_run | b_mask) & (mode == 4), idx_in),
                'keep_mode5': samples(m & (_run | b_mask) & (mode == 5), idx_in),
                'st_n': int(st.sum()), 'st_u': [float(x) for x in _u[st]], 'st_e': [float(x) for x in _e[st]],
                'st_keep_u': [float(x) for x in _u[st & (_run | b_mask)]],
                'st_keep_e': [float(x) for x in _e[st & (_run | b_mask)]],
                'st_drop_u': [float(x) for x in _u[st & ~_run & ~b_mask]]}
    _stop = ai & ~near & (np.abs(erpm) < PC_STOP_ERPM)
    PC_OUT[log] = {'session': session, 'new': _grp(_new), 'base_train': _grp(_base), 'base_val': _grp(_val),
                   'iv_in': _grp(_ivin),
                   'standstill_ai': {'frames': int(_stop.sum()), 'samples': samples(_stop, idx_in),
                                     'seconds': float(dt[_stop].sum())}}
    # ---- plan_checks で足した数(ここまで)
'''


def iv_windows():
    card = json.load(open('/home/jetson/jetformer/data/datasets/intervention_val_20261011/dataset_card.json', encoding='utf-8'))
    out = {}
    for w in card['intervention']['windows']:
        if w.get('included'):
            out.setdefault(w['log'], []).append(float(w['t_s']))
    return out


def med(xs):
    return round(float(np.median(xs)), 0) if len(xs) else None


def p90(xs):
    return round(float(np.percentile(xs, 90)), 0) if len(xs) else None


def summarize(groups):
    out = {}
    for name, items in groups.items():
        agg = {}
        for it in items:
            for k, v in it.items():
                if isinstance(v, list):
                    agg.setdefault(k, []).extend(v)
                else:
                    agg[k] = agg.get(k, 0) + v
        res = {k: v for k, v in agg.items() if not isinstance(v, list)}
        res['straight_u_p50'] = med(agg.get('st_u', []))
        res['straight_erpm_p50'] = med(agg.get('st_e', []))
        res['straight_erpm_p90'] = p90(agg.get('st_e', []))
        res['keep_straight_u_p50'] = med(agg.get('st_keep_u', []))
        res['keep_straight_erpm_p50'] = med(agg.get('st_keep_e', []))
        res['keep_straight_erpm_p90'] = p90(agg.get('st_keep_e', []))
        res['drop_straight_u_p50'] = med(agg.get('st_drop_u', []))
        out[name] = res
    return out


def seed_numbers():
    ss = json.load(open(REP / 'train_seed_spread.json', encoding='utf-8'))
    fam = ['A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B']
    pr = ss['per_run']
    out = {}
    for which in ('best', 'last'):
        for key in ('l1', 'steer_bias', 'thr_bias', 'l1_interv', 'r1_app', 'de_thr'):
            xs = [pr[k][which][key] for k in fam]
            m = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
            out['%s_%s' % (which, key)] = {
                'mean': round(m, 3), 'sd': round(sd, 3), 'range': round(max(xs) - min(xs), 3),
                # 腕 n 本の平均と 5 本の平均の差の標準誤差、腕 n 本どうし(同じ種の組)の差の標準誤差(独立とみなす)
                'se_2v5': round(sd * math.sqrt(1 / 2 + 1 / 5), 3), 'se_4v5': round(sd * math.sqrt(1 / 4 + 1 / 5), 3),
                'se_2v2': round(sd * math.sqrt(2 / 2), 3), 'se_4v4': round(sd * math.sqrt(2 / 4), 3),
                'thr_2se_2v5': round(2 * sd * math.sqrt(1 / 2 + 1 / 5), 3),
                'thr_2se_4v5': round(2 * sd * math.sqrt(1 / 4 + 1 / 5), 3),
                'thr_2se_4v4': round(2 * sd * math.sqrt(2 / 4), 3)}
    return out


def main():
    inv = load_inventory()
    inv.scene_brightness = fast_scene_span
    src = inspect.getsource(inv.process_log)
    anchor = "    rec['candidates'] = {"
    assert src.count(anchor) == 1, 'process_log の形が変わった'
    src = src.replace(anchor, EXTRA + anchor)
    ns = inv.__dict__
    ns.update(PC_RUN_K=RUN_K, PC_RUN_ERPM=RUN_ERPM, PC_STOP_ERPM=STOP_ERPM, PC_IV=iv_windows(), PC_OUT={})
    exec(compile(src, str(HERE / 'inventory.py') + ' (plan_checks の写し)', 'exec'), ns)
    modes = inv.mode_table()
    membership = inv.dataset_membership()
    logs = sorted(d for d in os.listdir(inv.RAW) if d[:8] in inv.DAYS and (inv.RAW / d).is_dir())
    sess = inv.sessions(logs)
    for log in logs:
        rec, _ = ns['process_log'](log, membership, modes, sess.get(log))
        print(log, rec.get('session'), 'ok' if log in ns['PC_OUT'] else '飛ばした', flush=True)
    po = ns['PC_OUT']
    iv_logs = set(iv_windows())
    g_new = {'10/10 介入の 8 本(窓の外)': [], '10/10 ほかの AI 走行 3 本': [], '10/10 人の記録(新しい分)': [],
             '10/23': [], '10/24 2 回目': [], '封 10/24 1 回目': [], '封 10/24 3 回目': []}
    g_base = {'今の学習 41 シーン': [], '保留 11 シーン': []}
    g_ivin = {'10/10 介入の窓の中': []}
    stand = {}
    for log, d in po.items():
        s = d['session'] or ''
        if log.startswith('20261010'):
            ai_log = any(x in log for x in ('155820', '160306', '163247'))
            k = '10/10 介入の 8 本(窓の外)' if log in iv_logs else ('10/10 ほかの AI 走行 3 本' if ai_log else '10/10 人の記録(新しい分)')
        elif log.startswith('20261023'):
            k = '10/23'
        elif s == TRAIN_SESS_2:
            k = '10/24 2 回目'
        elif s == SEALED[0]:
            k = '封 10/24 1 回目'
        elif s == SEALED[1]:
            k = '封 10/24 3 回目'
        else:
            k = None
        if k:
            g_new[k].append(d['new'])
        g_base['今の学習 41 シーン'].append(d['base_train'])
        g_base['保留 11 シーン'].append(d['base_val'])
        g_ivin['10/10 介入の窓の中'].append(d['iv_in'])
        st = stand.setdefault(s, {'frames': 0, 'samples': 0, 'seconds': 0.0})
        for kk in st:
            st[kk] += d['standstill_ai'][kk]
    res = {'schema': 'jf-dataset-v2-plan-checks/v1', 'tool': 'experiments/20261025_dataset_v2/plan_checks.py',
           'rule': {'run': '前後 1 秒(%d フレーム)の前進 eRPM の中央値 >= %d' % (RUN_K, RUN_ERPM),
                    'tk': '交代の直後 3 秒(棚卸しの b)', 'base': '棚卸しの (a)・衝突の前後を除く・ジャンル 4・5・clean_driving に無い・介入の窓の外',
                    'standstill': 'AI 側・dry run でない・|eRPM| < %d・衝突の前後でない(ToF の条件の前)' % STOP_ERPM},
           'new_human': summarize(g_new), 'base': summarize(g_base), 'iv_in': summarize(g_ivin),
           'standstill_ai_by_session': {k: {kk: (round(v, 1) if isinstance(v, float) else v) for kk, v in d.items()}
                                        for k, d in sorted(stand.items())},
           'seed': seed_numbers()}
    tmp = str(OUT) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    Path(tmp).replace(OUT)
    print(json.dumps({k: res[k] for k in ('new_human', 'base', 'iv_in', 'standstill_ai_by_session')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
