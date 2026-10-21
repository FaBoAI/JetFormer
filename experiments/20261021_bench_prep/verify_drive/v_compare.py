#!/usr/bin/env python3
"""V-drive: v_replay.py の結果(変更前の木 / 今の木)を比べる。

    python -I v_compare.py off   old.pkl new.pkl          # OFF: 車への出力・記録の欄・状態がビット単位で同じか
    python -I v_compare.py on    off_new.pkl on_new.pkl   # ON: 舵は同じ・前進を強めない・人/中立の行は変えない
    python -I v_compare.py toggle old.pkl toggle_new.pkl  # 途中で ON → OFF: OFF の区間は変更前と同じ
"""
import collections
import json
import pickle
import sys

import numpy as np


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def rows_diff(r0, r1, skip=('infer_ms',), idx=None):
    """r0 の欄が r1 で同じか。違いの数と最初の例"""
    n, first = 0, None
    rng = range(len(r0)) if idx is None else idx
    for i in rng:
        (t0, a), (t1, b) = r0[i], r1[i]
        if t0 != t1:
            n += 1
            first = first or (i, 't', t0, t1)
            continue
        for k, v in a.items():
            if k in skip:
                continue
            if k not in b or not same(v, b[k]):
                n += 1
                first = first or (i, k, repr(v)[:80], repr(b.get(k))[:80])
    return n, first


CAP_KEYS = ('cap_on', 'cap_zone', 'cap_zone_gyro', 'cap_pred_sig', 'cap_rel_psi', 'cap_mode', 'cap_limit', 'cap_in_us',
            'cap_thr_us', 'cap_brake_start', 'cap_applied')
ST_SKIP = ('zone_caps_error',)
STATS_SKIP = ('infer_ms', 'zone_ms', 'zone_ms_max', 'zone_brakes', 'zone_applied')


def cmp_off(a, b):
    out = {'log': a['log'], 'variant': a['variant'], 'N': a['N'], 'err_old': a['err'], 'err_new': b['err']}
    out['pairs_equal'] = a['pairs'] == b['pairs']
    out['single_equal'] = a['single'] == b['single']
    out['n_pairs'] = len(a['pairs'])
    out['n_rows'] = (len(a['rows']), len(b['rows']))
    n, first = rows_diff(a['rows'], b['rows']) if len(a['rows']) == len(b['rows']) else (-1, 'len')
    out['row_field_diffs'] = n
    out['row_first_diff'] = first
    out['new_rows_have_cap'] = all(all(k in r for k in CAP_KEYS) for _, r in b['rows'])
    out['new_cap_applied_any'] = any(r['cap_applied'] for _, r in b['rows'])
    out['new_cap_on_any'] = any(r['cap_on'] for _, r in b['rows'])
    out['events_old'] = [e[0] for e in a['events']]
    out['events_new'] = [e[0] for e in b['events']]
    # 状態(画面に出る値): 変更前にある項目は同じ(区間の速度上限の項目は足しただけ)
    sa, sb = a['st'], b['st']
    st_diff = [k for k in sa if k not in ST_SKIP and k not in ('stats', 'now') and not same(sa[k], sb.get(k))]
    stats_diff = [k for k in sa.get('stats', {}) if k not in STATS_SKIP and not same(sa['stats'][k], sb['stats'].get(k))]
    now_a, now_b = sa.get('now') or {}, sb.get('now') or {}
    now_diff = [k for k in now_a if not same(now_a[k], now_b.get(k))]
    out['st_diff'], out['stats_diff'], out['now_diff'] = st_diff, stats_diff, now_diff
    out['zone_caps_error'] = sb.get('zone_caps_error')
    out['params_end_equal'] = {k: v for k, v in b['params_end'].items() if k != 'zone_caps'} == a['params_end']
    rows = [r for _, r in b['rows']]
    out['cap_mode_counts'] = dict(collections.Counter(int(r['cap_mode']) for r in rows))
    out['cap_zone_counts'] = dict(collections.Counter(int(r['cap_zone']) for r in rows))
    out['holds'] = sum(bool(r['hold']) for r in rows)
    out['shadow_weaken_frames'] = sum(1 for r in rows if not r['hold'] and r['cap_thr_us'] > r['out'][1])
    out['shadow_brake_starts'] = sum(bool(r['cap_brake_start']) for r in rows)
    out['assist_counts'] = dict(collections.Counter(int(r['assist']) for r in rows))
    lm = np.array([x[1] for x in b['loop_ms']]) if b['loop_ms'] else np.zeros(1)
    out['zone_ms_per_wait'] = {'mean': float(lm.mean()), 'p99': float(np.percentile(lm, 99)), 'max': float(lm.max())}
    la = np.array([x[0] for x in a['loop_ms']]) if a['loop_ms'] else np.zeros(1)
    lb = np.array([x[0] for x in b['loop_ms']]) if b['loop_ms'] else np.zeros(1)
    out['loop_ms_old'] = {'mean': float(la.mean()), 'p99': float(np.percentile(la, 99))}
    out['loop_ms_new'] = {'mean': float(lb.mean()), 'p99': float(np.percentile(lb, 99))}
    out['ok'] = bool(out['pairs_equal'] and out['single_equal'] and n == 0 and out['new_rows_have_cap']
                     and not out['new_cap_applied_any'] and not st_diff and not stats_diff and not now_diff
                     and out['events_old'] == [e for e in out['events_new'] if e != 'zone_caps_error']
                     and a['err'] == b['err'] and out['params_end_equal'])
    return out


def cmp_on(off, on):
    """同じ入力で OFF / ON(今の木どうし)。ON の差は、上限の層が変えたスロットルだけ"""
    out = {'log': off['log'], 'variant': off['variant'], 'N': off['N'], 'err': on['err']}
    r0, r1 = off['rows'], on['rows']
    out['n_rows'] = (len(r0), len(r1))
    assert len(r0) == len(r1)
    bad = collections.Counter()
    first = {}
    n_app = 0
    modes = collections.Counter()
    for i, ((t0, a), (t1, b)) in enumerate(zip(r0, r1)):
        def fail(k):
            bad[k] += 1
            first.setdefault(k, (i, a['out'], b['out'], b.get('cap_mode'), b.get('select'), b.get('hold')))
        if t0 != t1:
            fail('time')
            continue
        if a['out'][0] != b['out'][0]:
            fail('steer_changed')
        if b['out'][1] < a['out'][1]:
            fail('more_forward')                     # 前進は小さい値 → ON が OFF より小さい = 前進を強めた
        if b['hold'] != a['hold']:
            fail('hold_changed')
        if b['hold'] and b['out'] != a['out']:
            fail('hold_out_changed')
        if (not b['select'] or b['dry_run']) and b['out'] != a['out']:
            fail('human_or_dry_changed')
        if b['cap_applied'] != (b['out'][1] != a['out'][1]):
            fail('applied_flag')
        if b['cap_in_us'] != a['out'][1] and not b['hold']:
            fail('cap_in_mismatch')                  # 層への入力は OFF の出力(= 上限の前)と同じはず
        if not b['hold'] and b['cap_applied'] and b['out'][1] != b['cap_thr_us']:
            fail('out_ne_cap_thr')
        if a['out'][1] > a['model_out'][1] + 1e-9 and False:
            pass
        # 影(OFF)と ON の区間・出し方は同じ(層への入力が同じなので)
        for k in ('cap_zone', 'cap_mode', 'cap_thr_us', 'cap_brake_start', 'cap_zone_gyro'):
            if not same(a[k], b[k]):
                fail('shadow_ne_on_' + k)
        n_app += bool(b['cap_applied'])
        if b['cap_applied']:
            modes[int(b['cap_mode'])] += 1
        # 後退側・中立付近はそのまま
        if a['out'][1] >= 1498.0 and b['out'][1] != a['out'][1] and b['cap_mode'] != 3:
            fail('reverse_changed_not_brake')
    out['violations'] = dict(bad)
    out['first'] = {k: [str(x) for x in v] for k, v in first.items()}
    out['applied_frames'] = n_app
    out['applied_modes'] = dict(modes)
    out['stats_zone_applied'] = on['st']['stats'].get('zone_applied')
    out['stats_zone_brakes'] = on['st']['stats'].get('zone_brakes')
    out['zone_caps_error'] = on['st'].get('zone_caps_error')
    out['pairs_len'] = (len(off['pairs']), len(on['pairs']))
    # PCA9685 に出した値: 舵は同じ、スロットルは ≥
    pa, pb = np.array(off['pairs']), np.array(on['pairs'])
    out['pca_steer_equal'] = bool(len(pa) == len(pb) and np.array_equal(pa[:, 0], pb[:, 0]))
    out['pca_thr_never_more_forward'] = bool(len(pa) == len(pb) and np.all(pb[:, 1] >= pa[:, 1]))
    out['pca_thr_max'] = float(pb[:, 1].max()) if len(pb) else None
    out['ok'] = bool(not bad and out['pca_steer_equal'] and out['pca_thr_never_more_forward'] and on['err'] is None
                     and on['st'].get('zone_caps_error') is None and n_app == out['stats_zone_applied'])
    return out


def cmp_toggle(old, tg):
    """途中で zone_caps 0 → 1 → 0。ON の区間の外は変更前と同じ"""
    r0, r1 = old['rows'], tg['rows']
    n = len(r0)
    on_idx = [i for i, (_, r) in enumerate(r1) if r['cap_on']]
    out = {'log': old['log'], 'N': old['N'], 'on_rows': len(on_idx)}
    if not on_idx:
        out['ok'] = False
        return out
    lo, hi = on_idx[0], on_idx[-1]
    nb, fb = rows_diff(r0, r1, idx=range(0, lo))
    na, fa = rows_diff(r0, r1, idx=range(hi + 1, n))
    out['before_on_diffs'], out['after_off_diffs'] = nb, na
    out['first'] = [fb, fa]
    out['applied_in_on'] = sum(bool(r1[i][1]['cap_applied']) for i in range(lo, hi + 1))
    out['applied_outside'] = sum(bool(r1[i][1]['cap_applied']) for i in list(range(0, lo)) + list(range(hi + 1, n)))
    out['ok'] = nb == 0 and na == 0 and out['applied_outside'] == 0 and tg['err'] is None
    return out


def cmp_fail(old, fl):
    """ON で走っていて、途中で上限の計算が例外 → そのあとは変更前と同じ出力・記録に -2・メッセージ・events に 1 回"""
    r0, r1 = old['rows'], fl['rows']
    out = {'log': old['log'], 'N': old['N'], 'err': fl['err']}
    k = next((i for i, (_, r) in enumerate(r1) if r['cap_mode'] == -2), None)
    out['fail_row'] = k
    if k is None:
        out['ok'] = False
        return out
    n, first = rows_diff(r0, r1, idx=range(k, len(r0)))
    out['after_fail_diffs'], out['first'] = n, first
    out['all_minus2_after'] = all(r['cap_mode'] == -2 and not r['cap_applied'] for _, r in r1[k:])
    out['applied_before'] = sum(bool(r['cap_applied']) for _, r in r1[:k])
    # PCA9685: 失敗の行より後の書き込みが同じ(行と書き込みの対応は、最後から数える)
    tail = len(r0) - k
    pa, pb = old['pairs'], fl['pairs']
    out['pca_tail_equal'] = pa[-tail:] == pb[-tail:] if tail else True
    out['events'] = [e[0] for e in fl['events']]
    st = fl['st']
    out['zone_caps_error'] = st.get('zone_caps_error')
    out['note_has'] = '上限を外しました' in (st.get('recording_note') or '')
    out['ok'] = bool(n == 0 and out['all_minus2_after'] and out['events'] == ['zone_caps_error'] and out['zone_caps_error']
                     and out['note_has'] and fl['err'] is None and out['pca_tail_equal'])
    return out


if __name__ == '__main__':
    kind, pa, pb = sys.argv[1], sys.argv[2], sys.argv[3]
    a, b = load(pa), load(pb)
    r = {'off': cmp_off, 'on': cmp_on, 'toggle': cmp_toggle, 'fail': cmp_fail}[kind](a, b)
    print(json.dumps(r, ensure_ascii=False, default=str))
