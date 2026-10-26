#!/usr/bin/env python3
"""D3(拡張)の数え直し(verify_d3、2026-10-26)。

eval_v2.py・judge_d3_v2.py・desk_v2.py・make_stage3_judge_v2.py・recount_stage12_v2.py を import しない(numpy と標準の
ライブラリだけ)。protocol_v2.json の決まりを、保存した予測(eval/pred の best・last)と、C3 のずれの試しの保存した予測
(desk/pred/<名前>_shift.npz)から自分で当てる。

  1. 固定(protocol_v2.sha256・lock2)と、decisions.json に書かれた sha256
  2. H1・介入の欄(l1_best・l1_last・l1_interv・steer_bias・l1_day・l1_night)を予測から計算
  3. robust_dl1_us を 8 本とも shift.npz から計算(desk_v2_stage3.json・写しの値と照らす)
  4. 5 本(A_s0・A_s1・v0-s2・v0-s3・B)の保存した予測から拒否権の 5 本の平均
  5. protocol decisions.D3 の 6 つの条件 → AUG2 か なし。decisions.json の D3 と照らす
  6. 段 3 と段 2(v2h-e60)の学習のコマンドが protocol の型どおりか(train.log の最初の開始の行)
  7. 段 4 の名前・コマンドを protocol の型から自分で作り、gpu_queue_v2.py resolve の答え(読むだけ)と照らす

書くのは /home/jetson/jetformer_reports/train_v2_20261025/verify_d3.json だけ。車・装置・走行の API に触れない。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        /home/jetson/jetformer/experiments/20261025_train_v2/verify_d3/recount_d3.py
"""

import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
REPO = Path('/home/jetson/jetformer')
EXP = REPO / 'experiments' / '20261025_train_v2'
PROTO_P = REP / 'protocol_v2.json'
LOCK2_P = REP / 'protocol_v2_lock2.json'
DEC_P = REP / 'decisions.json'
STAGE3_DESK = REP / 'desk_v2_stage3.json'
SNAP_DESK = REP / 'desk_v2_snapshot_20261026-0359_for_d3.json'
LIVE_DESK = REP / 'desk' / 'desk_v2.json'          # 読むだけ(別の係が書く)
EVAL_PRED = REP / 'eval' / 'pred'
SHIFT_DIR = REP / 'desk' / 'pred'
BUNDLE_P = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
ZOO = REPO / 'data' / 'zoo' / 'policy_zoo_20261012' / 'pred'
RACC = REPO / 'data' / 'zoo' / 'real_accuracy_20261015' / 'pred'
RUNS = REPO / 'data' / 'runs'
OUT = REP / 'verify_d3.json'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'

# 5 本の保存した予測(protocol reference_predictions_sha256 の 5 つ。ファイルの場所は 10/12・10/15 の保存)
FIVE_FILES = {
    'A_s0': (ZOO / 'jetformer_v0_s0.npz', ZOO / 'jetformer_v0_s0_last.npz'),
    'A_s1': (ZOO / 'jetformer_v0_s1.npz', ZOO / 'jetformer_v0_s1_last.npz'),
    'v0-s2': (RACC / 'realacc_20261015-v0-s2.npz', RACC / 'realacc_20261015-v0-s2_last.npz'),
    'v0-s3': (RACC / 'realacc_20261015-v0-s3.npz', RACC / 'realacc_20261015-v0-s3_last.npz'),
    'B': (ZOO / 'B.npz', ZOO / 'B_last.npz'),
}
SEEDS = (0, 1, 2, 3)


def h(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            d.update(c)
    return d.hexdigest()


def jload(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def amean(xs):
    xs = [float(x) for x in xs]
    return math.fsum(xs) / len(xs)


CHECKS = []          # (名前, 合否, 中身)


def check(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    print('%s %s %s' % ('OK ' if ok else 'NG ', name, '' if detail is None else json.dumps(detail, ensure_ascii=False)[:300]),
          flush=True)
    return ok


# ------------------------------------------------------------------ 1. 固定

def seals(proto_txt_sha, dec):
    lines = [ln for ln in (REP / 'protocol_v2.sha256').read_text(encoding='utf-8').splitlines()
             if ln.strip() and not ln.startswith('#')]
    res = {}
    for ln in lines:
        want, path = ln.split(None, 1)
        p = Path(path) if path.startswith('/') else REP / path
        res[path] = (h(p) == want)
    check('protocol_v2.sha256 の %d つ' % len(res), all(res.values()), res)
    lk = [ln for ln in (REP / 'protocol_v2_lock2.sha256').read_text(encoding='utf-8').splitlines()
          if ln.strip() and not ln.startswith('#')]
    want, path = lk[0].split(None, 1)
    lock2_sha = h(REP / path)
    check('protocol_v2_lock2.sha256', lock2_sha == want, {'sha256': lock2_sha})
    check('decisions.json の protocol_v2_sha256 = 今のファイル', dec['protocol_v2_sha256'] == proto_txt_sha)
    check('decisions.json の protocol_v2_lock2_sha256 = 今のファイル', dec['protocol_v2_lock2_sha256'] == lock2_sha)
    # lock2 に書かれた台本の sha256(判定の台本・キュー)が今のファイルと同じか
    L = jload(LOCK2_P)['files']
    sc = {}
    for k, v in L['C4(キュー v2)'].items():
        sc[k] = (h(v['path']) == v['sha256'])
    sc['protocol_numbers.py'] = h(L['数え直しの台本']['protocol_numbers.py']['path']) == L['数え直しの台本']['protocol_numbers.py']['sha256']
    for k, v in L['tables_7'].items():
        sc[k] = (h(v['path']) == v['sha256'])
    sc['dataset_card'] = h(L['dataset_card']['path']) == L['dataset_card']['sha256']
    check('lock2 の台本・表・データセットの sha256 = 今のファイル', all(sc.values()), sc)
    return lock2_sha


# ------------------------------------------------------------------ 2. 予測と欄

def load_bundle(proto):
    check('desk_inputs_20261022.npz の sha256 = protocol', h(BUNDLE_P) == proto['reference_predictions_sha256']['bundle'])
    with np.load(BUNDLE_P, allow_pickle=False) as z:
        B = {k: z[k] for k in ('key', 'scene', 'y', 'primary', 'interv', 'part')}
    B['key'] = str(B['key'])
    B['scene'] = B['scene'].astype(str)
    B['y'] = B['y'].astype(np.float64)
    prim, iv = B['primary'].astype(bool), B['interv'].astype(bool)
    B['primary'], B['interv'] = prim, iv
    nan_p = int(np.isnan(B['y'][prim]).sum())
    nan_i = int(np.isnan(B['y'][iv]).sum())
    check('H1 4,670・介入 1,570 フレーム(protocol holdouts)', int(prim.sum()) == proto['holdouts']['H1']['frames_desk']
          and int(iv.sum()) == proto['holdouts']['interv']['frames_desk'], {'H1': int(prim.sum()), 'interv': int(iv.sum())})
    check('H1・介入の正解に NaN が無い(nan_to_num に依らない)', nan_p == 0 and nan_i == 0, {'H1': nan_p, 'interv': nan_i})
    dn = proto['holdouts']['day_night']
    day = prim & np.isin(B['scene'], dn['day'])
    night = prim & np.isin(B['scene'], dn['night'])
    h1_scenes = sorted(set(B['scene'][prim].tolist()))
    check('H1 の 11 シーン = protocol(昼 4・夜 7)、昼と夜で H1 を分けきる',
          h1_scenes == sorted(proto['holdouts']['H1']['scenes']) and not (day & night).any() and ((day | night) == prim).all(),
          {'day_frames': int(day.sum()), 'night_frames': int(night.sum())})
    B['day'], B['night'] = day, night
    sealed = set(proto['seal']['logs'])
    check('予測の表に封の記録が無い', not any(s[:15] in sealed for s in set(B['scene'].tolist())))
    return B


def load_pred(path, key, run=None, ck=None):
    with np.load(path, allow_pickle=False) as z:
        p = z['pred'].astype(np.float64)
        meta = {k: str(z[k]) for k in ('key', 'run', 'ckpt') if k in z.files}
    ok = meta.get('key') == key and p.shape == (10985, 10, 2) and bool(np.isfinite(p).all())
    if run is not None:
        ok = ok and meta.get('run') == run
    if ck is not None:
        ok = ok and meta.get('ckpt') == ck
    if not ok:
        raise SystemExit('予測のファイルがおかしい: %s %s %s' % (path, meta, p.shape))
    return p


def fields(best, last, B):
    """H1 の L1 = H1 のフレーム × 10 ステップ × 2(ステア・スロットル)の |予測 − 人| の平均(desk_bench と同じ定義)"""
    y, P, I = B['y'], B['primary'], B['interv']
    ab = np.abs(best - y)
    al = np.abs(last - y)
    return {
        'l1_best_us': float(ab[P].mean()),
        'l1_last_us': float(al[P].mean()),
        'l1_interv_us': float(ab[I].mean()),
        'steer_bias_us': float((best[P, 0, 0] - y[P, 0, 0]).mean()),      # 行 0 の 予測 − 人(+ = 左)
        'l1_day_us': float(ab[B['day']].mean()),
        'l1_night_us': float(ab[B['night']].mean()),
    }


# ------------------------------------------------------------------ 3. robust_dl1_us

def robust(run, B, best, proto_shift_count):
    p = SHIFT_DIR / ('%s_shift.npz' % run)
    with np.load(p, allow_pickle=False) as z:
        meta = {k: str(z[k]) for k in ('key', 'run', 'ckpt', 'best_sha256', 'script_sha256')}
        rows = z['rows'].astype(np.int64)
        kinds = sorted(k[5:] for k in z.files if k.startswith('pred_') and k not in ('pred_none', 'pred_none_rows'))
        sh = {k: z['pred_' + k].astype(np.float64) for k in kinds}
        nrows = z['pred_none_rows'].astype(np.int64)
        pnone = z['pred_none'].astype(np.float64)
    want_rows = np.nonzero(B['primary'])[0]
    ok_rows = rows.shape == want_rows.shape and bool((rows == want_rows).all())
    best_now = h(RUNS / run / 'best.pth')
    y = B['y'][rows]
    base_l1 = float(np.abs(best[rows] - y).mean())
    per = {k: float(np.abs(sh[k] - y).mean()) - base_l1 for k in kinds}
    val = amean(per.values())
    none_vs_saved = float(np.abs(pnone - best[rows][nrows]).max())
    return {
        'robust_dl1_us': val, 'per_shift_us': per, 'base_l1_us': base_l1, 'kinds': kinds,
        'rows_are_H1_in_order': ok_rows, 'n_kinds_ok': len(kinds) == proto_shift_count,
        'key_ok': meta['key'] == B['key'], 'run_ok': meta['run'] == run, 'ckpt_ok': meta['ckpt'] == 'best',
        'best_pth_sha256_same_as_now': meta['best_sha256'] == best_now,
        'pred_none_rows': int(len(nrows)), 'pred_none_vs_saved_best_max_abs_us': none_vs_saved,
        'script_sha256': meta['script_sha256'],
    }


# ------------------------------------------------------------------ 6・7. コマンド

def proto_cmd(proto, name):
    for r in proto['runs']:
        if r['name'] == name:
            return r
    return None


def fill(proto, text, x, E, aug):
    ph = proto['choice_placeholders']
    out = text.replace('{X_DATA}', ph['{X_DATA}'][x]).replace('{x}', x).replace('{E}', str(E))
    out = out.replace('{AUG_ARGS}', ph['{AUG_ARGS}'][aug]).replace('{A}', '-aug2' if aug == 'AUG2' else '')
    return ' '.join(out.split())


def workers(cmd, n):
    return re.sub(r'--workers \d+', '--workers %d' % n, cmd)


def first_start_cmd(run):
    p = EXP / 'logs' / ('train_%s.log' % run)
    for ln in p.read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.match(r'^==== \S+ \S+ 開始\((.*)\)====$', ln)
        if m:
            return m.group(1)
    return None


def main():
    t0 = time.time()
    proto_sha = h(PROTO_P)
    proto = jload(PROTO_P)
    dec = jload(DEC_P)
    lock2_sha = seals(proto_sha, dec)
    x, E = dec['D1']['x'], dec['D2']['E']
    check('D1.x = v2h・D2.E = 60(D3 の型を埋める値)', x == 'v2h' and E == 60, {'x': x, 'E': E})
    D3rule = proto['decisions']['D3']['rule']
    T = proto['thresholds_us']['values']
    arm_aug = '%s-aug2-e%d' % (x, E)
    arm_none = '%s-e%d' % (x, E)
    check('protocol の腕に {x}-aug2-e{E}(段 3・種 0〜3)と {x}-e{E}(段 2・種 0〜3)がある',
          proto['arms']['{x}-aug2-e{E}']['seeds'] == list(SEEDS) and proto['arms']['{x}-e60']['seeds'] == list(SEEDS))
    runs_aug = ['v2_20261025-%s-s%d' % (arm_aug, k) for k in SEEDS]
    runs_none = ['v2_20261025-%s-s%d' % (arm_none, k) for k in SEEDS]

    # 学習が止まった・削った種が無いか(judgement.missing_seed)
    ep = {}
    for r in runs_aug + runs_none:
        res = jload(RUNS / r / 'result.json')
        ep[r] = res.get('epochs_done')
    check('8 本とも 60 エポックまで学習した(missing_seed なし・k = 4)', all(v == 60 for v in ep.values()), ep)

    B = load_bundle(proto)

    # ---- 2. 欄
    rows = {}
    for r in runs_aug + runs_none:
        best = load_pred(EVAL_PRED / ('%s.npz' % r), B['key'], r, 'best')
        last = load_pred(EVAL_PRED / ('%s_last.npz' % r), B['key'], r, 'last')
        rows[r] = fields(best, last, B)
        rows[r]['_best'] = best
    # ---- 3. robust
    nshift = 7
    check('protocol sets.shift_tests は 7 つ', proto['sets']['shift_tests'].startswith('H1 に 7 つ'))
    D_stage3 = jload(STAGE3_DESK)['runs']
    D_snap = jload(SNAP_DESK)['runs']
    rob = {}
    for r in runs_aug + runs_none:
        rr = robust(r, B, rows[r]['_best'], nshift)
        rob[r] = rr
        rows[r]['robust_dl1_us'] = rr['robust_dl1_us']
        okmeta = all(rr[k] for k in ('rows_are_H1_in_order', 'n_kinds_ok', 'key_ok', 'run_ok', 'ckpt_ok',
                                     'best_pth_sha256_same_as_now')) and rr['pred_none_vs_saved_best_max_abs_us'] <= 0.01
        check('%s の shift.npz(H1 の行・7 つ・key・best.pth の sha256・そのまま = 保存した best)' % r, okmeta,
              {k: rr[k] for k in ('rows_are_H1_in_order', 'kinds', 'best_pth_sha256_same_as_now', 'pred_none_rows',
                                  'pred_none_vs_saved_best_max_abs_us')})
        a = D_stage3.get(r, {}).get('robust_dl1_us')
        b = D_snap.get(r, {}).get('robust_dl1_us')
        d = {'mine': round(rr['robust_dl1_us'], 6), 'desk_v2_stage3': a, 'snapshot': b}
        okv = (a is not None and abs(a - rr['robust_dl1_us']) <= 0.000006) and (b is not None and abs(b - rr['robust_dl1_us']) <= 0.000006)
        per_a = D_stage3.get(r, {}).get('robust_dl1_per_shift_us') or {}
        per_ok = all(abs(per_a.get(k, 1e9) - v) <= 0.00006 for k, v in rr['per_shift_us'].items())
        check('%s の robust_dl1_us が desk の値と同じ(desk は 5 桁に丸め)・7 つの内訳も' % r, okv and per_ok, d)
    kinds_set = sorted(set(tuple(v['kinds']) for v in rob.values()))
    check('8 本とも同じ 7 つのずれ', len(kinds_set) == 1, kinds_set)

    # ---- 4. 5 本
    ref = proto['reference_predictions_sha256']
    five = {}
    for rid, (pb, pl) in FIVE_FILES.items():
        okh = h(pb) == ref[rid]['best'] and h(pl) == ref[rid]['last']
        check('5 本の %s の保存した予測の sha256 = protocol' % rid, okh)
        best = load_pred(pb, B['key'])
        last = load_pred(pl, B['key'])
        five[rid] = fields(best, last, B)
    five_interv = amean(v['l1_interv_us'] for v in five.values())
    five_bias = amean(v['steer_bias_us'] for v in five.values())
    fm = T['five_mean']
    check('5 本の平均(介入 best・steer_bias best)の丸めが protocol five_mean と同じ',
          round(five_interv, 3) == fm['interv_best'] and round(five_bias, 3) == fm['steer_bias_best'],
          {'interv': five_interv, 'steer_bias': five_bias, 'protocol': [fm['interv_best'], fm['steer_bias_best']]})

    # ---- 5. D3 の決まり
    k = len(SEEDS)
    thr_p = T['pairs']['k%d' % k]
    thr_v = T['vs5']['n%d' % k]
    nw = proto['thresholds_us']['not_worse_us']
    d_best = [rows[a]['l1_best_us'] - rows[n]['l1_best_us'] for a, n in zip(runs_aug, runs_none)]
    d_last = [rows[a]['l1_last_us'] - rows[n]['l1_last_us'] for a, n in zip(runs_aug, runs_none)]
    mdb, mdl = amean(d_best), amean(d_last)
    h1_not_worse = (mdb <= nw) and (mdl <= nw)                       # judgement.pairs.not_worse(best と last の両方)
    good = mdb <= -thr_p['l1_best'] and mdl <= -thr_p['l1_last'] and all(v < 0 for v in d_best) and all(v < 0 for v in d_last)
    bad = (mdb >= thr_p['l1_best'] and all(v > 0 for v in d_best)) or (mdl >= thr_p['l1_last'] and all(v > 0 for v in d_last))
    h1_verdict = '悪い' if bad else ('良い' if good else '同等')
    ra = [rows[a]['robust_dl1_us'] for a in runs_aug]
    rn = [rows[n]['robust_dl1_us'] for n in runs_none]
    rob_all = all(a < n for a, n in zip(ra, rn))
    ratio = amean(ra) / amean(rn)
    rob_ratio = ratio <= 0.8
    dday = amean(rows[a]['l1_day_us'] - rows[n]['l1_day_us'] for a, n in zip(runs_aug, runs_none))
    dnight = amean(rows[a]['l1_night_us'] - rows[n]['l1_night_us'] for a, n in zip(runs_aug, runs_none))
    iv = [rows[a]['l1_interv_us'] for a in runs_aug]
    iv_diff = amean(iv) - five_interv
    veto_iv = iv_diff > thr_v['interv_best'] and all(v > five_interv for v in iv)
    sb = [rows[a]['steer_bias_us'] for a in runs_aug]
    sb_mean = amean(sb)
    veto_bias = abs(sb_mean - five_bias) > thr_v['steer_bias_best']
    cond = {
        'H1 が悪くない': h1_not_worse,
        'robust_dl1_us が全部の組で AUG2 < なし': rob_all,
        '平均(AUG2) ÷ 平均(なし) ≤ 0.8': rob_ratio,
        'l1_day_us の差の平均 ≤ +1.0': dday <= 1.0,
        'l1_night_us の差の平均 ≤ +1.0': dnight <= 1.0,
        'AUG2 の腕に拒否権が無い': not (veto_iv or veto_bias),
    }
    aug = 'AUG2' if all(cond.values()) else 'なし'
    mine = {
        'aug': aug, 'conditions': cond,
        'h1_pair': {'d_best': d_best, 'mean_d_best': mdb, 'd_last': d_last, 'mean_d_last': mdl, 'not_worse_us': nw,
                    'thr_best': thr_p['l1_best'], 'thr_last': thr_p['l1_last'], 'verdict': h1_verdict},
        'robust_dl1_us': {'aug2': ra, 'none': rn, 'mean_aug2': amean(ra), 'mean_none': amean(rn), 'ratio': ratio},
        'l1_day_diff_mean': dday, 'l1_night_diff_mean': dnight,
        'veto_aug2': {'interv': veto_iv, 'interv_diff': iv_diff, 'interv_thr': thr_v['interv_best'],
                      'interv_seeds': iv, 'five_mean_interv': five_interv,
                      'bias': veto_bias, 'bias_mean': sb_mean, 'bias_minus_five': sb_mean - five_bias,
                      'five_mean_bias': five_bias, 'bias_thr': thr_v['steer_bias_best']},
        'margins': {
            'H1 best(≤ +0.5 まで)': nw - mdb, 'H1 last(≤ +0.5 まで)': nw - mdl,
            'robust の組の差の最小(なし − AUG2)': min(n - a for a, n in zip(ra, rn)),
            '比(≤ 0.8 まで)': 0.8 - ratio, '昼(≤ +1.0 まで)': 1.0 - dday, '夜(≤ +1.0 まで)': 1.0 - dnight,
            '介入(閾値まで)': thr_v['interv_best'] - iv_diff, '偏り(±の外まで)': thr_v['steer_bias_best'] - abs(sb_mean - five_bias),
        },
    }
    # decisions.json の D3 と照らす
    D3 = dec['D3']
    V = D3['values']
    cmp = {
        'h1.d_best': np.max(np.abs(np.array(V['h1_pair']['d_best']) - d_best)),
        'h1.d_last': np.max(np.abs(np.array(V['h1_pair']['d_last']) - d_last)),
        'h1.mean_d_best': abs(V['h1_pair']['mean_d_best'] - mdb),
        'h1.mean_d_last': abs(V['h1_pair']['mean_d_last'] - mdl),
        'robust.aug2': np.max(np.abs(np.array(V['robust_dl1_us']['aug2']) - ra)),
        'robust.none': np.max(np.abs(np.array(V['robust_dl1_us']['none']) - rn)),
        'robust.ratio': abs(V['robust_dl1_us']['ratio'] - ratio),
        'day': abs(V['l1_day_diff_mean'] - dday),
        'night': abs(V['l1_night_diff_mean'] - dnight),
        'veto.interv_diff': abs(V['veto_aug2']['interv_diff'] - iv_diff),
        'veto.bias_value(= 平均 − 5 本の平均 として)': abs(V['veto_aug2']['bias_value'] - (sb_mean - five_bias)),
    }
    cmp = {k_: float(v) for k_, v in cmp.items()}
    same_cond = {c: (D3['conditions'].get(c) == v) for c, v in cond.items()}
    check('D3 の答え(aug)が同じ', D3['aug'] == aug, {'decisions': D3['aug'], 'mine': aug})
    check('D3 の 6 つの条件が全部同じ', all(same_cond.values()) and set(D3['conditions']) == set(cond), same_cond)
    check('D3 の数の差の最大 ≤ 0.00001 µs', max(cmp.values()) <= 1e-5, {k_: '%.2e' % v for k_, v in cmp.items()})
    check('D3 の閾値(0.5・0.765・1.087・1.63・3.913)が protocol と同じ',
          V['h1_pair']['not_worse_us'] == nw and V['h1_pair']['thr_best'] == thr_p['l1_best']
          and V['h1_pair']['thr_last'] == thr_p['l1_last'] and V['veto_aug2']['interv_thr'] == thr_v['interv_best']
          and V['veto_aug2']['bias_thr'] == thr_v['steer_bias_best'])
    check('H1 の同じ種の判定(参考)が同じ', V['h1_pair']['verdict'] == h1_verdict, {'decisions': V['h1_pair']['verdict'], 'mine': h1_verdict})
    # 参考: decisions.json の sources の sha256
    S = D3['sources']
    check('D3 の sources の sha256 = 今のファイル(desk_v2_stage3.json・写し・stage3_judge.json)',
          h(STAGE3_DESK) == S['desk_aug2_rows_sha256'] and h(SNAP_DESK) == S['desk_none_rows_sha256']
          and h(REP / 'stage3_judge.json') == S['judge_sha256'])
    # 参考: 別の係の今の desk/desk_v2.json の robust も同じか(読むだけ)
    try:
        live = jload(LIVE_DESK)['runs']
        lv = {r: live.get(r, {}).get('robust_dl1_us') for r in runs_aug + runs_none}
        check('参考: 今の desk/desk_v2.json の 8 本の robust_dl1_us も同じ(読むだけ)',
              all(v is not None and abs(v - rows[r]['robust_dl1_us']) <= 6e-6 for r, v in lv.items()), lv)
    except (OSError, ValueError) as e:
        check('参考: 今の desk/desk_v2.json が読めない', True, str(e))

    # ---- 6. 段 3・段 2 の学習のコマンドが protocol の型どおりか(workers は lock2 段 0 の 6 で 2)
    lock2 = jload(LOCK2_P)
    w = int(lock2['stage0']['6']['answer']['workers'])
    check('lock2 段 0 の 6: workers = 2(メモリの最大 3,720 MB > 3.5 GB)', w == 2)
    tcmp = {}
    for r in runs_aug + runs_none:
        m = re.match(r'^v2_20261025-(.+)-s(\d)$', r)
        seed = int(m.group(2))
        tmpl_name = ('v2_20261025-{x}-aug2-e{E}-s%d' if 'aug2' in r else 'v2_20261025-{x}-e60-s%d') % seed
        pr = proto_cmd(proto, tmpl_name)
        want = workers(fill(proto, pr['command'], x, E, 'AUG2'), w)
        got = first_start_cmd(r)
        tcmp[r] = (got is not None and ' '.join(got.split()) == want)
    check('段 3(AUG2)・段 2(v2h-e60)の 8 本の学習のコマンド = protocol の型(x = v2h・E = 60・workers 2)', all(tcmp.values()), tcmp)

    # ---- 7. 段 4
    PREFIX = 'cd ~/jetformer && '
    st4 = [r for r in proto['runs'] if r['stage'] == 4]
    check('protocol の段 4 は 2 本(種 0・1)・msfusion', [r['seed'] for r in st4] == [0, 1]
          and all(r['policy'] == 'msfusion' for r in st4) and proto['arms']['{x}-msf{A}-e{E}']['seeds'] == [0, 1])
    s4 = {}
    tsv = {}
    for ln in (EXP / 'gpu_queue_v2.tsv').read_text(encoding='utf-8').splitlines():
        parts = ln.lstrip('#').strip().split('\t')
        if len(parts) == 3:
            tsv[parts[1].strip()] = (parts[0].strip(), parts[2].strip(), ln.startswith('#'))
    for r in st4:
        name = fill(proto, r['name'], x, E, D3['aug'])
        cmd = workers(fill(proto, r['command'], x, E, D3['aug']), w)
        assert cmd.startswith(PREFIX)
        want = cmd[len(PREFIX):]
        t = tsv.get(r['name'])
        tsv_ok = t is not None and t[0] == '4' and t[1] == r['command'][len(PREFIX):] and not t[2]
        env = dict(os.environ, GPU_QUEUE_WORKERS=str(w))
        env.pop('GPU_QUEUE_SMOKE', None)
        env.pop('GPU_QUEUE_DECISIONS', None)
        out = subprocess.run([PY, '-I', str(EXP / 'gpu_queue_v2.py'), 'resolve', r['name']], cwd=str(REPO), env=env,
                             capture_output=True, text=True, timeout=120)
        f = out.stdout.rstrip('\n').split('\t')
        got_ok = len(f) == 3 and f[0] == 'RUN' and f[1] == name and ' '.join(f[2].split()) == want
        a = shlex.split(want)
        ta = a[a.index('tools/train.py') + 1:]
        av = lambda key: ta[ta.index(key) + 1] if key in ta else None    # noqa: E731
        args_ok = (av('--policy') == 'msfusion' and av('--epochs') == '60' and av('--lr') == '3e-4' and av('--stride') == '3'
                   and av('--batch') == '64' and av('--workers') == str(w) and '--no-command-input' in ta
                   and av('--impact-g') == '2.5' and av('--seed') == str(r['seed'])
                   and av('--data') == 'data/datasets/dataset_v2_20261025'
                   and av('--drop-mask') == 'data/derived/dataset_v2_20261025/drop_v2h.npz'
                   and av('--aug-brightness') == '0.15' and av('--aug-contrast') == '0.2' and av('--aug-shift-px') == '4'
                   and av('--aug-tof-noise') == '0.02' and av('--aug-top-brightness') == '0.15'
                   and av('--out') == 'data/runs/%s' % name and '--init' not in ta and '--sample-weights-npz' not in ta)
        prefix_ok = want.startswith('/home/jetson/bin/jf-heavy --mem 3872M --lock big --need-mb 3200 -- ')
        fresh = not (RUNS / name).exists()
        s4[name] = {'protocol_template': r['name'], 'expected_cmd': want, 'resolve_stdout': out.stdout.strip(),
                    'resolve_rc': out.returncode, 'tsv_row_same_as_protocol': tsv_ok, 'resolve_same_as_expected': got_ok,
                    'args_ok': args_ok, 'prefix_ok': prefix_ok, 'run_dir_not_yet': fresh}
        check('段 4 %s: 表の行 = protocol・resolve = 型を自分で埋めたもの・引数・まだ学習していない' % name,
              tsv_ok and got_ok and args_ok and prefix_ok and fresh,
              {k_: v for k_, v in s4[name].items() if k_ not in ('expected_cmd', 'resolve_stdout')})
    # 2 本の名前が腕の型 {x}-msf{A}-e{E} = v2h-msf-aug2-e60
    check('段 4 の名前 = v2_20261025-v2h-msf-aug2-e60-s0・-s1', sorted(s4) == ['v2_20261025-v2h-msf-aug2-e60-s0',
                                                                             'v2_20261025-v2h-msf-aug2-e60-s1'])
    # train.py に引数がある
    src = (REPO / 'tools' / 'train.py').read_text(encoding='utf-8')
    flags = set(re.findall(r"add_argument\('(--[\w-]+)'", src))
    need = set(a for a in shlex.split(list(s4.values())[0]['expected_cmd']) if a.startswith('--') and a not in
               ('--mem', '--lock', '--need-mb'))
    need -= {'--'}
    check('段 4 の train.py の引数が全部 tools/train.py にある', need <= flags, sorted(need - flags))

    for r in rows.values():
        r.pop('_best', None)
    out = {
        'schema': 'jf-train-v2-verify-d3/v1',
        'made_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': str(Path(__file__).resolve()),
        'script_sha256': h(Path(__file__).resolve()),
        'protocol_v2_sha256': proto_sha, 'lock2_sha256': lock2_sha,
        'decisions_sha256': h(DEC_P),
        'rule_D3': D3rule,
        'mine': mine,
        'decisions_D3_aug': D3['aug'],
        'diff_vs_decisions': cmp,
        'rows': rows,
        'robust_detail': {r: {k_: v for k_, v in d.items()} for r, d in rob.items()},
        'five': five, 'five_mean_interv_best': five_interv, 'five_mean_steer_bias_best': five_bias,
        'stage4': s4,
        'checks': CHECKS,
        'all_ok': all(c['ok'] for c in CHECKS),
        'seconds': round(time.time() - t0, 1),
    }
    tmp = str(OUT) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)
    print('答え: %s(decisions.json は %s)・確かめ %d 個のうち NG %d・%s' % (
        aug, D3['aug'], len(CHECKS), sum(not c['ok'] for c in CHECKS), OUT), flush=True)
    return 0 if out['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
