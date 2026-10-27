#!/usr/bin/env python3
"""机上の評価 v3(2026-10-26): protocol_v3.json の 7 章の欄・8 章の閾値・9 章の拒否権と門(gates_v3)・10 章の E1〜E3 を、
学習の結果に当てて表を出す。v2 の台本 experiments/20261025_train_v2/eval_v2.py の作りを引き継いだ(H1・介入の式は同じ)。

車・PCA9685・I2C・カメラ・ToF・基板に触れない。走行の API は呼ばない(GPU の仕事の前に GET の状態だけ見る)。
data/・既定値・protocol_v2・lock2・protocol_v3・decisions_v3.json は読むだけ。decisions_v3.json(キューが読む答え)は書かない
(案は eval_v3/decisions_v3_proposed.json。main session が見て decisions_v3.json に写す)。

    cd ~/jetformer
    P=/home/jetson/miniconda/envs/jetracer/bin/python
    E=experiments/20261026_fix/c3/eval_v3.py
    G="/home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 $P -u"
    # ① 予測(GPU。1 本ずつ。H1・介入の best・last → AI の記録の集合(desk_v3 predict)→ TensorRT(desk_v3 trt)。
    #    gpu_queue_v3.sh の「机上の予測」と同じ形。初回だけ B を予測し直して保存と照らす(≤ 0.01 µs))
    $G $E predict --runs v3_20261026-v3n-aug2-e60-s0
    $P -I $E predicted v3_20261026-v3n-aug2-e60-s0      # 0 = 予測がそろっている(キューの predicted と同じ)
    # ② 欄と判定(CPU・numpy だけ。torch を読まない)
    L="/home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 800 --"
    $L $P -I experiments/20261026_fix/c3/desk_v3.py report      # → fix_20261026/desk_v3.json
    $L $P -I $E judge                                           # → fix_20261026/eval_v3/(eval_v3.json・md・decisions_v3_proposed.json)
    # ③ 段 0 の 8 の確かめ(保存した 5 本・v2 の 22 本で protocol_numbers_v3.json と同じ値か・自己試験)
    $L $P -I $E check                                           # → fix_20261026/fix2/check_v3.json・md
    $P -I $E commands                                           # 学習のあとに流す GPU のコマンドの一覧

この台本が出す H1・介入の欄(v2 の欄 + v3 の欄):
  l1_best_us・l1_last_us・l1_steer/thr_best_us・l1_interv_us・l1_interv_last_us・l1_day/night_us・l1_genre4/5_us・
  steer_bias_us・thr_bias_us(と _last)・steer_amp_ratio・r1_app_thr_diff_us・nonzone_thr_diff_us・l1_zone_<区間>_us
  v3: l1_interv_steer_best_us・l1_interv_steer_last_us(拒否権)・l1_interv_thr_best_us(参考)・l1_interv_thr_last_us(参考)
desk_v3.json(desk_v3.py report)から読む欄: 直線(回ごと)・止まった場面(前・後ろ側)・sanity・推論・低速のブレーキ・
  エンジン・学習中の物差し(probe_best・probe_last)。無い欄は「未」で、その門・決まりは合格にしない。
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import desk_v3 as D  # noqa: E402

REPO = D.REPO
REP3 = D.REP3
EVAL_DIR = REP3 / 'eval_v3'
PRED = D.EVAL_PRED
EVAL_JSON = EVAL_DIR / 'eval_v3.json'
EVAL_MD = EVAL_DIR / 'eval_v3.md'
PROPOSED = EVAL_DIR / 'decisions_v3_proposed.json'
CROSS_H1 = EVAL_DIR / 'crosscheck_B_h1.json'
CHECK_DIR = REP3 / 'fix2'
DECISIONS = REP3 / 'decisions_v3.json'
H1_DS = D.DATA / 'datasets' / 'clean_driving_20261011'
ZONES = ('R1_app', 'R1_turn', 'GM', 'DE', 'none', 'unknown')
LOOKAHEAD_ROW = 2
# v2 の門・印の数(protocol_v2 gates_v2・marks。protocol_v3 gates_v3.v2_gates の文と test_desk_v3.py で照らす)
TOO_LONG_US = 1.6
RISE_US, RISE_EPOCHS = 1.0, 10
AMP = (0.80, 1.25)
L1_VS_B_US = 10.0
BRAKE_FRAC = 0.01
STANDSTILL_GO_DROP = 0.10
STANDSTILL_MIN_FRAMES = 300
INFER_FACTOR = 1.3
STRAIGHT_US = 10.0
TOL_US, TOL_FRAC = 0.01, 0.0001          # 段 0 の 8 の「同じ値」
QUEUE_FUTILITY = {'st23_thr_vs_b_us': 42.4, 'stand_brake_side_frac': 0.192}   # gpu_queue_v3.FUTILITY(参考に照らす)
FIELDS_H1 = ('l1_best_us', 'l1_steer_best_us', 'l1_thr_best_us', 'l1_last_us', 'l1_interv_us', 'l1_interv_last_us',
             'l1_interv_steer_best_us', 'l1_interv_steer_last_us', 'l1_interv_thr_best_us', 'l1_interv_thr_last_us',
             'l1_day_us', 'l1_night_us', 'l1_genre4_us', 'l1_genre5_us', 'l1_genre4_last_us', 'l1_genre5_last_us',
             'steer_bias_us', 'thr_bias_us', 'steer_bias_last_us', 'thr_bias_last_us', 'steer_amp_ratio',
             'r1_app_thr_diff_us', 'nonzone_thr_diff_us') + tuple('l1_zone_%s_us' % z for z in ZONES)
# 5 本・v2 で low10 の保存した予測が無い 20 本(check の「足りない値」。probe_check.json・G1 にあるのは B・A_s0・v2h-e30-s0・s1・
# v2h-e60-s2・v2h-aug2-e60-s1・bfast-e15-s0 の 7 本)
LOW10_MISSING = ('clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30', 'realacc_20261015-v0-s2', 'realacc_20261015-v0-s3',
                 'bfast_20261025-e15-s1', 'bfast_20261025-e15-s2', 'v2_20261025-v2h-aug2-e60-s0', 'v2_20261025-v2h-aug2-e60-s2',
                 'v2_20261025-v2h-aug2-e60-s3', 'v2_20261025-v2h-e100-s0', 'v2_20261025-v2h-e30-s2', 'v2_20261025-v2h-e30-s3',
                 'v2_20261025-v2h-e60-s0', 'v2_20261025-v2h-e60-s1', 'v2_20261025-v2h-e60-s3', 'v2_20261025-v2h-msf-aug2-e60-s0',
                 'v2_20261025-v2h-msf-aug2-e60-s1', 'v2_20261025-v2m-e30-s0', 'v2_20261025-v2m-e30-s1', 'v2_20261025-v2m-e30-s2',
                 'v2_20261025-v2m-e30-s3')
GATE_ORDER = ('engine_ok', 'sanity_ok', 'inference', 'amp', 'l1_vs_b', 'brake', 'standstill', 'human_throttle', 'straight',
              'straight_1010', 'straight_1023', 'standstill_back', 'probe_st10', 'probe_st23', 'probe_stand_brake_side',
              'probe_low10')
NEW_GATES = GATE_ORDER[9:]

log = D.log
num = D.num
mean = lambda xs: sum(xs) / len(xs)  # noqa: E731


# ================================================================ H1・介入(numpy)

def load_bundle(proto2):
    """desk_inputs_20261022.npz(eval_v2.load_bundle と同じ照らし: sha256・key・H1 4,670・介入 1,570・封・ジャンル 4)"""
    want = proto2['reference_predictions_sha256']['bundle']
    if D.sha256(D.BUNDLE) != want:
        raise SystemExit('desk_inputs_20261022.npz の sha256 が protocol_v2 と違います')
    with np.load(D.BUNDLE, allow_pickle=False) as z:
        if str(z['key']) != D.TABLE_KEY:
            raise SystemExit('desk_inputs の key が違います')
        S = {k: z[k] for k in ('part', 'scene', 'frame', 'erpm', 'primary', 'interv', 'r1_pre1s', 'zone', 'nonzone')}
        S['y'] = np.nan_to_num(np.asarray(z['y'], np.float64))
    if int(S['primary'].sum()) != proto2['holdouts']['H1']['frames_desk'] or \
            int(S['interv'].sum()) != proto2['holdouts']['interv']['frames_desk']:
        raise SystemExit('H1・介入のフレーム数が protocol と違います')
    sealed = set(proto2['seal']['logs'])
    if any(str(s)[:15] in sealed for s in np.unique(S['scene'])):
        raise SystemExit('予測の表に封の記録があります(11/1 より前は読まない)')
    S['genre'] = h1_genre(S['scene'], S['frame'], S['primary'])
    if int((S['primary'] & (S['genre'] == 4)).sum()) != proto2['holdouts']['H1']['genre_frames']['4']:
        raise SystemExit('H1 のジャンル 4 のフレーム数が protocol と違います')
    day = set(proto2['holdouts']['day_night']['day'])
    S['day'] = S['primary'] & np.isin(S['scene'], sorted(day))
    S['night'] = S['primary'] & ~S['day']
    return S


def h1_genre(scene, frame, prim):
    g = np.full(len(scene), -1, np.int16)
    for s in sorted(set(scene[prim].tolist())):
        with np.load(H1_DS / 'aligned' / ('%s.npz' % s), allow_pickle=False) as a:
            mode = a['mode']
        m = scene == s
        g[m] = mode[frame[m]]
    return g


def five_preds(proto2):
    """5 本の保存した予測(sha256 を protocol_v2 の reference_predictions_sha256 と照らす)"""
    ref = proto2['reference_predictions_sha256']
    out = {}
    for rid, run, pb, pl, seed in D.FIVE:
        for ck, p in (('best', pb), ('last', pl)):
            if D.sha256(p) != ref[rid][ck]:
                raise SystemExit('%s の保存した予測(%s)の sha256 が protocol_v2 と違います' % (rid, ck))
        out[rid] = {'run': run, 'seed': seed, 'best': D.load_h1_pred(pb), 'last': D.load_h1_pred(pl)}
    return out


def metrics(pred, S):
    """eval_v2.metrics と同じ + 介入のステア・スロットルの L1(protocol_numbers_v3.part1 と同じ式)"""
    y, prim, iv = S['y'], S['primary'], S['interv']
    d = np.abs(pred - y)
    out = {'l1': float(d[prim].mean()), 'l1_steer': float(d[prim][:, :, 0].mean()), 'l1_thr': float(d[prim][:, :, 1].mean()),
           'l1_interv': float(d[iv].mean()), 'l1_interv_steer': float(d[iv][:, :, 0].mean()),
           'l1_interv_thr': float(d[iv][:, :, 1].mean()),
           'steer_bias': float((pred[prim][:, 0, 0] - y[prim][:, 0, 0]).mean()),
           'thr_bias': float((pred[prim][:, 0, 1] - y[prim][:, 0, 1]).mean())}
    for g in (4, 5):
        out['l1_genre%d' % g] = float(d[prim & (S['genre'] == g)].mean())
    out['l1_day'] = float(d[S['day']].mean())
    out['l1_night'] = float(d[S['night']].mean())
    for z in ZONES:
        m = prim & (S['zone'] == z)
        out['l1_zone_%s' % z] = float(d[m].mean()) if m.any() else None
    hs = np.std(y[prim][:, :, 0])
    out['steer_amp_ratio'] = float(np.std(pred[prim][:, :, 0]) / hs) if hs > 0 else None
    r1 = prim & S['r1_pre1s']
    out['r1_app_thr_diff'] = float((pred[r1][:, LOOKAHEAD_ROW, 1] - y[r1][:, LOOKAHEAD_ROW, 1]).mean()) if r1.any() else None
    nz = prim & S['nonzone']
    out['nonzone_thr_diff'] = float((pred[nz][:, 0, 1] - y[nz][:, 0, 1]).mean()) if nz.any() else None
    return out


def rise_mark(history):
    """marks.rise(参考。eval_v2 と同じ)"""
    v = [num(h.get('val_us')) for h in (history or [])]
    v = [x for x in v if x is not None]
    if len(v) < 5 + RISE_EPOCHS:
        return False
    ma = np.convolve(v, np.ones(5) / 5.0, mode='valid')
    run = 0
    for i in range(1, len(ma)):
        if ma[i] >= ma[:i].min() + RISE_US:
            run += 1
            if run >= RISE_EPOCHS:
                return True
        else:
            run = 0
    return False


def h1_fields(best, last, S):
    mb = metrics(best, S)
    ml = metrics(last, S) if last is not None else None
    g = lambda k: None if ml is None else ml[k]  # noqa: E731
    r = {'l1_best_us': mb['l1'], 'l1_steer_best_us': mb['l1_steer'], 'l1_thr_best_us': mb['l1_thr'],
         'l1_last_us': g('l1'), 'l1_interv_us': mb['l1_interv'], 'l1_interv_last_us': g('l1_interv'),
         'l1_interv_steer_best_us': mb['l1_interv_steer'], 'l1_interv_steer_last_us': g('l1_interv_steer'),
         'l1_interv_thr_best_us': mb['l1_interv_thr'], 'l1_interv_thr_last_us': g('l1_interv_thr'),
         'l1_day_us': mb['l1_day'], 'l1_night_us': mb['l1_night'], 'l1_genre4_us': mb['l1_genre4'],
         'l1_genre5_us': mb['l1_genre5'], 'l1_genre4_last_us': g('l1_genre4'), 'l1_genre5_last_us': g('l1_genre5'),
         'steer_bias_us': mb['steer_bias'], 'thr_bias_us': mb['thr_bias'], 'steer_bias_last_us': g('steer_bias'),
         'thr_bias_last_us': g('thr_bias'), 'steer_amp_ratio': mb['steer_amp_ratio'],
         'r1_app_thr_diff_us': mb['r1_app_thr_diff'], 'nonzone_thr_diff_us': mb['nonzone_thr_diff']}
    for z in ZONES:
        r['l1_zone_%s_us' % z] = mb['l1_zone_%s' % z]
    r['mark_too_long'] = None if ml is None else bool(ml['l1'] - mb['l1'] > TOO_LONG_US)
    return r


def run_row(name, best, last, S, desk_row, run_dir=None):
    """1 本の学習の行: H1・介入の欄 + desk_v3 の欄 + 印"""
    fam = D.family(name)
    arm, seed = D.arm_seed(name)
    r = {'run': name, 'family': fam, 'arm': arm, 'seed': seed, 'rid': D.RID_OF.get(name)}
    r.update(h1_fields(best, last, S))
    dr = desk_row or {}
    for k in D.FIELDS_DESK:
        r[k] = dr.get(k)
    r['desk_notes'] = dr.get('notes')
    if run_dir is not None:
        res = D.load_json(Path(run_dir) / 'result.json') or {}
        r['mark_rise'] = rise_mark(res.get('history'))
        r['train_best_epoch'] = res.get('best_epoch')
        r['train_best_val_us'] = res.get('best_val_us')
        r['epochs_done'] = res.get('epochs_done')
        r['train_samples'] = res.get('train_samples')
    return r


# ================================================================ 決まり(numpy なし。test_desk_v3.py で試す)

def thresholds(proto):
    """protocol_v3.json thresholds_us から(丸めた値のまま。v2 と同じ)"""
    T = proto['thresholds_us']
    v2 = T['v2_same']
    return {'vs5': v2['vs5'], 'pairs': v2['pairs'], 'five_mean': v2['five_mean'], 'not_worse_us': v2['not_worse_us'],
            'iv_steer_mean_protocol': T['interv_steer']['five_mean_best'],
            'iv_steer_thr': T['interv_steer']['thresholds_best'],
            'g1010': T['gate_straight_1010']['value'], 'g1023': T['gate_straight_1023']['value'],
            'g_back': T['gate_standstill_back']['value'], 'p_st10': T['probe_st10']['value'],
            'p_st23': T['probe_st23']['value'], 'p_stand': T['probe_stand_brake_side']['value'],
            'p_low10': T['probe_low10']['value'], 'fix_st23_k4': T['fix_confirmed_straight_1023_k4']['value'],
            'fix_back_k4': T['fix_confirmed_standstill_back_k4']['value'],
            'futility_st23': T['futility']['st23'], 'futility_stand': T['futility']['stand_brake_side'],
            'human_throttle': v2['five_mean']['thr_bias_best'] + v2['pairs']['k2']['thr_bias_best']}


def vs5(vals_best, vals_last, five_best, five_last, thr_vs5, key_best='l1_best', key_last='l1_last'):
    """judgement.vs5(eval_v2.vs5 と同じ)→ {'verdict': 良い・悪い・同等・未, ...}"""
    sb = [v for v in vals_best.values() if v is not None]
    sl = [v for v in vals_last.values() if v is not None]
    n = len(sb)
    if n == 0 or ('n%d' % n) not in thr_vs5:
        return {'verdict': '未', 'n': n, 'why': '種が無いか、n = %d の閾値が無い' % n}
    fb, fl = mean(five_best), mean(five_last)
    tb = thr_vs5['n%d' % n][key_best]
    db = mean(sb) - fb
    good_b = db <= -tb and all(v < fb for v in sb)
    bad_b = db >= tb and all(v > fb for v in sb)
    out = {'n': n, 'mean_best': mean(sb), 'five_mean_best': fb, 'diff_best': db, 'thr_best': tb}
    if len(sl) == n:
        tl = thr_vs5['n%d' % n][key_last]
        dl = mean(sl) - fl
        good_l = dl <= -tl and all(v < fl for v in sl)
        bad_l = dl >= tl and all(v > fl for v in sl)
        out.update({'mean_last': mean(sl), 'five_mean_last': fl, 'diff_last': dl, 'thr_last': tl})
        if bad_b or bad_l:
            out['verdict'] = '悪い'
        elif good_b and good_l:
            out['verdict'] = '良い'
        else:
            out['verdict'] = '同等'
    else:
        out['verdict'] = '悪い' if bad_b else '未'
        out['why'] = 'last が全部の種にそろっていない'
    return out


def veto_interv_steer(vals, five_mean, thr_by_n):
    """vetoes_v3.interv_steer: 平均 − 5 本の平均 > 閾値(n)、かつ全部の種が 5 本の平均より大きい → 拒否"""
    v = [x for x in vals.values() if x is not None]
    n = len(v)
    key = 'vs5_n%d' % n
    if n == 0 or key not in thr_by_n:
        return {'veto': None, 'n': n, 'why': '種が無い'}
    diff = mean(v) - five_mean
    return {'veto': bool(diff > thr_by_n[key] and all(x > five_mean for x in v)), 'n': n, 'mean': mean(v),
            'five_mean': five_mean, 'diff': diff, 'thr': thr_by_n[key]}


def veto_bias(vals, five_mean, thr_vs5):
    """vetoes_v3.bias(v2 と同じ): steer_bias の腕の平均 − 5 本の平均 が ±閾値(n)の外 → 拒否"""
    v = [x for x in vals.values() if x is not None]
    n = len(v)
    if n == 0 or ('n%d' % n) not in thr_vs5:
        return {'veto': None, 'n': n, 'why': '種が無い'}
    t = thr_vs5['n%d' % n]['steer_bias_best']
    d = mean(v) - five_mean
    return {'veto': bool(abs(d) > t), 'n': n, 'diff': d, 'thr': t}


def gate(v):
    return None if v is None else bool(v)


def standstill_gate(r, b, frames):
    n = num(frames)
    a, bb = num(r.get('standstill_go_frac')), num(b.get('standstill_go_frac'))
    if n is None or a is None or bb is None:
        return None
    if n < STANDSTILL_MIN_FRAMES:
        return '参考'
    return a >= bb - STANDSTILL_GO_DROP


def probe_gates(p, T):
    """学習中の門(best.pth の probe)→ 4 つ(True・False・None)"""
    p = p if isinstance(p, dict) else {}
    st10, st23 = num(p.get('st10_thr_vs_b_us')), num(p.get('st23_thr_vs_b_us'))
    sb, lo = num(p.get('stand_brake_side_frac')), num(p.get('low10_brake_frac'))
    return {'probe_st10': None if st10 is None else abs(st10) <= T['p_st10'],
            'probe_st23': None if st23 is None else abs(st23) <= T['p_st23'],
            'probe_stand_brake_side': None if sb is None else sb <= T['p_stand'],
            'probe_low10': None if lo is None else lo <= T['p_low10']}


def gates_v3(r, b, T, standstill_frames):
    """gates_v3(候補の種ごと)。r・b は run_row の行(b = B)。値: True・False・None(未)・'参考'"""
    g = {}
    g['engine_ok'] = gate(r.get('engine_ok'))
    g['sanity_ok'] = gate(r.get('sanity_ok'))
    bp99, p99 = num(r.get('b_trt_p99_ms_same_day')), num(r.get('trt_p99_ms'))
    g['inference'] = None if (bp99 is None or p99 is None) else p99 <= INFER_FACTOR * bp99
    amp = num(r.get('steer_amp_ratio'))
    g['amp'] = None if amp is None else AMP[0] <= amp <= AMP[1]
    l1, l1b = num(r.get('l1_best_us')), num(b.get('l1_best_us'))
    g['l1_vs_b'] = None if (l1 is None or l1b is None) else abs(l1 - l1b) < L1_VS_B_US
    bf = num(r.get('brake_lowspeed_frac'))
    g['brake'] = None if bf is None else bf <= BRAKE_FRAC
    g['standstill'] = standstill_gate(r, b, standstill_frames)
    tb = num(r.get('thr_bias_us'))
    g['human_throttle'] = None if tb is None else tb <= T['human_throttle']
    st = num(r.get('straight_thr_vs_b_us'))
    g['straight'] = None if st is None else abs(st) <= STRAIGHT_US
    s10, s23 = num(r.get('straight_thr_vs_b_us_1010')), num(r.get('straight_thr_vs_b_us_1023'))
    g['straight_1010'] = None if s10 is None else abs(s10) <= T['g1010']
    g['straight_1023'] = None if s23 is None else abs(s23) <= T['g1023']
    bk = num(r.get('standstill_back_frac'))
    g['standstill_back'] = None if bk is None else bk <= T['g_back']
    g.update(probe_gates(r.get('probe_best'), T))
    return {k: g[k] for k in GATE_ORDER}


def gates_pass(g):
    """全部 True(「参考」は数えない)→ True、どれか False → False、ほかは None(未)"""
    vals = [v for v in g.values() if v != '参考']
    if any(v is False for v in vals):
        return False
    if all(v is True for v in vals):
        return True
    return None


def futility(seed_rows, T):
    """protocol futility: 種 0・1 の best の probe が両方とも st23 > 閾値 か stand_brake_side > 閾値 → 種 2・3 を始めない"""
    out = {}
    for name, (t23, tst) in (('protocol', (T['futility_st23'], T['futility_stand'])),
                             ('queue', (QUEUE_FUTILITY['st23_thr_vs_b_us'], QUEUE_FUTILITY['stand_brake_side_frac']))):
        vals = []
        for s in (0, 1):
            p = (seed_rows.get(s) or {}).get('probe_best')
            if not isinstance(p, dict) or num(p.get('st23_thr_vs_b_us')) is None or num(p.get('stand_brake_side_frac')) is None:
                vals = None
                break
            vals.append(num(p['st23_thr_vs_b_us']) > t23 or num(p['stand_brake_side_frac']) > tst)
        out[name] = None if vals is None else all(vals)
    return out


def expected_seeds(seed_rows, T, cut_seeds=()):
    """腕の要る種: 0〜3 から、途中でやめる決まり(種 2・3)と表で削った種(cut_order)を除く"""
    f = futility(seed_rows, T)
    exp = [0, 1] if f['protocol'] is True else [0, 1, 2, 3]
    return [s for s in exp if s not in set(cut_seeds)], f


def decide_e1(arms):
    """E1: 拒否権(介入のステア・偏り)が無く、H1 の L1 で 5 本に「悪い」でない腕だけ残す。arms[腕] = {'complete', 'vs5', 'veto'}"""
    out = {}
    for arm in D.ARMS_V3:
        a = arms.get(arm)
        if not a or not a.get('seeds'):
            out[arm] = {'keep': None, 'why': '種がまだ無い'}
            continue
        v = a['vs5']['verdict']
        vt = a['veto']
        vt_any = None if (vt['interv_steer']['veto'] is None or vt['bias']['veto'] is None) else \
            bool(vt['interv_steer']['veto'] or vt['bias']['veto'])
        if vt_any is True:
            why = '拒否権(%s)' % '・'.join(k for k in ('interv_steer', 'bias') if vt[k]['veto'])
            out[arm] = {'keep': False, 'why': why}
        elif v == '悪い':
            out[arm] = {'keep': False, 'why': 'H1 の L1 が 5 本に「悪い」'}
        elif not a.get('complete'):
            out[arm] = {'keep': None, 'why': '種がそろっていない(%s / %s)' % (a['seeds'], a.get('expected'))}
        elif v == '未' or vt_any is None:
            out[arm] = {'keep': None, 'why': '判定が未(5 本の判定 %s・拒否権 %s)' % (v, vt_any)}
        else:
            out[arm] = {'keep': True, 'why': '拒否権なし・5 本に「%s」' % v}
    return {'arms': out, 'kept': [a for a in D.ARMS_V3 if out[a]['keep'] is True],
            'undecided': [a for a in D.ARMS_V3 if out[a]['keep'] is None and arms.get(a, {}).get('seeds')],
            'final': all(out[a]['keep'] is not None for a in D.ARMS_V3)}      # 3 つの腕が全部そろって決まったときだけ


def decide_e2(e1, arms, gate_pass_by_arm):
    """E2: 残った腕を「門に全部合格した種の数」の多い順。同じ数なら H1 の L1(best)の腕の平均が小さい順。
    gate_pass_by_arm[腕] = {種: True・False・None}"""
    rows = []
    for arm in e1['kept']:
        gp = gate_pass_by_arm.get(arm) or {}
        rows.append({'arm': arm, 'pass': sum(1 for v in gp.values() if v is True),
                     'undecided': sum(1 for v in gp.values() if v is None),
                     'l1_best_mean': mean([r['l1_best_us'] for r in arms[arm]['rows'].values()])})
    rows.sort(key=lambda x: (-x['pass'], x['l1_best_mean']))
    final = e1['final'] and all(x['undecided'] == 0 for x in rows)
    why = '' if final else ('E1 が未の腕がある' if not e1['final'] else '門が未の種がある(仮の順)')
    return {'order': [x['arm'] for x in rows], 'rows': rows, 'final': final, 'why': why}


def decide_e3(e2, gate_pass_by_arm, name_of):
    """E3: 1 番目の腕の種を 0 → 3 の順に見て、門に全部合格した最初 = V1、次 = V2。
    1 番目の腕に 1 本しかなければ、V2 = 2 番目の腕の最初の合格。H1 で種を選ばない。name_of(腕, 種) → 学習の名前"""
    out = {'V1': None, 'V2': None, 'final': e2['final'], 'why': e2['why']}
    if not e2['order']:
        out['why'] = (out['why'] + '・' if out['why'] else '') + '残った腕が無い'
        return out
    passes = []
    for arm in e2['order'][:2]:
        gp = gate_pass_by_arm.get(arm) or {}
        passes.append([name_of(arm, s) for s in sorted(gp) if gp[s] is True])
    first = passes[0]
    if first:
        out['V1'] = first[0]
        if len(first) > 1:
            out['V2'] = first[1]
        elif len(passes) > 1 and passes[1]:
            out['V2'] = passes[1][0]
    if not first:
        out['why'] = (out['why'] + '・' if out['why'] else '') + '1 番目の腕に門の全部に合格した種が無い(どの腕も 0)'
    return out


def fix_confirmed(arm_rows, base_rows, T, sd1023, sd_back, not_worse_us=0.5):
    """fix_confirmed(結論の言い方。選びには使わない): 同じ種の v2h-aug2-e60 との d"""
    seeds = sorted(s for s in arm_rows if s in base_rows)
    k = len(seeds)
    out = {'k': k, 'seeds': seeds}
    if k < 2:
        out['why'] = '同じ種の組が %d(2 組以上で見る)' % k
        return out

    def d(key):
        a = [num(arm_rows[s].get(key)) for s in seeds]
        b = [num(base_rows[s].get(key)) for s in seeds]
        if any(x is None for x in a + b):
            return None
        return [x - y for x, y in zip(a, b)]
    t_st = T['fix_st23_k4'] if k == 4 else round(2.0 * sd1023 * math.sqrt(2.0 / k), 3)
    t_bk = T['fix_back_k4'] if k == 4 else round(2.0 * sd_back * math.sqrt(2.0 / k), 4)
    if k != 4:
        out['note'] = 'k = %d の値は protocol に無いので、同じ式 2·sd·√(2/k) で出した' % k
    for name, key, t in (('straight_1023', 'straight_thr_vs_b_us_1023', t_st), ('standstill_back', 'standstill_back_frac', t_bk)):
        dd = d(key)
        if dd is None:
            out[name] = {'ok': None, 'why': '欄が未'}
        else:
            out[name] = {'ok': bool(mean(dd) <= -t and all(x < 0 for x in dd)), 'd': dd, 'mean_d': mean(dd), 'thr': -t}
    dh = d('l1_best_us')
    dl = d('l1_last_us')
    out['h1_not_worse'] = None if (dh is None or dl is None) else bool(mean(dh) <= not_worse_us and mean(dl) <= not_worse_us)
    out['h1_d_best'], out['h1_d_last'] = dh, dl
    out['say'] = []
    if out['straight_1023'].get('ok'):
        out['say'].append('10/23 の直線の緩めが減った')
    if out['standstill_back'].get('ok'):
        out['say'].append('止まった場面の後ろ側が減った')
    return out


def loosen_at_end(best_probe, last_probe, T):
    """印(参考): best は学習中の門を通るのに last が落ちる"""
    gb, gl = probe_gates(best_probe, T), probe_gates(last_probe, T)
    if any(v is None for v in gb.values()) or not isinstance(last_probe, dict):
        return None
    return bool(all(gb.values()) and any(v is False for v in gl.values()))


def judge(rows, five_rows, T, standstill_frames, v2h_rows=None, sd1023=None, sd_back=None, cut=()):
    """rows: v3 の学習の run_row の行。five_rows: {rid: 行}。cut: 表で削った学習の名前。→ 判定の全部"""
    five = list(five_rows.values())
    b = five_rows['B']
    fbest, flast = [r['l1_best_us'] for r in five], [r['l1_last_us'] for r in five]
    five_iv = mean([r['l1_interv_steer_best_us'] for r in five])
    five_bias = mean([r['steer_bias_us'] for r in five])
    out = {'five': {'l1_best_mean': mean(fbest), 'l1_last_mean': mean(flast), 'interv_steer_best_mean': five_iv,
                    'steer_bias_mean': five_bias, 'interv_steer_best_mean_protocol': T['iv_steer_mean_protocol']},
           'arms': {}, 'gates': {}, 'marks': {}, 'fix_confirmed': {}, 'notes': []}
    if abs(five_iv - T['iv_steer_mean_protocol']) > TOL_FRAC + 1e-9:
        raise SystemExit('5 本の介入のステアの L1 の平均 %.5f が protocol(%.4f)と違います' % (five_iv, T['iv_steer_mean_protocol']))
    by_arm = {}
    for r in rows:
        if r['family'] == 'v3':
            by_arm.setdefault(r['arm'], {})[r['seed']] = r
    gate_pass = {}
    for arm in D.ARMS_V3:
        rs = by_arm.get(arm) or {}
        cut_seeds = [s for s in range(4) if 'v3_20261026-%s-aug2-e60-s%d' % (arm, s) in set(cut)]
        exp, fut = expected_seeds(rs, T, cut_seeds)
        ok_rows = {s: r for s, r in rs.items() if r.get('l1_best_us') is not None}
        extra = sorted(set(ok_rows) - set(exp))          # 削った・途中でやめたはずの種(judgement.missing_seed: 数に入れない)
        if extra:
            out['notes'].append('%s: 種 %s は要る種(%s)の外なので数に入れない(参考)' % (arm, extra, exp))
        ok_rows = {s: ok_rows[s] for s in exp if s in ok_rows}
        a = {'seeds': sorted(ok_rows), 'expected': exp, 'futility': fut, 'cut': cut_seeds, 'extra_seeds': extra, 'rows': ok_rows,
             'complete': all(s in ok_rows and ok_rows[s].get('l1_last_us') is not None for s in exp)}
        if fut['protocol'] is not None and fut['queue'] is not None and fut['protocol'] != fut['queue']:
            out['notes'].append('%s: 途中でやめる決まりが protocol(42.369・0.1919)とキュー(42.4・0.192)で分かれた' % arm)
        a['vs5'] = vs5({s: r['l1_best_us'] for s, r in ok_rows.items()}, {s: r['l1_last_us'] for s, r in ok_rows.items()},
                       fbest, flast, T['vs5'])
        a['veto'] = {'interv_steer': veto_interv_steer({s: r['l1_interv_steer_best_us'] for s, r in ok_rows.items()},
                                                       five_iv, T['iv_steer_thr']),
                     'bias': veto_bias({s: r['steer_bias_us'] for s, r in ok_rows.items()}, five_bias, T['vs5'])}
        gate_pass[arm] = {}
        for s in sorted(ok_rows):
            r = ok_rows[s]
            g = gates_v3(r, b, T, standstill_frames)
            gp = gates_pass(g)
            out['gates'][r['run']] = {'gates': g, 'pass': gp, 'fails': [k for k, v in g.items() if v is False],
                                      'undecided': [k for k, v in g.items() if v is None]}
            gate_pass[arm][s] = gp
            out['marks'][r['run']] = {'too_long': r.get('mark_too_long'), 'rise': r.get('mark_rise'),
                                      'loosen_at_end': loosen_at_end(r.get('probe_best'), r.get('probe_last'), T)}
        a['gates_pass'] = gate_pass[arm]
        out['arms'][arm] = a
        if v2h_rows and arm in ('v3n', 'v3d') and ok_rows:
            out['fix_confirmed'][arm] = fix_confirmed(ok_rows, v2h_rows, T, sd1023, sd_back, T['not_worse_us'])
    e1 = decide_e1(out['arms'])
    e2 = decide_e2(e1, out['arms'], gate_pass)
    names = {(r['arm'], r['seed']): r['run'] for r in rows if r['family'] == 'v3'}
    e3 = decide_e3(e2, gate_pass, lambda arm, s: names[(arm, s)])
    out['decisions'] = {'E1': e1, 'E2': e2, 'E3': e3}
    for arm, a in out['arms'].items():
        a['rows'] = sorted(a['rows'])             # json に行を二重に書かない(行は runs に)
    return out


# ================================================================ 表

def fmt(v, nd=2):
    if v is None:
        return '未'
    if v == '参考':
        return '参考'
    if isinstance(v, (bool, np.bool_)):
        return '○' if v else '×'
    if isinstance(v, (float, np.floating)):
        return ('%%.%df' % nd) % v
    return str(v)


def report_md(rows, five_rows, J, meta):
    L = ['# 机上の評価 v3%s' % ('(空回し: v3 の枠を全部モデル B で埋めた。判定ではない)' if meta.get('dry_run_b') else ''), '',
         '- 作った時刻: %s' % meta['made_at'],
         '- protocol_v3.json: `%s`・lock2_v3: %s' % (meta['protocol_v3_sha256'], meta['lock2_v3'] or '**まだ無い**'),
         '- desk_v3.json: %s' % (meta['desk_path'] if meta['desk'] else '**まだ無い**(その欄の門は「未」)'),
         '- 台本: `%s`(sha256 `%s`)' % (meta['script'], meta['script_sha256']),
         '- 机上の順位で実車の順位は決めない。決まりの答えは案(decisions_v3_proposed.json)。decisions_v3.json に写すのは main session。', '',
         '## 1. 学習ごとの欄(H1・介入。µs)', '',
         '| 名前 | L1 best | L1 last | 介入 | 介入のステア best | last | 介入のスロットル(参考) | ステアの偏り | スロットルの偏り | 振れ幅 | 長すぎ |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in list(five_rows.values()) + rows:
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            ('5 本: %s' % r['rid']) if r.get('rid') else r['run'], fmt(r['l1_best_us']), fmt(r['l1_last_us']),
            fmt(r['l1_interv_us']), fmt(r['l1_interv_steer_best_us']), fmt(r['l1_interv_steer_last_us']),
            fmt(r['l1_interv_thr_best_us']), fmt(r['steer_bias_us']), fmt(r['thr_bias_us']), fmt(r['steer_amp_ratio'], 3),
            fmt(r.get('mark_too_long'))))
    L += ['', '## 2. 腕(拒否権・5 本の判定・門)', '',
          '| 腕 | 種(要る種) | H1 best の差 | 閾値 | last の差 | 閾値 | 判定 | 介入のステアの差 | 閾値 | 拒否権 | 偏りの差 | 閾値 | 拒否権 | 門に全部合格 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, a in J['arms'].items():
        v, iv, bi = a['vs5'], a['veto']['interv_steer'], a['veto']['bias']
        L.append('| %s | %s(%s) | %s | %s | %s | %s | **%s** | %s | %s | %s | %s | %s | %s | %s |' % (
            arm, '・'.join(map(str, a['seeds'])) or '—', '・'.join(map(str, a['expected'])), fmt(v.get('diff_best')),
            fmt(v.get('thr_best'), 3), fmt(v.get('diff_last')), fmt(v.get('thr_last'), 3), v['verdict'], fmt(iv.get('diff')),
            fmt(iv.get('thr'), 3), fmt(iv.get('veto')), fmt(bi.get('diff')), fmt(bi.get('thr'), 3), fmt(bi.get('veto')),
            '・'.join('%d %s' % (s, fmt(p)) for s, p in sorted(a['gates_pass'].items())) or '—'))
    E = J['decisions']
    L += ['', '## 3. 決まりの答え(案)', '', '| 決まり | 答え | 理由 |', '|---|---|---|',
          '| E1 残す腕 | %s | %s |' % ('・'.join(E['E1']['kept']) or 'なし', ' / '.join(
              '%s: %s' % (k, v['why']) for k, v in E['E1']['arms'].items())),
          '| E2 腕の順 | %s | %s |' % ('→'.join(E['E2']['order']) or 'なし', '; '.join(
              '%s 合格 %d・未 %d・L1 %.2f' % (x['arm'], x['pass'], x['undecided'], x['l1_best_mean']) for x in E['E2']['rows'])
              + ('(%s)' % E['E2']['why'] if E['E2']['why'] else '')),
          '| E3 v3 の候補 | V1 %s・V2 %s | %s |' % (fmt(E['E3']['V1']), fmt(E['E3']['V2']),
                                                   ('決まった' if E['E3']['final'] else '仮(%s)' % E['E3']['why']))]
    if J['gates']:
        L += ['', '## 4. 門(gates_v3。○ 合格・× 不合格・未 = 欄がまだ)', '',
              '| 名前 | ' + ' | '.join(GATE_ORDER) + ' | 全部 |', '|---|' + '---|' * (len(GATE_ORDER) + 1)]
        for name, g in J['gates'].items():
            L.append('| %s | %s | %s |' % (name, ' | '.join(fmt(g['gates'][k]) for k in GATE_ORDER),
                                           {True: '**合格**', False: '不合格', None: '未'}[g['pass']]))
    if J['fix_confirmed']:
        L += ['', '## 5. 「原因 A の直しが効いた」と書いてよいか(同じ種の v2h-aug2-e60 と。選びには使わない)', '']
        for arm, f in J['fix_confirmed'].items():
            L.append('- %s(k = %s): 10/23 の直線 %s・後ろ側 %s・H1 悪くない %s → %s' % (
                arm, f['k'], fmt((f.get('straight_1023') or {}).get('ok')), fmt((f.get('standstill_back') or {}).get('ok')),
                fmt(f.get('h1_not_worse')), '・'.join(f.get('say') or []) or '書かない'))
    L += ['', '## 6. 印(参考)', '']
    for name, m in J['marks'].items():
        L.append('- %s: 長すぎ %s・上がり %s・終わりで緩む %s' % (name, fmt(m['too_long']), fmt(m['rise']), fmt(m['loosen_at_end'])))
    L += ['', '## 7. 注', ''] + ['- %s' % n for n in J['notes']]
    L += ['- 閾値は protocol_v3.json thresholds_us のまま(丸めた値)。5 本の平均は保存した予測から同じフレームで(丸めない)。',
          '- 封の T・V の拒否権(11/1)と E4(B-FAST2)・E5(11/3 の 1 枠)はこの台本の外。',
          '- 門「推論」はその学習の TensorRT の仕事の日の B の p99 の中央値の 1.3 倍。']
    return '\n'.join(L) + '\n'


# ================================================================ 行を作る(CPU)

def desk_rows(desk_path):
    d = D.load_json(desk_path) if desk_path and Path(desk_path).exists() else None
    return d, ((d or {}).get('runs') or {})


def v3_names_with_preds(proto):
    out = []
    for name in D.protocol_runs(proto):
        if D.family(name) == 'v3' and D.npz_ok(PRED / ('%s.npz' % name)):
            out.append(name)
    return sorted(out)


def build_eval_rows(proto, proto2, desk_runs, S, names):
    five = five_preds(proto2)
    five_rows = {}
    for rid, f in five.items():
        r = run_row(f['run'], f['best'], f['last'], S, desk_runs.get(f['run']), run_dir=D.RUNS / f['run'])
        r['seed'] = f['seed']
        five_rows[rid] = r
    rows = []
    for name in names:
        best = D.load_h1_pred(D.h1_pred_path(name, 'best'), 'best')
        lp = D.h1_pred_path(name, 'last')
        last = D.load_h1_pred(lp, 'last') if Path(lp).exists() else None
        if best is None:
            continue
        rows.append(run_row(name, best, last, S, desk_runs.get(name), run_dir=D.run_dir_of(name, proto)))
    return five_rows, rows


def sd_of_five(desk_runs, key):
    x = [num((desk_runs.get(run) or {}).get(key)) for run in D.FIVE_RUNS]
    if any(v is None for v in x):
        return None
    return float(np.std(np.array(x), ddof=1))


def cmd_judge(args):
    proto = D.protocol()
    proto2 = D.protocol2()
    T = thresholds(proto)
    desk_path = Path(args.desk) if args.desk else D.DESK_JSON
    desk, desk_runs = desk_rows(desk_path)
    S = load_bundle(proto2)
    v2h = ['v2_20261025-v2h-aug2-e60-s%d' % s for s in range(4)]
    if args.dry_run_b:                    # 空回し: v3 の 12 本の枠を全部モデル B の予測・B の desk の行で埋めて、決まりの道筋を通す
        names = [n for n in D.protocol_runs(proto) if D.family(n) == 'v3']
        five_rows, _ = build_eval_rows(proto, proto2, desk_runs, S, [])
        bb, bl = D.load_h1_pred(D.FIVE[-1][2]), D.load_h1_pred(D.FIVE[-1][3])
        rows = [run_row(n, bb, bl, S, desk_runs.get(D.B_RUN), run_dir=D.RUNS / D.B_RUN) for n in names]
    else:
        names = v3_names_with_preds(proto)
        five_rows, rows = build_eval_rows(proto, proto2, desk_runs, S, names)
    _, base = build_eval_rows(proto, proto2, desk_runs, S, v2h)
    v2h_rows = {r['seed']: r for r in base}
    J = judge(rows, five_rows, T, (desk or {}).get('standstill_frames'), v2h_rows,
              sd_of_five(desk_runs, 'straight_thr_vs_b_us_1023'), sd_of_five(desk_runs, 'standstill_back_frac'),
              cut=() if args.dry_run_b else D.queue_cut())
    if desk is None:
        J['notes'].append('desk_v3.json がまだ無い(desk_v3.py report)。その欄の門は「未」')
    elif not args.dry_run_b:
        newer = [f.name for d in (PRED, D.PRED, D.TRT) if d.exists() for f in d.glob('*')
                 if f.is_file() and f.stat().st_mtime > desk_path.stat().st_mtime]
        if newer:
            J['notes'].append('**desk_v3.json が予測より古い**(%d 個のファイルがあとから)。先に desk_v3.py report を流す' % len(newer))
    J['notes'].insert(0, ('空回し: v3 の %d 本の枠 = モデル B の保存した予測と B の desk の行(判定ではない)' % len(rows))
                      if args.dry_run_b else 'v3 の学習で予測のあるもの: %d 本' % len(rows))
    me = Path(__file__).resolve()
    meta = {'made_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'protocol_v3_sha256': D.sha256(D.PROTOCOL),
            'lock2_v3': D.sha256(D.LOCK2) if D.LOCK2.exists() else None, 'desk': desk is not None,
            'desk_path': str(desk_path), 'desk_made_at': (desk or {}).get('made_at'),
            'script': str(me), 'script_sha256': D.sha256(me), 'desk_script_sha256': D.sha256(Path(D.__file__).resolve())}
    meta['dry_run_b'] = bool(args.dry_run_b)
    out_dir = Path(args.out) if args.out else (EVAL_DIR / 'dryrun_B' if args.dry_run_b else EVAL_DIR)
    D.write_json(out_dir / 'eval_v3.json', D.clean({'meta': meta, 'five': five_rows, 'runs': rows, 'judge': J}))
    D.write_text(out_dir / 'eval_v3.md', report_md(rows, five_rows, J, meta))
    E = J['decisions']
    D.write_json(out_dir / 'decisions_v3_proposed.json', D.clean({
        'made_at': meta['made_at'], 'note': 'eval_v3.py の案。キューが読むのは decisions_v3.json(main session が写す)',
        'E1': {'kept': E['E1']['kept'], 'final': E['E1']['final'], 'why': {k: v['why'] for k, v in E['E1']['arms'].items()}},
        'E2': {'order': E['E2']['order'], 'final': E['E2']['final'], 'why': E['E2']['why']},
        'E3': {'V1': E['E3']['V1'], 'V2': E['E3']['V2'], 'final': E['E3']['final'], 'why': E['E3']['why']}}))
    log('書きました: %s(v3 の学習 %d 本)' % (out_dir / 'eval_v3.md', len(rows)))
    log('E1 %s / E2 %s / V1 %s・V2 %s(%s)' % (E['E1']['kept'], E['E2']['order'], E['E3']['V1'], E['E3']['V2'],
                                             '決まった' if E['E3']['final'] else '仮'))
    return 0


# ================================================================ 段 0 の 8 の確かめ(CPU)

class Cmp(object):
    def __init__(self):
        self.items = []

    def add(self, part, run, key, got, want, tol):
        g, w = num(got), num(want)
        if g is None or w is None:
            ok = (got is None and want is None)
            diff = None
        else:
            diff = abs(g - w)
            ok = diff <= tol + 1e-12
        self.items.append({'part': part, 'run': run, 'key': key, 'got': got, 'want': want, 'diff': diff, 'tol': tol, 'ok': bool(ok)})
        return ok

    def exact(self, part, run, key, got, want):
        self.items.append({'part': part, 'run': run, 'key': key, 'got': got, 'want': want, 'diff': None, 'tol': 0,
                           'ok': bool(got == want)})

    def summary(self):
        by = {}
        for it in self.items:
            b = by.setdefault(it['part'], {'n': 0, 'ng': 0, 'max_diff_us': 0.0, 'max_diff_frac': 0.0})
            b['n'] += 1
            b['ng'] += 0 if it['ok'] else 1
            if it['diff'] is not None:
                k = 'max_diff_frac' if it['tol'] < 0.001 else 'max_diff_us'
                b[k] = max(b[k], it['diff'])
        return by


def tol_of(key):
    return TOL_FRAC if ('frac' in key) else TOL_US


def se_table(sd):
    out = {('vs5_n%d' % n): round(2.0 * sd * math.sqrt(1.0 / n + 0.2), 3) for n in (1, 2, 3, 4)}
    out.update({('pairs_k%d' % k): round(2.0 * sd * math.sqrt(2.0 / k), 3) for k in (2, 3, 4)})
    return out


def self_test_new_gates(desk_runs, T):
    """protocol gates_v3.self_test と同じ数え方: 回ごとの直線・後ろ側・学習中の門(st10・st23・stand。low10 は入れない)。
    B-FAST は直線・st を数えない"""
    out = {}
    for run, r in desk_runs.items():
        p = r.get('probe_tab') or {}
        ok = {'straight_1010': abs(r['straight_thr_vs_b_us_1010']) <= T['g1010'],
              'straight_1023': abs(r['straight_thr_vs_b_us_1023']) <= T['g1023'],
              'standstill_back': r['standstill_back_frac'] <= T['g_back'],
              'probe_st10': abs(p['st10_thr_vs_b_us']) <= T['p_st10'], 'probe_st23': abs(p['st23_thr_vs_b_us']) <= T['p_st23'],
              'probe_stand_brake_side': p['stand_brake_side_frac'] <= T['p_stand']}
        fails = [k for k, v in ok.items() if not v]
        if run.startswith('bfast'):
            fails = [f for f in fails if 'straight' not in f and 'st1' not in f and 'st2' not in f]
        out[run] = fails or 'pass'
    return out


def cmd_check(args):
    """段 0 の 8: 保存した予測(5 本・v2 の 22 本)で protocol_numbers_v3.json と同じ値(±0.01 µs・±0.0001)を出すか。
    あわせて protocol_v3 の閾値の数え直し・自己試験・v2 の desk_v2.json / eval_v2.json の欄と同じか"""
    t0 = time.time()
    proto = D.protocol()
    proto2 = D.protocol2()
    T = thresholds(proto)
    N = D.load_json(D.NUMBERS)
    if D.sha256(D.NUMBERS) != proto['numbers_sha256']['protocol_numbers_v3.json']:
        raise SystemExit('protocol_numbers_v3.json の sha256 が protocol_v3 と違います')
    C = Cmp()
    S = load_bundle(proto2)
    rows, top, dnotes = D.build_rows(proto)
    desk_runs = {n: r for n, r in rows.items()}
    five = five_preds(proto2)
    # ---- part1: 5 本の H1・介入
    n1 = N['part1_interv_steer']
    mine1 = {}
    for rid, f in five.items():
        mb, ml = metrics(f['best'], S), metrics(f['last'], S)
        m = {'interv_steer_best': mb['l1_interv_steer'], 'interv_thr_best': mb['l1_interv_thr'], 'interv_best': mb['l1_interv'],
             'h1_best': mb['l1'], 'interv_steer_last': ml['l1_interv_steer'], 'interv_thr_last': ml['l1_interv_thr'],
             'interv_last': ml['l1_interv'], 'h1_last': ml['l1']}
        mine1[rid] = m
        for k, v in m.items():
            C.add('1 5 本の H1・介入', rid, k, v, n1['five'][rid][k], TOL_US)
    for key, st in n1['stats'].items():
        x = np.array([mine1[rid][key] for rid, *_ in D.FIVE])
        sd = float(np.std(x, ddof=1))
        C.add('1 5 本の平均・sd・閾値', key, 'mean', float(x.mean()), st['mean'], TOL_US)
        C.add('1 5 本の平均・sd・閾値', key, 'sd', sd, st['sd'], TOL_US)
        for tk, tv in se_table(sd).items():
            C.add('1 5 本の平均・sd・閾値', key, tk, tv, st['thresholds'][tk], TOL_US)
    # ---- part2: 新しい欄(5 本・v2 の 22 本)
    n2 = N['part2_desk_new_fields']
    keys2 = ('straight_thr_vs_b_us', 'straight_thr_vs_b_us_1010', 'straight_thr_vs_b_us_1023', 'standstill_back_frac',
             'standstill_go_frac')
    for rid, run, *_ in D.FIVE:
        r = desk_runs[run]
        for k in keys2:
            C.add('2 新しい欄(5 本)', rid, k, r.get(k), n2['five'][rid][k], tol_of(k))
        C.exact('2 新しい欄(5 本)', rid, 'n', r.get('n'), n2['five'][rid]['n'])
        C.exact('2 新しい欄(5 本)', rid, 'tab_sha256', r.get('tab_sha256'), n2['five'][rid]['tab_sha256'])
    for run, want in n2['v2_and_bfast'].items():
        r = desk_runs.get(run) or {}
        for k in keys2:
            C.add('2 新しい欄(v2 の 22 本)', run, k, r.get(k), want[k], tol_of(k))
        C.exact('2 新しい欄(v2 の 22 本)', run, 'n', r.get('n'), want['n'])
        C.exact('2 新しい欄(v2 の 22 本)', run, 'tab_sha256', r.get('tab_sha256'), want['tab_sha256'])
    C.exact('2 新しい欄(v2 の 22 本)', '—', '本数', len(n2['v2_and_bfast']), 22)
    for k in keys2:
        x5 = np.array([desk_runs[run][k] for run in D.FIVE_RUNS])
        x4 = np.array([desk_runs[run][k] for run in D.FIVE_RUNS if run != D.B_RUN])
        st = n2['stats'][k]
        t = tol_of(k)
        C.add('2 5 本の平均・sd', k, 'mean5', float(x5.mean()), st['mean5'], t)
        C.add('2 5 本の平均・sd', k, 'sd5', float(np.std(x5, ddof=1)), st['sd5'], t)
        C.add('2 5 本の平均・sd', k, 'mean4_nonB', float(x4.mean()), st['mean4_nonB'], t)
        C.add('2 5 本の平均・sd', k, 'n1_vs5_2se', 2.0 * float(np.std(x5, ddof=1)) * math.sqrt(1.2), st['n1_vs5_2se'], t)
    # ---- part3: 学習中の物差し(保存した予測から)・GPU の probe_check と・low10
    n3 = N['part3_probe_from_tab']
    pcheck = D.probe_check_gpu()
    for run, want in n3['runs'].items():
        p = (desk_runs.get(run) or {}).get('probe_tab') or {}
        for k in ('st10_thr_vs_b_us', 'st23_thr_vs_b_us', 'stand_brake_side_frac', 'stand_go_frac'):
            C.add('3 物差し(保存した予測)', run, k, p.get(k), want[k], tol_of(k))
        if run in pcheck:
            for k in ('st10_thr_vs_b_us', 'st23_thr_vs_b_us', 'stand_brake_side_frac', 'stand_go_frac'):
                C.add('3 物差し 対 GPU の probe_check', run, k, p.get(k), pcheck[run][k], tol_of(k))
            g1 = p.get('low10_g1')
            if g1 is not None:
                C.add('3 low10: G1 の保存した予測 対 GPU の probe_check', run, 'low10_brake_frac', g1,
                      pcheck[run]['low10_brake_frac'], TOL_FRAC)
    for k, st in n3['stats_five'].items():
        x = np.array([desk_runs[run]['probe_tab'][k] for run in D.FIVE_RUNS])
        t = tol_of(k)
        C.add('3 5 本の平均・sd', k, 'mean5', float(x.mean()), st['mean5'], t)
        C.add('3 5 本の平均・sd', k, 'sd5', float(np.std(x, ddof=1)), st['sd5'], t)
    C.exact('3 物差し(保存した予測)', '—', '行の数', top['probe_frames'], {**n3['n'], 'low10': proto['probe']['frames']['low10']})
    # ---- part4: v2 の 22 本の介入のステア・スロットル
    n4 = N['part4_v2_interv_steer']
    v2names = sorted(n4)
    mine4 = {}
    for run in v2names:
        best = D.load_h1_pred(D.EVAL2_PRED / ('%s.npz' % run), 'best')
        last = D.load_h1_pred(D.EVAL2_PRED / ('%s_last.npz' % run), 'last')
        mb, ml = metrics(best, S), metrics(last, S)
        mine4[run] = {'mb': mb, 'ml': ml, 'best': best, 'last': last}
        for k, v in (('interv_steer_best', mb['l1_interv_steer']), ('interv_thr_best', mb['l1_interv_thr']),
                     ('interv_steer_last', ml['l1_interv_steer']), ('interv_thr_last', ml['l1_interv_thr'])):
            C.add('4 v2 の 22 本の介入', run, k, v, n4[run][k], TOL_US)
    C.exact('4 v2 の 22 本の介入', '—', '本数', len(v2names), 22)
    # ---- protocol_v3 thresholds_us の数え直し(8 章)
    Tv = proto['thresholds_us']
    iv = np.array([mine1[rid]['interv_steer_best'] for rid, *_ in D.FIVE])
    ivl = np.array([mine1[rid]['interv_steer_last'] for rid, *_ in D.FIVE])
    C.add('8 閾値の数え直し', 'interv_steer', 'five_mean_best', float(iv.mean()), Tv['interv_steer']['five_mean_best'], TOL_US)
    for tk, tv in se_table(float(np.std(iv, ddof=1))).items():
        C.add('8 閾値の数え直し', 'interv_steer', tk, tv, Tv['interv_steer']['thresholds_best'][tk], TOL_US)
    for tk, tv in se_table(float(np.std(ivl, ddof=1))).items():
        C.add('8 閾値の数え直し', 'interv_steer_last', tk, tv, Tv['interv_steer']['thresholds_last'][tk], TOL_US)

    def ms(key, src=None):
        src = desk_runs if src is None else src
        x = np.array([src[run][key] for run in D.FIVE_RUNS], np.float64)
        return float(x.mean()), float(np.std(x, ddof=1))
    _, sd10 = ms('straight_thr_vs_b_us_1010')
    _, sd23 = ms('straight_thr_vs_b_us_1023')
    mbk, sdbk = ms('standstill_back_frac')
    # protocol は probe の 5 本の値を part3 の丸め(µs は 0.01・割合は 0.0001)のあとで平均・sd にした。同じ手順で数え直す
    ptab = {run: {k: round(desk_runs[run]['probe_tab'][k], 2 if k.endswith('_us') else 4)
                  for k in ('st10_thr_vs_b_us', 'st23_thr_vs_b_us', 'stand_brake_side_frac')} for run in D.FIVE_RUNS}
    praw = {run: desk_runs[run]['probe_tab'] for run in D.FIVE_RUNS}
    _, sp10 = ms('st10_thr_vs_b_us', ptab)
    _, sp23 = ms('st23_thr_vs_b_us', ptab)
    mps, sdps = ms('stand_brake_side_frac', ptab)
    rc = [('gate_straight_1010', max(10.0, 2 * sd10 * math.sqrt(1.2)), TOL_US),
          ('gate_straight_1023', max(10.0, 2 * sd23 * math.sqrt(1.2)), TOL_US),
          ('gate_standstill_back', mbk + 2 * sdbk * math.sqrt(1.2), TOL_FRAC),
          ('probe_st10', max(10.0 / 0.9, 2 * sp10 * math.sqrt(1.2)), TOL_US),
          ('probe_st23', max(10.0 / 0.9, 2 * sp23 * math.sqrt(1.2)), TOL_US),
          ('probe_stand_brake_side', mps + 2 * sdps * math.sqrt(1.2), TOL_FRAC),
          ('fix_confirmed_straight_1023_k4', 2 * sd23 * math.sqrt(0.5), TOL_US),
          ('fix_confirmed_standstill_back_k4', 2 * sdbk * math.sqrt(0.5), TOL_FRAC)]
    for k, v, t in rc:
        C.add('8 閾値の数え直し', k, 'value', v, Tv[k]['value'], t)
    C.add('8 閾値の数え直し', 'futility', 'st23', 3 * max(10.0 / 0.9, 2 * sp23 * math.sqrt(1.2)), Tv['futility']['st23'], TOL_US)
    C.add('8 閾値の数え直し', 'futility', 'stand_brake_side', 3 * (mps + 2 * sdps * math.sqrt(1.2)),
          Tv['futility']['stand_brake_side'], TOL_FRAC)
    C.add('8 閾値の数え直し', 'human_throttle', 'value', T['human_throttle'], 15.964, 0.0005)
    mr, sr = ms('stand_brake_side_frac', praw)
    probe_stand_unrounded = mr + 2 * sr * math.sqrt(1.2)      # 参考(丸めない値からだと 0.0639。門は protocol の 0.0640)
    # ---- v2 の C3・eval の欄と同じか(この台本が v2 の式を写せたか)
    d2 = D.load_json(D.REP2 / 'desk' / 'desk_v2.json') or {}
    for run, r in desk_runs.items():
        w = (d2.get('runs') or {}).get(run)
        if not w:
            continue
        for k in ('straight_thr_vs_b_us', 'standstill_go_frac', 'de_pre_thr_vs_b_us', 'r1_pre_thr_vs_b_us',
                  'sw_pre_thr_vs_b_us', 'de_overshoot_vs_b_us', 'brake_lowspeed_frac', 'trt_p99_ms', 'trt_median_ms'):
            C.add('v2 の desk_v2.json と同じ', run, k, r.get(k), w.get(k), tol_of(k) if k != 'brake_lowspeed_frac' else TOL_FRAC)
        C.exact('v2 の desk_v2.json と同じ', run, 'sanity_ok', r.get('sanity_ok'), w.get('sanity_ok'))
        C.exact('v2 の desk_v2.json と同じ', run, 'engine_ok', r.get('engine_ok'), w.get('engine_ok'))
    C.add('v2 の desk_v2.json と同じ', '—', 'b_trt_p99_ms_same_day', top['b_trt_p99_by_day'].get('20261026'),
          d2.get('b_trt_p99_ms_same_day'), TOL_US)
    e2 = D.load_json(D.REP2 / 'eval' / 'eval_v2.json') or {}
    e2runs = {r['run']: r for r in e2.get('runs') or []}
    e2runs.update({r['run']: r for r in (e2.get('five') or {}).values()})
    allrows = {}
    for rid, f in five.items():
        allrows[f['run']] = run_row(f['run'], f['best'], f['last'], S, desk_runs.get(f['run']), D.RUNS / f['run'])
    for run, m in mine4.items():
        allrows[run] = run_row(run, m['best'], m['last'], S, desk_runs.get(run), D.RUNS / run)
    for run, r in allrows.items():
        w = e2runs.get(run)
        if not w:
            continue
        for k in ('l1_best_us', 'l1_last_us', 'l1_interv_us', 'l1_interv_last_us', 'steer_bias_us', 'thr_bias_us',
                  'steer_amp_ratio', 'l1_steer_best_us', 'l1_thr_best_us', 'r1_app_thr_diff_us'):
            C.add('v2 の eval_v2.json と同じ', run, k, r.get(k), w.get(k), TOL_US if k != 'steer_amp_ratio' else TOL_FRAC)
    # ---- 自己試験
    st_new = self_test_new_gates(desk_runs, T)
    want_st = proto['gates_v3']['self_test']
    for rid, run, *_ in D.FIVE:
        C.exact('自己試験(protocol の数え方)', rid, '新しい門', st_new[run], want_st['five'][rid])
    for run, w in want_st['v2_and_bfast'].items():
        C.exact('自己試験(protocol の数え方)', run, '新しい門', st_new.get(run), w)
    n_v2 = [run for run in want_st['v2_and_bfast'] if run.startswith('v2_')]
    n_v2_fail = sum(1 for run in n_v2 if st_new.get(run) != 'pass')
    C.exact('自己試験(protocol の数え方)', '—', 'v2 の落ちた本数', '%d / %d' % (n_v2_fail, len(n_v2)), '19 / 19')
    bf_ok = [run for run in want_st['v2_and_bfast'] if run.startswith('bfast') and st_new.get(run) == 'pass']
    C.exact('自己試験(protocol の数え方)', '—', 'B-FAST の後ろ側が通る', len(bf_ok), 3)
    # 全部の門(gates_v3)を 27 本に当てる(欄が無いものは「未」)
    b_row = allrows[D.B_RUN]
    full = {}
    for run, r in allrows.items():
        g = gates_v3(r, b_row, T, top['standstill_frames'])
        full[run] = {'gates': g, 'pass': gates_pass(g), 'fails': [k for k, v in g.items() if v is False],
                     'undecided': [k for k, v in g.items() if v is None],
                     'new_pass': gates_pass({k: g[k] for k in NEW_GATES})}
    for run in n_v2:
        C.exact('自己試験(全部の門で v2 の 19 本が落ちる)', run, '全部の門で落ちる', full[run]['pass'], False)
    # ---- 足りない値
    missing = []
    for run, r in allrows.items():
        p = r.get('probe_best') or {}
        if num(p.get('low10_brake_frac')) is None:
            missing.append({'run': run, 'what': 'probe の low10_brake_frac', 'why': '10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い',
                            'how': 'probe_gates.py run --ckpt best(GPU。学習のあと)'})
        if r.get('probe_last') is None:
            missing.append({'run': run, 'what': 'probe_last(last.pth の物差し)', 'why': 'C3 の保存した予測は best だけ',
                            'how': 'probe_gates.py run --ckpt last(GPU。門には使わない)'})
        for k in ('sanity_ok', 'trt_p99_ms', 'brake_lowspeed_frac', 'engine_ok'):
            if r.get(k) is None:
                missing.append({'run': run, 'what': k, 'why': 'エンジン・TensorRT の仕事が無い', 'how': 'エンジンを作り desk_v3.py trt(GPU)'})
    out = {'schema': 'jf-desk-v3-check/v1', 'made_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'protocol_v3_sha256': D.sha256(D.PROTOCOL), 'numbers_sha256': D.sha256(D.NUMBERS),
           'eval_v3_sha256': D.sha256(Path(__file__).resolve()), 'desk_v3_sha256': D.sha256(Path(D.__file__).resolve()),
           'tolerance': {'us': TOL_US, 'frac': TOL_FRAC}, 'summary': C.summary(),
           'all_ok': all(it['ok'] for it in C.items), 'ng': [it for it in C.items if not it['ok']],
           'self_test_new_gates': st_new,
           'full_gates': {run: {'pass': f['pass'], 'new_pass': f['new_pass'], 'fails': f['fails'], 'undecided': f['undecided']}
                          for run, f in full.items()},
           'missing': missing, 'items': C.items, 'seconds': round(time.time() - t0, 1),
           'notes': ['probe の閾値(st10・st23・stand_brake_side)は protocol と同じく、5 本の値を丸めた(0.01 µs・0.0001)あとで数え直した。'
                     '丸めない値からだと stand_brake_side の門は %.5f(protocol は 0.0640。門は protocol の値を使う)' % probe_stand_unrounded]}
    od = Path(args.out) if args.out else CHECK_DIR
    D.write_json(od / 'check_v3.json', D.clean(out, 6))
    D.write_text(od / 'check_v3.md', check_md(out, allrows))
    log('書きました: %s(確かめ %d 個・外れ %d・%.0f 秒)' % (od / 'check_v3.json', len(C.items), len(out['ng']), out['seconds']))
    for part, s in out['summary'].items():
        log('  %-40s %4d 個 外れ %d 最大の差 %.6f µs・%.7f' % (part, s['n'], s['ng'], s['max_diff_us'], s['max_diff_frac']))
    return 0 if out['all_ok'] else 1


def check_md(o, allrows):
    L = ['# 段 0 の 8: 机上の台本(desk_v3・eval_v3)の確かめ', '',
         '- 作った時刻: %s(%s 秒)' % (o['made_at'], o['seconds']),
         '- protocol_v3.json `%s`・protocol_numbers_v3.json `%s`' % (o['protocol_v3_sha256'][:16], o['numbers_sha256'][:16]),
         '- desk_v3.py `%s`・eval_v3.py `%s`' % (o['desk_v3_sha256'], o['eval_v3_sha256']),
         '- 許す差: ±%.2f µs・±%.4f' % (o['tolerance']['us'], o['tolerance']['frac']),
         '- 答え: **%s**(外れ %d 個)' % ('全部同じ' if o['all_ok'] else '外れあり', len(o['ng'])), '',
         '## 1. 確かめの表', '', '| 部分 | 個数 | 外れ | 最大の差 [µs] | 最大の差 [割合] |', '|---|---|---|---|---|']
    for part, s in o['summary'].items():
        L.append('| %s | %d | %d | %.6f | %.7f |' % (part, s['n'], s['ng'], s['max_diff_us'], s['max_diff_frac']))
    if o['ng']:
        L += ['', '## 外れ', '', '| 部分 | 学習 | 欄 | この台本 | 正 | 差 |', '|---|---|---|---|---|---|']
        for it in o['ng'][:60]:
            L.append('| %s | %s | %s | %s | %s | %s |' % (it['part'], it['run'], it['key'], it['got'], it['want'], it['diff']))
    L += ['', '## 2. 門(gates_v3)を 27 本に当てた', '', '| 学習 | 新しい門 | 全部の門 | 落ちた門 | 未の門 |', '|---|---|---|---|---|']
    for run, f in o['full_gates'].items():
        L.append('| %s | %s | %s | %s | %s |' % (('5 本: %s' % D.RID_OF[run]) if run in D.RID_OF else run,
                                                {True: '○', False: '×', None: '未'}[f['new_pass']],
                                                {True: '○', False: '×', None: '未'}[f['pass']], '・'.join(f['fails']) or '—',
                                                '・'.join(f['undecided']) or '—'))
    L += ['', '## 3. 足りない値(GPU で予測し直すのは学習のあと)', '', '| 学習 | 欄 | 理由 | 出し方 |', '|---|---|---|---|']
    seen = set()
    for m in o['missing']:
        key = (m['run'], m['what'])
        if key in seen:
            continue
        seen.add(key)
        L.append('| %s | %s | %s | %s |' % (('5 本: %s' % D.RID_OF[m['run']]) if m['run'] in D.RID_OF else m['run'],
                                            m['what'], m['why'], m['how']))
    return '\n'.join(L) + '\n'


# ================================================================ 予測(GPU)

def h1_file(name, ck):
    return PRED / ('%s%s.npz' % (name, '' if ck == 'best' else '_last'))


def h1_current(name, ck, proto):
    """H1 の予測があり、今の <ck>.pth(大きさ・更新時刻)のものか"""
    p = h1_file(name, ck)
    if not D.npz_ok(p):
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            ok = (str(z['key']) == D.TABLE_KEY and str(z['ckpt']) == ck and z['pred'].shape == (10985, 10, 2)
                  and 'weights_size' in z.files)
            return bool(ok and D.stamp_same(int(z['weights_size']), int(z['weights_mtime_ns']),
                                            D.run_dir_of(name, proto) / ('%s.pth' % ck)))
    except Exception:          # noqa: BLE001
        return False


def is_predicted(name, proto):
    """→ (全部そろった?, {部分: bool})。部分 = h1_best・h1_last・tab・trt(重みやエンジンが無い部分は False)"""
    rd = D.run_dir_of(name, proto)
    parts = {'h1_best': h1_current(name, 'best', proto),
             'h1_last': h1_current(name, 'last', proto) if (rd / 'last.pth').exists() else True,
             'tab': D.tab_current(name, proto), 'trt': D.trt_current(name, proto)}
    return all(parts.values()), parts


def cmd_predicted(args):
    proto = D.protocol(check=False)
    ok, parts = is_predicted(args.name, proto)
    print(json.dumps({'name': args.name, 'predicted': ok, **parts}, ensure_ascii=False))
    return 0 if ok else 1


def crosscheck_b_h1(G, tab):
    """B を予測し直して 10/12 の保存(data/zoo の B.npz)と照らす(≤ 0.01 µs。段 data で train_data.py が変わったため)"""
    t0 = time.time()
    res, genres, _, _ = G.predict_one(D.RUNS / D.B_RUN, 'best', tab)
    pred = res['actions'].astype(np.float64)
    saved_path = D.FIVE[-1][2]
    with np.load(saved_path) as z:
        ref = z['pred'].astype(np.float64)
        same_key = str(z['key']) == tab['key']
        same_genre = bool(np.array_equal(z['genre'], genres.astype(np.int8)))
    mx = float(np.abs(pred - ref).max())
    out = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'run': D.B_RUN, 'saved': str(saved_path), 'max_abs_us': round(mx, 6),
           'limit_us': D.CROSS_LIMIT_US, 'same_key': same_key, 'same_genre': same_genre,
           'pass': bool(same_key and same_genre and mx <= D.CROSS_LIMIT_US), 'seconds': round(time.time() - t0, 1),
           'train_data_sha256': D.sha256(REPO / 'jetformer' / 'train_data.py')}
    D.write_json(CROSS_H1, out)
    log('B の H1 の予測し直し: %s' % json.dumps(out, ensure_ascii=False))
    if not out['pass']:
        raise SystemExit('B の H1 の予測し直しが 10/12 の保存と合いません(≤ %.2f µs)' % D.CROSS_LIMIT_US)


def cmd_predict_h1(args):
    """H1・介入の予測(GPU。eval_v2.cmd_predict と同じ手順: desk_gpu.table・predict_one。best・last)"""
    proto = D.protocol()
    proto2 = D.protocol2()
    names = list(dict.fromkeys(args.runs or []))
    check_names(names, proto, args.allow_before_lock2)
    import desk_gpu as G                  # 10/21 の desk の GPU の手順(predict_one・table)をそのまま使う
    import rl_policies as rp
    tab = G.table()
    if any(str(s)[:15] in set(proto2['seal']['logs']) for s in np.unique(tab['scene'])):
        raise SystemExit('予測の表に封の記録があります(11/1 より前は読まない)')
    G.check_ready()                       # GET の状態だけ・GPU の空き
    cj = D.load_json(CROSS_H1) or {}
    if not cj.get('pass') or cj.get('train_data_sha256') != D.sha256(REPO / 'jetformer' / 'train_data.py'):
        crosscheck_b_h1(G, tab)
    PRED.mkdir(parents=True, exist_ok=True)
    for name in names:
        rd = D.run_dir_of(name, proto)
        for ck in ('best', 'last'):
            if h1_current(name, ck, proto) and not args.force:
                log('%s(%s): 予測はあります' % (name, ck))
                continue
            w = rd / ('%s.pth' % ck)
            if not w.exists():
                log('%s: %s.pth がありません(飛ばす)' % (name, ck))
                continue
            stamp = D.weights_stamp(w)
            t0 = time.time()
            res, genres, sub, meta = G.predict_one(rd, ck, tab)
            pred = res.pop('actions')
            extra = {'out_' + k: v for k, v in res.items()}
            rp._save_npz(h1_file(name, ck), pred=pred, genre=genres.astype(np.int8), substituted=sub, **extra, **meta,
                         weights_size=np.array(stamp['size']), weights_mtime_ns=np.array(stamp['mtime_ns']),
                         weights_sha256=np.array(D.sha256(w)), script_sha256=np.array(D.sha256(Path(__file__).resolve())))
            ok = h1_current(name, ck, proto)
            log('%s(%s): %d フレーム %.0f 秒・読み直し %s' % (name, ck, len(pred), time.time() - t0, 'OK' if ok else 'NG'))
            if not ok:
                raise SystemExit('%s の予測を読み直せません' % h1_file(name, ck))
    return 0


def check_names(names, proto, allow_before_lock2=False):
    if not names:
        raise SystemExit('--runs を付けてください')
    bad = [n for n in names if not D.allowed_run(n, proto) or n in D.FIVE_RUNS]
    if bad:
        raise SystemExit('protocol_v3 の runs の名前ではありません(5 本は保存した予測を使う): %s' % ' '.join(bad))
    if not D.LOCK2.exists() and not allow_before_lock2:
        raise SystemExit('protocol_v3_lock2.json がまだ無いので、v3 の予測はしません(protocol_v3 lock.second_lock)')


def cmd_predict(args):
    """キューの「机上の予測」: 1 本ずつ H1(この台本)→ AI の記録の集合(desk_v3 predict)→ TensorRT(desk_v3 trt)。
    どれも別のプロセスで順に(外側の jf-heavy --lock big の中。中で jf-heavy を呼ばない)。そろった部分は飛ばす"""
    proto = D.protocol()
    names = list(dict.fromkeys(args.runs or []))
    check_names(names, proto, args.allow_before_lock2)
    py = sys.executable
    me, desk = str(Path(__file__).resolve()), str(Path(D.__file__).resolve())
    extra = ['--allow-before-lock2'] if args.allow_before_lock2 else []
    rc_all = 0
    for name in names:
        ok, parts = is_predicted(name, proto)
        log('%s: 予測の状態 %s' % (name, json.dumps(parts, ensure_ascii=False)))
        steps = []
        if not (parts['h1_best'] and parts['h1_last']):
            steps.append(('H1・介入', [py, '-u', me, 'predict-h1', '--runs', name] + extra))
        if not parts['tab']:
            steps.append(('AI の記録の集合', [py, '-u', desk, 'predict', '--runs', name] + extra))
        if not parts['trt']:
            if not (D.run_dir_of(name, proto) / D.ENGINE).exists():
                log('%s: エンジンが無いので TensorRT の仕事はしません(キューのエンジンの手順のあとに)' % name)
                rc_all = rc_all or 3
            else:
                steps.append(('TensorRT', [py, '-u', desk, 'trt', '--runs', name] + extra))
        for what, cmd in steps:
            log('%s: %s(%s)' % (name, what, ' '.join(cmd[2:])))
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=str(REPO)).returncode
            log('%s: %s rc=%d %.0f 秒' % (name, what, rc, time.time() - t0))
            if rc != 0:
                rc_all = rc
                break
    return rc_all


def gpu_commands(proto):
    G = ('/home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env '
         'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 /home/jetson/miniconda/envs/jetracer/bin/python -u')
    L = ['cd ~/jetformer']
    for name in D.protocol_runs(proto):
        if D.family(name) != 'v3':
            continue
        L.append('%s experiments/20261026_fix/c3/eval_v3.py predict --runs %s' % (G, name))
    return L


def cmd_commands(args):
    proto = D.protocol(check=False)
    print('# 学習のあとの机上の予測(GPU。1 本ずつ。キューが lock2_v3 のあとに自分で回すときは要らない)')
    print('\n'.join(gpu_commands(proto)))
    print('# 欄と判定(CPU)')
    L = '/home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 800 -- /home/jetson/miniconda/envs/jetracer/bin/python -I'
    print('%s experiments/20261026_fix/c3/desk_v3.py report' % L)
    print('%s experiments/20261026_fix/c3/eval_v3.py judge' % L)
    print('# 5 本・v2 の low10(学習中の門の 1 つ。保存した FP32 の予測が無い 20 本。参考。GPU。1 本 約 20 秒)')
    print('/home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '
          '/home/jetson/miniconda/envs/jetracer/bin/python -u experiments/20261026_fix/diag_data/probe_gates.py run --ckpt best '
          '--runs %s --out experiments/20261026_fix/fix2/probe_low10_ref.json' % ' '.join(LOW10_MISSING))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict', help='机上の予測をまとめて(GPU。jf-heavy --lock big の中で。キューの形)')
    p.add_argument('--runs', nargs='*')
    p.add_argument('--allow-before-lock2', action='store_true', help='lock2_v3 の前でも予測する(deviations_v3.md に書くこと)')
    h = sub.add_parser('predict-h1', help='H1・介入の予測だけ(GPU)')
    h.add_argument('--runs', nargs='*')
    h.add_argument('--force', action='store_true')
    h.add_argument('--allow-before-lock2', action='store_true')
    q = sub.add_parser('predicted', help='予測がそろっているか(0 = そろった。キューの gpu_queue_v3.py predicted が呼ぶ)')
    q.add_argument('name')
    j = sub.add_parser('judge', help='欄と判定(CPU・numpy)')
    j.add_argument('--desk', help='desk_v3.json の代わり(試しだけ。既定 %s)' % D.DESK_JSON)
    j.add_argument('--out', help='書く場所(既定 %s)' % EVAL_DIR)
    j.add_argument('--dry-run-b', action='store_true', help='空回し: v3 の 12 本の枠を全部 B で埋める(既定の書く場所 eval_v3/dryrun_B)')
    c = sub.add_parser('check', help='段 0 の 8: 保存した予測で protocol_numbers_v3.json と同じ値か・自己試験(CPU)')
    c.add_argument('--out', help='書く場所(既定 %s)' % CHECK_DIR)
    sub.add_parser('commands', help='学習のあとに流すコマンド')
    args = ap.parse_args()
    return {'predict': cmd_predict, 'predict-h1': cmd_predict_h1, 'predicted': cmd_predicted, 'judge': cmd_judge,
            'check': cmd_check, 'commands': cmd_commands}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
