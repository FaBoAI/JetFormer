#!/usr/bin/env python3
"""C3 v3: 机上のベンチ v3(2026-10-26)。protocol_v3.json の 7 章の欄(fields_added_v3)と、v2 の門に要る欄を出す。
v2 の台本 experiments/20261025_train_v2/c3/desk_v2.py の作りを引き継いだ(式は同じ。集合 sets_v2.npz は作り直さない)。

車・PCA9685・I2C・カメラ・ToF・基板に触れない。走行の API は呼ばない(GPU の仕事の前に GET の状態だけ見る)。
data/・既定値・protocol_v2・lock2・protocol_v3・decisions*.json は読むだけ。封の 14 本は読まない。data/raw の npz は読まない
(0 バイトの AI_CMD.npz などは通らない。aligned の npz が 0 バイトなら止まるか飛ばして数える)。
書くのは /home/jetson/jetformer_reports/fix_20261026/ の desk_v3.json・desk_v3.md と desk_v3/(pred/・trt/)だけ。

    cd ~/jetformer
    P=/home/jetson/miniconda/envs/jetracer/bin/python
    C=experiments/20261026_fix/c3/desk_v3.py
    G="/home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 $P -u"
    # ① AI の記録の集合の予測(GPU。PyTorch FP32 の best・ジャンル 5。1 本 約 2 分。初回は B を予測し直して v2 の保存と照らす)
    $G $C predict --runs v3_20261026-v3n-aug2-e60-s0
    # ② TensorRT(GPU。B はいつも入る。候補は 1 回に 2 本まで: sanity・同じ日の推論の速さ・低速のブレーキ)
    $G $C trt --runs v3_20261026-v3n-aug2-e60-s0
    # ③ 欄と表(CPU・numpy。torch を読まない)
    /home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 800 -- $P -I $C report
  ①②は、ふつうは eval_v3.py predict(キューの「机上の予測」)がまとめて呼ぶ。

欄(どれも PyTorch FP32 の best・ジャンル 5・C3 の集合 sets_v2.npz。車に出る = drive.apply_gains と同じ・ゲイン 0.90):
  v2 と同じ  de/r1/sw_pre_thr_vs_b_us・de_overshoot_vs_b_us・straight_thr_vs_b_us・straight_fwd_vs_b_us・standstill_go_frac・
             lap_delta_est_s・lap3_est_s(参考)・sanity_ok・trt_median_ms・trt_p99_ms・brake_lowspeed_frac・engine_ok
  v3 で足す  straight_thr_vs_b_us_1010・_1023   直線を回で分けた値(sess 0 = 10/10 882・sess 1 = 10/23 1,161 フレーム。行 2)
             standstill_back_frac               止まった場面(1,115)で 行 0 の車に出るスロットル > そのモデルの中立 + 30 µs の割合
             brake_lowspeed_moving_frac(参考)  低速のブレーキを 0.3〜1.0 m/s だけで数えた値(brake_lowspeed_frac と同じ母集団)
             probe_best・probe_last・probe_history  学習中の物差し。v3 の学習は result.json の probe_best・probe_last を読み、
                                                キューの logs/probe/<名前>_<best|last>.json と照らす。5 本・v2 は probe の行を
                                                保存した予測(tab)から出し直す(low10 は probe_check.json か G1 の保存した予測)
  使わない欄 robust_dl1_us(D3 は v2 の答えを受け継ぐ)・g9_minus_g5(v3 にジャンル 9 は無い)・bfast_*(B-FAST2 は学習しない)
"""

import argparse
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

HERE = Path(__file__).resolve().parent                       # experiments/20261026_fix/c3
FIX = HERE.parent                                            # experiments/20261026_fix
REPO = Path('/home/jetson/jetformer')
DESK1021 = REPO / 'experiments' / '20261021_bench_prep' / 'desk'
for _p in (str(REPO), str(REPO / 'tools'), str(DESK1021)):
    if _p not in sys.path:
        sys.path.append(_p)

REP3 = Path('/home/jetson/jetformer_reports/fix_20261026')
PROTOCOL = REP3 / 'protocol_v3.json'
PROTOCOL_SHA = REP3 / 'protocol_v3.sha256'
NUMBERS = REP3 / 'protocol_numbers_v3.json'
LOCK2 = REP3 / 'protocol_v3_lock2.json'
DESK_JSON = REP3 / 'desk_v3.json'
DESK_MD = REP3 / 'desk_v3.md'
WORK = REP3 / 'desk_v3'
PRED = WORK / 'pred'                    # v3 の学習の AI の記録の集合の予測(<名前>_tab.npz)
TRT = WORK / 'trt'                      # v3 の TensorRT の仕事(<仕事>.json・<仕事>/<名前>.npz)
EVAL_PRED = REP3 / 'eval_v3' / 'pred'   # eval_v3.py の H1 の予測(<名前>.npz・<名前>_last.npz)
REP2 = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PROTOCOL2 = REP2 / 'protocol_v2.json'
SEAL_OPENED = REP2 / 'seal_opened.json'
SETS_NPZ = REP2 / 'desk' / 'sets_v2.npz'
SETS_JSON = REP2 / 'desk' / 'sets_v2.json'
PRED2 = REP2 / 'desk' / 'pred'          # v2 の C3 の予測(5 本・v2 の 22 本・B)
TRT2 = REP2 / 'desk' / 'trt'
EVAL2_PRED = REP2 / 'eval' / 'pred'
PROBE_SET = FIX / 'diag_data' / 'out' / 'probe_set.npz'
PROBE_CHECK = FIX / 'diag_data' / 'out' / 'probe_check.json'
PROBE_EXTRA = FIX / 'fix2' / 'probe_low10_ref.json'   # 5 本・v2 の low10 を GPU で足すとき(学習のあと)
G1_DIR = FIX / 'diag_data' / 'out'      # g1_<名前>.npz(G1 の保存した予測。part 3 = 10/10 の AI 走行の低速)
QUEUE_PROBE = FIX / 'logs' / 'probe'    # gpu_queue_v3.sh の物差し(probe_gates.py run)の出力
DATA = REPO / 'data'
RUNS = DATA / 'runs'
DATASETS = DATA / 'datasets'
DS_V2 = 'dataset_v2_20261025'
DS_1010 = 'ai_mix_20261011'
BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
TABLE_KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
ENGINE = 'jetformer_fp16.engine'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
ZOO = DATA / 'zoo' / 'policy_zoo_20261012' / 'pred'
RACC = DATA / 'zoo' / 'real_accuracy_20261015' / 'pred'
# B と同じ作り方の 5 本(eval_v2.FIVE・protocol_numbers_v3.FIVE と同じ。sha256 は protocol_v2 の reference_predictions_sha256)
FIVE = [
    ('A_s0', 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30', ZOO / 'jetformer_v0_s0.npz', ZOO / 'jetformer_v0_s0_last.npz', 0),
    ('A_s1', 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30', ZOO / 'jetformer_v0_s1.npz', ZOO / 'jetformer_v0_s1_last.npz', 1),
    ('v0-s2', 'realacc_20261015-v0-s2', RACC / 'realacc_20261015-v0-s2.npz', RACC / 'realacc_20261015-v0-s2_last.npz', 2),
    ('v0-s3', 'realacc_20261015-v0-s3', RACC / 'realacc_20261015-v0-s3.npz', RACC / 'realacc_20261015-v0-s3_last.npz', 3),
    ('B', B_RUN, ZOO / 'B.npz', ZOO / 'B_last.npz', None),
]
FIVE_RUNS = [r[1] for r in FIVE]
RID_OF = {r[1]: r[0] for r in FIVE}

# ---- 数(protocol のとおり。test_desk_v3.py で protocol の文と照らす)
ROW0, ROW2 = 0, 2                    # 行 0 = 1 フレーム先、行 2 = 走る日の先読み 3
GAIN = 0.90                          # B・候補(ジャンル 5)の走るゲイン
DRIVE_LO, DRIVE_HI = 1000.0, 2000.0  # config.DRIVE の min_us・max_us(apply_gains の lo・hi)
STAND_MARGIN_US = 30.0               # standstill_go_frac・standstill_back_frac: 中立 ∓ 30 µs
SESS_1010, SESS_1023 = 0, 1          # sets_v2 の sess(desk_v2.SESS_CODE)
PROBE_NEUTRAL_US = 1498.0            # probe_gates.NEUTRAL(門の決まり desk_lib.NEUTRAL_US)
PROBE_GO_NEUTRAL_US = 1496.0         # probe_gates の stand_go_frac(参考)
PROBE_LOW10_MARGIN_US = 100.0        # low10: 行 0 > 1598
PROBE_KEYS = ('st10_thr_vs_b_us', 'st23_thr_vs_b_us', 'stand_brake_side_frac', 'low10_brake_frac')
PROBE_TOL = {'st10_thr_vs_b_us': 0.01, 'st23_thr_vs_b_us': 0.01, 'stand_brake_side_frac': 0.0001,
             'low10_brake_frac': 0.0001, 'stand_go_frac': 0.0001}
MOVING_MPS = 0.3                     # brake_lowspeed_moving_frac: 0.3〜1.0 m/s
STANDSTILL_MIN_FRAMES = 300
SANITY_GENRE = 5
SANITY_LO, SANITY_HI = 500.0, 2500.0
INFER_FACTOR = 1.3
TIME_WARMUP, TIME_RUNS = 100, 600
CROSS_LIMIT_US = 0.01                # B を予測し直したときの v2 の保存との差の上限
LAP_COEF_S_PER_US = -0.02696
LAP3_BASE_S = 25.75
BASE_PARAMS_OVERRIDE = {'steer_gain': 1.0, 'throttle_gain': 1.0, 'lookahead': 3, 'tof_guard': 1, 'ensemble_j': 0,
                        'speed_cap': 0.0, 'zone_caps': 0, 'tcp_mode': 0, 'speed_scale': 1.0}   # desk_v2・desk_gpu と同じ
RE_V3 = re.compile(r'^v3_20261026-(v3n|v3d|v3b)-aug2-e60-s([0-3])$')
RE_BF2 = re.compile(r'^(bfast2|bnull)_20261026-e15-s([0-2])$')
ARMS_V3 = ('v3n', 'v3d', 'v3b')

# desk_v3.json の各学習の行に出す欄(eval_v3.py が読む。test_desk_v3.py で照らす)
FIELDS_TAB = ('de_pre_thr_vs_b_us', 'r1_pre_thr_vs_b_us', 'sw_pre_thr_vs_b_us', 'de_overshoot_vs_b_us',
              'straight_thr_vs_b_us', 'straight_thr_vs_b_us_1010', 'straight_thr_vs_b_us_1023', 'straight_fwd_vs_b_us',
              'standstill_go_frac', 'standstill_back_frac', 'lap_delta_est_s', 'lap3_est_s')
FIELDS_TRT = ('sanity_ok', 'trt_median_ms', 'trt_p99_ms', 'b_trt_p99_ms_same_day', 'inference_ok_same_day',
              'brake_lowspeed_frac', 'brake_lowspeed_moving_frac')
FIELDS_DESK = FIELDS_TAB + FIELDS_TRT + ('engine_ok', 'probe_best', 'probe_last', 'probe_history', 'probe_source')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def npz_ok(path):
    """0 バイト・無い npz は読まない(10/24 の 0 バイトの AI_CMD.npz などと同じ壊れ方)"""
    try:
        return Path(path).is_file() and Path(path).stat().st_size > 0
    except OSError:
        return False


def num(x):
    if isinstance(x, (bool, np.bool_)) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rnd(x, nd=5):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    v = num(x)
    return None if v is None else round(v, nd)


def clean(v, nd=6):
    """json に書く形(numpy の型を外す。丸めは nd 桁)"""
    if isinstance(v, dict):
        return {str(k): clean(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x, nd) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return rnd(v, nd)
    if isinstance(v, np.ndarray):
        return clean(v.tolist(), nd)
    return v


# ================================================================ 式(numpy だけ。test_desk_v3.py で作り物の入力で試す)

def car_out(raw, ref, thr_gain, steer_gain=1.0, trim=0.0, lo=DRIVE_LO, hi=DRIVE_HI):
    """drive.apply_gains と同じ式を配列で(desk_v2.car_out の写し)。raw [..., 2](steer, throttle µs)→ 車に出る値 [..., 2]。
    ゲインは中心・中立の周り。スロットルのゲインは前進側(raw < 中立)だけ"""
    raw = np.asarray(raw, np.float64)
    s = ref[0] + trim + np.asarray(steer_gain, np.float64) * (raw[..., 0] - ref[0])
    dt = raw[..., 1] - ref[1]
    t = ref[1] + np.where(dt < 0, np.asarray(thr_gain, np.float64) * dt, dt)
    return np.stack([np.clip(s, lo, hi), np.clip(t, lo, hi)], axis=-1)


def thr_vs_b(pc, pb, refc, refb, mask, gc=GAIN, gb=GAIN, row=ROW2):
    """行 row の車に出るスロットル(候補 − B)の平均 [µs]。+ = B より緩める"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    a = car_out(np.asarray(pc)[m][:, row], refc, gc)[:, 1]
    b = car_out(np.asarray(pb)[m][:, row], refb, gb)[:, 1]
    return float(np.mean(a - b))


def steer_vs_b(pc, pb, mask, row=ROW0):
    """行 row の予測ステア(候補 − B)の平均 [µs]。+ = 左"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    return float(np.mean(np.asarray(pc, np.float64)[m][:, row, 0] - np.asarray(pb, np.float64)[m][:, row, 0]))


def go_frac(p, ref, mask, gain=GAIN, row=ROW0, margin=STAND_MARGIN_US):
    """止まった場面で、行 row の車に出るスロットルが 中立 − margin より前へ出す割合(v2 と同じ)"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    thr = car_out(np.asarray(p)[m][:, row], ref, gain)[:, 1]
    return float(np.mean(thr < ref[1] - margin))


def back_frac(p, ref, mask, gain=GAIN, row=ROW0, margin=STAND_MARGIN_US):
    """standstill_back_frac: 止まった場面で、行 row の車に出るスロットルが そのモデルの中立 + margin より後ろ側の割合"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    thr = car_out(np.asarray(p)[m][:, row], ref, gain)[:, 1]
    return float(np.mean(thr > ref[1] + margin))


def lap_est(straight_fwd_us):
    """速さの見込み(参考・門にしない。v2 と同じ)"""
    if straight_fwd_us is None:
        return None, None
    d = LAP_COEF_S_PER_US * straight_fwd_us
    return d, LAP3_BASE_S + 3.0 * d


def straight_masks(S):
    st = np.asarray(S['straight'], bool)
    sess = np.asarray(S['sess'])
    return st, st & (sess == SESS_1010), st & (sess == SESS_1023)


def row_fields(pc, pb, refc, refb, S):
    """1 本の学習の、AI の記録の集合の欄(numpy だけ)。pc・pb: 表の全部の行の予測 [N,10,2](ジャンル 5・best)"""
    st, st10, st23 = straight_masks(S)
    stand = np.asarray(S['stand'], bool)
    r = {'de_pre_thr_vs_b_us': thr_vs_b(pc, pb, refc, refb, S['pre_de']),
         'r1_pre_thr_vs_b_us': thr_vs_b(pc, pb, refc, refb, S['pre_r1']),
         'sw_pre_thr_vs_b_us': thr_vs_b(pc, pb, refc, refb, S['pre_sw']),
         'de_overshoot_vs_b_us': steer_vs_b(pc, pb, S['de_pass'], ROW0),
         'straight_thr_vs_b_us': thr_vs_b(pc, pb, refc, refb, st),
         'straight_thr_vs_b_us_1010': thr_vs_b(pc, pb, refc, refb, st10),
         'straight_thr_vs_b_us_1023': thr_vs_b(pc, pb, refc, refb, st23),
         'standstill_go_frac': go_frac(pc, refc, stand),
         'standstill_back_frac': back_frac(pc, refc, stand)}
    r['straight_fwd_vs_b_us'] = None if r['straight_thr_vs_b_us'] is None else -r['straight_thr_vs_b_us']
    r['lap_delta_est_s'], r['lap3_est_s'] = lap_est(r['straight_fwd_vs_b_us'])
    r['n'] = {'straight': int(st.sum()), 'straight_1010': int(st10.sum()), 'straight_1023': int(st23.sum()),
              'stand': int(stand.sum())}
    return r


def probe_from_tab(pc, pb, rows):
    """probe_gates.Probe と同じ式を、C3 の保存した予測の同じ行で(生の予測・ゲインの前)。rows = probe_rows() の行。
    st10・st23 は B の tab の予測との差(probe_gates は G1 の B を写した値。どちらも PyTorch FP32・ジャンル 5・best)"""
    pc, pb = np.asarray(pc, np.float64), np.asarray(pb, np.float64)
    r = {}
    for k in ('st10', 'st23'):
        r['%s_thr_vs_b_us' % k] = float(np.mean(pc[rows[k]][:, ROW2, 1] - pb[rows[k]][:, ROW2, 1]))
    s = pc[rows['stand']][:, ROW0, 1]
    r['stand_brake_side_frac'] = float(np.mean(s > PROBE_NEUTRAL_US + STAND_MARGIN_US))
    r['stand_go_frac'] = float(np.mean(s < PROBE_GO_NEUTRAL_US - STAND_MARGIN_US))
    r['low10_brake_frac'] = None              # 10/10 の低速は C3 の集合に無い(probe_check.json か G1 から)
    return r


def low10_of(pred_rows0_thr):
    """probe_gates の low10_brake_frac: 10/10 の低速 582 フレームで 行 0 > 1598 の割合"""
    u = np.asarray(pred_rows0_thr, np.float64)
    return float(np.mean(u > PROBE_NEUTRAL_US + PROBE_LOW10_MARGIN_US)) if u.size else None


def probe_diff(a, b, keys=PROBE_KEYS):
    """2 つの物差しの値の差の最大(欄ごと)。どちらかが無い欄は飛ばす"""
    out = {}
    for k in keys:
        x, y = num((a or {}).get(k)), num((b or {}).get(k))
        if x is not None and y is not None:
            out[k] = abs(x - y)
    return out


def probe_same(a, b, keys=PROBE_KEYS):
    d = probe_diff(a, b, keys)
    return bool(d) and all(v <= PROBE_TOL[k] + 1e-9 for k, v in d.items())


def pick_probe(result_json, queue_best, queue_last, tab_best):
    """学習中の門に使う物差しを選ぶ: result.json の probe_best・probe_last → キューの logs/probe → 保存した予測(tab)。
    → {'best': dict or None, 'last': dict or None, 'history': list or None, 'source': str, 'notes': [...]}"""
    res = result_json or {}
    out = {'best': None, 'last': None, 'history': res.get('probe_history'), 'source': None, 'notes': []}
    rb, rl = res.get('probe_best'), res.get('probe_last')
    if isinstance(rb, dict):
        out['best'], out['source'] = rb, 'result.json'
        if isinstance(queue_best, dict):
            d = probe_diff(rb, queue_best)
            out['best_vs_queue_max_diff'] = d
            if not probe_same(rb, queue_best):
                out['notes'].append('result.json の probe_best とキューの物差し(best.pth)が ±0.01 µs・±0.0001 を超えて違う: %s'
                                    % json.dumps({k: round(v, 5) for k, v in d.items()}))
    elif isinstance(queue_best, dict):
        out['best'], out['source'] = queue_best, 'キューの物差し(logs/probe)'
        out['notes'].append('result.json に probe_best が無いので、キューの物差しを使った')
    elif isinstance(tab_best, dict):
        out['best'], out['source'] = tab_best, '保存した予測(tab。low10 は probe_check.json か G1)'
    if isinstance(rl, dict):
        out['last'] = rl
        if isinstance(queue_last, dict) and not probe_same(rl, queue_last):
            out['notes'].append('result.json の probe_last とキューの物差し(last.pth)が違う: %s'
                                % json.dumps({k: round(v, 5) for k, v in probe_diff(rl, queue_last).items()}))
    elif isinstance(queue_last, dict):
        out['last'] = queue_last
    return out


def brake_parts(saved_best_thr0, v_hold, primary, trt_thr0, trt_v, trt_valid, vmin=None):
    """brake_lowspeed_frac(v2 と同じ: 保留の FP32 best + 10/10 の AI 走行の TensorRT。v < 1.0 m/s で 行 0 > 1598 µs)。
    vmin を付けると v ≥ vmin だけ(brake_lowspeed_moving_frac)。→ (割合, {部分: [数, 分母]})"""
    import desk_lib as DL
    low = np.asarray(primary, bool) & np.isfinite(v_hold) & (v_hold < DL.LOWSPEED_MPS)
    if vmin is not None:
        low &= v_hold >= vmin
    kh, nh = DL.brake_lowspeed(saved_best_thr0, v_hold, low)
    va = np.asarray(trt_valid, bool)
    if vmin is not None:
        va = va & (np.asarray(trt_v, np.float64) >= vmin)
    ka, na = DL.brake_lowspeed(trt_thr0, trt_v, va)
    return DL.frac(kh + ka, nh + na), {'heldout': [kh, nh], 'ai_1010': [ka, na]}


# ================================================================ protocol・名前・場所

def protocol(check=True):
    """protocol_v3.json(固定を sha256sum -c で確かめる)。v2 の集合・5 本を使うので protocol_v2 の固定も見る"""
    if check:
        for d, f in ((REP3, 'protocol_v3.sha256'), (REP2, 'protocol_v2.sha256')):
            r = subprocess.run(['sha256sum', '-c', f], cwd=str(d), capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit('protocol の固定が合いません(%s/%s):\n%s%s' % (d, f, r.stdout, r.stderr))
    return load_json(PROTOCOL)


def protocol2():
    return load_json(PROTOCOL2)


def protocol_runs(proto):
    return {r['name']: r for r in list(proto.get('runs') or []) + list(proto.get('runs_bfast2_conditional') or [])}


def out_of_command(cmd):
    """学習のコマンドの --out(gpu_queue_v3.run_dir と同じ: ~/jetformer からの相対)"""
    a = shlex.split(cmd)
    if '--out' not in a:
        return None
    return a[a.index('--out') + 1]


def run_dir_of(name, proto=None):
    """学習の出力の場所。protocol_v3 の runs の名前は、そのコマンドの --out(= キューの run_dir)。ほかは data/runs/<名前>"""
    rr = protocol_runs(proto or {}).get(name)
    if rr:
        o = out_of_command(rr['command'])
        if o:
            return REPO / o
    return RUNS / name


def epochs_of(name, proto=None):
    """学習のコマンドの --epochs(gpu_queue_v3.epochs_in と同じ読み方: tools/train.py のあと。無ければ 30)。protocol に無い名前は None"""
    rr = protocol_runs(proto or {}).get(name)
    if not rr:
        return None
    a = shlex.split(rr['command'])
    if 'tools/train.py' not in a:
        return None
    a = a[a.index('tools/train.py') + 1:]
    if '--epochs' in a and a.index('--epochs') + 1 < len(a):
        return int(a[a.index('--epochs') + 1])
    return 30


def family(name):
    if RE_V3.match(name):
        return 'v3'
    if RE_BF2.match(name):
        return 'bfast2'
    if name in FIVE_RUNS:
        return 'ref'
    if name.startswith('v2_20261025-'):
        return 'v2'
    if name.startswith('bfast_20261025-'):
        return 'bfast'
    return 'other'


def arm_seed(name):
    m = RE_V3.match(name)
    if m:
        return m.group(1), int(m.group(2))
    m = RE_BF2.match(name)
    if m:
        return m.group(1), int(m.group(2))
    m = re.match(r'^(v2|bfast)_20261025-(.+)-s(\d+)$', name)
    if m:
        return (('bfast-' if m.group(1) == 'bfast' else '') + m.group(2)), int(m.group(3))
    return None, None


QUEUE_TSV = FIX / 'gpu_queue_v3.tsv'


def queue_rows(path=QUEUE_TSV):
    """gpu_queue_v3.tsv の行(gpu_queue_v3.queue_rows(include_commented=True) と同じ読み方)→ [(段, 名前, コマンド, 削った?)]"""
    out = []
    try:
        text = Path(path).read_text(encoding='utf-8')
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip():
            continue
        commented = line.startswith('#')
        body = line.lstrip('#').strip() if commented else line
        parts = body.split('\t')
        if len(parts) != 3:
            continue
        stage, name, cmd = (x.strip() for x in parts)
        out.append((stage, name, cmd, commented))
    return out


def queue_cut(path=QUEUE_TSV):
    """表で削った(行の頭に #)学習の名前(protocol_v3 cut_order)"""
    return {n for _, n, _, c in queue_rows(path) if c}


def allowed_run(name, proto):
    return name in FIVE_RUNS or name in protocol_runs(proto)


def model_cfg(run_dir):
    cfg = load_json(Path(run_dir) / 'model_config.json')
    if not cfg:
        raise SystemExit('%s の model_config.json がありません' % run_dir)
    return cfg


def ref_of(cfg):
    return (float(cfg['steer_center_us']), float(cfg['throttle_neutral_us']))


def seal_guard(logs, sealed):
    """封の記録を読もうとしたら止まる(protocol seal.rules)"""
    bad = sorted({str(x)[:15] for x in logs} & set(sealed))
    if bad:
        raise SystemExit('封の記録は読みません(この台本は封を開けない): %s' % ' '.join(bad))


def weights_stamp(path):
    st = Path(path).stat()
    return {'size': int(st.st_size), 'mtime_ns': int(st.st_mtime_ns)}


def stamp_same(meta_size, meta_mtime, path):
    try:
        st = Path(path).stat()
    except OSError:
        return False
    try:
        return int(meta_size) == st.st_size and int(meta_mtime) == st.st_mtime_ns
    except (TypeError, ValueError):
        return False


# ================================================================ 入力(CPU)

def load_sets(proto):
    """C3 の集合(v2 の sets_v2.npz をそのまま。protocol_v3 sets.sets_v2_sha256 と照らす)"""
    if sha256(SETS_NPZ) != proto['sets']['sets_v2_sha256']:
        raise SystemExit('sets_v2.npz の sha256 が protocol_v3 と違います')
    with np.load(SETS_NPZ, allow_pickle=False) as z:
        S = {k: z[k] for k in z.files}
    S['key'] = str(S['key'])
    sj = load_json(SETS_JSON) or {}
    if sj.get('key') != S['key']:
        raise SystemExit('sets_v2.json と sets_v2.npz の key が違います')
    for k in ('pre_r1', 'pre_de', 'pre_sw', 'de_pass', 'straight', 'stand'):
        S[k] = S[k].astype(bool)
    seal_guard(np.unique(S['log']), proto['seal']['logs'])
    return S


def probe_rows(S, proto):
    """probe_set.npz の st10・st23・stand のフレーム → sets_v2 の行(protocol_numbers_v3.part3 と同じ)"""
    if sha256(PROBE_SET) != proto['probe']['set_sha256']:
        raise SystemExit('probe_set.npz の sha256 が protocol_v3 と違います')
    key = {(str(s), int(f)): i for i, (s, f) in enumerate(zip(S['scene'], S['frame']))}
    rows = {}
    with np.load(PROBE_SET, allow_pickle=False) as P:
        for k in ('st10', 'st23', 'stand'):
            idx = [key.get((str(s), int(f)), -1) for s, f in zip(P[k + '_scene'], P[k + '_frame'])]
            if min(idx) < 0:
                raise SystemExit('probe の行が sets_v2 に無い: %s' % k)
            rows[k] = np.array(idx, np.int64)
        rows['low10_B'] = P['low10_B'].astype(np.float32)
    want = proto['probe']['frames']
    if any(len(rows[k]) != want[k] for k in ('st10', 'st23', 'stand')) or len(rows['low10_B']) != want['low10']:
        raise SystemExit('probe のフレーム数が protocol_v3 と違います')
    return rows


def tab_path(name):
    """v3 の予測を先に、無ければ v2 の C3 の予測"""
    p = PRED / ('%s_tab.npz' % name)
    return p if p.exists() else PRED2 / ('%s_tab.npz' % name)


def load_tab(name, key):
    p = tab_path(name)
    if not npz_ok(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        if str(z['key']) != key:
            log('%s: tab の key が集合と違う(古い。使わない)' % name)
            return None
        if 'ckpt' in z.files and str(z['ckpt']) != 'best':
            raise SystemExit('%s は best の予測ではありません' % p)
        pred = z['pred'].astype(np.float64)
        meta = {k: str(z[k]) for k in ('best_sha256', 'model_config_sha256', 'script_sha256') if k in z.files}
        for k in ('weights_size', 'weights_mtime_ns'):
            if k in z.files:
                meta[k] = int(z[k])
    if pred.shape[1:] != (10, 2) or not np.isfinite(pred).all():
        raise SystemExit('%s: 予測の形か値がおかしい' % p)
    return {'pred': pred, 'path': str(p), 'sha256': sha256(p), **meta}


def load_bundle_min(proto2):
    """desk_inputs_20261022.npz の erpm・primary(低速のブレーキの保留の部分。eval_v2.load_bundle と同じ照らし)"""
    if sha256(BUNDLE) != proto2['reference_predictions_sha256']['bundle']:
        raise SystemExit('desk_inputs_20261022.npz の sha256 が protocol_v2 と違います')
    with np.load(BUNDLE, allow_pickle=False) as z:
        if str(z['key']) != TABLE_KEY:
            raise SystemExit('desk_inputs の key が違います')
        return {'erpm': z['erpm'].astype(np.float64), 'primary': z['primary'].astype(bool)}


def load_h1_pred(path, ck=None):
    if not npz_ok(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z['key']) != TABLE_KEY:
            raise SystemExit('%s: 予測の表の key が違います' % path)
        if ck and 'ckpt' in z.files and str(z['ckpt']) != ck:
            raise SystemExit('%s: ckpt が %s ではありません' % (path, ck))
        p = np.asarray(z['pred'], np.float64)
    if p.shape != (10985, 10, 2) or not np.isfinite(p).all():
        raise SystemExit('%s: 予測の形か値がおかしい' % path)
    return p


def h1_pred_path(name, ck='best'):
    """H1 の予測の場所: 5 本は data/zoo の保存、v3 は eval_v3/pred、v2 は v2 の eval/pred"""
    for rid, run, pb, pl, seed in FIVE:
        if run == name:
            return pb if ck == 'best' else pl
    suf = '' if ck == 'best' else '_last'
    p = EVAL_PRED / ('%s%s.npz' % (name, suf))
    return p if p.exists() else EVAL2_PRED / ('%s%s.npz' % (name, suf))


def saved_best(name):
    p = h1_pred_path(name, 'best')
    return load_h1_pred(p, None if name in FIVE_RUNS else 'best') if Path(p).exists() else None


def queue_probe(name, ck):
    """キューの物差しの出力(gpu_queue_v3.probe_result と同じ読み方: 1 つだけの値)"""
    r = load_json(QUEUE_PROBE / ('%s_%s.json' % (name, ck)))
    if isinstance(r, dict) and len(r) == 1:
        return list(r.values())[0]
    return None


def probe_check_gpu():
    """GPU の物差し(probe_gates.py run)の 5 本・v2 の値: diag_data の probe_check.json(4 本)+ 学習のあとに足す
    fix2/probe_low10_ref.json(eval_v3.py commands の最後の行。low10 が足りない 20 本)。同じ名前は probe_check.json が先"""
    d = dict(load_json(PROBE_CHECK, {}) or {})
    for k, v in (load_json(PROBE_EXTRA, {}) or {}).items():
        d.setdefault(Path(k).name if '/' in k else k, v)
    return d


def low10_from_g1(name, probe_low10_b):
    """G1 の保存した予測(PyTorch FP32・ジャンル 5・best)の part 3 を 4 個に 1 個(probe_gates の間引き)→ low10。
    B の G1 が probe_set の low10_B と同じ並びかを確かめる(ほかの G1 は a1010_sel が B と同じか)"""
    p = G1_DIR / ('g1_%s.npz' % name)
    pB = G1_DIR / ('g1_%s.npz' % B_RUN)
    if not (npz_ok(p) and npz_ok(pB)):
        return None
    with np.load(pB, allow_pickle=False) as zb:
        selB = zb['a1010_sel']
        b3 = zb['pred'][zb['part'] == 3][::4][:, :3, 1].astype(np.float32)
    if b3.shape != probe_low10_b.shape or not np.array_equal(b3, probe_low10_b):
        raise SystemExit('G1 の B の 10/10 の低速が probe_set の low10_B と同じ並びではありません')
    with np.load(p, allow_pickle=False) as z:
        if not np.array_equal(z['a1010_sel'], selB) or int(z['genre']) != SANITY_GENRE:
            raise SystemExit('%s の 10/10 の低速の並びかジャンルが B の G1 と違います' % p)
        u = z['pred'][z['part'] == 3][::4][:, ROW0, 1]
    return low10_of(u)


# ---------------------------------------------------------------- エンジン・TensorRT

def engine_fields(run_dir):
    """engine_check.json が今のエンジンのものなら engine_ok と run_numpy の速さ(eval_v2.engine_fields と同じ規則)"""
    run_dir = Path(run_dir)
    c = load_json(run_dir / 'engine_check.json')
    eng = run_dir / ENGINE
    cur = False
    if isinstance(c, dict) and eng.exists():
        st = eng.stat()
        e = c.get('engine') or {}
        try:
            cur = int(e.get('size', -1)) == st.st_size and int(e.get('mtime_ns', -1)) == st.st_mtime_ns
        except (TypeError, ValueError):
            cur = False
    if not cur:
        return {'engine_ok': False if eng.exists() else None, 'engine_note': 'engine_check.json が無いか今のエンジンのものでない'}
    mx = num(c.get('actions_max_us'))
    return {'engine_ok': bool(c.get('accuracy_ok') is True and mx is not None and mx <= 2.5), 'engine_actions_max_us': mx,
            'engine_note': 'engine_check.json(%s)' % c.get('checked_at')}


def trt_results(dirs=None):
    """trt/*.json(v3 と v2)→ {学習: [(仕事の json, エンジンの st)]}(新しい順)。試し(--limit)と v1 の形は使わない"""
    out = {}
    jobs = []
    for d in (dirs or (TRT, TRT2)):
        if Path(d).exists():
            jobs += sorted(Path(d).glob('*.json'))
    jobs.sort(key=lambda p: p.stem, reverse=True)
    for p in jobs:
        j = load_json(p) or {}
        if j.get('limit') or j.get('schema') not in ('jf-c3-trt/v2', 'jf-c3-trt/v3'):
            continue
        for name, st in (j.get('engines') or {}).items():
            out.setdefault(name, []).append((j, st))
    return out


def engine_current(run_dir, st):
    eng = Path(run_dir) / ENGINE
    if not eng.exists() or not st.get('engine'):
        return False
    s = eng.stat()
    return int(st['engine'].get('size', -1)) == s.st_size and int(st['engine'].get('mtime_ns', -1)) == s.st_mtime_ns


def b_p99_same_day(T, day):
    vals = [s.get('run_numpy_p99_ms') for j, s in T.get(B_RUN, []) if j.get('day') == day and s.get('run_numpy_p99_ms')]
    return float(np.median(vals)) if vals else None


def trt_fields(name, run_dir, T, bundle, saved):
    """TensorRT の欄(sanity_ok・trt_*・brake_lowspeed_frac・_moving_frac)と、同じ日の B の p99(その日の B の中央値)"""
    import desk_lib as DL
    out = {k: None for k in FIELDS_TRT}
    out.update({'trt_job': None, 'b_trt_p99_ms_same_job': None, 'trt_notes': []})
    lst = [(j, st) for j, st in T.get(name, []) if engine_current(run_dir, st)]
    if not lst:
        out['trt_notes'].append('今のエンジンの TensorRT の仕事がまだ')
        return out
    j, st = lst[0]
    out['trt_job'], out['trt_day'] = j['job'], j['day']
    out['sanity_ok'] = DL.sanity_of(st)
    out['sanity_detail'] = {k: st.get(k) for k in ('frames', 'check_io_ok', 'nonfinite_frames', 'out_of_range_values',
                                                     'outside_drive_range_values', 'error')}
    out['trt_median_ms'], out['trt_p99_ms'] = st.get('run_numpy_median_ms'), st.get('run_numpy_p99_ms')
    out['b_trt_p99_ms_same_job'] = ((j.get('engines') or {}).get(B_RUN) or {}).get('run_numpy_p99_ms')
    out['b_trt_p99_ms_same_day'] = b_p99_same_day(T, j['day'])
    if out['trt_p99_ms'] is not None and out['b_trt_p99_ms_same_day']:
        out['inference_ok_same_day'] = bool(out['trt_p99_ms'] <= INFER_FACTOR * out['b_trt_p99_ms_same_day'])
    f = Path(st.get('file') or '')
    if npz_ok(f) and saved is not None:
        with np.load(f) as z:
            pr, erpm, part = z['pred'].astype(np.float64), z['erpm'].astype(np.float64), z['part']
        v_hold = DL.speed_mps(bundle['erpm'])
        m0 = part == 0
        v_ai = DL.speed_mps(erpm[m0])
        thr0 = pr[m0][:, 0, 1]
        out['brake_lowspeed_frac'], out['brake_lowspeed'] = brake_parts(saved[:, 0, 1], v_hold, bundle['primary'],
                                                                          thr0, v_ai, np.isfinite(thr0))
        out['brake_lowspeed_moving_frac'], out['brake_lowspeed_moving'] = brake_parts(
            saved[:, 0, 1], v_hold, bundle['primary'], thr0, v_ai, np.isfinite(thr0), vmin=MOVING_MPS)
    else:
        out['trt_notes'].append('低速のブレーキ: 保留の予測か TensorRT のファイルが無い')
    return out


# ================================================================ 欄と表(CPU)

def build_rows(proto, names=None, with_v2=True, trt_dirs=None):
    """→ (rows {名前: 欄}, top, notes)。B の tab が無ければ止まる。names を付けるとその学習だけ(B・5 本はいつも)。
    trt_dirs: 読む TensorRT の仕事の場所(既定 v3 の trt と v2 の trt。eval_v3 check は v2 だけ)"""
    proto2 = protocol2()
    S = load_sets(proto)
    PR = probe_rows(S, proto)
    bundle = load_bundle_min(proto2)
    T = trt_results(trt_dirs)
    pcheck = probe_check_gpu()
    prun = protocol_runs(proto)
    cand = set(FIVE_RUNS)
    if names:
        cand |= set(names)
    else:
        if PRED.exists():
            cand |= {p.name[:-len('_tab.npz')] for p in PRED.glob('*_tab.npz')}
        if with_v2 and PRED2.exists():
            cand |= {p.name[:-len('_tab.npz')] for p in PRED2.glob('*_tab.npz') if not p.name.startswith('bench_')}
        cand |= {n for n in prun if (run_dir_of(n, proto) / 'result.json').exists()}
    order = {'ref': 0, 'v3': 1, 'bfast2': 2, 'v2': 3, 'bfast': 4, 'other': 5}
    cand = sorted(cand, key=lambda n: (order[family(n)], FIVE_RUNS.index(n) if n in FIVE_RUNS else 0, n))
    tabB = load_tab(B_RUN, S['key'])
    if tabB is None:
        raise SystemExit('B の tab の予測がありません(%s)' % tab_path(B_RUN))
    rdB = run_dir_of(B_RUN, proto)
    refB = ref_of(model_cfg(rdB))
    rows, notes = {}, []
    for n in cand:
        rd = run_dir_of(n, proto)
        fam = family(n)
        arm, seed = arm_seed(n)
        r = {'run': n, 'family': fam, 'rid': RID_OF.get(n), 'arm': arm, 'seed': seed, 'run_dir': str(rd)}
        for k in FIELDS_DESK:
            r[k] = None
        cfg = load_json(rd / 'model_config.json')
        if not cfg:
            r['notes'] = ['model_config.json が無い(学習がまだ)']
            pk = pick_probe(load_json(rd / 'result.json'), queue_probe(n, 'best'), queue_probe(n, 'last'), None)
            r.update({'probe_best': pk['best'], 'probe_last': pk['last'], 'probe_history': pk['history'],
                      'probe_source': pk['source']})
            rows[n] = r
            continue
        ref = ref_of(cfg)
        r.update({'steer_center_us': ref[0], 'throttle_neutral_us': ref[1], 'trained_modes': cfg.get('trained_modes'),
                  'notes': []})
        tab = load_tab(n, S['key'])
        tprobe = None
        if tab is not None:
            if 'weights_size' in tab and not stamp_same(tab['weights_size'], tab['weights_mtime_ns'], rd / 'best.pth'):
                r['notes'].append('tab の予測が今の best.pth のものでない(予測し直す)')
            else:
                r.update(row_fields(tab['pred'], tabB['pred'], ref, refB, S))
                r['tab_file'], r['tab_sha256'] = tab['path'], tab['sha256']
                tprobe = probe_from_tab(tab['pred'], tabB['pred'], PR)
        else:
            r['notes'].append('tab の予測がまだ')
        # low10(5 本・v2 は保存した予測から: GPU の probe_check.json → G1 の保存した予測。v3 は result.json)
        if tprobe is not None:
            g1 = low10_from_g1(n, PR['low10_B'])
            gpu = (pcheck.get(n) or {}).get('low10_brake_frac')
            if gpu is not None:
                tprobe['low10_brake_frac'], tprobe['low10_source'] = float(gpu), 'probe_check.json(GPU)'
            elif g1 is not None:
                tprobe['low10_brake_frac'], tprobe['low10_source'] = g1, 'G1 の保存した予測'
            r['probe_tab'] = dict(tprobe, low10_g1=g1, low10_probe_check=gpu)
        pk = pick_probe(load_json(rd / 'result.json'), queue_probe(n, 'best'), queue_probe(n, 'last'), tprobe)
        r.update({'probe_best': pk['best'], 'probe_last': pk['last'], 'probe_history': pk['history'], 'probe_source': pk['source']})
        r['notes'] += pk['notes']
        if fam in ('v3', 'bfast2') and tprobe is not None and isinstance(pk['best'], dict) and pk['source'] == 'result.json':
            d = probe_diff(pk['best'], tprobe, PROBE_KEYS[:3])
            r['probe_best_vs_tab_max_diff'] = d          # 参考(学習の中の物差しと C3 の保存した予測。行は同じ)
        saved = saved_best(n)
        tf = trt_fields(n, rd, T, bundle, saved)
        for k in FIELDS_TRT:
            r[k] = tf[k]
        r['trt'] = {k: v for k, v in tf.items() if k not in FIELDS_TRT}
        ef = engine_fields(rd) if (rd / ENGINE).exists() else {'engine_ok': None, 'engine_note': 'エンジンが無い'}
        r['engine_ok'] = ef['engine_ok']
        r['engine_note'] = ef.get('engine_note')
        rows[n] = r
    st, st10, st23 = straight_masks(S)
    top = {'standstill_frames': int(S['stand'].sum()),
           'standstill_gate': '門' if int(S['stand'].sum()) >= STANDSTILL_MIN_FRAMES else '参考',
           'straight_frames': {'all': int(st.sum()), '10/10': int(st10.sum()), '10/23': int(st23.sum())},
           'probe_frames': {k: int(len(PR[k])) for k in ('st10', 'st23', 'stand')} | {'low10': int(len(PR['low10_B']))},
           'sets_key': S['key'], 'sets_npz_sha256': proto['sets']['sets_v2_sha256']}
    days = sorted({j['day'] for j, _ in T.get(B_RUN, [])})
    top['b_trt_p99_by_day'] = {d: b_p99_same_day(T, d) for d in days}
    if SEAL_OPENED.exists():
        notes.append('seal_opened.json があります(11/1 のあと)。この台本は封の記録を読まない(封の欄は 11/1 の別の手順)')
    return rows, top, notes


LOCK2_DESK_KEY = 'desk(机上の台本)'      # fix2/write_lock2_v3.py が files に書く欄(desk_v3.py・eval_v3.py の entry)


def lock2_scripts(lock2=None):
    """lock2_v3 に記録した机上の台本(desk_v3.py・eval_v3.py)の sha256 と、今の台本(この台本と同じ場所)が同じか(fix_round1 F7)。
    → {'lock2': lock2 がある?, 'same': True・False・None(lock2 が無い), 'diff': [...], 'now': {...}, 'recorded': {...}}"""
    lock2 = Path(lock2 or LOCK2)
    here = Path(__file__).resolve().parent
    now = {n: (sha256(here / n) if (here / n).exists() else None) for n in ('desk_v3.py', 'eval_v3.py')}
    if not lock2.exists():
        return {'lock2': False, 'same': None, 'diff': [], 'now': now, 'recorded': None}
    j = load_json(lock2)
    rec = ((j or {}).get('files') or {}).get(LOCK2_DESK_KEY) if isinstance(j, dict) else None
    recorded = {n: ((rec or {}).get(n) or {}).get('sha256') for n in now}
    diff = []
    for n in now:
        if not recorded[n]:
            diff.append('%s: lock2_v3 に sha256 が無い' % n)
        elif recorded[n] != now[n]:
            diff.append('%s: lock2_v3 %s… ≠ 今 %s…' % (n, recorded[n][:12], (now[n] or 'ファイルが無い')[:12]))
    return {'lock2': True, 'same': not diff, 'diff': diff, 'now': now, 'recorded': recorded}


def lock2_stop(what, allow_deviation=False):
    """予測(GPU)の前: lock2_v3 があって机上の台本の sha256 が違えば止まる(--allow-deviation のときだけ進め、注を出す)"""
    lk = lock2_scripts()
    if lk['lock2'] and not lk['same']:
        msg = 'lock2_v3 に記録した机上の台本と今の台本が違います(%s)' % ' / '.join(lk['diff'])
        if not allow_deviation:
            raise SystemExit('%s。%s はしません(protocol_v3 lock.second_lock。直したなら deviations_v3.md に書き、'
                             '--allow-deviation で流す)' % (msg, what))
        log('注意: %s。--allow-deviation で %s を進めます(deviations_v3.md に書く)' % (msg, what))
    return lk


def lock2_note(lk):
    """表・判定の先頭に書く注(lock2 があって台本が違うときだけ)。無ければ None"""
    if lk.get('lock2') and not lk.get('same'):
        return '**lock2_v3 に記録した机上の台本の sha256 と今の台本が違う**(%s)。protocol_v3 lock.second_lock の外' % ' / '.join(lk['diff'])
    return None


def lock_info():
    me = Path(__file__).resolve()
    lk = lock2_scripts()
    return {'protocol_v3_sha256': sha256(PROTOCOL), 'lock2_v3_sha256': sha256(LOCK2) if LOCK2.exists() else None,
            'script': str(me), 'script_sha256': sha256(me), 'lock2_scripts_same': lk['same'], 'lock2_scripts_diff': lk['diff']}


def definitions():
    return {
        'car_output': 'drive.apply_gains と同じ(ステア 1.0・trim 0・スロットルのゲインは前進側だけ・1000〜2000)。ゲイン 0.90(B・候補)',
        'pred': 'PyTorch FP32 の best・ジャンル 5・C3 の集合(sets_v2.npz。v2 と同じ 16,621 行)',
        'straight_thr_vs_b_us_1010・_1023': '直線(学習に使っていない AI のきれいな周)を回で分けた 行 2 の車に出るスロットル(候補 − B)の平均。+ = 緩める',
        'standstill_back_frac': '止まった場面(1,115)で 行 0 の車に出るスロットル > そのモデルの中立 + 30 µs の割合',
        'standstill_go_frac': '同じ場面で 行 0 の車に出るスロットル < そのモデルの中立 − 30 µs の割合(v2 と同じ)',
        'brake_lowspeed_frac': 'v2 と同じ: v < 1.0 m/s で行 0 > 1598 µs の割合(保留の FP32 best + 10/10 の AI 走行の TensorRT)',
        'brake_lowspeed_moving_frac': '参考: 同じ式で 0.3 ≤ v < 1.0 m/s だけ',
        'probe': '学習中の物差し(probe_gates.py と同じ式・生の予測)。v3 は result.json の probe_best・probe_last。'
                 '5 本・v2 は C3 の保存した予測の同じ行(low10 は probe_check.json か G1 の保存した予測)',
        'trt': 'v2 と同じ測り方(同じ実フレーム・100 回の空回し + 600 回の run_numpy・B → 候補 → B)。門「推論」は同じ日の B の p99 の中央値の 1.3 倍',
    }


def fmt(v, nd=2):
    if v is None:
        return '未'
    if isinstance(v, (bool, np.bool_)):
        return '○' if v else '×'
    if isinstance(v, (float, np.floating)):
        return ('%%.%df' % nd) % v
    return str(v)


def fmt_probe(p):
    if not isinstance(p, dict):
        return '未'
    return 'st10 %s・st23 %s・止 %s・低 %s' % (fmt(num(p.get('st10_thr_vs_b_us'))), fmt(num(p.get('st23_thr_vs_b_us'))),
                                             fmt(num(p.get('stand_brake_side_frac')), 4), fmt(num(p.get('low10_brake_frac')), 4))


def report_md(o):
    L = ['# 机上のベンチ v3', '',
         '- 作った時刻: %s' % o['made_at'],
         '- protocol_v3.json: `%s`・lock2_v3: `%s`' % (o['protocol_v3_sha256'], o['lock2_v3_sha256']),
         '- 台本: `%s`(sha256 `%s`)' % (o['script'], o['script_sha256']),
         '- 集合: sets_v2.npz(key `%s`)。直線 %d(10/10 %d・10/23 %d)・止まった場面 %d(%s)' % (
             o['sets_key'], o['straight_frames']['all'], o['straight_frames']['10/10'], o['straight_frames']['10/23'],
             o['standstill_frames'], o['standstill_gate']),
         '- 門・決まりは eval_v3.py が protocol_v3 のとおりに当てる。机上の順位で実車の順位は決めない。', '',
         '## 学習ごとの欄(µs・割合・ms)', '',
         '| 名前 | 直線 | 10/10 | 10/23 | 止まって出る | 後ろ側 | DE 手前 | R1 手前 | sanity | p99 | 同じ日の B | ブレーキ | 動いて | エンジン | 物差し best | 出どころ |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for n, r in o['runs'].items():
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            ('5 本: %s' % r['rid']) if r.get('rid') else n, fmt(r.get('straight_thr_vs_b_us')),
            fmt(r.get('straight_thr_vs_b_us_1010')), fmt(r.get('straight_thr_vs_b_us_1023')),
            fmt(r.get('standstill_go_frac'), 4), fmt(r.get('standstill_back_frac'), 4), fmt(r.get('de_pre_thr_vs_b_us')),
            fmt(r.get('r1_pre_thr_vs_b_us')), fmt(r.get('sanity_ok')), fmt(r.get('trt_p99_ms'), 3),
            fmt(r.get('b_trt_p99_ms_same_day'), 3), fmt(r.get('brake_lowspeed_frac'), 4),
            fmt(r.get('brake_lowspeed_moving_frac'), 4), fmt(r.get('engine_ok')), fmt_probe(r.get('probe_best')),
            r.get('probe_source') or '未'))
    L += ['', '## 注', ''] + ['- %s' % x for x in o.get('notes') or []]
    for n, r in o['runs'].items():
        for x in r.get('notes') or []:
            L.append('- %s: %s' % (n, x))
    L += ['- 「_thr_」は PWM の差(+ = 緩める)。直線の 10/10・10/23 は sets_v2 の sess で分けた。',
          '- 物差しの出どころ: v3 は result.json(キューの logs/probe と照らす)。5 本・v2 は保存した予測から出し直した値。']
    return '\n'.join(L) + '\n'


def cmd_report(args):
    t0 = time.time()
    proto = protocol()
    rows, top, notes = build_rows(proto, names=args.runs or None)
    li = lock_info()
    ln = lock2_note({'lock2': li['lock2_v3_sha256'] is not None, 'same': li['lock2_scripts_same'], 'diff': li['lock2_scripts_diff']})
    if ln:
        notes.insert(0, ln + '。直したなら deviations_v3.md に書く')
    out = {'schema': 'jf-c3-desk-v3/v1', 'made_at': time.strftime('%Y-%m-%d %H:%M:%S'), **li, **top,
           'definitions': definitions(), 'notes': notes, 'runs': {n: clean(r) for n, r in rows.items()}}
    dj, dm = (Path(args.out) / 'desk_v3.json', Path(args.out) / 'desk_v3.md') if args.out else (DESK_JSON, DESK_MD)
    write_json(dj, out)
    write_text(dm, report_md(out))
    log('書きました: %s(学習 %d 本・%.0f 秒)' % (dj, len(rows), time.time() - t0))
    return 0


# ================================================================ 予測(GPU。jf-heavy --lock big で)

class _Frames(object):
    """予測するフレーム(desk_v2._Frames と同じ中身。ずれの試しは無い)"""

    def __init__(self, items, use_command=False):
        self.items, self.use_command = items, use_command

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        import cv2
        import torch
        from jetformer.train_data import normalize_inputs
        path, tof, imu, veh, _ = self.items[n]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError('画像が読めません: %s' % path)
        img, t, im, st = normalize_inputs(img, tof, imu, veh, self.use_command)
        return torch.from_numpy(img), torch.from_numpy(t), torch.from_numpy(im), torch.from_numpy(st)


def frame_items(ds, scenes, frames):
    """(データセット, シーン, フレーム) → [(画像のパス, ToF, IMU, 車両, ジャンル)](desk_v2.frame_items と同じ)。
    aligned が 0 バイトなら止まる(行が集合とずれるため)"""
    items, cur, a = [], None, None
    for s, i in zip(scenes, frames):
        s = str(s)
        if s != cur:
            p = DATASETS / ds / 'aligned' / (s + '.npz')
            if not npz_ok(p):
                raise SystemExit('aligned が無いか 0 バイトです: %s' % p)
            with np.load(p, allow_pickle=False) as z:
                a = {k: z[k] for k in ('cam_files', 'tof', 'imu', 'vehicle', 'mode')}
            cur = s
        i = int(i)
        items.append((str(DATASETS / ds / str(a['cam_files'][i])), a['tof'][i].copy(), a['imu'][i].copy(),
                      a['vehicle'][i].copy(), int(a['mode'][i])))
    return items


def tab_current(name, proto):
    """v3 の tab の予測があり、今の best.pth のもの(大きさ・更新時刻)か"""
    p = PRED / ('%s_tab.npz' % name)
    if not npz_ok(p):
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            if 'weights_size' not in z.files:
                return False
            ok_shape = z['pred'].shape[1:] == (10, 2)
            return ok_shape and stamp_same(int(z['weights_size']), int(z['weights_mtime_ns']), run_dir_of(name, proto) / 'best.pth')
    except Exception:          # noqa: BLE001
        return False


def trt_current(name, proto):
    rd = run_dir_of(name, proto)
    return any(engine_current(rd, st) for j, st in trt_results((TRT,)).get(name, []))


def predict_tab(names, proto, S, batch=32, workers=3, force=False):
    import gc
    import torch
    import policy_zoo as pz
    import rl_evaluate as ev
    from jetformer.policies import runtime
    for name in names:
        rd = run_dir_of(name, proto)
        if not (rd / 'best.pth').exists():
            log('%s: best.pth がありません(飛ばす)' % name)
            continue
        if tab_current(name, proto) and not force:
            log('%s: tab の予測はあります(今の best.pth)' % name)
            continue
        cfg = model_cfg(rd)
        trained = np.array([int(x) for x in cfg['trained_modes']])
        use_cmd = bool(cfg.get('command_input', True))
        meta = dict(run=np.array(name), ckpt=np.array('best'), model_config_sha256=np.array(sha256(rd / 'model_config.json')),
                    best_sha256=np.array(sha256(rd / 'best.pth')), script_sha256=np.array(sha256(Path(__file__).resolve())),
                    weights_size=np.array(weights_stamp(rd / 'best.pth')['size']),
                    weights_mtime_ns=np.array(weights_stamp(rd / 'best.pth')['mtime_ns']))
        t0 = time.time()
        model = runtime.load_run(rd, 'best', device='cuda')[0]
        items = frame_items(DS_V2, S['scene'], S['frame'])
        g5, sub5 = ev._genres(np.full(len(items), SANITY_GENRE), None, trained)
        res = pz._predict_frames(model, _Frames(items, use_cmd), g5, cfg['num_modes'], batch, workers)
        pred = res['actions'].astype(np.float32)
        if pred.shape != (len(items), 10, 2) or not np.isfinite(pred).all():
            raise SystemExit('%s: 予測の形か値がおかしい' % name)
        save_npz(PRED / ('%s_tab.npz' % name), key=np.array(S['key']), pred=pred, genre=g5.astype(np.int8),
                 substituted=sub5, **meta)
        log('%s: tab %d フレーム %.0f 秒' % (name, len(pred), time.time() - t0))
        del model
        gc.collect()
        torch.cuda.empty_cache()


def crosscheck_b(proto, S, batch=32, workers=3):
    """B を予測し直して、v2 の C3 の保存(desk/pred の B の tab)と照らす(≤ 0.01 µs)。段 data で train_data.py が変わったため"""
    import gc
    import torch
    import policy_zoo as pz
    import rl_evaluate as ev
    from jetformer.policies import runtime
    out_path = WORK / 'crosscheck_B_tab.json'
    rd = run_dir_of(B_RUN, proto)
    cfg = model_cfg(rd)
    trained = np.array([int(x) for x in cfg['trained_modes']])
    t0 = time.time()
    model = runtime.load_run(rd, 'best', device='cuda')[0]
    items = frame_items(DS_V2, S['scene'], S['frame'])
    g5, _ = ev._genres(np.full(len(items), SANITY_GENRE), None, trained)
    pred = pz._predict_frames(model, _Frames(items, bool(cfg.get('command_input', True))), g5, cfg['num_modes'],
                              batch, workers)['actions'].astype(np.float64)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    with np.load(PRED2 / ('%s_tab.npz' % B_RUN), allow_pickle=False) as z:
        saved = z['pred'].astype(np.float64)
        same_key = str(z['key']) == S['key']
    mx = float(np.abs(pred - saved).max())
    out = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'run': B_RUN, 'saved': str(PRED2 / ('%s_tab.npz' % B_RUN)),
           'frames': int(len(pred)), 'max_abs_us': round(mx, 6), 'limit_us': CROSS_LIMIT_US, 'same_key': same_key,
           'pass': bool(same_key and mx <= CROSS_LIMIT_US), 'seconds': round(time.time() - t0, 1),
           'train_data_sha256': sha256(REPO / 'jetformer' / 'train_data.py')}
    write_json(out_path, out)
    log('B の tab の予測し直し: %s' % json.dumps(out, ensure_ascii=False))
    if not out['pass']:
        raise SystemExit('B の tab の予測し直しが v2 の保存と合いません(≤ %.2f µs)' % CROSS_LIMIT_US)


def cmd_predict(args):
    proto = protocol()
    names = list(dict.fromkeys(args.runs or []))
    if not names:
        raise SystemExit('--runs を付けてください')
    bad = [n for n in names if not allowed_run(n, proto)]
    if bad:
        raise SystemExit('protocol_v3 の runs・5 本の名前ではありません: %s' % ' '.join(bad))
    if any(family(n) in ('v3', 'bfast2') for n in names) and not LOCK2.exists() and not args.allow_before_lock2:
        raise SystemExit('protocol_v3_lock2.json がまだ無いので、v3 の予測はしません(protocol_v3 lock.second_lock)')
    if any(family(n) in ('v3', 'bfast2') for n in names):
        lock2_stop('v3 の予測', getattr(args, 'allow_deviation', False))
    import rl_policies as rp
    import desk_gpu
    busy = desk_gpu.web_busy()           # GET の状態だけ
    if busy:
        raise SystemExit('始めません: %s' % busy)
    rp._check_gpu_free(min_gb=2.0)
    S = load_sets(proto)
    cj = load_json(WORK / 'crosscheck_B_tab.json') or {}
    if not cj.get('pass') or cj.get('train_data_sha256') != sha256(REPO / 'jetformer' / 'train_data.py'):
        crosscheck_b(proto, S, args.batch, args.workers)
    predict_tab(names, proto, S, args.batch, args.workers, args.force)
    return 0


def sanity_frames(S):
    """sanity の集合: 10/10 の AI 走行(desk_gpu.ai_frames と同じ 21,759)+ 10/24 2 回目(sets の san_*)。desk_v2 と同じ"""
    import desk_gpu
    rows = [(DS_1010, s, i, t, 0) for s, i, t in desk_gpu.ai_frames()]
    rows += [(DS_V2, str(s), int(i), int(t), 1) for s, i, t in zip(S['san_scene'], S['san_frame'], S['san_t_us'])]
    return rows


def time_run_numpy(runner, feed, warmup=TIME_WARMUP, runs=TIME_RUNS):
    for _ in range(warmup):
        runner.run_numpy(**feed)
    ts = np.zeros(runs)
    for j in range(runs):
        t0 = time.perf_counter()
        runner.run_numpy(**feed)
        ts[j] = (time.perf_counter() - t0) * 1000.0
    return ts


def trt_times(ms):
    ms = np.asarray(ms, np.float64)
    ms = ms[np.isfinite(ms)]
    if not len(ms):
        return None, None
    return float(np.median(ms)), float(np.percentile(ms, 99))


def cmd_trt(args):
    """desk_v2.cmd_trt と同じ手順(B → 候補。sanity の流れ・同じ実フレームの 100 + 600 回)。出力は desk_v3/trt"""
    import cv2
    import torch
    from jetformer import config, drive, policies
    from jetformer.policies import post as post_mod
    from jetformer.train_data import normalize_inputs
    from jetformer.trt_runtime import TRTRunner
    import rl_policies as rp
    import desk_lib as DL
    import desk_gpu
    proto = protocol()
    names = [B_RUN] + [n for n in dict.fromkeys(args.runs or []) if n != B_RUN]
    if len(names) > 1 + args.max_runs:
        raise SystemExit('1 回の仕事は候補 %d 本まで(短く分ける)' % args.max_runs)
    bad = [n for n in names if not allowed_run(n, proto)]
    if bad:
        raise SystemExit('protocol_v3 の runs・5 本の名前ではありません: %s' % ' '.join(bad))
    if any(family(n) in ('v3', 'bfast2') for n in names) and not LOCK2.exists() and not args.allow_before_lock2:
        raise SystemExit('protocol_v3_lock2.json がまだ無いので、v3 の TensorRT の仕事はしません')
    if any(family(n) in ('v3', 'bfast2') for n in names):
        lock2_stop('v3 の TensorRT の仕事', getattr(args, 'allow_deviation', False))
    for n in names:
        if run_dir_of(n, proto) != RUNS / n:
            raise SystemExit('%s の出力が data/runs の下ではありません(drive.model_info が読めない)' % n)
    busy = desk_gpu.web_busy()
    if busy:
        raise SystemExit('始めません: %s' % busy)
    S = load_sets(proto)
    rp._check_gpu_free(min_gb=2.0)
    rows = sanity_frames(S)
    if args.limit:
        rows = rows[:args.limit // 2] + [r for r in rows if r[4] == 1][:args.limit - args.limit // 2]
    job = time.strftime('%Y%m%d-%H%M%S')
    log('TensorRT の仕事 %s: エンジン %d 本・フレーム %d' % (job, len(names), len(rows)))
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    params = dict(drive.DEFAULT_PARAMS)          # 写しを変えるだけ(drive の既定値は変えない)
    params.update(BASE_PARAMS_OVERRIDE)
    lo, hi = float(config.DRIVE['min_us']), float(config.DRIVE['max_us'])
    E = {}
    for name in names:
        st = {'run': name}
        try:
            info = drive.model_info(name)
            if info is None:
                raise RuntimeError('drive.model_info が None(エンジンが無い)')
            st['drivable'], st['why_not'] = bool(info.get('drivable')), info.get('why_not')
            eng = RUNS / name / ENGINE
            stt = eng.stat()
            st['engine'] = {'size': stt.st_size, 'mtime_ns': stt.st_mtime_ns, 'sha256': sha256(eng)}
            runner = TRTRunner(str(eng))
            nm = int(info['num_modes'])
            ok, why = policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs},
                                        {n: list(runner.shapes[n]) for n in runner.outputs}, nm)
            st['check_io_ok'], st['check_io_why'] = bool(ok), why
            if not ok:
                raise RuntimeError('check_io: %s' % why)
            ref = (float(info['steer_center_us']), float(info['throttle_neutral_us']))
            stage = post_mod.Stage(info, ref, fps=config.CAMERA['fps'])
            oh = np.zeros((1, nm), np.float32)
            oh[0, SANITY_GENRE] = 1.0
            zeros = {n: np.zeros(runner.shapes[n], np.float32) for n in runner.inputs}
            for _ in range(3):
                runner.run_numpy(**zeros)
            E[name] = {'runner': runner, 'stage': stage, 'mode': oh, 'cmd': bool(info.get('command_input', False)),
                       'pred': np.full((len(rows), 10, 2), np.nan, np.float32),
                       'model_pred': np.full((len(rows), 10, 2), np.nan, np.float32), 'nonfinite_out': 0,
                       'post_error': 0, 'ms': np.full(len(rows), np.nan, np.float32), 'st': st}
        except Exception as e:      # noqa: BLE001
            st['error'] = '%s: %s' % (type(e).__name__, e)
            log('%s: 読めません %s' % (name, st['error']))
            E[name] = {'st': st}
    cur, a, prev_scene, skipped = None, None, None, 0
    erpm = np.zeros(len(rows))
    done = np.zeros(len(rows), bool)
    t_start = time.time()
    for k, (ds, s, i, t_us, prt) in enumerate(rows):
        if (ds, s) != cur:
            p = DATASETS / ds / 'aligned' / (s + '.npz')
            a = None
            if npz_ok(p):
                with np.load(p, allow_pickle=False) as z:
                    a = {n: z[n] for n in ('cam_files', 'tof', 'imu', 'vehicle')}
            cur = (ds, s)
        if a is None:                    # 0 バイトの aligned は飛ばす(数える。sanity の nonfinite には入れない)
            skipped += 1
            continue
        img = cv2.imread(str(DATASETS / ds / str(a['cam_files'][i])))
        erpm[k] = float(a['vehicle'][i, 3])
        done[k] = True
        if (ds, s) != prev_scene:
            for e in E.values():
                if 'stage' in e:
                    e['stage'].reset()
            prev_scene = (ds, s)
        normed = {}
        for name, e in E.items():
            if 'runner' not in e:
                continue
            uc = e['cmd']
            if uc not in normed:
                im, tf, iu, sv = normalize_inputs(img, a['tof'][i], a['imu'][i], a['vehicle'][i], uc)
                normed[uc] = {'image': im[None], 'tof': tf[None], 'imu': iu[None], 'state': sv[None]}
            feed = dict(normed[uc])
            feed['mode'] = e['mode']
            t0 = time.perf_counter()
            res = e['runner'].run_numpy(**feed)
            e['ms'][k] = (time.perf_counter() - t0) * 1000.0
            if not all(np.isfinite(v).all() for v in res.values()):
                e['nonfinite_out'] += 1
            mp = e['stage'].model_pred(res)
            e['model_pred'][k] = mp
            try:
                pred, _ = e['stage'].apply(res, mp, post_mod.speed_of(erpm[k]), params, int(t_us) * 1000)
                e['pred'][k] = pred
            except post_mod.PostError:
                e['post_error'] += 1
        if (k + 1) % 3000 == 0:
            log('%d / %d(%.0f 秒)' % (k + 1, len(rows), time.time() - t_start))
    ds0, s0, i0 = rows[0][0], rows[0][1], rows[0][2]
    with np.load(DATASETS / ds0 / 'aligned' / (s0 + '.npz'), allow_pickle=False) as z:
        a0 = {n: z[n][i0] for n in ('cam_files', 'tof', 'imu', 'vehicle')}
    img0 = cv2.imread(str(DATASETS / ds0 / str(a0['cam_files'])))
    order = [n for n in E if 'runner' in E[n]]
    if B_RUN in order:
        order = [B_RUN] + [n for n in order if n != B_RUN] + ([B_RUN] if len(order) > 1 else [])
    times = {}
    for name in order:
        e = E[name]
        im, tf, iu, sv = normalize_inputs(img0, a0['tof'], a0['imu'], a0['vehicle'], e['cmd'])
        feed = {'image': im[None], 'tof': tf[None], 'imu': iu[None], 'state': sv[None], 'mode': e['mode']}
        times.setdefault(name, []).append(time_run_numpy(e['runner'], feed))
    part = np.array([r[4] for r in rows], np.int8)
    summary = {'schema': 'jf-c3-trt/v3', 'timing': 'engine_check と同じ(100 回の空回し + 600 回・同じ実フレーム・B → 候補 → B)',
               'job': job, 'day': job[:8], 'made_at': time.strftime('%Y-%m-%d %H:%M:%S'), **lock_info(),
               'frames': len(rows), 'frames_1010': int((part == 0).sum()), 'frames_1024_2': int((part == 1).sum()),
               'skipped_zero_byte_frames': skipped, 'limit': args.limit or None, 'mode_id': SANITY_GENRE,
               'params': {k: params[k] for k in sorted(BASE_PARAMS_OVERRIDE)}, 'drive_range_us': [lo, hi],
               'seconds': round(time.time() - t_start, 1), 'engines': {}}
    d_dir = TRT / job
    for name, e in E.items():
        st = e['st']
        if 'runner' in e:
            p, mp = e['pred'], e['model_pred']
            bad = (~np.isfinite(p).all(axis=(1, 2)) | ~np.isfinite(mp).all(axis=(1, 2))) & done   # 飛ばした行は数えない
            oor = int(((p < SANITY_LO) | (p > SANITY_HI)).sum() + ((mp < SANITY_LO) | (mp > SANITY_HI)).sum())
            lmed, lp99 = trt_times(e['ms'])
            med, p99 = trt_times(times[name][0]) if name in times else (None, None)
            if name in times and len(times[name]) > 1:
                st['run_numpy_after_median_ms'], st['run_numpy_after_p99_ms'] = trt_times(times[name][-1])
            st.update({'sanity_loop_median_ms': lmed, 'sanity_loop_p99_ms': lp99, 'frames': int(done.sum()),
                       'nonfinite_out_frames': e['nonfinite_out'], 'post_error_frames': e['post_error'],
                       'nonfinite_frames': int(bad.sum() + e['nonfinite_out'] + e['post_error']),
                       'out_of_range_values': oor, 'sanity_range_us': [SANITY_LO, SANITY_HI],
                       'outside_drive_range_values': int(((p < lo) | (p > hi)).sum()), 'values': int(p[done].size),
                       'by_part': {nm: {'frames': int(((part == c) & done).sum()), 'nonfinite_frames': int(bad[part == c].sum())}
                                   for nm, c in (('10/10', 0), ('10/24 2 回目', 1))},
                       'steer_min_us': float(np.nanmin(p[:, :, 0])), 'steer_max_us': float(np.nanmax(p[:, :, 0])),
                       'thr_min_us': float(np.nanmin(p[:, :, 1])), 'thr_max_us': float(np.nanmax(p[:, :, 1])),
                       'run_numpy_median_ms': med, 'run_numpy_p99_ms': p99})
            f = d_dir / ('%s.npz' % name)
            save_npz(f, pred=p, model_pred=mp, ms=e['ms'], ms_timed=np.asarray(times.get(name, [[]])[0], np.float32),
                     erpm=erpm.astype(np.float32), part=part, run=np.array(name), mode_id=np.array(SANITY_GENRE),
                     job=np.array(job))
            st['file'] = str(f)
        st['sanity_ok'] = DL.sanity_of(st)
        summary['engines'][name] = st
        log('%s: sanity_ok %s 中央 %s ms・p99 %s ms %s' % (name, st['sanity_ok'], rnd(st.get('run_numpy_median_ms'), 3),
                                                       rnd(st.get('run_numpy_p99_ms'), 3), st.get('error') or ''))
    write_json(TRT / ('%s.json' % job), summary)
    log('書きました: %s' % (TRT / ('%s.json' % job)))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict', help='AI の記録の集合の予測(GPU。jf-heavy --lock big で)')
    p.add_argument('--runs', nargs='*')
    p.add_argument('--force', action='store_true', help='あっても予測し直す')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--allow-before-lock2', action='store_true', help='lock2_v3 の前でも v3 を予測する(deviations_v3.md に書くこと)')
    p.add_argument('--allow-deviation', action='store_true',
                   help='lock2_v3 に記録した机上の台本と今の台本が違っても予測する(deviations_v3.md に書くこと)')
    t = sub.add_parser('trt', help='TensorRT の sanity・推論の速さ・低速のブレーキ(GPU。B はいつも入る)')
    t.add_argument('--runs', nargs='*')
    t.add_argument('--max-runs', type=int, default=2)
    t.add_argument('--limit', type=int, default=0, help='試し: 先頭のフレームだけ(欄に使わない)')
    t.add_argument('--allow-before-lock2', action='store_true')
    t.add_argument('--allow-deviation', action='store_true', help='lock2_v3 の台本と違っても(deviations_v3.md に書くこと)')
    r = sub.add_parser('report', help='欄と表(CPU・numpy)')
    r.add_argument('--out', help='試しだけ: desk_v3.json・md を書く場所(既定 %s)' % REP3)
    r.add_argument('--runs', nargs='*', help='この学習だけ(B・5 本はいつも入る)')
    args = ap.parse_args()
    return {'predict': cmd_predict, 'trt': cmd_trt, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
