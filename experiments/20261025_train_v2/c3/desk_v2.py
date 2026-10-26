#!/usr/bin/env python3
"""C3: 机上のベンチ v2(2026-10-25)。protocol_v2.json の sets・fields_added・stage0_checks・gates_v2・gates_bfast・
decisions.D3 が使う欄を出す。10/21 の机上のベンチ(experiments/20261021_bench_prep/desk/)を土台にした。

車・PCA9685・I2C・カメラ・ToF・基板に触れない。走行の API は呼ばない(jf-heavy が始める前に状態を見るだけ)。
data/・既定値・protocol・lock2・decisions.json は読むだけ。封の 14 本は読まない(読もうとしたら止まる。11/1 の封の欄はこの台本に無い)。
書くのは /home/jetson/jetformer_reports/train_v2_20261025/desk/ の下だけ(desk_v2.json・desk_v2.md・sets_v2.*・pred/・trt/)。

    cd ~/jetformer
    P=/home/jetson/miniconda/envs/jetracer/bin/python
    C3=experiments/20261025_train_v2/c3/desk_v2.py
    # ① 集合(CPU・numpy。1 回だけ。数分)
    /home/jetson/bin/jf-heavy --lock none --mem 3G --need-mb 1500 -- nice -n 10 $P -I $C3 sets
    # ② PyTorch FP32 の予測(GPU。1 本ずつ・短く。tab = AI の記録の集合 約 2 分、shift = H1 のずれの試し 7 つ 約 3 分)
    G="/home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 $P -u"
    $G $C3 predict --what tab --runs clean_driving_20261011-nocmd-s3-20261011-024626      # B(段 0 の 2・4 に要る)
    $G $C3 predict --what tab --runs bench_20261021-ens-p2                                # 2 本の平均(段 0 の 4)
    $G $C3 predict --what shift --runs v2_20261025-v2h-aug2-e60-s0                        # D3 の robust_dl1_us
    # ③ TensorRT(GPU。B はいつも入る。候補は 1 回に 2 本まで: sanity・同じ日の推論の速さ・低速のブレーキ)
    $G $C3 trt --runs v2_20261025-v2h-e60-s0 v2_20261025-v2h-e60-s1
    # ④ 欄と表(CPU・numpy。torch を読まない)
    /home/jetson/bin/jf-heavy --lock none --mem 2500M --need-mb 1500 -- $P -I $C3 report
  まとめて流すときは c3/run_c3_gpu.sh(1 回の GPU の仕事ごとに jf-heavy --lock big・あいだ 60 秒)。

集合(protocol sets。AI の記録の集合はデータセット dataset_v2_20261025 の aligned のフレーム(カメラ 30 fps)で、
ラベル data/labels/<記録>.npz の行(label_index)の値で選ぶ。周・起点・直線は inventory.py の process_log の写しで出す
(make_v2_tables.py・plan_checks.py と同じ取り出し)。どのフレームも vehicle_valid・label_index あり):
  AI の側         ラベルの ai(CH3 が AI 側・ondisk)かつ dry run でない かつ aligned の source == 1
  ai_pre_<場所>   学習の回(46 本)の AI の衝突(ラベルの events.collisions で ai)の手前 1 秒 [tc − 1, tc) の AI の側。
                  場所はラベルの loc: R1・DE、SW = loc_detail SW(loc では other)
  de_pass         学習の回の AI の周(inventory の周 = R1 の起点 → 次の起点)のうち「学習に使っていない周」
                  (きれいでない周・B の 9.5 秒以上・ほかのモデル)の中の DE の入り(loc_detail が DE になった行)から 0.5 秒
                  [te, te + 0.5)、同じ周の中の AI の側
  straight        学習の回の AI のきれいな周で(B の 9.5 秒以上・ほかのモデル)、直線(inventory の straight:
                  eRPM < −1000・|ヨーの 11 サンプル平均| < 40 deg/s)の AI の側
  standstill      学習の回の AI の側・|ラベルの eRPM| < 400・衝突の前後でない(near_collision)・
                  ToFGuard(data/tof_floor_ref.npy).features(aligned の ToF, erpm 0, gz 0, ステア中心) の x_free > 1.0 m
  gb_main         10/23 公式の B の 3 本(No.8・11・15)の AI の側・衝突の前後でない・間引きなし。
                  G-b は loc_detail の DE・SW・HS、G-c は R1 の起点 ta(inventory の起点)の [ta − 1, ta)、
                  bfast_de_line は DE の入りから [te + 0.5, te + 0.7)
  gb_ref          10/24 2 回目の 6 本の AI の側・衝突の前後でない・B-FAST の手本のフレームの外(drop_bfast.npz で残したフレームの外)
  stage0_2        10/24 2 回目の、B が走った記録(No.9・No.10)の AI の側で、記録の model_out が数値・hold でない
  sanity          10/10 の AI 走行 21,759 フレーム(ai_mix_20261011 の source == 1 & vehicle_valid。10/21 と同じ)+
                  10/24 2 回目の 6 本の source == 1 & vehicle_valid
  H1              desk_inputs_20261022.npz の primary(4,670 フレーム。eval_v2.py と同じ)

欄の式(protocol fields_added・sign_rules。どれも PyTorch FP32 の best・ジャンル 5。TensorRT はエンジンの run_numpy):
  車に出る        drive.apply_gains と同じ(ステアのゲイン 1.0・trim 0・スロットルのゲインは前進側だけ・1000〜2000 に切る)。
                  ゲイン: B 0.90・候補のジャンル 5 0.90・B-FAST 0.90・ジャンル 9 1.00。ref は各モデルの model_config
  _thr_vs_b       行 2 の車に出るスロットル(候補 − B)のフレームの平均。+ = 緩める。_fwd_ = −(PWM の差)
  steer の vs_b   行 0 の予測ステア(候補 − B)の平均(車に出す前の予測)
  standstill_go_frac  行 0 の車に出るスロットル < そのモデルの中立 − 30 µs の割合
  robust_dl1_us   ずれの試し 7 つ(画像 3 行上・3 行下・明るさ ×0.8・×1.2・ToF 1 行下・上 1/3 ×0.85・×1.15)の
                  H1 の L1(試し)− L1(そのまま)の平均。そのまま = eval/pred の保存した best。上 1/3 は deviations.md 7:
                  行 r の掛け算 1 + (f − 1)·w(r)、w = train_data.top_ramp(一番上 1・上 1/3 の境で 0 の直線)
  bfast_gf_limit_us   B の直線の 行 2 の前進の量の(ゲイン 1.00 − 0.90)の平均(= 0.10 × B の生の前進の量)
  sanity_ok・brake_lowspeed_frac・engine_ok  10/21 の desk_lib.py と同じ(sanity の集合だけ 10/24 2 回目を足した)
  trt_median_ms・trt_p99_ms  engine_check.json と同じ測り方(同じ実フレームで 100 回の空回し + 600 回の run_numpy)を、
                  この台本の TensorRT の仕事で B → 候補 → B の順に同じ回に測った中央値・99% 点
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
HERE = REPO / 'experiments' / '20261025_train_v2'
C3DIR = HERE / 'c3'
DESK1021 = REPO / 'experiments' / '20261021_bench_prep' / 'desk'
INV_PY = REPO / 'experiments' / '20261025_dataset_v2' / 'inventory.py'
for _p in (str(REPO), str(REPO / 'tools'), str(HERE), str(DESK1021)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PROTOCOL = REP / 'protocol_v2.json'
LOCK2 = REP / 'protocol_v2_lock2.json'
SEAL_OPENED = REP / 'seal_opened.json'
OUT = REP / 'desk'
DESK_JSON = OUT / 'desk_v2.json'
DESK_MD = OUT / 'desk_v2.md'
SETS_NPZ = OUT / 'sets_v2.npz'
SETS_JSON = OUT / 'sets_v2.json'
PRED = OUT / 'pred'
TRT = OUT / 'trt'
EVAL_PRED = REP / 'eval' / 'pred'
DATA = REPO / 'data'
RUNS = DATA / 'runs'
DATASETS = DATA / 'datasets'
LABELS = DATA / 'labels'
DS_V2 = 'dataset_v2_20261025'
DS_1010 = 'ai_mix_20261011'
TABLES = DATA / 'derived' / DS_V2
TOF_REF = DATA / 'tof_floor_ref.npy'
BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
TABLE_KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
ENGINE = 'jetformer_fp16.engine'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
ENS_RUN = 'bench_20261021-ens-p2'
GB_LOGS = ('20261023-210840', '20261023-211209', '20261023-211547')     # 10/23 公式の B の No.8・No.11・No.15
SESS_1024_2 = '10/24 2 回目(16:01〜)'
SESS_CODE = {'10/10': 0, '10/23': 1, '10/24 2 回目': 2}

# ---- 数(protocol のとおり。test_desk_v2.py で protocol の文と照らす)
ROW0, ROW2 = 0, 2                    # 行 0 = 1 フレーム先、行 2 = 走る日の先読み 3
GAINS = {'B': 0.90, 'g5': 0.90, 'bfast': 0.90, 'g9': 1.00}
DRIVE_LO, DRIVE_HI = 1000.0, 2000.0  # config.DRIVE の min_us・max_us(apply_gains の lo・hi)
PRE_S = 1.0                          # AI の場面: 衝突の手前 1 秒。G-c: R1 の起点の手前 1 秒
DE_PASS_S = (0.0, 0.5)               # DE の通過: 入って 0.5 秒
DE_LINE_S = (0.5, 0.7)               # bfast_de_line: 入って 0.5〜0.7 秒
LAP_B_UNUSED_S = 9.5                 # B のきれいな周で 9.5 秒未満は学習に使った(v2-M・B-FAST)ので外す
STAND_ERPM = 400.0
STAND_GO_US = 30.0
TOF_FREE_M = 1.0
STANDSTILL_MIN_FRAMES = 300
SANITY_GENRE = 5
SANITY_LO, SANITY_HI = 500.0, 2500.0
INFER_FACTOR = 1.3
STAGE0_2_LIMIT_US = 10.0
STAGE0_1_LIMIT_US = 0.01
LAP_COEF_S_PER_US = -0.02696
LAP3_BASE_S = 25.75
LOC_DETAIL = {'?': 0, 'R1': 1, 'GM_in': 2, 'GM_out': 3, 'DE': 4, 'SW': 5, 'HS': 6, 'LH': 7}   # label_events.LOC_DETAIL
SHIFTS = ('img_up3', 'img_down3', 'bright_0.8', 'bright_1.2', 'tof_down1', 'top_0.85', 'top_1.15')
SHIFT_LABELS = {'img_up3': '画像 3 行上', 'img_down3': '画像 3 行下', 'bright_0.8': '明るさ ×0.8', 'bright_1.2': '明るさ ×1.2',
                'tof_down1': 'ToF 1 行下', 'top_0.85': '上 1/3 ×0.85', 'top_1.15': '上 1/3 ×1.15'}
TOP_FRACTION = 1.0 / 3.0             # train_data.TOP_FRACTION と同じ(試験で照らす)
TIME_WARMUP, TIME_RUNS = 100, 600       # 推論の速さ: engine_load_check.py(engine_check.json)と同じ 100 回の空回しの後 600 回
SHIFT_CHECK_ROWS = 256               # ずれの試しの仕事で「そのまま」も予測して保存と照らす行の数
SHIFT_PIPE_LIMIT_US = 0.5            # その差がこれを超えたら止める(入力・ジャンルの作り方の違い。値は記録する)
BASE_PARAMS_OVERRIDE = {'steer_gain': 1.0, 'throttle_gain': 1.0, 'lookahead': 3, 'tof_guard': 1, 'ensemble_j': 0,
                        'speed_cap': 0.0, 'zone_caps': 0, 'tcp_mode': 0, 'speed_scale': 1.0}   # desk_gpu.py と同じ

# desk_v2.json の各学習の行に出す欄(eval_v2.py の C3_FIELDS をすべて含む。test_desk_v2.py で照らす)
FIELDS_C3 = ('sanity_ok', 'brake_lowspeed_frac', 'robust_dl1_us', 'de_pre_thr_vs_b_us', 'r1_pre_thr_vs_b_us',
             'sw_pre_thr_vs_b_us', 'de_overshoot_vs_b_us', 'straight_thr_vs_b_us', 'straight_fwd_vs_b_us',
             'standstill_go_frac', 'g9_minus_g5_pre_thr_us', 'bfast_gb_fwd_vs_b_us', 'bfast_gb_fwd_vs_b_us_de',
             'bfast_gb_fwd_vs_b_us_sw', 'bfast_gb_fwd_vs_b_us_hs', 'bfast_gc_fwd_vs_b_us', 'bfast_gb_ref_fwd_vs_b_us',
             'bfast_gb_fwd_g100_vs_b090_us', 'bfast_de_line_vs_b_us', 'lap_delta_est_s', 'lap3_est_s',
             'trt_median_ms', 'trt_p99_ms', 'engine_ok')
FIELDS_H1 = ('l1_best_us', 'l1_steer_best_us', 'l1_thr_best_us', 'l1_last_us', 'l1_interv_us', 'l1_interv_last_us',
             'r1_app_thr_diff_us', 'nonzone_thr_diff_us', 'steer_amp_ratio', 'l1_day_us', 'l1_night_us',
             'l1_genre4_us', 'l1_genre5_us', 'l1_genre4_last_us', 'l1_genre5_last_us', 'steer_bias_us', 'thr_bias_us',
             'steer_bias_last_us', 'thr_bias_last_us') + tuple('l1_zone_%s_us' % z for z in
                                                                 ('R1_app', 'R1_turn', 'GM', 'DE', 'none', 'unknown'))


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


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rnd(x, nd=4):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    v = num(x)
    return None if v is None else round(v, nd)


# ================================================================ 式(numpy だけ。test_desk_v2.py で作り物の入力で試す)

def car_out(raw, ref, thr_gain, steer_gain=1.0, trim=0.0, lo=DRIVE_LO, hi=DRIVE_HI):
    """drive.apply_gains と同じ式を配列で。raw [..., 2](steer, throttle µs)→ 車に出る値 [..., 2]。
    ゲインは中心・中立の周り。スロットルのゲインは前進側(raw < 中立)だけ。thr_gain・steer_gain・trim は配列でもよい"""
    raw = np.asarray(raw, np.float64)
    s = ref[0] + trim + np.asarray(steer_gain, np.float64) * (raw[..., 0] - ref[0])
    dt = raw[..., 1] - ref[1]
    t = ref[1] + np.where(dt < 0, np.asarray(thr_gain, np.float64) * dt, dt)
    return np.stack([np.clip(s, lo, hi), np.clip(t, lo, hi)], axis=-1)


def top_ramp(height):
    """train_data.top_ramp と同じ(torch を読まないための写し。試験で照らす): 一番上の行 1.0 → 上 1/3 の境まで直線で 0"""
    w = 1.0 - np.arange(height, dtype=np.float32) / (height * TOP_FRACTION)
    return np.clip(w, 0.0, 1.0).reshape(height, 1, 1)


def perturb(kind, img, tof):
    """ずれの試し。img = BGR uint8 [H,W,3]、tof = ToF の生の値 [42,54](uint16)→ (img, tof) の写し。
    img_up3・img_down3・bright_0.8・bright_1.2・tof_down1 は tools/policy_zoo.perturb と同じ(試験で照らす)。
    top_<f>: 上 1/3 を掛け算(deviations.md 7)。行 r に 1 + (f − 1)·top_ramp(r) を掛けて四捨五入・0〜255 に切る"""
    img = np.asarray(img)
    tof = np.asarray(tof)
    if kind == 'img_down3':
        return np.concatenate([np.repeat(img[:1], 3, axis=0), img[:-3]], axis=0), tof.copy()
    if kind == 'img_up3':
        return np.concatenate([img[3:], np.repeat(img[-1:], 3, axis=0)], axis=0), tof.copy()
    if kind in ('bright_0.8', 'bright_1.2'):
        f = float(kind.split('_')[1])
        return np.clip(np.rint(img.astype(np.float32) * f), 0, 255).astype(np.uint8), tof.copy()
    if kind == 'tof_down1':
        out = np.zeros_like(tof)
        out[1:] = tof[:-1]
        return img.copy(), out
    if kind in ('top_0.85', 'top_1.15'):
        f = np.float32(float(kind.split('_')[1]))
        mult = np.float32(1.0) + (f - np.float32(1.0)) * top_ramp(img.shape[0])
        return np.clip(np.rint(img.astype(np.float32) * mult), 0, 255).astype(np.uint8), tof.copy()
    raise ValueError('知らないずれ: %s' % kind)


def window_mask(t, starts, lo, hi):
    """各時刻 t が、どれかの start の [start + lo, start + hi) に入るか"""
    t = np.asarray(t, np.float64)
    m = np.zeros(len(t), bool)
    for s in np.asarray(starts, np.float64).ravel():
        m |= (t >= s + lo) & (t < s + hi)
    return m


def run_starts(code, value):
    """code が value になった行(前の行が value でない・最初の行)の番号"""
    m = np.asarray(code) == value
    return np.nonzero(m & ~np.concatenate([[False], m[:-1]]))[0]


def lap_of(t, bounds):
    """各時刻が入る周 [a, b) の番号(bounds の順。入らなければ −1)"""
    t = np.asarray(t, np.float64)
    out = np.full(len(t), -1, np.int64)
    for j, (a, b) in enumerate(bounds):
        out[(t >= a) & (t < b)] = j
    return out


def lap_unused(lap):
    """DE の通過に使う周: 学習に使っていない周(きれいでない周・B の 9.5 秒以上・ほかのモデル)"""
    return (not lap.get('clean')) or lap.get('model') != 'B' or float(lap['lap_s']) >= LAP_B_UNUSED_S


def lap_straight(lap):
    """直線に使う周: 学習に使っていない AI のきれいな周(B の 9.5 秒以上・ほかのモデル)"""
    return bool(lap.get('clean')) and (lap.get('model') != 'B' or float(lap['lap_s']) >= LAP_B_UNUSED_S)


def mean_or_none(x):
    x = np.asarray(x, np.float64)
    return float(x.mean()) if x.size else None


def thr_vs_b(pc, pb, refc, refb, mask, gc, gb, row=ROW2):
    """行 row の車に出るスロットル(候補 − B)の平均 [µs]。+ = B より緩める"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    a = car_out(np.asarray(pc)[m][:, row], refc, gc)[:, 1]
    b = car_out(np.asarray(pb)[m][:, row], refb, gb)[:, 1]
    return float(np.mean(a - b))


def fwd_vs_b(pc, pb, refc, refb, mask, gc, gb, row=ROW2):
    """前進の量の差(候補 − B)= −(PWM の差)。+ = 強く踏む"""
    v = thr_vs_b(pc, pb, refc, refb, mask, gc, gb, row)
    return None if v is None else -v


def steer_vs_b(pc, pb, mask, row=ROW0):
    """行 row の予測ステア(候補 − B)の平均 [µs]。+ = 左"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    return float(np.mean(np.asarray(pc, np.float64)[m][:, row, 0] - np.asarray(pb, np.float64)[m][:, row, 0]))


def go_frac(p, ref, mask, gain, row=ROW0, margin=STAND_GO_US):
    """止まった場面で、行 row の車に出るスロットルが 中立 − margin より前へ出す割合"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    thr = car_out(np.asarray(p)[m][:, row], ref, gain)[:, 1]
    return float(np.mean(thr < ref[1] - margin))


def gf_limit(pb, refb, mask, row=ROW2, g_hi=1.00, g_lo=0.90):
    """G-f の上限: B を g_hi にしたときの直線の前進の量 − g_lo のときの量 の平均(= 0.10 × B の生の前進の量)"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    r = np.asarray(pb)[m][:, row]
    return float(np.mean(car_out(r, refb, g_lo)[:, 1] - car_out(r, refb, g_hi)[:, 1]))


def raw_forward(pb, refb, mask, row=ROW2):
    """参考: B の生の前進の量(中立 − 予測のスロットル。後ろ向きは 0)の平均"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    return float(np.mean(np.maximum(refb[1] - np.asarray(pb, np.float64)[m][:, row, 1], 0.0)))


def l1(pred, y):
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(y, np.float64))))


def robust_dl1(base, shifted, y):
    """→ (平均の増え, {ずれ: 増え})。base・shifted[k]・y は同じ行の [N,10,2]"""
    b = l1(base, y)
    per = {k: l1(v, y) - b for k, v in shifted.items()}
    return (float(np.mean(list(per.values()))) if per else None), per


def lap_est(straight_fwd_us):
    """速さの見込み(参考・門にしない): lap_delta_est_s = −0.02696 × straight_fwd_vs_b_us、lap3_est_s = 25.75 + 3 × それ"""
    if straight_fwd_us is None:
        return None, None
    d = LAP_COEF_S_PER_US * straight_fwd_us
    return d, LAP3_BASE_S + 3.0 * d


def stage0_2_diffs(pred, model_out, ref, gains, trim, lookahead):
    """段 0 の 2: PyTorch の予測 pred [N,10,2] を、記録のフレームごとのゲイン・trim・先読みで車に出す前の値
    (drive の model_us = apply_gains(pred[lookahead − 1]))にして、記録の model_out [N,2] との差 [N,2]"""
    pred = np.asarray(pred, np.float64)
    k = np.clip(np.asarray(lookahead, np.int64), 1, pred.shape[1]) - 1
    raw = pred[np.arange(len(pred)), k]
    g = np.asarray(gains, np.float64)
    est = car_out(raw, ref, thr_gain=g[:, 1], steer_gain=g[:, 0], trim=np.asarray(trim, np.float64))
    return est - np.asarray(model_out, np.float64)


def stage0_2_summary(d, limit=STAGE0_2_LIMIT_US):
    """差 [N,2] → 中央値。決まりの「差の中央値」を 2 通り(ステア・スロットルを合わせた |差| と、片方ずつの |差| の大きい方)で出す。
    2 通りが limit の同じ側なら、その答え。分かれたら「決まりで決まらない」"""
    a = np.abs(np.asarray(d, np.float64))
    if not len(a):
        return {'frames': 0, 'pass': None, 'why': 'フレームが無い'}
    pooled = float(np.median(a.ravel()))
    per = [float(np.median(a[:, 0])), float(np.median(a[:, 1]))]
    worst = max(per)
    ok_p, ok_w = pooled <= limit, worst <= limit
    out = {'frames': int(len(a)), 'median_abs_us_pooled': pooled, 'median_abs_us_steer': per[0],
           'median_abs_us_throttle': per[1], 'median_signed_us': [float(np.median(d[:, 0])), float(np.median(d[:, 1]))],
           'p90_abs_us': [float(np.percentile(a[:, 0], 90)), float(np.percentile(a[:, 1], 90))], 'limit_us': limit}
    if ok_p == ok_w:
        out['pass'] = bool(ok_p)
    else:
        out['pass'] = None
        out['why'] = '決まりで決まらない(合わせた中央値と片方ずつの中央値で答えが分かれた)'
    return out


def time_run_numpy(runner, feed, warmup=TIME_WARMUP, runs=TIME_RUNS):
    """engine_load_check.py の速さと同じ測り方: 同じ入力で warmup 回の空回しの後、runs 回の run_numpy の時間 [ms]"""
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


def standstill_gate_kind(n_frames):
    """止まった場面の門: 300 フレーム以上なら門、未満なら参考"""
    if n_frames is None:
        return None
    return '門' if n_frames >= STANDSTILL_MIN_FRAMES else '参考'


def seal_guard(logs, sealed):
    """封の記録を読もうとしたら止まる(protocol seal.rules。この台本は 11/1 の封を開けない)"""
    bad = sorted({str(x)[:15] for x in logs} & set(sealed))
    if bad:
        raise SystemExit('封の記録は読みません(C3 は封を開けない): %s' % ' '.join(bad))
    if SEAL_OPENED.exists():
        raise SystemExit('seal_opened.json があります。11/1 のあとは、この台本の集合を作り直さない(main session に聞く)')


def scene_masks(d, k, has, src, tl, laps_b, de_entries, anchors, is_gb, is_1024_2, keep_bfast):
    """1 つのシーンの aligned のフレームの集合のマスク(numpy だけ。試験で作り物の d を入れる)。
    d: 記録のラベルの行の値(ai・near・erpm・straight・loc_detail・model_out・hold・run・colls)。k: ラベルの行の番号。
    has: label_index あり & vehicle_valid。src: aligned の source。tl: ラベルの時刻 [s]。
    laps_b: [(a, b, 周)]。de_entries: [(te, 周の番号 or −1)]。anchors: R1 の起点 [s]。keep_bfast: B-FAST の手本のフレーム"""
    n = len(k)
    kk = np.clip(k, 0, None)
    ai = has & (src == 1) & d['ai'][kk]
    near = d['near'][kk]
    out = {}
    colls = [c for c in d['colls'] if c.get('ai')]
    for name, rule in (('pre_r1', lambda c: c.get('loc') == 'R1'), ('pre_de', lambda c: c.get('loc') == 'DE'),
                       ('pre_sw', lambda c: c.get('loc_detail') == 'SW')):
        out[name] = ai & window_mask(tl, [c['t_s'] for c in colls if rule(c)], -PRE_S, 0.0)
    lapf = lap_of(tl, [(a, b) for a, b, _ in laps_b])
    unused = np.array([lap_unused(x) for _, _, x in laps_b] + [False], bool)
    st_ok = np.array([lap_straight(x) for _, _, x in laps_b] + [False], bool)
    dp = np.zeros(n, bool)
    for te, j in de_entries:
        if j >= 0 and unused[j]:
            dp |= (tl >= te + DE_PASS_S[0]) & (tl < te + DE_PASS_S[1]) & (lapf == j)
    out['de_pass'] = ai & dp
    out['straight'] = ai & d['straight'][kk] & st_ok[lapf]
    out['stand_pre'] = ai & (np.abs(d['erpm'][kk]) < STAND_ERPM) & ~near
    gb = ai & ~near & bool(is_gb)
    loc = d['loc_detail'][kk]
    out['gb'] = gb
    for name in ('DE', 'SW', 'HS'):
        out['gb_%s' % name.lower()] = gb & (loc == LOC_DETAIL[name])
    out['gb_dsh'] = out['gb_de'] | out['gb_sw'] | out['gb_hs']
    out['gb_r1pre'] = gb & window_mask(tl, anchors, -PRE_S, 0.0)
    out['gb_deline'] = gb & window_mask(tl, [te for te, _ in de_entries], DE_LINE_S[0], DE_LINE_S[1])
    gr = ai & ~near & bool(is_1024_2) & ~np.asarray(keep_bfast, bool)
    out['gbref'] = gr
    out['gbref_dsh'] = gr & np.isin(loc, [LOC_DETAIL['DE'], LOC_DETAIL['SW'], LOC_DETAIL['HS']])
    mo = d['model_out'][kk]
    out['s0_2'] = (ai & bool(is_1024_2) & (d['run'][kk] == B_RUN) & np.isfinite(mo).all(axis=1) & ~d['hold'][kk])
    return out


TAB_MASKS = ('pre_r1', 'pre_de', 'pre_sw', 'de_pass', 'straight', 'stand', 'gb', 'gb_de', 'gb_sw', 'gb_hs', 'gb_dsh',
             'gb_r1pre', 'gb_deline', 'gbref', 'gbref_dsh', 's0_2')
IN_TAB = ('pre_r1', 'pre_de', 'pre_sw', 'de_pass', 'straight', 'stand', 'gb', 'gbref', 's0_2')


def row_fields(pc, pb, refc, refb, M, family, trained, pc9=None, rows9=None):
    """1 本の学習の、AI の記録の集合の欄(numpy だけ)。pc・pb: 表の全部の行の予測 [N,10,2](ジャンル 5)。
    M: 表のマスク。family: 'v2'・'bfast'・'ref'。pc9: ジャンル 9 の予測(rows9 の行だけ)"""
    g5, gb = GAINS['g5'], GAINS['B']
    if family == 'bfast':
        g5 = GAINS['bfast']
    r = {'de_pre_thr_vs_b_us': thr_vs_b(pc, pb, refc, refb, M['pre_de'], g5, gb),
         'r1_pre_thr_vs_b_us': thr_vs_b(pc, pb, refc, refb, M['pre_r1'], g5, gb),
         'sw_pre_thr_vs_b_us': thr_vs_b(pc, pb, refc, refb, M['pre_sw'], g5, gb),
         'de_overshoot_vs_b_us': steer_vs_b(pc, pb, M['de_pass'], ROW0),
         'straight_thr_vs_b_us': thr_vs_b(pc, pb, refc, refb, M['straight'], g5, gb),
         'standstill_go_frac': go_frac(pc, refc, M['stand'], g5)}
    r['straight_fwd_vs_b_us'] = None if r['straight_thr_vs_b_us'] is None else -r['straight_thr_vs_b_us']
    r['lap_delta_est_s'], r['lap3_est_s'] = lap_est(r['straight_fwd_vs_b_us'])
    r['g9_minus_g5_pre_thr_us'] = None
    if pc9 is not None and 9 in list(trained):
        m = np.asarray(M['pre_r1'] | M['pre_de'], bool)
        sel = m[rows9]
        if sel.any():
            a = car_out(pc9[sel][:, ROW2], refc, GAINS['g9'])[:, 1]
            b = car_out(np.asarray(pc)[rows9][sel][:, ROW2], refc, GAINS['g5'])[:, 1]
            r['g9_minus_g5_pre_thr_us'] = float(np.mean(a - b))
    for k in ('bfast_gb_fwd_vs_b_us', 'bfast_gb_fwd_vs_b_us_de', 'bfast_gb_fwd_vs_b_us_sw', 'bfast_gb_fwd_vs_b_us_hs',
              'bfast_gc_fwd_vs_b_us', 'bfast_gb_ref_fwd_vs_b_us', 'bfast_gb_fwd_g100_vs_b090_us', 'bfast_de_line_vs_b_us'):
        r[k] = None
    if family == 'bfast':
        gbf = GAINS['bfast']
        r['bfast_gb_fwd_vs_b_us'] = fwd_vs_b(pc, pb, refc, refb, M['gb_dsh'], gbf, gb)
        for z in ('de', 'sw', 'hs'):
            r['bfast_gb_fwd_vs_b_us_%s' % z] = fwd_vs_b(pc, pb, refc, refb, M['gb_%s' % z], gbf, gb)
        r['bfast_gc_fwd_vs_b_us'] = fwd_vs_b(pc, pb, refc, refb, M['gb_r1pre'], gbf, gb)
        r['bfast_gb_ref_fwd_vs_b_us'] = fwd_vs_b(pc, pb, refc, refb, M['gbref_dsh'], gbf, gb)
        r['bfast_gb_fwd_g100_vs_b090_us'] = fwd_vs_b(pc, pb, refc, refb, M['gb_dsh'], 1.00, gb)
        r['bfast_de_line_vs_b_us'] = steer_vs_b(pc, pb, M['gb_deline'], ROW0)
    return r


# ================================================================ protocol・固定

def protocol(check=True):
    if check:
        r = subprocess.run(['sha256sum', '-c', 'protocol_v2.sha256'], cwd=str(REP), capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit('protocol の固定が合いません:\n%s%s' % (r.stdout, r.stderr))
    return load_json(PROTOCOL)


def lock_info():
    return {'protocol_v2_sha256': sha256(PROTOCOL), 'lock2_sha256': sha256(LOCK2) if LOCK2.exists() else None,
            'script': str(Path(__file__).resolve()), 'script_sha256': sha256(Path(__file__).resolve())}


def ai_logs_of(proto):
    return [x for v in proto['tables']['dataset']['ai_logs'].values() for x in v]


def run_family(name):
    if name.startswith('bfast_20261025-'):
        return 'bfast'
    if name.startswith('v2_20261025-'):
        return 'v2'
    return 'ref'


def allowed_run(name, proto):
    import eval_v2 as E
    refs = {r[1] for r in E.FIVE} | {ENS_RUN}
    return name in refs or any(p.match(name) for p in E.protocol_patterns(proto))


def model_cfg(name):
    cfg = load_json(RUNS / name / 'model_config.json')
    if not cfg:
        raise SystemExit('%s の model_config.json がありません' % name)
    return cfg


def ref_of(cfg):
    return (float(cfg['steer_center_us']), float(cfg['throttle_neutral_us']))


# ================================================================ 集合(CPU)

EXTRA = '''
    # ---- desk_v2(C3)で足した取り出し(ここから。make_v2_tables.py の EXTRA と同じ場所)
    PC_OUT[log] = {'session': session, 't_s': t.copy(), 't_us': t_us.copy(), 'ai': ai.copy(), 'near': near.copy(),
                   'erpm': erpm.copy(), 'straight': straight.copy(), 'anchors': np.asarray(anchors, float).copy(),
                   'laps': [dict(x) for x in laps], 'colls': [dict(c) for c in colls],
                   'run': np.array([((name2row.get(s) or {}).get('run') or '') for s in sc_name]),
                   'loc_detail': lab['loc_detail'].astype(np.int8).copy(),
                   'model_out': lab['model_out'].astype(np.float64).copy(), 'gains': gains.copy(),
                   'trim': lab['trim'].astype(np.float64).copy(), 'lookahead': lab['lookahead'].astype(np.int64).copy(),
                   'hold': lab['hold'].astype(bool).copy(),
                   'loc_detail_names': list(info.get('loc_detail_names') or [])}
    # ---- desk_v2(C3)で足した取り出し(ここまで)
'''


def _fast_scene_span(sdir):
    """明るさの画像を読まない scene_brightness(plan_checks.py・make_v2_tables.py と同じ)"""
    try:
        with np.load(sdir / 'CAM_FRONT.npz', allow_pickle=False) as z:
            ts = z['timestamps_us'].astype(np.int64)
    except Exception:          # noqa: BLE001
        names = sorted(os.listdir(sdir / 'CAM_FRONT')) if (sdir / 'CAM_FRONT').is_dir() else []
        names = [x for x in names if x.endswith('.jpg') and x[:-4].isdigit()]
        ts = np.array([int(x[:-4]) for x in names], np.int64)
    idx = np.arange(0, len(ts), 3)
    span = (int(ts[0]), int(ts[-1])) if len(ts) else None
    nan = np.full(len(idx), np.nan)
    return (ts[idx] if len(ts) else ts), nan, nan.copy(), len(ts), span


def extract_logs(logs, sealed):
    """inventory.py の process_log の写し(make_v2_tables.py と同じ方法)で、記録ごとのラベルの行の値を取り出す"""
    seal_guard(logs, sealed)
    spec = importlib.util.spec_from_file_location('inv_copy_c3', str(INV_PY))
    inv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inv)
    inv.scene_brightness = _fast_scene_span
    src = inspect.getsource(inv.process_log)
    anchor = "    rec['candidates'] = {"
    if src.count(anchor) != 1:
        raise SystemExit('inventory.process_log の形が変わった')
    ns = inv.__dict__
    ns.update(PC_OUT={})
    exec(compile(src.replace(anchor, EXTRA + anchor), str(INV_PY) + ' (desk_v2 の写し)', 'exec'), ns)
    modes = inv.mode_table()
    membership = inv.dataset_membership()
    all_logs = sorted(d for d in os.listdir(inv.RAW) if d[:8] in inv.DAYS and (inv.RAW / d).is_dir())
    sess = inv.sessions(all_logs)              # 記録の名前(時刻)だけで回を決める(中身は読まない)
    for lg in logs:
        ns['process_log'](lg, membership, modes, sess.get(lg))
        if lg not in ns['PC_OUT']:
            raise SystemExit('%s の取り出しができません(ラベルか記録が読めない)' % lg)
        log('  取り出し', lg, sess.get(lg))
    return ns['PC_OUT']


def lap_bounds(d):
    """inventory の AI の周 → [(a, b, 周)]。a・b は起点そのもの(make_v2_tables.lap_rows と同じ照らし)"""
    A = np.asarray(d['anchors'], np.float64)
    out = []
    for x in d['laps']:
        if x['who'] != 'ai':
            continue
        j = int(np.argmin(np.abs(A - x['t_start'])))
        if abs(A[j] - x['t_start']) > 2e-3 or j + 1 >= len(A) or abs((A[j + 1] - A[j]) - x['lap_s']) > 2e-3:
            raise SystemExit('周 %.3f の起点が見つかりません' % x['t_start'])
        out.append((float(A[j]), float(A[j + 1]), x))
    return out


def kept_tables(scenes):
    """表 drop_v2h・drop_v2m・drop_bfast で学習に残したフレーム(シーン → bool)"""
    out = {}
    for name in ('drop_v2h', 'drop_v2m', 'drop_bfast'):
        with np.load(TABLES / ('%s.npz' % name), allow_pickle=False) as z:
            sc, fr, dr = z['scene'].astype(str), z['frame'].astype(np.int64), z['drop'].astype(bool)
        want = np.isin(sc, list(scenes))
        tab = {}
        for s in np.unique(sc[want]):
            m = want & (sc == s)
            arr = np.zeros(int(fr[m].max()) + 1, bool)
            arr[fr[m]] = ~dr[m]
            tab[str(s)] = arr
        out[name] = tab
    return out


def session_key(sess):
    if sess.startswith('10/10'):
        return '10/10'
    if sess.startswith('10/23'):
        return '10/23'
    if sess == SESS_1024_2:
        return '10/24 2 回目'
    raise SystemExit('学習の回ではない記録です: %s' % sess)


def cmd_sets(args):
    from jetformer.tof_guard import ToFGuard
    t0 = time.time()
    proto = protocol()
    sealed = proto['seal']['logs']
    logs = ai_logs_of(proto)
    card = load_json(DATASETS / DS_V2 / 'dataset_card.json')
    splits = load_json(DATASETS / DS_V2 / 'splits.json')
    if card['ai_mix']['ai_logs'] != logs:
        raise SystemExit('データセットの ai_logs が protocol の 46 本と違います')
    seal_guard(logs, sealed)
    po = extract_logs(logs, sealed)
    names = set()
    for lg, d in po.items():
        names.add(tuple(d['loc_detail_names']))
    if names != {tuple(LOC_DETAIL)}:
        raise SystemExit('ラベルの loc_detail_names が label_events.LOC_DETAIL と違います: %s' % names)
    scenes = sorted(s for s in splits['train'] if s.rsplit('-', 1)[0] in set(logs))
    seal_guard([s[:15] for s in scenes], sealed)
    kept = kept_tables(scenes)
    guard = ToFGuard(np.load(TOF_REF))
    rows = {k: [] for k in ('scene', 'frame', 'log', 't_s', 't_us', 'sess', 'loc_detail', 'erpm_label', 'model_out',
                            'gains', 'trim', 'lookahead', 'x_free') + TAB_MASKS}
    san = {'scene': [], 'frame': [], 't_us': []}
    cnt = {'stand_pre_aligned': 0, 'stand_pre_by_sess': {}, 'stand_by_sess': {}, 'overlap_train': {}}
    pre = {}
    for lg in logs:
        d = po[lg]
        pre[lg] = {'bounds': lap_bounds(d)}
        te_idx = run_starts(d['loc_detail'], LOC_DETAIL['DE'])
        te = d['t_s'][te_idx]
        lap_te = lap_of(te, [(a, b) for a, b, _ in pre[lg]['bounds']])
        pre[lg]['de_entries'] = list(zip(te.tolist(), lap_te.tolist()))
    for sc in scenes:
        lg = sc.rsplit('-', 1)[0]
        d = po[lg]
        with np.load(DATASETS / DS_V2 / 'aligned' / (sc + '.npz'), allow_pickle=False) as z:
            ts = z['timestamps_us'].astype(np.int64)
            src = z['source'].astype(np.int64)
            li = z['label_index'].astype(np.int64)
            vv = z['vehicle_valid'].astype(bool)
            tof = z['tof']
        n = len(ts)
        has = (li >= 0) & vv
        k = np.clip(li, 0, None)
        if has.any() and (k.max() >= len(d['t_s']) or not np.array_equal(ts[has], d['t_us'][k[has]])):
            raise SystemExit('%s: aligned の時刻と label_index の行の時刻が違います' % sc)
        tl = np.where(li >= 0, d['t_s'][k], np.nan)
        sk = session_key(d['session'])
        kb = kept['drop_bfast'].get(sc, np.zeros(0, bool))
        keep_bf = np.zeros(n, bool)
        keep_bf[:min(n, len(kb))] = kb[:n]
        M = scene_masks(d, li, has, src, tl, pre[lg]['bounds'], pre[lg]['de_entries'], d['anchors'],
                        lg in GB_LOGS, sk == '10/24 2 回目', keep_bf)
        xf = np.full(n, np.nan, np.float32)
        for i in np.nonzero(M['stand_pre'])[0]:
            xf[i] = guard.features(tof[i], 0.0, 0.0, guard.center)['x_free']
        M['stand'] = M['stand_pre'] & (xf > TOF_FREE_M)
        cnt['stand_pre_aligned'] += int(M['stand_pre'].sum())
        cnt['stand_pre_by_sess'][sk] = cnt['stand_pre_by_sess'].get(sk, 0) + int(M['stand_pre'].sum())
        cnt['stand_by_sess'][sk] = cnt['stand_by_sess'].get(sk, 0) + int(M['stand'].sum())
        trained_any = np.zeros(n, bool)
        for tname, tab in kept.items():
            a = tab.get(sc)
            if a is not None:
                trained_any[:min(n, len(a))] |= a[:n]
        for mname in TAB_MASKS:
            ov = int((M[mname] & trained_any).sum())
            cnt['overlap_train'][mname] = cnt['overlap_train'].get(mname, 0) + ov
        sel = np.zeros(n, bool)
        for mname in IN_TAB:
            sel |= M[mname]
        idx = np.nonzero(sel)[0]
        rows['scene'] += [sc] * len(idx)
        rows['frame'].append(idx.astype(np.int32))
        rows['log'] += [lg] * len(idx)
        rows['t_s'].append(tl[idx])
        rows['t_us'].append(ts[idx])
        rows['sess'].append(np.full(len(idx), SESS_CODE[sk], np.int8))
        rows['loc_detail'].append(d['loc_detail'][k[idx]].astype(np.int8))
        rows['erpm_label'].append(d['erpm'][k[idx]].astype(np.float32))
        rows['model_out'].append(d['model_out'][k[idx]].astype(np.float32))
        rows['gains'].append(d['gains'][k[idx]].astype(np.float32))
        rows['trim'].append(d['trim'][k[idx]].astype(np.float32))
        rows['lookahead'].append(d['lookahead'][k[idx]].astype(np.int16))
        rows['x_free'].append(xf[idx])
        for mname in TAB_MASKS:
            rows[mname].append(M[mname][idx])
        if sk == '10/24 2 回目':
            s_idx = np.nonzero(vv & (src == 1))[0]
            san['scene'] += [sc] * len(s_idx)
            san['frame'].append(s_idx.astype(np.int32))
            san['t_us'].append(ts[s_idx])
    for mname in ('de_pass', 'straight'):        # protocol sets: 「学習に使っていない」周だけ
        if cnt['overlap_train'].get(mname):
            raise SystemExit('止める: 「学習に使っていない」はずの集合 %s に、表で学習に残したフレームが %d あります'
                             % (mname, cnt['overlap_train'][mname]))
    arr = {}
    for k_, v in rows.items():
        if k_ in ('scene', 'log'):
            arr[k_] = np.array(v)
        else:
            arr[k_] = np.concatenate(v) if v else np.zeros(0)
    key = hashlib.sha1('|'.join('%s:%d' % (s, f) for s, f in zip(arr['scene'], arr['frame'])).encode()).hexdigest()
    sarr = {'san_scene': np.array(san['scene']), 'san_frame': np.concatenate(san['frame']),
            'san_t_us': np.concatenate(san['t_us'])}
    save_npz(SETS_NPZ, key=np.array(key), **arr, **sarr)
    # 数
    coll = {'all_ai': 0, 'R1': 0, 'DE': 0, 'SW': 0, 'by_sess': {}}
    label_stand = {}
    laps = {'ai': 0, 'unused': 0, 'straight_ok': 0, 'b_clean_lt_9_5': 0, 'clean': 0}
    for lg in logs:
        d = po[lg]
        sk = session_key(d['session'])
        c_ai = [c for c in d['colls'] if c.get('ai')]
        coll['all_ai'] += len(c_ai)
        coll['by_sess'][sk] = coll['by_sess'].get(sk, 0) + len(c_ai)
        coll['R1'] += sum(1 for c in c_ai if c.get('loc') == 'R1')
        coll['DE'] += sum(1 for c in c_ai if c.get('loc') == 'DE')
        coll['SW'] += sum(1 for c in c_ai if c.get('loc_detail') == 'SW')
        st = d['ai'] & ~d['near'] & (np.abs(d['erpm']) < STAND_ERPM)
        label_stand[sk] = label_stand.get(sk, 0) + int(st.sum())
        for _, _, x in pre[lg]['bounds']:
            laps['ai'] += 1
            laps['clean'] += int(bool(x['clean']))
            laps['unused'] += int(lap_unused(x))
            laps['straight_ok'] += int(lap_straight(x))
            laps['b_clean_lt_9_5'] += int(bool(x['clean']) and x.get('model') == 'B' and x['lap_s'] < LAP_B_UNUSED_S)

    def by_sess(m):
        return {name: int((arr[m] & (arr['sess'] == code)).sum()) for name, code in SESS_CODE.items()}
    counts = {m: {'frames': int(arr[m].sum()), 'by_session': by_sess(m)} for m in TAB_MASKS}
    nst = int(arr['stand'].sum())
    out = {
        'schema': 'jf-c3-sets-v2/v1', 'made_at': time.strftime('%Y-%m-%d %H:%M:%S'), **lock_info(),
        'key': key, 'table_rows': int(len(arr['frame'])), 'sanity_1024_frames': int(len(sarr['san_frame'])),
        'counts': counts,
        'collisions_ai': coll, 'collisions_ai_protocol': 207,
        'laps_ai': laps,
        'standstill': {'label_rows_before_tof_by_session': label_stand,
                       'label_rows_before_tof': int(sum(label_stand.values())), 'protocol_upper_bound': 1622,
                       'aligned_before_tof': cnt['stand_pre_aligned'], 'aligned_before_tof_by_session': cnt['stand_pre_by_sess'],
                       'aligned_after_tof': nst, 'aligned_after_tof_by_session': cnt['stand_by_sess'],
                       'gate': standstill_gate_kind(nst), 'rule': 'x_free > %.1f m(ToFGuard.features、erpm 0・gz 0・ステア中心)' % TOF_FREE_M},
        'overlap_with_training_tables': cnt['overlap_train'],
        'inputs_sha256': {'dataset_card': sha256(DATASETS / DS_V2 / 'dataset_card.json'),
                          'splits': sha256(DATASETS / DS_V2 / 'splits.json'),
                          'drop_v2h': sha256(TABLES / 'drop_v2h.npz'), 'drop_v2m': sha256(TABLES / 'drop_v2m.npz'),
                          'drop_bfast': sha256(TABLES / 'drop_bfast.npz'), 'tof_floor_ref': sha256(TOF_REF),
                          'inventory.py': sha256(INV_PY),
                          'labels': {lg: [sha256(LABELS / (lg + '.npz')), sha256(LABELS / (lg + '.json'))] for lg in logs}},
        'sets_npz_sha256': sha256(SETS_NPZ),
        'seconds': round(time.time() - t0, 1),
    }
    write_json(SETS_JSON, out)
    log('集合を書きました: %s(表 %d 行・止まった場面 %d フレーム・%s)' % (SETS_NPZ, len(arr['frame']), nst, out['standstill']['gate']))
    log(json.dumps({'collisions_ai': coll, 'counts': {m: c['frames'] for m, c in counts.items()},
                    'standstill': out['standstill']}, ensure_ascii=False))
    return 0


def load_sets():
    if not SETS_NPZ.exists():
        raise SystemExit('先に sets を動かしてください(%s が無い)' % SETS_NPZ)
    with np.load(SETS_NPZ, allow_pickle=False) as z:
        S = {k: z[k] for k in z.files}
    S['key'] = str(S['key'])
    return S


# ================================================================ 予測(GPU)

class _Frames(object):
    """予測するフレーム(画像・ToF・IMU・車両の値)。torch の Dataset として使う(tools/policy_zoo._Frames と同じ中身 + ずれ)"""

    def __init__(self, items, use_command=False, pert=None):
        self.items, self.use_command, self.pert = items, use_command, pert

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
        if self.pert:
            img, tof = perturb(self.pert, img, tof)
        img, t, im, st = normalize_inputs(img, tof, imu, veh, self.use_command)
        return torch.from_numpy(img), torch.from_numpy(t), torch.from_numpy(im), torch.from_numpy(st)


def frame_items(ds, scenes, frames):
    """(データセット, シーン, フレーム) → [(画像のパス, ToF, IMU, 車両, ジャンル)]。シーンの順に読む"""
    items, cur, a = [], None, None
    for s, i in zip(scenes, frames):
        s = str(s)
        if s != cur:
            with np.load(DATASETS / ds / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
                a = {k: z[k] for k in ('cam_files', 'tof', 'imu', 'vehicle', 'mode')}
            cur = s
        i = int(i)
        items.append((str(DATASETS / ds / str(a['cam_files'][i])), a['tof'][i].copy(), a['imu'][i].copy(),
                      a['vehicle'][i].copy(), int(a['mode'][i])))
    return items


def saved_best(name):
    """eval_v2.py の保存した best の予測(10,985 フレームの表)。5 本は data/zoo の保存。無ければ None"""
    import eval_v2 as E
    for rid, run, pb, pl, seed in E.FIVE:
        if run == name and Path(pb).exists():
            return E.load_pred(pb)
    p = EVAL_PRED / ('%s.npz' % name)
    return E.load_pred(p, 'best') if p.exists() else None


def tab_file(name):
    return PRED / ('%s_tab.npz' % name)


def shift_file(name):
    return PRED / ('%s_shift.npz' % name)


def cmd_predict(args):
    proto = protocol()
    names = list(dict.fromkeys(args.runs or []))
    if not names:
        raise SystemExit('--runs を付けてください')
    bad = [n for n in names if not allowed_run(n, proto)]
    if bad:
        raise SystemExit('protocol の runs・5 本・2 本の平均の名前ではありません: %s' % ' '.join(bad))
    if any(run_family(n) != 'ref' for n in names) and not LOCK2.exists():
        raise SystemExit('protocol_v2_lock2.json がまだ無いので、v2・B-FAST の予測はしません')
    seal_guard([], proto['seal']['logs'])
    import gc
    import torch
    import policy_zoo as pz
    import rl_evaluate as ev
    import rl_policies as rp
    from jetformer.policies import runtime
    rp._check_gpu_free(min_gb=2.0)
    S = load_sets() if args.what in ('tab', 'all') else None
    if S is not None:
        seal_guard(S['log'], proto['seal']['logs'])
    for name in names:
        run_dir = RUNS / name
        if not (run_dir / 'best.pth').exists():
            log('%s: best.pth がありません(飛ばす)' % name)
            continue
        cfg = model_cfg(name)
        trained = np.array([int(x) for x in cfg['trained_modes']])
        use_cmd = bool(cfg.get('command_input', True))
        meta = dict(run=np.array(name), ckpt=np.array('best'), model_config_sha256=np.array(sha256(run_dir / 'model_config.json')),
                    best_sha256=np.array(sha256(run_dir / 'best.pth')), script_sha256=np.array(sha256(Path(__file__).resolve())))
        model = None
        if args.what in ('tab', 'all'):
            old_key = None
            if tab_file(name).exists():
                with np.load(tab_file(name)) as z:
                    old_key = str(z['key'])
            if old_key == S['key'] and not args.force:
                log('%s: tab の予測はあります' % name)
            else:
                if model is None:
                    model = runtime.load_run(run_dir, 'best', device='cuda')[0]
                t0 = time.time()
                items = frame_items(DS_V2, S['scene'], S['frame'])
                g5, sub5 = ev._genres(np.full(len(items), SANITY_GENRE), None, trained)
                res = pz._predict_frames(model, _Frames(items, use_cmd), g5, cfg['num_modes'], args.batch, args.workers)
                extra = {}
                if 9 in trained.tolist():
                    rows9 = np.nonzero(S['pre_r1'] | S['pre_de'])[0]
                    g9, _ = ev._genres(np.full(len(rows9), 9), None, trained)
                    r9 = pz._predict_frames(model, _Frames([items[i] for i in rows9], use_cmd), g9, cfg['num_modes'],
                                            args.batch, args.workers)
                    extra = {'rows_g9': rows9.astype(np.int64), 'pred_g9': r9['actions'].astype(np.float32)}
                pred = res['actions'].astype(np.float32)
                if pred.shape != (len(items), 10, 2) or not np.isfinite(pred).all():
                    raise SystemExit('%s: 予測の形か値がおかしい' % name)
                save_npz(tab_file(name), key=np.array(S['key']), pred=pred, genre=g5.astype(np.int8),
                         substituted=sub5, **extra, **meta)
                log('%s: tab %d フレーム(ジャンル 9 %s)%.0f 秒' % (name, len(pred), len(extra.get('rows_g9', [])) if extra else 'なし',
                                                          time.time() - t0))
        if args.what in ('shift', 'all'):
            if shift_file(name).exists() and not args.force:
                log('%s: ずれの試しの予測はあります' % name)
            else:
                if model is None:
                    model = runtime.load_run(run_dir, 'best', device='cuda')[0]
                t0 = time.time()
                with np.load(BUNDLE, allow_pickle=False) as z:
                    if str(z['key']) != TABLE_KEY:
                        raise SystemExit('desk_inputs の key が違います')
                    part, scene, frame, prim = z['part'], z['scene'].astype(str), z['frame'], z['primary'].astype(bool)
                rows = np.nonzero(prim)[0]
                if (part[rows] != 0).any():
                    raise SystemExit('H1 に part 0 でない行があります')
                items = frame_items(DS_1010, scene[rows], frame[rows])
                genres, sub = ev._genres([it[4] for it in items], None, trained)
                base = saved_best(name)
                out = {'rows': rows.astype(np.int64), 'genre': genres.astype(np.int8), 'key': np.array(TABLE_KEY)}
                chk = np.arange(min(SHIFT_CHECK_ROWS, len(rows))) if base is not None else np.arange(len(rows))
                rn = pz._predict_frames(model, _Frames([items[i] for i in chk], use_cmd), genres[chk], cfg['num_modes'],
                                        args.batch, args.workers)['actions']
                out['pred_none_rows'] = chk.astype(np.int64)
                out['pred_none'] = rn.astype(np.float32)
                if base is not None:
                    mx = float(np.abs(rn.astype(np.float64) - base[rows][chk]).max())
                    out['none_vs_saved_max_abs_us'] = np.array(mx)
                    if mx > SHIFT_PIPE_LIMIT_US:      # 入力・ジャンルの作り方が保存と違う(バッチの組み方の数値の揺れより大きい)
                        raise SystemExit('%s: ずれの試しの「そのまま」が保存した予測と %.4f µs 違います(> %.1f)'
                                         % (name, mx, SHIFT_PIPE_LIMIT_US))
                for kind in SHIFTS:
                    r = pz._predict_frames(model, _Frames(items, use_cmd, pert=kind), genres, cfg['num_modes'],
                                           args.batch, args.workers)['actions']
                    out['pred_' + kind] = r.astype(np.float32)
                save_npz(shift_file(name), **out, **meta)
                log('%s: ずれの試し %d × %d フレーム %.0f 秒(そのままの照らし %s)' % (
                    name, len(SHIFTS), len(rows), time.time() - t0,
                    ('%.4f µs' % float(out['none_vs_saved_max_abs_us'])) if base is not None else '保存なし・全部予測'))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return 0


# ================================================================ TensorRT(GPU)

def sanity_frames(S):
    """sanity の集合: 10/10 の AI 走行(desk_gpu.ai_frames と同じ 21,759)+ 10/24 2 回目(sets の san_*)。
    → [(データセット, シーン, フレーム, 時刻 µs, 部分 0=10/10・1=10/24)]"""
    import desk_gpu
    rows = [(DS_1010, s, i, t, 0) for s, i, t in desk_gpu.ai_frames()]
    rows += [(DS_V2, str(s), int(i), int(t), 1) for s, i, t in zip(S['san_scene'], S['san_frame'], S['san_t_us'])]
    return rows


def cmd_trt(args):
    import cv2
    import torch
    from jetformer import config, drive, policies
    from jetformer.policies import post as post_mod
    from jetformer.train_data import normalize_inputs
    from jetformer.trt_runtime import TRTRunner
    import rl_policies as rp
    import desk_lib as DL
    proto = protocol()
    seal_guard([], proto['seal']['logs'])
    names = [B_RUN] + [n for n in dict.fromkeys(args.runs or []) if n != B_RUN]
    if len(names) > 1 + args.max_runs:
        raise SystemExit('1 回の仕事は候補 %d 本まで(短く分ける)' % args.max_runs)
    bad = [n for n in names if not allowed_run(n, proto)]
    if bad:
        raise SystemExit('protocol の runs・5 本・2 本の平均の名前ではありません: %s' % ' '.join(bad))
    S = load_sets()
    seal_guard(S['log'], proto['seal']['logs'])
    rp._check_gpu_free(min_gb=2.0)
    rows = sanity_frames(S)
    if args.limit:
        rows = rows[:args.limit // 2] + [r for r in rows if r[4] == 1][:args.limit - args.limit // 2]
    job = time.strftime('%Y%m%d-%H%M%S')
    log('TensorRT の仕事 %s: エンジン %d 本・フレーム %d(10/10 %d・10/24 2 回目 %d)' % (
        job, len(names), len(rows), sum(1 for r in rows if r[4] == 0), sum(1 for r in rows if r[4] == 1)))
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    params = dict(drive.DEFAULT_PARAMS)
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
    cur, a, prev_scene = None, None, None
    erpm = np.zeros(len(rows))
    t_start = time.time()
    for k, (ds, s, i, t_us, prt) in enumerate(rows):
        if (ds, s) != cur:
            with np.load(DATASETS / ds / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
                a = {n: z[n] for n in ('cam_files', 'tof', 'imu', 'vehicle')}
            cur = (ds, s)
        img = cv2.imread(str(DATASETS / ds / str(a['cam_files'][i])))
        erpm[k] = float(a['vehicle'][i, 3])
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
    # 推論の速さ: sanity の流れの時間は、フレームの読み込みのあいだに GPU が休むので遅く出る(B だけの仕事で 中央 4.1 ms)。
    # desk_bench と同じ定義(engine_check.json の run_numpy = 100 回の空回し + 600 回)で、B → 候補 → B の順に同じ回に測る
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
    summary = {'schema': 'jf-c3-trt/v2', 'timing': 'engine_check と同じ(100 回の空回し + 600 回・同じ実フレーム・B → 候補 → B)', 'job': job, 'day': job[:8], 'made_at': time.strftime('%Y-%m-%d %H:%M:%S'),
               **lock_info(), 'frames': len(rows), 'frames_1010': int((part == 0).sum()), 'frames_1024_2': int((part == 1).sum()),
               'limit': args.limit or None, 'mode_id': SANITY_GENRE,
               'params': {k: params[k] for k in sorted(BASE_PARAMS_OVERRIDE)}, 'drive_range_us': [lo, hi],
               'seconds': round(time.time() - t_start, 1), 'engines': {}}
    d_dir = TRT / job
    for name, e in E.items():
        st = e['st']
        if 'runner' in e:
            p, mp = e['pred'], e['model_pred']
            bad = ~np.isfinite(p).all(axis=(1, 2)) | ~np.isfinite(mp).all(axis=(1, 2))
            oor = int(((p < SANITY_LO) | (p > SANITY_HI)).sum() + ((mp < SANITY_LO) | (mp > SANITY_HI)).sum())
            lmed, lp99 = trt_times(e['ms'])
            med, p99 = trt_times(times[name][0]) if name in times else (None, None)
            if name in times and len(times[name]) > 1:
                st['run_numpy_after_median_ms'], st['run_numpy_after_p99_ms'] = trt_times(times[name][-1])
            st.update({'sanity_loop_median_ms': lmed, 'sanity_loop_p99_ms': lp99,'frames': len(rows), 'nonfinite_out_frames': e['nonfinite_out'], 'post_error_frames': e['post_error'],
                       'nonfinite_frames': int(bad.sum() + e['nonfinite_out'] + e['post_error']),
                       'out_of_range_values': oor, 'sanity_range_us': [SANITY_LO, SANITY_HI],
                       'outside_drive_range_values': int(((p < lo) | (p > hi)).sum()), 'values': int(p.size),
                       'by_part': {nm: {'frames': int((part == c).sum()),
                                        'nonfinite_frames': int(bad[part == c].sum()),
                                        'out_of_range_values': int(((p[part == c] < SANITY_LO) | (p[part == c] > SANITY_HI)).sum()
                                                                   + ((mp[part == c] < SANITY_LO) | (mp[part == c] > SANITY_HI)).sum())}
                                   for nm, c in (('10/10', 0), ('10/24 2 回目', 1))},
                       'steer_min_us': float(np.nanmin(p[:, :, 0])), 'steer_max_us': float(np.nanmax(p[:, :, 0])),
                       'thr_min_us': float(np.nanmin(p[:, :, 1])), 'thr_max_us': float(np.nanmax(p[:, :, 1])),
                       'run_numpy_median_ms': med, 'run_numpy_p99_ms': p99})
            f = d_dir / ('%s.npz' % name)
            save_npz(f, pred=p, model_pred=mp, ms=e['ms'], ms_timed=np.asarray(times.get(name, [[]])[0], np.float32),
                     erpm=erpm.astype(np.float32), part=part,
                     run=np.array(name), mode_id=np.array(SANITY_GENRE), job=np.array(job))
            st['file'] = str(f)
        st['sanity_ok'] = DL.sanity_of(st)
        summary['engines'][name] = st
        log('%s: sanity_ok %s 中央 %s ms・p99 %s ms %s' % (name, st['sanity_ok'], rnd(st.get('run_numpy_median_ms'), 3),
                                                       rnd(st.get('run_numpy_p99_ms'), 3), st.get('error') or ''))
    write_json(TRT / ('%s.json' % job), summary)
    log('書きました: %s' % (TRT / ('%s.json' % job)))
    return 0


# ================================================================ 欄と表(CPU)

def trt_results():
    """trt/*.json → {学習: [(仕事の json, エンジンの st)]}(新しい順)"""
    out = {}
    for p in sorted(TRT.glob('*.json'), reverse=True) if TRT.exists() else []:
        j = load_json(p) or {}
        if j.get('limit') or j.get('schema') != 'jf-c3-trt/v2':
            continue                       # 試し(--limit)と、速さを sanity の流れで測った最初の版(v1)は欄に使わない
        for name, st in (j.get('engines') or {}).items():
            out.setdefault(name, []).append((j, st))
    return out


def engine_current(name, st):
    eng = RUNS / name / ENGINE
    if not eng.exists() or not st.get('engine'):
        return False
    s = eng.stat()
    return int(st['engine'].get('size', -1)) == s.st_size and int(st['engine'].get('mtime_ns', -1)) == s.st_mtime_ns


def trt_fields(name, T, bundle, saved):
    """TensorRT の欄(sanity_ok・trt_*・brake_lowspeed_frac)と B の同じ回・同じ日の p99"""
    import desk_lib as DL
    out = {'sanity_ok': None, 'trt_median_ms': None, 'trt_p99_ms': None, 'brake_lowspeed_frac': None, 'trt_job': None,
           'b_trt_p99_ms_same_job': None, 'b_trt_p99_ms_same_day': None, 'inference_ok_same_day': None, 'trt_notes': []}
    lst = [(j, st) for j, st in T.get(name, []) if engine_current(name, st)]
    if not lst:
        out['trt_notes'].append('今のエンジンの TensorRT の仕事がまだ')
        return out
    j, st = lst[0]
    out['trt_job'] = j['job']
    out['sanity_ok'] = DL.sanity_of(st)
    out['sanity_detail'] = {k: st.get(k) for k in ('frames', 'check_io_ok', 'nonfinite_frames', 'out_of_range_values',
                                                     'outside_drive_range_values', 'steer_min_us', 'steer_max_us',
                                                     'thr_min_us', 'thr_max_us', 'by_part', 'error')}
    out['trt_median_ms'], out['trt_p99_ms'] = st.get('run_numpy_median_ms'), st.get('run_numpy_p99_ms')
    b = (j.get('engines') or {}).get(B_RUN) or {}
    out['b_trt_p99_ms_same_job'] = b.get('run_numpy_p99_ms')
    same_day = [s.get('run_numpy_p99_ms') for jj, s in T.get(B_RUN, []) if jj['day'] == j['day'] and s.get('run_numpy_p99_ms')]
    out['b_trt_p99_ms_same_day'] = float(np.median(same_day)) if same_day else None
    if out['trt_p99_ms'] is not None and out['b_trt_p99_ms_same_day']:
        out['inference_ok_same_day'] = bool(out['trt_p99_ms'] <= INFER_FACTOR * out['b_trt_p99_ms_same_day'])
    # 低速のブレーキ(10/21 と同じ: 保留の FP32 best + 10/10 の AI 走行の TensorRT。中立 1498 + 100 µs)
    f = Path(st.get('file') or '')
    if f.exists() and saved is not None:
        with np.load(f) as z:
            pr, erpm, part = z['pred'].astype(np.float64), z['erpm'].astype(np.float64), z['part']
        v_hold = DL.speed_mps(bundle['erpm'])
        low_hold = bundle['primary'] & np.isfinite(v_hold) & (v_hold < DL.LOWSPEED_MPS)
        kh, nh = DL.brake_lowspeed(saved[:, 0, 1], v_hold, low_hold)
        m0 = part == 0
        ka, na = DL.brake_lowspeed(pr[m0][:, 0, 1], DL.speed_mps(erpm[m0]), np.isfinite(pr[m0][:, 0, 1]))
        kb, nb = DL.brake_lowspeed(pr[~m0][:, 0, 1], DL.speed_mps(erpm[~m0]), np.isfinite(pr[~m0][:, 0, 1]))
        out['brake_lowspeed_frac'] = DL.frac(kh + ka, nh + na)
        out['brake_lowspeed'] = {'heldout': [kh, nh], 'ai_1010': [ka, na],
                                 'ref_ai_1024_2(参考。定義に入れない)': [kb, nb], 'neutral_plus_us': DL.NEUTRAL_US + DL.BRAKE_MARGIN_US}
    else:
        out['trt_notes'].append('低速のブレーキ: 保留の予測(eval/pred)か TensorRT のファイルが無い')
    return out


def ens_row(a, b, tabs, M, refB, T, bundle):
    """2 本の平均(D4 の予備): 2 本の保存した予測の平均で欄を出す(eval_v2.d4_ensemble と同じく予測の平均。
    TensorRT は 2 本のエンジンの出力のフレームごとの平均で sanity・低速のブレーキ。推論の速さは出さない(D4 は推論を除く))"""
    import desk_lib as DL
    name = 'ens:%s+%s' % (a, b)
    r = {'run': name, 'family': 'ens', 'parts': [a, b], 'ens_notes': []}
    for k in FIELDS_C3:
        r[k] = None
    ca, cb = model_cfg(a), model_cfg(b)
    if ref_of(ca) != ref_of(cb):
        r['ens_notes'].append('2 本の中心・中立が違うので出さない')
        return r
    ref = ref_of(ca)
    r['steer_center_us'], r['throttle_neutral_us'] = ref
    tm = sorted(set(ca.get('trained_modes') or []) & set(cb.get('trained_modes') or []))
    r['trained_modes'] = tm
    if a in tabs and b in tabs and B_RUN in tabs:
        r.update(row_fields((tabs[a]['pred'] + tabs[b]['pred']) / 2.0, tabs[B_RUN]['pred'], ref, refB, M, 'v2', tm))
    else:
        r['ens_notes'].append('2 本の tab の予測がそろっていない')
    sa, sb, ba, bb = shift_file(a), shift_file(b), saved_best(a), saved_best(b)
    if sa.exists() and sb.exists() and ba is not None and bb is not None:
        with np.load(sa) as za, np.load(sb) as zb:
            if np.array_equal(za['rows'], zb['rows']):
                rows = za['rows']
                sh = {k: (za['pred_' + k].astype(np.float64) + zb['pred_' + k].astype(np.float64)) / 2.0 for k in SHIFTS}
                r['robust_dl1_us'], per = robust_dl1((ba[rows] + bb[rows]) / 2.0, sh, bundle['y'][rows])
                r['robust_dl1_per_shift_us'] = {k: rnd(v) for k, v in per.items()}
    la = [(j, st) for j, st in T.get(a, []) if engine_current(a, st)]
    lb = [(j, st) for j, st in T.get(b, []) if engine_current(b, st)]
    if la and lb and Path(la[0][1].get('file') or '').exists() and Path(lb[0][1].get('file') or '').exists():
        with np.load(la[0][1]['file']) as za, np.load(lb[0][1]['file']) as zb:
            if za['pred'].shape == zb['pred'].shape and np.array_equal(za['part'], zb['part']):
                pr = (za['pred'].astype(np.float64) + zb['pred'].astype(np.float64)) / 2.0
                erpm, part = za['erpm'].astype(np.float64), za['part']
                fin = np.isfinite(pr).all(axis=(1, 2))
                oor = int(((pr < SANITY_LO) | (pr > SANITY_HI)).sum())
                r['sanity_ok'] = bool(DL.sanity_of(la[0][1]) and DL.sanity_of(lb[0][1]) and fin.all() and oor == 0)
                r['sanity_detail'] = {'nonfinite_frames': int((~fin).sum()), 'out_of_range_values': oor,
                                      'parts_sanity_ok': [DL.sanity_of(la[0][1]), DL.sanity_of(lb[0][1])]}
                if ba is not None and bb is not None:
                    v_hold = DL.speed_mps(bundle['erpm'])
                    low = bundle['primary'] & np.isfinite(v_hold) & (v_hold < DL.LOWSPEED_MPS)
                    kh, nh = DL.brake_lowspeed(((ba + bb) / 2.0)[:, 0, 1], v_hold, low)
                    m0 = part == 0
                    ka, na = DL.brake_lowspeed(pr[m0][:, 0, 1], DL.speed_mps(erpm[m0]), np.isfinite(pr[m0][:, 0, 1]))
                    r['brake_lowspeed_frac'] = DL.frac(kh + ka, nh + na)
            else:
                r['ens_notes'].append('2 本の TensorRT のフレームの並びが違う')
    else:
        r['ens_notes'].append('2 本の今のエンジンの TensorRT の仕事がそろっていない')
    import eval_v2 as E
    oks = [E.engine_fields(RUNS / x).get('engine_ok') if (RUNS / x / ENGINE).exists() else None for x in (a, b)]
    r['engine_ok'] = None if any(v is None for v in oks) else bool(all(oks))
    r['ens_notes'].append('推論の速さは出さない(protocol decisions.D4: 門は推論を除く)。2 本の平均は 11/3 の 1 枠に入れない')
    return r


def cmd_report(args):
    import eval_v2 as E
    t0 = time.time()
    proto = protocol()
    seal_guard([], proto['seal']['logs'])
    S = load_sets()
    sj = load_json(SETS_JSON) or {}
    if sj.get('key') != S['key']:
        raise SystemExit('sets_v2.json と sets_v2.npz の key が違います')
    bundle = E.load_bundle(proto)
    five = E.five_preds(proto)
    T = trt_results()
    names = set()
    for p in PRED.glob('*_tab.npz') if PRED.exists() else []:
        names.add(p.name[:-len('_tab.npz')])
    for p in PRED.glob('*_shift.npz') if PRED.exists() else []:
        names.add(p.name[:-len('_shift.npz')])
    names |= set(T)
    names.add(B_RUN)
    names = sorted(names, key=lambda n: (n != B_RUN, n))
    tabs = {}
    for n in names:
        f = tab_file(n)
        if f.exists():
            with np.load(f) as z:
                if str(z['key']) != S['key']:
                    log('%s: tab の key が集合と違う(古い。使わない)' % n)
                    continue
                tabs[n] = {'pred': z['pred'].astype(np.float64),
                           'rows_g9': z['rows_g9'] if 'rows_g9' in z.files else None,
                           'pred_g9': z['pred_g9'].astype(np.float64) if 'pred_g9' in z.files else None,
                           'substituted': int(np.asarray(z['substituted']).sum())}
    if B_RUN not in tabs:
        log('注意: B の tab の予測がまだ。_vs_b の欄・段 0 の 2・4 は未')
    M = {m: S[m].astype(bool) for m in TAB_MASKS}
    refB = ref_of(model_cfg(B_RUN))
    runs, notes = {}, []
    for n in names:
        cfg = model_cfg(n)
        fam = run_family(n)
        ref = ref_of(cfg)
        r = {'run': n, 'family': fam, 'steer_center_us': ref[0], 'throttle_neutral_us': ref[1],
             'trained_modes': cfg.get('trained_modes')}
        for k in FIELDS_C3:
            r[k] = None
        if n in tabs and B_RUN in tabs:
            t = tabs[n]
            r.update(row_fields(t['pred'], tabs[B_RUN]['pred'], ref, refB, M, fam, cfg.get('trained_modes') or [],
                                t['pred_g9'], t['rows_g9']))
            r['tab_genre_substituted'] = t['substituted']
        # ずれの試し
        sf = shift_file(n)
        saved = saved_best(n)
        if sf.exists():
            with np.load(sf) as z:
                rows = z['rows']
                y = bundle['y'][rows]
                if saved is not None:
                    base, src = saved[rows], 'eval/pred(保存した best)'
                elif len(z['pred_none_rows']) == len(rows):
                    base, src = z['pred_none'].astype(np.float64), 'この台本で予測したそのまま'
                else:
                    base, src = None, None
                if base is not None:
                    sh = {k: z['pred_' + k].astype(np.float64) for k in SHIFTS}
                    r['robust_dl1_us'], per = robust_dl1(base, sh, y)
                    r['robust_dl1_per_shift_us'] = {k: rnd(v) for k, v in per.items()}
                    r['robust_base'] = src
                    if 'none_vs_saved_max_abs_us' in z.files:
                        r['robust_none_vs_saved_max_abs_us'] = rnd(float(z['none_vs_saved_max_abs_us']), 5)
        # TensorRT
        tf = trt_fields(n, T, bundle, saved)
        for k in ('sanity_ok', 'trt_median_ms', 'trt_p99_ms', 'brake_lowspeed_frac'):
            r[k] = tf[k]
        r['trt'] = {k: v for k, v in tf.items() if k not in ('sanity_ok', 'trt_median_ms', 'trt_p99_ms', 'brake_lowspeed_frac')}
        ef = E.engine_fields(RUNS / n) if (RUNS / n / ENGINE).exists() else {'engine_ok': None}
        r['engine_ok'] = ef.get('engine_ok')
        runs[n] = r
    for spec in (getattr(args, 'ens', None) or []):
        if '+' not in spec:
            raise SystemExit('--ens は <軽量型>+<多段型> の形: %s' % spec)
        a_, b_ = spec.split('+', 1)
        er_ = ens_row(a_, b_, tabs, M, refB, T, bundle)
        runs[er_['run']] = er_
    # 上の欄
    st_frames = int(M['stand'].sum())
    top = {'standstill_frames': st_frames, 'standstill_gate': standstill_gate_kind(st_frames)}
    top['bfast_gf_limit_us'] = gf_limit(tabs[B_RUN]['pred'], refB, M['straight']) if B_RUN in tabs else None
    top['b_raw_forward_straight_us'] = raw_forward(tabs[B_RUN]['pred'], refB, M['straight']) if B_RUN in tabs else None
    days = sorted({j['day'] for j, _ in T.get(B_RUN, [])})
    top['b_trt_p99_ms_same_day'] = None
    if days:
        vals = [s.get('run_numpy_p99_ms') for j, s in T.get(B_RUN, []) if j['day'] == days[-1] and s.get('run_numpy_p99_ms')]
        top['b_trt_p99_ms_same_day'] = float(np.median(vals)) if vals else None
        top['b_trt_p99_day'] = days[-1]
        other = [n for n, r in runs.items() if (r.get('trt') or {}).get('trt_job') and r['trt']['trt_job'][:8] != days[-1]]
        if other:
            notes.append('TensorRT の日が B の最新の日(%s)と違う学習: %s(各行の inference_ok_same_day はその日の B で見た)'
                         % (days[-1], ' '.join(other)))
    # 段 0 の 2・4・8
    stage0 = {}
    if B_RUN in tabs:
        m2 = M['s0_2']
        dd = stage0_2_diffs(tabs[B_RUN]['pred'][m2], S['model_out'][m2], refB, S['gains'][m2], S['trim'][m2], S['lookahead'][m2])
        stage0['2'] = {'what': proto['stage0_checks']['list'][1], **stage0_2_summary(dd),
                       'frames_rule': '10/24 2 回目の、B が走った記録(No.9 20261024-162033・No.10 162305)の AI の側・model_out が数値・hold でない'
                                      '(No.5〜8 は遅く入る型が走ったので、記録の model_out は B の出力ではない)',
                       'how': '予測(PyTorch FP32 best・ジャンル 5)を記録のフレームごとのゲイン・trim・先読みで drive の model_us にして比べた'}
    else:
        stage0['2'] = {'what': proto['stage0_checks']['list'][1], 'pass': None, 'why': 'B の tab の予測がまだ'}
    s4 = {'what': proto['stage0_checks']['list'][3], 'standstill_frames': st_frames, 'gate': standstill_gate_kind(st_frames),
          'label_rows_before_tof': (sj.get('standstill') or {}).get('label_rows_before_tof'),
          'aligned_before_tof': (sj.get('standstill') or {}).get('aligned_before_tof'),
          'B_standstill_go_frac': runs[B_RUN].get('standstill_go_frac'),
          'ENS_standstill_go_frac': (runs.get(ENS_RUN) or {}).get('standstill_go_frac')}
    stage0['4'] = s4
    stage0['8'] = {'what': proto['stage0_checks']['list'][7], 'b_trt_p99_ms_same_day': top['b_trt_p99_ms_same_day'],
                   'day': top.get('b_trt_p99_day'), 'inference_gate_ms': (INFER_FACTOR * top['b_trt_p99_ms_same_day'])
                   if top['b_trt_p99_ms_same_day'] else None, 'b_jobs': [(j['job'], s.get('run_numpy_p99_ms')) for j, s in T.get(B_RUN, [])]}
    # 門(eval_v2.py の gates_v2・gates_bfast をそのまま使う。H1 の欄は eval_v2.run_row)
    desk_like = {'runs': {n: {k: r.get(k) for k in FIELDS_C3} for n, r in runs.items()}, **top}
    for n, r in runs.items():
        desk_like['runs'][n]['trt_p99_ms'] = r.get('trt_p99_ms')
    T_ = proto['thresholds_us']['values']
    thr_bias_plus = T_['five_mean']['thr_bias_best'] + T_['pairs']['k2']['thr_bias_best']
    f_b = five['B']
    b_eval = E.run_row(B_RUN, f_b['best'], f_b['last'], bundle, desk_like, run_dir=RUNS / B_RUN)
    for n, r in runs.items():
        if r['family'] == 'ens':
            pa, pb_ = saved_best(r['parts'][0]), saved_best(r['parts'][1])
            if pa is None or pb_ is None:
                r['gates_pass'] = None
                r['gates_note'] = '2 本の H1 の予測(eval/pred)がそろっていない'
                continue
            er = E.run_row(n, (pa + pb_) / 2.0, None, bundle, desk_like, run_dir=None)
            er['engine_ok'] = r.get('engine_ok')
            for k in FIELDS_H1:
                r[k] = er.get(k)
            g = E.gates_v2(er, b_eval, desk_like, thr_bias_plus)
            g.pop('inference', None)
            r['gates'] = g
            r['gates_pass'] = None
            r['gates_pass_except_inference'] = E.gates_pass(g)
            r['gates_note'] = '2 本の平均: D4 の「門(推論を除く)」= gates_pass_except_inference。主の候補・1 枠にしない'
            continue
        best = saved_best(n)
        lastp = EVAL_PRED / ('%s_last.npz' % n)
        last = None
        for rid, run, pb, pl, seed in E.FIVE:
            if run == n:
                last = E.load_pred(pl)
        if last is None and lastp.exists():
            last = E.load_pred(lastp, 'last')
        if best is None:
            r['gates_pass'] = None
            r['gates_note'] = 'H1 の予測(eval/pred)がまだ'
            continue
        er = E.run_row(n, best, last, bundle, desk_like, run_dir=RUNS / n)
        for k in FIELDS_H1:
            r[k] = er.get(k)
        if r['family'] == 'v2':
            g = E.gates_v2(er, b_eval, desk_like, thr_bias_plus)
        elif r['family'] == 'bfast':
            g = E.gates_bfast(er, b_eval, desk_like)
        else:
            r['gates_pass'] = None
            r['gates_note'] = '基準・参考(門に掛けない)'
            continue
        r['gates'] = g
        r['gates_pass'] = E.gates_pass(g)
    out = {'schema': 'jf-c3-desk-v2/v1', 'made_at': time.strftime('%Y-%m-%d %H:%M:%S'), **lock_info(),
           'sets_key': S['key'], 'sets_json_sha256': sha256(SETS_JSON), 'sets_npz_sha256': sha256(SETS_NPZ),
           **top, 'stage0': stage0,
           'set_counts': {m: c['frames'] for m, c in (sj.get('counts') or {}).items()},
           'collisions_ai': sj.get('collisions_ai'),
           'definitions': definitions(), 'notes': notes,
           'runs': {n: clean_row(r) for n, r in runs.items()}}
    dj, dm = (Path(args.out) / 'desk_v2.json', Path(args.out) / 'desk_v2.md') if args.out else (DESK_JSON, DESK_MD)
    write_json(dj, out)
    write_text(dm, report_md(out))
    log('書きました: %s(学習 %d 本・%.0f 秒)' % (dj, len(runs), time.time() - t0))
    log('段 0: 2 %s / 4 %s(%d フレーム・%s) / 8 %s ms' % (
        stage0['2'].get('pass'), s4.get('B_standstill_go_frac'), st_frames, s4['gate'], rnd(top['b_trt_p99_ms_same_day'], 3)))
    return 0


def clean_row(r):
    def c(v):
        if isinstance(v, dict):
            return {k: c(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [c(x) for x in v]
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return rnd(v, 5)
        return v
    return c(r)


def definitions():
    return {
        'car_output': 'drive.apply_gains と同じ(ステア 1.0・trim 0・スロットルのゲインは前進側だけ・1000〜2000)。'
                      'ゲイン B 0.90・候補のジャンル 5 0.90・B-FAST 0.90・ジャンル 9 1.00。ref は各モデルの model_config',
        'pred': 'PyTorch FP32 の best。AI の記録の集合はジャンル 5(v2-M のジャンル 9 は R1・DE の手前 1 秒だけ)。H1 のずれは aligned のジャンル',
        'ai_pre': '学習の回(46 本)の AI の衝突の手前 1 秒 [tc − 1, tc)・AI の側。R1・DE はラベルの loc、SW は loc_detail',
        'de_pass': '学習に使っていない AI の周(きれいでない・B の 9.5 秒以上・ほかのモデル)の DE の入りから 0.5 秒',
        'straight': '学習に使っていない AI のきれいな周(B の 9.5 秒以上・ほかのモデル)の直線(inventory の straight)',
        'standstill': 'AI の側・|eRPM| < 400・衝突の前後でない・ToFGuard の x_free > 1.0 m(erpm 0・gz 0・ステア中心)',
        'gb_main': '10/23 公式の B の 3 本の AI の側・衝突の前後でない。G-b = loc_detail DE・SW・HS、G-c = R1 の起点の手前 1 秒',
        'gb_ref': '10/24 2 回目の AI の側・衝突の前後でない・B-FAST の手本のフレームの外',
        'robust_dl1_us': 'H1 の 7 つのずれの L1 − そのまま(保存した best)の L1 の平均。上 1/3 は 1 + (f − 1)·top_ramp を掛ける',
        'bfast_gf_limit_us': 'B の直線の 行 2 の前進の量の(ゲイン 1.00 − 0.90)の平均',
        'sanity': '10/10 の AI 走行 21,759 + 10/24 2 回目の AI のフレームを TensorRT(ジャンル 5・基準の設定)に。NaN・Inf なし・500〜2500 µs・check_io',
        'brake_lowspeed_frac': '10/21 と同じ: v < 1.0 m/s で行 0 のスロットル > 1598 µs の割合(保留 FP32 + 10/10 の AI 走行 TensorRT)',
        'trt': 'engine_check.json と同じ測り方(同じ実フレーム・100 回の空回し + 600 回の run_numpy)を、この台本の TensorRT の仕事で '
               'B → 候補 → B の順に同じ回に測った中央値・99% 点。門「推論」は同じ日の B の 1.3 倍。sanity の流れの時間は sanity_loop_* に参考',
    }


def fmt(v, nd=2):
    if v is None:
        return '未'
    if isinstance(v, bool):
        return '○' if v else '×'
    if isinstance(v, float):
        return ('%%.%df' % nd) % v
    return str(v)


def report_md(o):
    L = ['# 机上のベンチ v2(C3)', '',
         '- 作った時刻: %s' % o['made_at'],
         '- protocol_v2.json: `%s`・lock2: `%s`' % (o['protocol_v2_sha256'], o['lock2_sha256']),
         '- 台本: `%s`(sha256 `%s`)' % (o['script'], o['script_sha256']),
         '- 集合の key: `%s`' % o['sets_key'],
         '- 机上の順位で実車の順位は決めない(plan.md 6.4)。門・決まりは eval_v2.py が protocol のとおりに当てる。', '',
         '## 段 0 の 2・4・8', '',
         '| # | 答え |', '|---|---|']
    s = o['stage0']
    s2 = s['2']
    L.append('| 2 | 差の絶対値の中央値 %s µs(ステア・スロットルを合わせて)・ステア %s・スロットル %s(%s フレーム)→ **%s** |' % (
        fmt(s2.get('median_abs_us_pooled')), fmt(s2.get('median_abs_us_steer')), fmt(s2.get('median_abs_us_throttle')),
        s2.get('frames', '未'), {True: '合格(≤ 10 µs)', False: '外れ(止めて調べる)', None: s2.get('why') or '未'}[s2.get('pass')]))
    s4 = s['4']
    L.append('| 4 | 止まった場面 %s フレーム(ToF の前 %s・ラベルの行 %s)→ **%s**。B %s・2 本の平均 %s |' % (
        s4['standstill_frames'], s4.get('aligned_before_tof'), s4.get('label_rows_before_tof'), s4.get('gate'),
        fmt(s4.get('B_standstill_go_frac'), 3), fmt(s4.get('ENS_standstill_go_frac'), 3)))
    s8 = s['8']
    L.append('| 8 | 同じ日の B の p99 %s ms(%s)→ 門「推論」%s ms |' % (fmt(s8.get('b_trt_p99_ms_same_day'), 3), s8.get('day') or '未',
                                                          fmt(s8.get('inference_gate_ms'), 3)))
    L += ['', '## 集合のフレーム数', '', '| 集合 | フレーム |', '|---|---|']
    for k, v in (o.get('set_counts') or {}).items():
        L.append('| %s | %s |' % (k, v))
    L += ['', '- AI の衝突(学習の回): %s' % json.dumps(o.get('collisions_ai'), ensure_ascii=False),
          '- G-f の上限 bfast_gf_limit_us: %s µs(B の生の前進の量 %s µs)' % (fmt(o.get('bfast_gf_limit_us'), 3),
                                                                   fmt(o.get('b_raw_forward_straight_us'), 1)), '',
          '## 学習ごとの欄(単位 µs・割合・ms)', '',
          '| 名前 | robust | DE 手前 | R1 手前 | SW 手前 | DE 行き過ぎ | 直線 | 止まって出る | g9−g5 | G-b | G-c | G-b 参考 | 1.00 対 0.90 | DE の線 | 3 周の見込み | sanity | p99 | ブレーキ | 門 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for n, r in o['runs'].items():
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            n, fmt(r.get('robust_dl1_us')), fmt(r.get('de_pre_thr_vs_b_us')), fmt(r.get('r1_pre_thr_vs_b_us')),
            fmt(r.get('sw_pre_thr_vs_b_us')), fmt(r.get('de_overshoot_vs_b_us')), fmt(r.get('straight_thr_vs_b_us')),
            fmt(r.get('standstill_go_frac'), 3), fmt(r.get('g9_minus_g5_pre_thr_us')), fmt(r.get('bfast_gb_fwd_vs_b_us')),
            fmt(r.get('bfast_gc_fwd_vs_b_us')), fmt(r.get('bfast_gb_ref_fwd_vs_b_us')), fmt(r.get('bfast_gb_fwd_g100_vs_b090_us')),
            fmt(r.get('bfast_de_line_vs_b_us')), fmt(r.get('lap3_est_s'), 2), fmt(r.get('sanity_ok')),
            fmt(r.get('trt_p99_ms'), 3), fmt(r.get('brake_lowspeed_frac'), 4),
            ({True: '全部合格', False: '不合格', None: '未'}[r.get('gates_pass')] if r.get('family') in ('v2', 'bfast') else
             ('推論を除き %s' % {True: '全部合格', False: '不合格', None: '未'}[r.get('gates_pass_except_inference')]
              if r.get('family') == 'ens' else '—'))))
    L += ['', '## 注', ''] + ['- %s' % x for x in o.get('notes') or []]
    L += ['- 「_thr_」は PWM の差(+ = 緩める)、「_fwd_」は前進の量の差(+ = 強く踏む)。ステアは + = 左。',
          '- 欄の名前と式は protocol_v2.json の fields_added・sign_rules のとおり。くわしくは c3.md。',
          '- 速さの見込み(lap3_est_s)は参考で、門にしない。']
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('sets', help='集合を作る(CPU・numpy)')
    p = sub.add_parser('predict', help='PyTorch FP32 の予測(GPU。jf-heavy --lock big で)')
    p.add_argument('--runs', nargs='*')
    p.add_argument('--what', choices=('tab', 'shift', 'all'), default='tab')
    p.add_argument('--force', action='store_true', help='あっても予測し直す')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=3)
    t = sub.add_parser('trt', help='TensorRT の sanity・推論の速さ(GPU。B はいつも入る)')
    t.add_argument('--runs', nargs='*')
    t.add_argument('--max-runs', type=int, default=2)
    t.add_argument('--limit', type=int, default=0, help='試し: 先頭のフレームだけ(欄に使わない)')
    r = sub.add_parser('report', help='欄と表(CPU・numpy)')
    r.add_argument('--out', help='試しだけ: desk_v2.json・md を書く場所(既定 %s)' % OUT)
    r.add_argument('--ens', nargs='*', help='2 本の平均の行も出す(D4): <軽量型の名前>+<多段型の名前> ...')
    args = ap.parse_args()
    return {'sets': cmd_sets, 'predict': cmd_predict, 'trt': cmd_trt, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
