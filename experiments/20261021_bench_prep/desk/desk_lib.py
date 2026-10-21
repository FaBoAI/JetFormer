#!/usr/bin/env python3
"""段 Desk(2026-10-21)の机上のベンチマークの計算(numpy だけ。torch を読まない。test_desk_lib.py で試す)

欄の名前と定義は protocol.json の desk_fields と protocol_amendments.md の A1 のとおり。
計算のくわしい決め方(結果を見る前に決めた。desk_bench.md の「定義」にも書く):
  - 保留の集合 primary: 予測の表(tools/rl_evaluate.pred_table、key 3d339512…、10,985 フレーム)の part 0 のうち、
    人のきれいな 11 シーン(heldout.json の human_clean_scenes)・future_valid & vehicle_valid・衝突(2.5 G)の前後でない
    = 4,670 フレーム(10/15 realacc_eval.evaluate_all と同じ)
  - 介入の集合 interv: part 1(intervention_val_20261011 の採点するフレーム)で future_valid・正解が有限 = 1,570 フレーム
  - L1 は 10 ステップ × 2(ステア・スロットル)の平均 [µs](realacc_lib.l1_rows)。ステア・スロットルに分けた欄は片方だけの平均
  - steer_amp_ratio: primary の予測ステアの標準偏差 ÷ 人のステア(正解)の標準偏差。10 ステップをまとめて数える(best の重み)
  - r1_app_thr_diff_us: primary のうち R1 の起点 ta(label_events.course_anchors、ログ全体の IMU から後で見た判定)の
    手前 1 秒 [ta − 1.0, ta) のフレームで、予測スロットル − 人のスロットル の平均 [µs]。行は走る日の先読み 3 で使う
    index 2(3 フレーム先)。正 = 人より緩める(PWM は中立 1498・前進は小さい値)。参考に 10 行の平均も出す
  - brake_lowspeed_frac(A1): 速さ v = −eRPM × 0.5e-3 < 1.0 m/s のフレームのうち、予測の 1 行目(index 0)のスロットルが
    1598 µs(中立 1498 + 100)を超えた割合。分母 = primary の低速フレーム + 10/10 の AI 走行の低速フレーム
    (保留は PyTorch FP32 の best、AI 走行は TensorRT のエンジン。どちらもジャンル: 保留は人のジャンル、AI 走行は 5)
  - engine_ok: engine_check.json が今のエンジン(大きさ・更新時刻)のもので、accuracy_ok が true・actions_max_us ≤ 2.5
  - sanity_ok: 10/10 の AI 走行 11 本の AI のフレーム(ai_mix_20261011 の source == 1 & vehicle_valid、21,759 個)を
    TensorRT のエンジン(走行と同じ run_numpy・ジャンル 5・基準の設定の後処理)に入れ、出力がすべて有限で、
    予測の操作値 [10,2] がすべて 500〜2500 µs(RC の PWM として意味のある幅)の中、かつ check_io が通る。
    (12:33 の 200 フレームの試しで、ステアが 1000 µs を少し下回る値(908〜1005 µs。人の正解も 983 µs まである)が
    ほぼ全部の候補に出た。走行は 1000〜2000 µs に切るので、壊れたモデルを見つける幅として 500〜2500 µs にした。
    1000〜2000 µs の外に出た値の数(走行が切る数)は参考に出す)
  - 門: protocol.json gates の 5 つ。家族 label_r1 だけ A1(5 番目をステアの L1 に置き換え、brake_lowspeed_frac ≤ 0.01 を足す)
"""

import math

import numpy as np

NEUTRAL_US = 1498.0
BRAKE_MARGIN_US = 100.0                 # A1: 中立 + 100 µs を超えたらブレーキ側
LOWSPEED_MPS = 1.0
SPEED_PER_ERPM = 0.5e-3                 # jetformer/policies/post.py と同じ
R1_LEAD_S = 1.0
LOOKAHEAD_ROW = 2                       # 走る日の基準の先読み 3 → index 2
GATE_ACC_US = 2.5
GATE_P99_MS = 10.0
GATE_AMP = (0.80, 1.25)
GATE_L1_US = 10.0
GATE_BRAKE_FRAC = 0.01
SANITY_LO, SANITY_HI = 500.0, 2500.0       # sanity の幅 [µs]
DRIVE_LO, DRIVE_HI = 1000.0, 2000.0         # 走行が切る幅(config.DRIVE。参考)

# id・表示名・学習結果・家族(protocol.json pool。家族は run の名前の規則 family_rule.by_run_name でも確かめる)
CANDIDATES = [
    ('B', 'モデルB(基準)', 'clean_driving_20261011-nocmd-s3-20261011-024626', None),
    ('MSF', '多段型', 'clean_driving_20261011-msfusion-s0-nocmd-stride3-e30', 'arch'),
    ('TCP', 'TCP型', 'clean_driving_20261011-tcp-s0-nocmd-stride3-e30', 'arch'),
    ('ACT', 'ACT型', 'clean_driving_20261011-act-s0-nocmd-stride3-e30', 'arch'),
    ('SIM', 'シミュ混合', 'simreal_mix_20261013-cotrain-ft-s0', 'robust'),
    ('ENS', '2本の平均', 'bench_20261021-ens-p2', 'ens'),
    ('SLOW', '遅く入る型', 'bench_20261021-slow-s0', 'label_r1'),
    ('WGT', '苦手区間重み型', 'bench_20261021-wgt-s0', 'weight'),
    ('AUG', 'ずれに強い型', 'bench_20261021-aug-s0', 'robust'),
]
IDS = [c[0] for c in CANDIDATES]
RUN_OF = {c[0]: c[2] for c in CANDIDATES}
LABEL_OF = {c[0]: c[1] for c in CANDIDATES}
FAMILY_OF = {c[0]: c[3] for c in CANDIDATES}
NEW_TRAINED = ('SLOW', 'WGT', 'AUG')     # 段 G2 の新しい学習(予測をここで作る)
ENS_PARTS = ('realacc_20261015-v0-s2', 'realacc_20261015-msf-s2')


def family_by_run(run, table_family=None):
    """protocol.json family_rule.by_run_name: -slow- → label_r1、-wgt- → weight、-aug- → robust、-ens- → ens。ほかは表の家族"""
    for key, fam in (('-slow-', 'label_r1'), ('-wgt-', 'weight'), ('-aug-', 'robust'), ('-ens-', 'ens')):
        if key in run:
            return fam
    return table_family


# ---------------------------------------------------------------- 衝突の前後(jetformer/train_data.impact_mask の写し)

def impact_mask(timestamps_us, imu, impact_g=2.5, before_s=1.0, after_s=2.0):
    """jetformer/train_data.impact_mask と同じ(torch を読まないために写した。make_bench_tables.py と同じ写し)"""
    t = np.asarray(timestamps_us, np.float64) / 1e6
    acc = np.asarray(imu, np.float32)[:, :, :2]
    hg = np.hypot(acc[..., 0], acc[..., 1]).max(axis=1) / 9.80665
    mask = np.zeros(len(t), bool)
    for ti in t[hg >= impact_g]:
        mask |= (t >= ti - before_s) & (t <= ti + after_s)
    return mask


# ---------------------------------------------------------------- 指標

def l1_parts(pred, y, mask):
    """→ {'l1': 10×2 の平均, 'l1_steer': ステアだけ, 'l1_thr': スロットルだけ, 'n': フレーム数} [µs]"""
    m = np.asarray(mask, bool)
    d = np.abs(np.asarray(pred, np.float64)[m] - np.nan_to_num(np.asarray(y, np.float64)[m]))
    if not len(d):
        return {'l1': None, 'l1_steer': None, 'l1_thr': None, 'n': 0}
    return {'l1': float(d.mean()), 'l1_steer': float(d[:, :, 0].mean()), 'l1_thr': float(d[:, :, 1].mean()),
            'n': int(m.sum())}


def steer_amp_ratio(pred, y, mask, row=None):
    """予測ステアの標準偏差 ÷ 人のステアの標準偏差(row=None なら 10 ステップをまとめて、row=k ならその行だけ)"""
    m = np.asarray(mask, bool)
    p = np.asarray(pred, np.float64)[m][:, :, 0]
    h = np.asarray(y, np.float64)[m][:, :, 0]
    if row is not None:
        p, h = p[:, row], h[:, row]
    sh = float(np.std(h))
    if not len(p) or not np.isfinite(sh) or sh <= 0:
        return None
    return float(np.std(p) / sh)


def pre_anchor_mask(t_s, anchors_s, lead_s=R1_LEAD_S):
    """各フレームの時刻 t_s [s] が、どれかの R1 の起点 ta の [ta − lead, ta) に入るか"""
    t = np.asarray(t_s, np.float64)
    m = np.zeros(len(t), bool)
    for ta in np.asarray(anchors_s, np.float64):
        m |= (t >= ta - lead_s) & (t < ta)
    return m


def thr_diff(pred, y, mask, row=LOOKAHEAD_ROW):
    """予測スロットル − 人のスロットル の平均 [µs](row=None なら 10 行の平均)。正 = 人より緩める"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    p = np.asarray(pred, np.float64)[m][:, :, 1]
    h = np.asarray(y, np.float64)[m][:, :, 1]
    if row is not None:
        p, h = p[:, row], h[:, row]
    return float(np.mean(p - h))


def speed_mps(erpm):
    return -np.asarray(erpm, np.float64) * SPEED_PER_ERPM


def brake_lowspeed(pred_row0_thr, v_mps, valid=None, neutral=NEUTRAL_US, margin=BRAKE_MARGIN_US, vmax=LOWSPEED_MPS):
    """A1: 速さ < vmax のフレームのうち、予測の 1 行目のスロットル > 中立 + margin の数 → (数, 分母)"""
    u = np.asarray(pred_row0_thr, np.float64)
    v = np.asarray(v_mps, np.float64)
    ok = np.isfinite(v) & (v < vmax)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    return int((ok & (u > neutral + margin)).sum()), int(ok.sum())


def frac(k, n):
    return None if not n else float(k) / float(n)


# ---------------------------------------------------------------- エンジンの確認

def engine_current(check, size, mtime_ns):
    """engine_check.json(dict)が今のエンジン(大きさ・更新時刻)のものか(drive.engine_check_of と同じ規則)"""
    if not isinstance(check, dict):
        return False
    e = check.get('engine') or {}
    try:
        return int(e.get('size', -1)) == int(size) and int(e.get('mtime_ns', -1)) == int(mtime_ns)
    except (TypeError, ValueError):
        return False


def engine_ok_of(check, current, gate_us=GATE_ACC_US):
    """engine_ok: 今のエンジンの確認で accuracy_ok が true・actions_max_us ≤ 2.5"""
    if not (current and isinstance(check, dict)):
        return False
    try:
        mx = float(check.get('actions_max_us'))
    except (TypeError, ValueError):
        return False
    return bool(check.get('accuracy_ok') is True and math.isfinite(mx) and mx <= gate_us)


def sanity_of(stats):
    """sanity の数え(desk_gpu.py sanity の結果)→ sanity_ok"""
    if not isinstance(stats, dict) or stats.get('error'):
        return False
    return bool(stats.get('check_io_ok') is True and stats.get('frames', 0) > 0 and stats.get('nonfinite_frames') == 0
                and stats.get('out_of_range_values') == 0)


# ---------------------------------------------------------------- 門(protocol.json gates + A1)

def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def gates(row, base_row, family):
    """→ (gates_pass, gate_fail_reasons)。欄が無い・null・NaN は不合格(理由 missing:<欄>)。
    family == 'label_r1' だけ A1: 5 番目を l1_steer_best_us の差に置き換え、brake_lowspeed_frac ≤ 0.01 を足す"""
    fails = []
    for k in ('engine_ok', 'sanity_ok'):
        v = row.get(k)
        if v is None:
            fails.append('missing:%s' % k)
        elif v is not True:
            fails.append(k)
    p99 = _num(row.get('trt_p99_ms'))
    if p99 is None:
        fails.append('missing:trt_p99_ms')
    elif not p99 <= GATE_P99_MS:
        fails.append('trt_p99_ms')
    amp = _num(row.get('steer_amp_ratio'))
    if amp is None:
        fails.append('missing:steer_amp_ratio')
    elif not (GATE_AMP[0] <= amp <= GATE_AMP[1]):
        fails.append('steer_amp_ratio')
    key = 'l1_steer_best_us' if family == 'label_r1' else 'l1_best_us'
    a, b = _num(row.get(key)), _num((base_row or {}).get(key))
    if a is None:
        fails.append('missing:%s' % key)
    elif b is None:
        fails.append('missing:B.%s' % key)
    elif not (a - b < GATE_L1_US):
        fails.append(key)
    if family == 'label_r1':
        bf = _num(row.get('brake_lowspeed_frac'))
        if bf is None:
            fails.append('missing:brake_lowspeed_frac')
        elif not bf <= GATE_BRAKE_FRAC:
            fails.append('brake_lowspeed_frac')
    return (not fails), fails


# ---------------------------------------------------------------- R1 の起点ごとのかたまりのブートストラップ(読み方の参考)

def anchor_ids(t_s, anchors_s, lead_s=R1_LEAD_S, start=0):
    """各フレームが入る起点の番号(start から。入らなければ −1)。起点が重なれば後の起点"""
    t = np.asarray(t_s, np.float64)
    out = np.full(len(t), -1, np.int64)
    for k, ta in enumerate(np.asarray(anchors_s, np.float64)):
        out[(t >= ta - lead_s) & (t < ta)] = start + k
    return out


def cluster_boot_ci(values, cluster, n_boot=2000, seed=0, q=(5.0, 95.0)):
    """かたまり(起点)を復元抽出して、フレームの平均の 90% 区間(百分位)→ (平均, 下, 上, かたまりの数)"""
    v = np.asarray(values, np.float64)
    c = np.asarray(cluster)
    ok = c >= 0
    v, c = v[ok], c[ok]
    if not len(v):
        return None, None, None, 0
    ids = np.unique(c)
    s = np.array([v[c == i].sum() for i in ids])
    n = np.array([(c == i).sum() for i in ids], np.float64)
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(n_boot, len(ids)))
    m = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(m, q)
    return float(v.mean()), float(lo), float(hi), int(len(ids))
