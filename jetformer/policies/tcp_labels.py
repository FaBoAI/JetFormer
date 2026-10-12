"""TCP 型の正解(1 秒先までの通過点・目標速度)と、制御器の較正(numpy だけ。torch を読まない)

    from jetformer.policies import tcp_labels
    path, built = tcp_labels.ensure_labels('data/datasets/clean_driving_20261011')     # 無い・古ければ作る(CPU 1分以内)
    cpath, _ = tcp_labels.ensure_calibration('data/datasets/clean_driving_20261011')   # 学習のシーンだけで当てはめ
    per_scene = tcp_labels.labels_by_scene('data/datasets/clean_driving_20261011')     # {シーン: {'waypoints', 'v_tgt', 'wp_valid'}}

元データ(データセットの can_bus/):
    <scene>_ms_imu.json           約100 Hz。rotation_rate[2] は rad/s(z 正 = 左旋回)、linear_accel は m/s²(+x 右・+y 前)
    <scene>_vehicle_monitor.json  約100 Hz。erpm(負 = 前進)・steering_us(大きい = 左)・throttle_us(中立より下 = 前進)・
                                  valid・ai_driving
車の約束: 速さ v = -eRPM × 0.5e-3 [m/s](後退は負)。自車の座標 x = 前・y = 左 [m]、原点 = 撮像時刻 t₀ の車。

正解(spec.md §3.2.2。research_src/tcp_label_check.py の load_scene・dead_reckon・waypoints と同じ手順):
    1. ジャイロのバイアス b: ログ(シーン名の先頭の日時)ごとに、止まっている間(|eRPM| < 1 かつ |gz| < 5°/s)の gz の中央値
       (50 サンプル未満なら 0)
    2. v(t) を IMU の時刻へ線形補間し、ψ = ∫(gz − b)dt・X = ∫v cosψ dt・Y = ∫v sinψ dt(dt は 0〜50 ms に切る)
    3. フレーム i(aligned の timestamps_us[i] = t₀)の自車座標へ: x_k = cosψ₀·ΔX + sinψ₀·ΔY、y_k = −sinψ₀·ΔX + cosψ₀·ΔY
       (k = 1..10、0.1 秒刻み)
    4. 目標速度 v_tgt = t₀ + 0.2〜0.5 秒の v の平均(0〜4.5 に切る)、v_now = t₀ の v
    5. wp_valid: 1 秒の窓がシーン(記録)の中・窓の中の IMU の間隔 < 50 ms・窓の中で v ≥ −0.1・窓が衝突(水平 2.5G)の
       前 1 秒〜後 2 秒に掛からない・窓の中の |gz| ≤ 15 rad/s(衝突の飽和)。無効のフレームの通過点は 0
キャッシュ: data/derived/<データセット名>/tcp_labels_v1.npz(データセットのフォルダーの中には書かない。ダウンロード・
エクスポートに入るため)。splits.json の sha1・各シーンのフレーム数が変わっていたら作り直す。

較正(spec.md §3.2.3。学習のシーンだけ・人の操作のサンプルだけ): ステア → 曲率の速度帯ごとの表(遅れ 0/40/80/120 ms から R² 最大)と、
スロットルの前向きの写像 thr = c0 + c1·v + c2·(v_tgt − v)/0.33。data/derived/<データセット名>/tcp_calib_v1.json
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import ROOT

VERSION = 1
LABELS_NAME = 'tcp_labels_v1'
CALIB_NAME = 'tcp_calib_v1'
DERIVED_ROOT = ROOT / 'data' / 'derived'     # テストでは monkeypatch で tmp に向ける

SPEED_PER_ERPM = 0.5e-3          # 速さ [m/s] = -eRPM × 0.5e-3(65 mm のタイヤ)
GRAVITY = 9.80665
WP_STEPS, WP_DT = 10, 0.1        # 通過点: 0.1 秒刻みで 1 秒先まで
V_TGT_WINDOW = (0.2, 0.5)        # 目標速度 = この時間(t₀ から)の v の平均
V_TGT_POINTS = 31                # 平均の点の数(10 ms 刻み)
V_MAX = 4.5                      # 目標速度の上限(speed_bins の最後)
DT_MAX = 0.05                    # 積分の dt の上限・窓の中の IMU の間隔の上限 [s]
V_MIN_WINDOW = -0.1              # 窓の中の速さの下限(後退する窓は無効)
GZ_MAX = 15.0                    # 窓の中の |gz| の上限 [rad/s](衝突でジャイロが飽和した窓を外す)
IMPACT_G, IMPACT_BEFORE_S, IMPACT_AFTER_S = 2.5, 1.0, 2.0     # モデル B と同じ衝突の除外(水平 2.5G の前 1 秒〜後 2 秒)
BIAS_STILL_ERPM, BIAS_STILL_GZ, BIAS_MIN_N = 1.0, float(np.deg2rad(5.0)), 50
INVALID_REASONS = ('no_log', 'outside', 'imu_gap', 'reverse', 'impact', 'gyro_sat')

# 較正(spec.md §3.2.3)
LAGS_MS = (0, 40, 80, 120)
LIN_US = 250.0                   # 遅れを選ぶ: |ステア − 中心| ≤ 250 µs(頭打ちの手前)
CAL_V_MIN = 0.8                  # κ = ω/v を使う速さの下限 [m/s]
OFFSET_EDGES = np.arange(-600.0, 601.0, 50.0)       # ステアのずれの区切り(50 µs)→ 中心 −575〜+575 の 24 個
BANDS_MPS = (2.0, 3.0)           # 速度帯: v < 2・2 ≤ v < 3・3 ≤ v
MIN_BIN_N = 30                   # これ未満の区切りは全速度の表の値
THR_V_MIN = 0.3                  # スロットルの写像: v ≥ 0.3 m/s
THR_FWD_US = 10.0                # 前進のサンプル: thr < 中立 − 10 µs
TAU_S = 0.33                     # 写像の時定数(c2 の分母)
FWD_MAX_PCT = 99.0


class LabelError(Exception):
    """正解・較正を作れない(tcp.prepare が SystemExit に直す)"""


# ==================================================================== 置き場所・版
def dataset_name(data_root):
    return os.path.basename(os.path.normpath(str(data_root)))


def derived_dir(data_root):
    return Path(DERIVED_ROOT) / dataset_name(data_root)


def labels_file(wp_steps=WP_STEPS, wp_dt=WP_DT):
    """既定(10 点・0.1 秒)は tcp_labels_v1.npz。ほかの形は名前に入れる(既定のキャッシュを壊さない)"""
    if int(wp_steps) == WP_STEPS and abs(float(wp_dt) - WP_DT) < 1e-9:
        return LABELS_NAME + '.npz'
    return '%s-n%d-dt%dms.npz' % (LABELS_NAME, int(wp_steps), int(round(float(wp_dt) * 1000)))


def labels_path(data_root, wp_steps=WP_STEPS, wp_dt=WP_DT):
    return derived_dir(data_root) / labels_file(wp_steps, wp_dt)


def calib_path(data_root):
    return derived_dir(data_root) / (CALIB_NAME + '.json')


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_splits(data_root):
    """→ (splits の dict, splits.json の sha1)"""
    p = os.path.join(str(data_root), 'splits.json')
    with open(p, encoding='utf-8') as f:
        return json.load(f), file_sha1(p)


def all_scenes(splits):
    """split の全部のシーン(train → val → ほか。重なりは1回)"""
    out = []
    keys = [k for k in ('train', 'val') if k in splits] + sorted(k for k in splits if k not in ('train', 'val'))
    for k in keys:
        for s in splits[k]:
            if s not in out:
                out.append(s)
    return out


def has_can_bus(data_root):
    d = os.path.join(str(data_root), 'can_bus')
    return os.path.isdir(d) and any(n.endswith('_ms_imu.json') for n in os.listdir(d))


def log_of(scene):
    """シーン名 '20261010-152405-0002' → ログ '20261010-152405'(ジャイロのバイアスの単位)"""
    return scene.rsplit('-', 1)[0] if '-' in scene else scene


def aligned_times(data_root, scene):
    """aligned の撮像時刻 [N] µs(int64)"""
    with np.load(os.path.join(str(data_root), 'aligned', scene + '.npz')) as z:
        return z['timestamps_us'].astype(np.int64)


def _write_atomic_bytes(path, write):
    """write(tmp のパス) で書いてから置き換える(電源断で半端なファイルを残さない)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('%s.%d.tmp%s' % (path.stem, os.getpid(), path.suffix))
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_json(path, obj):
    def w(tmp):
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
    _write_atomic_bytes(path, w)


# ==================================================================== 記録の読み込みと推測航法
def load_logs(data_root, scene):
    """can_bus の IMU・車両の記録 → 時刻の順の配列。どちらかが無ければ None。
    it/vt: 時刻 µs、gz: rad/s(左 正)、ax/ay: m/s²、erpm、st/th: µs、vok: 人の操作の有効なサンプル"""
    d = os.path.join(str(data_root), 'can_bus')
    pi, pv = os.path.join(d, scene + '_ms_imu.json'), os.path.join(d, scene + '_vehicle_monitor.json')
    if not (os.path.exists(pi) and os.path.exists(pv)):
        return None
    with open(pi, encoding='utf-8') as f:
        imu = json.load(f)
    with open(pv, encoding='utf-8') as f:
        veh = json.load(f)
    if len(imu) < 2 or len(veh) < 2:
        return None
    it = np.array([r['utime'] for r in imu], np.int64)
    oi = np.argsort(it, kind='stable')
    vt = np.array([r['utime'] for r in veh], np.int64)
    ov = np.argsort(vt, kind='stable')
    rot = np.array([r['rotation_rate'] for r in imu], np.float64).reshape(len(imu), -1)
    acc = np.array([r['linear_accel'] for r in imu], np.float64).reshape(len(imu), -1)
    valid = [r.get('valid') or [True, True] for r in veh]
    vok = np.array([bool(v[0]) and bool(v[1]) and not r.get('ai_driving', False) for v, r in zip(valid, veh)])
    return {'it': it[oi], 'gz': rot[oi, 2], 'ax': acc[oi, 0], 'ay': acc[oi, 1],
            'vt': vt[ov], 'erpm': np.array([r['erpm'] for r in veh], np.float64)[ov],
            'st': np.array([r['steering_us'] for r in veh], np.float64)[ov],
            'th': np.array([r['throttle_us'] for r in veh], np.float64)[ov], 'vok': vok[ov]}


def speed_of_erpm(erpm):
    return -np.asarray(erpm, np.float64) * SPEED_PER_ERPM


def gyro_biases(logs_by_scene):
    """{シーン: logs} → {ログ: (バイアス rad/s, 止まっているサンプル数)}。50 サンプル未満のログは 0"""
    still = {}
    for scene, lg in logs_by_scene.items():
        if lg is None:
            continue
        er = np.interp(lg['it'], lg['vt'], lg['erpm'])
        m = (np.abs(er) < BIAS_STILL_ERPM) & (np.abs(lg['gz']) < BIAS_STILL_GZ)
        still.setdefault(log_of(scene), []).append(lg['gz'][m])
    out = {}
    for log, parts in still.items():
        x = np.concatenate(parts) if parts else np.zeros(0)
        out[log] = (float(np.median(x)) if len(x) >= BIAS_MIN_N else 0.0, int(len(x)))
    return out


def dead_reckon(t, gz, bias, v):
    """IMU の時刻 t [s] で ψ・X・Y(記録の最初を原点)。v は IMU の時刻の速さ [m/s]"""
    dt = np.clip(np.diff(t, prepend=t[0]), 0.0, DT_MAX)
    psi = np.cumsum((gz - bias) * dt)
    x = np.cumsum(v * np.cos(psi) * dt)
    y = np.cumsum(v * np.sin(psi) * dt)
    return psi, x, y


def target_speed(t0, tv, v):
    """t₀ + 0.2〜0.5 秒の v の平均(線形補間した v の時間の平均。tv [s] の v [m/s])。t0 はスカラーか配列"""
    t0 = np.asarray(t0, np.float64)
    off = np.linspace(V_TGT_WINDOW[0], V_TGT_WINDOW[1], V_TGT_POINTS)
    return np.interp(t0[..., None] + off, tv, v).mean(axis=-1)


def impact_mask_frames(timestamps_us, imu, impact_g=IMPACT_G, before_s=IMPACT_BEFORE_S, after_s=IMPACT_AFTER_S):
    """train_data.impact_mask と同じ計算(torch を読まないための写し。テストで一致を確かめる)。
    衝突の前後(before_s 前 〜 after_s 後)のフレームを True"""
    t = np.asarray(timestamps_us, np.float64) / 1e6
    acc = np.asarray(imu, np.float32)[:, :, :2]
    hg = np.hypot(acc[..., 0], acc[..., 1]).max(axis=1) / GRAVITY
    mask = np.zeros(len(t), bool)
    for ti in t[hg >= impact_g]:
        mask |= (t >= ti - before_s) & (t <= ti + after_s)
    return mask


def impact_times(lg):
    """水平の加速度(重力を除く。IMU は +x 右・+y 前)が 2.5G 以上の時刻 [s]"""
    hg = np.hypot(lg['ax'], lg['ay']) / GRAVITY
    return lg['it'][hg >= IMPACT_G] / 1e6


def scene_labels(ts_us, lg, bias, wp_steps=WP_STEPS, wp_dt=WP_DT):
    """1 シーンの全フレームの正解。ts_us: aligned の撮像時刻 [N] µs、lg: load_logs の戻り値(None 可)。
    → ({'waypoints' [N,W,2], 'v_tgt' [N], 'v_now' [N], 'wp_valid' [N]}, 無効の理由ごとの数)"""
    n, w = len(ts_us), int(wp_steps)
    out = {'waypoints': np.zeros((n, w, 2), np.float32), 'v_tgt': np.zeros(n, np.float32),
           'v_now': np.zeros(n, np.float32), 'wp_valid': np.zeros(n, bool)}
    reasons = dict.fromkeys(INVALID_REASONS, 0)
    if lg is None:
        reasons['no_log'] = n
        return out, reasons
    t, tv = lg['it'] / 1e6, lg['vt'] / 1e6
    v_veh = speed_of_erpm(lg['erpm'])
    v_imu = np.interp(t, tv, v_veh)
    psi, X, Y = dead_reckon(t, lg['gz'], bias, v_imu)
    t0 = np.asarray(ts_us, np.float64) / 1e6
    horizon = w * float(wp_dt)
    taus = float(wp_dt) * np.arange(1, w + 1)
    tq = t0[:, None] + taus[None, :]
    p0, x0, y0 = np.interp(t0, t, psi), np.interp(t0, t, X), np.interp(t0, t, Y)
    dx, dy = np.interp(tq, t, X) - x0[:, None], np.interp(tq, t, Y) - y0[:, None]
    c, s = np.cos(p0)[:, None], np.sin(p0)[:, None]
    wp = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)

    inside = (t0 >= max(t[0], tv[0])) & (t0 + horizon <= min(t[-1], tv[-1]))
    gaps = np.diff(t)
    ti = impact_times(lg)
    valid = np.zeros(n, bool)
    for i in range(n):
        if not inside[i]:
            reasons['outside'] += 1
            continue
        a, b = t0[i], t0[i] + horizon
        lo = max(int(np.searchsorted(t, a, 'right')) - 1, 0)        # 最後の t ≤ t₀
        hi = min(int(np.searchsorted(t, b, 'left')), len(t) - 1)     # 最初の t ≥ t₀ + 1 秒
        if hi <= lo or gaps[lo:hi].max() >= DT_MAX:
            reasons['imu_gap'] += 1
            continue
        jv = slice(int(np.searchsorted(tv, a, 'left')), int(np.searchsorted(tv, b, 'right')))
        if min(v_veh[jv].min(initial=np.inf), float(np.interp(a, tv, v_veh)), float(np.interp(b, tv, v_veh))) < V_MIN_WINDOW:
            reasons['reverse'] += 1
            continue
        if len(ti) and np.any((ti >= a - IMPACT_AFTER_S) & (ti <= b + IMPACT_BEFORE_S)):
            reasons['impact'] += 1
            continue
        if np.abs(lg['gz'][lo:hi + 1]).max() > GZ_MAX:
            reasons['gyro_sat'] += 1
            continue
        valid[i] = True
    out['waypoints'][valid] = wp[valid].astype(np.float32)
    out['wp_valid'] = valid
    out['v_tgt'][inside] = np.clip(target_speed(t0[inside], tv, v_veh), 0.0, V_MAX).astype(np.float32)
    out['v_now'][inside] = np.interp(t0[inside], tv, v_veh).astype(np.float32)
    return out, reasons


# ==================================================================== 正解のキャッシュ
def build_labels(data_root, wp_steps=WP_STEPS, wp_dt=WP_DT, path=None):
    """split の全部のシーンの全フレームの正解を作って npz に書く(tmp → 置き換え)。→ 概要の dict"""
    if not has_can_bus(data_root):
        raise LabelError('TCP 型には can_bus の IMU・車両の記録が要ります(%s/can_bus)' % data_root)
    t_start = time.time()
    splits, sha = read_splits(data_root)
    scenes = all_scenes(splits)
    logs = {s: load_logs(data_root, s) for s in scenes}
    if all(lg is None for lg in logs.values()):
        raise LabelError('TCP 型には can_bus の IMU・車両の記録が要ります(どのシーンにもありません)')
    bias = gyro_biases(logs)
    cols = {k: [] for k in ('scene', 'frame', 'waypoints', 'v_tgt', 'v_now', 'wp_valid')}
    reasons = dict.fromkeys(INVALID_REASONS, 0)
    scene_n = []
    for s in scenes:
        ts = aligned_times(data_root, s)
        lab, r = scene_labels(ts, logs[s], bias.get(log_of(s), (0.0, 0))[0], wp_steps, wp_dt)
        for k, v in r.items():
            reasons[k] += v
        n = len(ts)
        scene_n.append(n)
        cols['scene'].append(np.full(n, s))
        cols['frame'].append(np.arange(n, dtype=np.int32))
        for k in ('waypoints', 'v_tgt', 'v_now', 'wp_valid'):
            cols[k].append(lab[k])
    arr = {k: np.concatenate(v) for k, v in cols.items()}
    summary = _summary(data_root, splits, scenes, logs, arr, bias, reasons)
    summary['seconds'] = round(time.time() - t_start, 1)
    path = Path(path or labels_path(data_root, wp_steps, wp_dt))

    def w(tmp):
        np.savez(tmp, scene=arr['scene'].astype(str), frame=arr['frame'], waypoints=arr['waypoints'],
                 v_tgt=arr['v_tgt'], v_now=arr['v_now'], wp_valid=arr['wp_valid'], version=np.int32(VERSION),
                 splits_sha1=np.array(sha), wp_steps=np.int32(wp_steps), wp_dt=np.float64(wp_dt),
                 scenes=np.array(scenes, dtype=str), scene_frames=np.array(scene_n, np.int64),
                 summary=np.array(json.dumps(summary, ensure_ascii=False)))
    _write_atomic_bytes(path, w)
    return summary


def _summary(data_root, splits, scenes, logs, arr, bias, reasons):
    """正解の概要と符号の確認(報告・CLI 用)"""
    val = set(splits.get('val', []))
    modes, steer, used = [], [], []
    for s in scenes:
        with np.load(os.path.join(str(data_root), 'aligned', s + '.npz')) as z:
            n = len(z['timestamps_us'])
            m = z['mode'] if 'mode' in z.files else np.zeros(n, np.int16)
            base = z['future_valid'] & z['vehicle_valid'] if 'future_valid' in z.files else np.ones(n, bool)
            modes.append(np.where(base, m, -1))
            steer.append(z['vehicle'][:, 0].astype(np.float64) if 'vehicle' in z.files else np.full(n, np.nan))
            imp = impact_mask_frames(z['timestamps_us'], z['imu']) if 'imu' in z.files else np.zeros(n, bool)
            used.append(base & ~imp)            # 学習に使うフレーム(衝突 2.5G の前後を除く。モデル B と同じ)
    modes, steer, used = np.concatenate(modes), np.concatenate(steer), np.concatenate(used)
    valid, wp, vn = arr['wp_valid'], arr['waypoints'], arr['v_now']
    per_mode = {}
    for m in sorted(set(int(x) for x in np.unique(modes)) - {-1, 0}):
        sel = modes == m
        su = sel & used
        per_mode[str(m)] = {'frames': int(sel.sum()), 'valid_frac': round(float(valid[sel].mean()), 3) if sel.any() else None,
                            'frames_used': int(su.sum()),
                            'valid_frac_used': round(float(valid[su].mean()), 3) if su.any() else None}
    # 符号: ステアが左(中心より +150 µs 以上)で走っているフレームの 1 秒先の y は正(左)のはず
    center = action_reference(data_root, splits.get('train', scenes))[0]
    mov = valid & (vn > 1.0)
    left, right = mov & (steer - center > 150), mov & (steer - center < -150)
    corr = []
    yaw = []
    for s, lg in logs.items():
        if lg is None:
            continue
        b = bias.get(log_of(s), (0.0, 0))[0]
        v = speed_of_erpm(lg['erpm'])
        gz = np.interp(lg['vt'], lg['it'], lg['gz']) - b
        m = lg['vok'] & (v > CAL_V_MIN)
        if m.sum() > 50 and np.std(lg['st'][m]) > 0 and np.std(gz[m]) > 0:
            corr.append(float(np.corrcoef(lg['st'][m], gz[m])[0, 1]))
        t = lg['it'] / 1e6
        yaw.append(float(np.sum((lg['gz'] - b) * np.clip(np.diff(t, prepend=t[0]), 0, DT_MAX))))
    vt = arr['v_tgt'][valid]
    return {
        'dataset': dataset_name(data_root), 'version': VERSION, 'scenes': len(scenes),
        'scenes_without_log': [s for s, lg in logs.items() if lg is None],
        'frames': int(len(valid)), 'valid_frames': int(valid.sum()), 'valid_frac': round(float(valid.mean()), 3),
        'invalid_reasons': reasons,
        'valid_frac_by_mode(frames = future_valid & vehicle_valid、used = さらに 2.5G の前後を除く)': per_mode,
        'gyro_bias_rad_s': {k: [round(b, 5), n] for k, (b, n) in sorted(bias.items())},
        'v_tgt_p10_50_90_99(valid)': [round(float(x), 2) for x in np.percentile(vt, [10, 50, 90, 99])] if len(vt) else None,
        'sign_checks': {
            'steer_vs_yawrate_corr_median(>0 = ステア大 = 左 = gz 正)': round(float(np.median(corr)), 3) if corr else None,
            'scenes_positive_total_yaw(左回りのコース)': [int(np.sum(np.array(yaw) > 0)), len(yaw)],
            'left_steer_frames_y1s_positive_frac': round(float((wp[left, -1, 1] > 0).mean()), 3) if left.any() else None,
            'right_steer_frames_y1s_negative_frac': round(float((wp[right, -1, 1] < 0).mean()), 3) if right.any() else None,
            'moving_frames_x1s_positive_frac(前進 = x 正)': round(float((wp[mov, -1, 0] > 0).mean()), 3) if mov.any() else None},
        'val_scenes': sorted(val & set(scenes)),
    }


def load_labels(path):
    """npz → dict(scene は str の配列)"""
    with np.load(str(path), allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out['scene'] = out['scene'].astype(str)
    return out


def labels_fresh(data_root, wp_steps=WP_STEPS, wp_dt=WP_DT, path=None):
    """→ (使えるか, 理由)。版・splits の sha1・通過点の形・シーンとフレーム数が今のデータセットと同じか"""
    path = Path(path or labels_path(data_root, wp_steps, wp_dt))
    if not path.exists():
        return False, 'ありません'
    try:
        with np.load(str(path), allow_pickle=False) as z:
            if int(z['version']) != VERSION:
                return False, '版が違います'
            splits, sha = read_splits(data_root)
            if str(z['splits_sha1']) != sha:
                return False, 'splits.json が変わりました'
            if int(z['wp_steps']) != int(wp_steps) or abs(float(z['wp_dt']) - float(wp_dt)) > 1e-9:
                return False, '通過点の形が違います'
            scenes = all_scenes(splits)
            if list(z['scenes'].astype(str)) != scenes:
                return False, 'シーンが違います'
            counts = z['scene_frames']
    except (OSError, KeyError, ValueError) as e:
        return False, '読めません(%s)' % e
    for s, n in zip(scenes, counts):
        if len(aligned_times(data_root, s)) != int(n):
            return False, 'シーン %s のフレーム数が変わりました' % s
    return True, ''


def ensure_labels(data_root, wp_steps=WP_STEPS, wp_dt=WP_DT, force=False):
    """→ (npz のパス, 作ったか)。済んでいれば何もしない"""
    path = labels_path(data_root, wp_steps, wp_dt)
    if not force and labels_fresh(data_root, wp_steps, wp_dt, path)[0]:
        return path, False
    build_labels(data_root, wp_steps, wp_dt, path)
    return path, True


def labels_by_scene(data_root, wp_steps=WP_STEPS, wp_dt=WP_DT, scenes=None, keys=('waypoints', 'v_tgt', 'wp_valid')):
    """{シーン: {キー: [aligned のフレーム数, ...]}}(npz に無いフレームは 0 と無効)。scenes = None なら split の全部"""
    lab = load_labels(labels_path(data_root, wp_steps, wp_dt))
    if scenes is None:
        scenes = all_scenes(read_splits(data_root)[0])
    order = np.argsort(lab['scene'], kind='stable')
    sorted_scenes = lab['scene'][order]
    out = {}
    for s in scenes:
        n = len(aligned_times(data_root, s))
        a, b = np.searchsorted(sorted_scenes, s, 'left'), np.searchsorted(sorted_scenes, s, 'right')
        rows = order[a:b]
        fr = lab['frame'][rows].astype(np.int64)
        keep = (fr >= 0) & (fr < n)
        rows, fr = rows[keep], fr[keep]
        d = {}
        for k in keys:
            src = lab[k]
            arr = np.zeros((n,) + src.shape[1:], src.dtype)
            arr[fr] = src[rows]
            d[k] = arr
        out[s] = d
    return out


# ==================================================================== 較正(学習のシーンだけ)
def action_reference(data_root, scenes):
    """(ステアの中心, スロットルの中立) [µs]。JetFormerDataset.action_reference と同じ計算(止まっている間の人の操作の中央値)
    → model_config.json の steer_center_us・throttle_neutral_us と同じ値"""
    rows = []
    for s in scenes:
        with np.load(os.path.join(str(data_root), 'aligned', s + '.npz')) as z:     # 要る配列だけ読む(ToF は読まない)
            if 'vehicle' not in z.files or 'tof' not in z.files or 'imu' not in z.files:
                continue
            veh = z['vehicle']
            ev = z['erpm_valid'] if 'erpm_valid' in z.files else True
            rows.append(veh[z['vehicle_valid'] & ev & (np.abs(veh[:, 3]) < 1)])
    rows = np.concatenate(rows) if rows else np.zeros((0, 4), np.float32)
    if len(rows) < 10:
        return 1500.0, 1500.0
    return round(float(np.median(rows[:, 0])), 1), round(float(np.median(rows[:, 1])), 1)


def _steer_samples(lg, bias, center, lag_ms):
    """人の操作のサンプルの時刻 t で (ステアのずれ S, κ(t+遅れ), v(t+遅れ))。κ = (gz − b)/v、v ≥ 0.8 m/s だけ"""
    t, tv = lg['it'] / 1e6, lg['vt'] / 1e6
    v_veh = speed_of_erpm(lg['erpm'])
    tq = tv + lag_ms / 1000.0
    w = np.interp(tq, t, lg['gz'] - bias)
    v = np.interp(tq, tv, v_veh)
    ok = lg['vok'] & (tq >= t[0]) & (tq <= min(t[-1], tv[-1])) & (v >= CAL_V_MIN) & (np.abs(w) <= GZ_MAX)
    return lg['st'][ok] - center, w[ok] / v[ok], v[ok]


def _linear_r2(S, K):
    """|S| ≤ 250 で、左右別の原点を通る直線 κ = k·S の当てはまり → (R², k_left, k_right, n)"""
    lin = np.abs(S) <= LIN_US
    S, K = S[lin], K[lin]
    if len(S) < 10:
        return -np.inf, 0.0, 0.0, int(len(S))
    pos, neg = S > 0, S < 0
    kl = float(np.sum(S[pos] * K[pos]) / np.sum(S[pos] ** 2)) if pos.any() else 0.0
    kr = float(np.sum(S[neg] * K[neg]) / np.sum(S[neg] ** 2)) if neg.any() else 0.0
    pred = np.where(S > 0, kl * S, kr * S)
    sst = float(np.sum((K - K.mean()) ** 2))
    r2 = 1.0 - float(np.sum((K - pred) ** 2)) / sst if sst > 0 else -np.inf
    return r2, kl, kr, int(len(S))


def _binned_median(S, K, centers_n):
    """ずれの区切りごとの κ の中央値(MIN_BIN_N 未満は nan)と数"""
    idx = np.digitize(S, OFFSET_EDGES) - 1
    med = np.full(centers_n, np.nan)
    cnt = np.zeros(centers_n, np.int64)
    for j in range(centers_n):
        m = idx == j
        cnt[j] = int(m.sum())
        if cnt[j] >= MIN_BIN_N:
            med[j] = float(np.median(K[m]))
    return med, cnt


def monotone_table(curv):
    """ずれについて単調に増える形にそろえ(np.maximum.accumulate)、逆引きのため 1e-6 ずつ足して狭義単調に"""
    k = np.maximum.accumulate(np.asarray(curv, np.float64))
    return k + np.arange(len(k)) * 1e-6


def fit_calibration(data_root):
    """学習のシーン(splits の train)の人の操作だけで、ステア → 曲率の表とスロットルの前向きの写像を当てはめる → dict"""
    splits, sha = read_splits(data_root)
    train = list(splits.get('train', []))
    logs = {s: load_logs(data_root, s) for s in train}
    if all(lg is None for lg in logs.values()):
        raise LabelError('TCP 型の較正には学習のシーンの can_bus の記録が要ります')
    bias = gyro_biases(logs)
    center, neutral = action_reference(data_root, train)
    bias_of = {s: bias.get(log_of(s), (0.0, 0))[0] for s in train}

    # ---- ステア → 曲率: 遅れを選ぶ
    by_lag = {}
    for lag in LAGS_MS:
        parts = [_steer_samples(lg, bias_of[s], center, lag) for s, lg in logs.items() if lg is not None]
        S, K, V = (np.concatenate([p[i] for p in parts]) for i in range(3))
        r2, kl, kr, n = _linear_r2(S, K)
        by_lag[lag] = {'r2': r2, 'k_left_per_us': kl, 'k_right_per_us': kr, 'n': n, 'samples': (S, K, V)}
    lag = max(LAGS_MS, key=lambda x: by_lag[x]['r2'])
    if not np.isfinite(by_lag[lag]['r2']):
        raise LabelError('ステアの較正に使えるサンプルがありません(v ≥ %.1f m/s・|ステア − 中心| ≤ %d µs)' % (CAL_V_MIN, LIN_US))
    S, K, V = by_lag[lag]['samples']
    centers = 0.5 * (OFFSET_EDGES[:-1] + OFFSET_EDGES[1:])
    nb = len(centers)
    all_med, all_n = _binned_median(S, K, nb)
    ok = np.isfinite(all_med)                  # 全速度で 30 個以上ある区切り
    if ok.sum() < 2:
        raise LabelError('ステアの較正の区切りが足りません(%d 個)' % int(ok.sum()))
    all_med = np.interp(centers, centers[ok], all_med[ok])            # 足りない区切りは隣から(端はそのまま = いっぱい)
    edges = (-np.inf,) + tuple(BANDS_MPS) + (np.inf,)
    curv, counts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (V >= lo) & (V < hi)
        med, cnt = _binned_median(S[m], K[m], nb)
        med = np.where(np.isfinite(med) | ~ok, med, all_med)        # 30 個未満の区切りは全速度の表の値
        own = np.isfinite(med)
        if own.sum() >= 2:      # どの速さにもサンプルが無い区切り(表の端)は、この速度帯の端の値をそのまま延ばす(= いっぱい)
            med = np.interp(centers, centers[own], med[own])
        else:
            med = all_med
        curv.append(monotone_table(med))
        counts.append(cnt)

    # ---- スロットル: thr = c0 + c1·v + c2·(v_tgt − v)/0.33(前進のサンプル)
    X, Y = [], []
    for s, lg in logs.items():
        if lg is None:
            continue
        tv = lg['vt'] / 1e6
        v = speed_of_erpm(lg['erpm'])
        m = lg['vok'] & (tv + V_TGT_WINDOW[1] <= tv[-1]) & (v >= THR_V_MIN) & (lg['th'] < neutral - THR_FWD_US)
        if not m.any():
            continue
        vt = target_speed(tv[m], tv, v)
        X.append(np.stack([np.ones(int(m.sum())), v[m], (vt - v[m]) / TAU_S], axis=1))
        Y.append(lg['th'][m])
    if not X or sum(len(y) for y in Y) < 10:
        raise LabelError('スロットルの較正に使える前進のサンプルがありません')
    X, Y = np.concatenate(X), np.concatenate(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    fwd_max = float(np.percentile(neutral - Y, FWD_MAX_PCT))
    return {
        'version': VERSION, 'dataset': dataset_name(data_root), 'splits_sha1': sha, 'scenes': 'train',
        'train_scenes': len(train), 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'steer_table': {'lag_ms': int(lag), 'offset_us': [float(c) for c in centers], 'bands_mps': [float(b) for b in BANDS_MPS],
                        'curv': [[round(float(x), 6) for x in c] for c in curv], 'center_us': center,
                        'counts': [[int(x) for x in c] for c in counts], 'all_speed_counts': [int(x) for x in all_n],
                        'v_min_mps': CAL_V_MIN,
                        'r2_by_lag': {str(k): round(float(v['r2']), 4) for k, v in by_lag.items()},
                        'slope_left_right_per_us': [round(by_lag[lag]['k_left_per_us'], 6), round(by_lag[lag]['k_right_per_us'], 6)]},
        'throttle_ff': {'c0': round(float(coef[0]), 3), 'c1': round(float(coef[1]), 3), 'c2': round(float(coef[2]), 3),
                        'tau_s': TAU_S, 'fwd_max_us': round(max(fwd_max, 0.0), 1), 'neutral_us': neutral,
                        'train_mae_us': round(float(np.mean(np.abs(X @ coef - Y))), 1), 'n': int(len(Y))},
        'gyro_bias_rad_s': {k: [round(b, 5), n] for k, (b, n) in sorted(bias.items())},
    }


def calibration_fresh(data_root, path=None):
    path = Path(path or calib_path(data_root))
    if not path.exists():
        return False, 'ありません'
    try:
        with open(path, encoding='utf-8') as f:
            c = json.load(f)
    except (OSError, ValueError) as e:
        return False, '読めません(%s)' % e
    if c.get('version') != VERSION:
        return False, '版が違います'
    if c.get('splits_sha1') != read_splits(data_root)[1]:
        return False, 'splits.json が変わりました'
    return True, ''


def ensure_calibration(data_root, force=False):
    """→ (json のパス, 作ったか)"""
    path = calib_path(data_root)
    if not force and calibration_fresh(data_root, path)[0]:
        return path, False
    _write_json(path, fit_calibration(data_root))
    return path, True


def load_calibration(data_root):
    with open(calib_path(data_root), encoding='utf-8') as f:
        return json.load(f)
