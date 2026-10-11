"""走る前・走った後のセンサーの向きの点検(カメラと ToF が学習データと同じ向きか)

2026-10-10 の夜、カメラと ToF の取り付けが少しずつ下を向き、学習データと見え方が変わっていた疑いがある
(画像の下端に写る自車の青いバンパーが、学習データの 6〜8行 → 9〜10行。ToF の遠い床が基準より 14〜23% 近い)。
ロードマップ S0: 走行の前後にこの2つを測り、ずれていたら直してから走る。

  ・カメラ: 224x224 の画像の下端から、中央の列(80〜143)の半分以上が青(HSV 色相95〜130・彩度≥80・明度≥60)の行が
    何行続くか。目安は学習データと同じ 7行 ±1
  ・ToF: 前に障害物のない床の上で、行9(遠い床)と行18(近い床)の中央の列(20〜34)の距離が、点検の基準
    (data/tof_check_ref.npy)から何% ずれているか。目安は ±6%
    点検の基準は、向きがそろっていた学習データ4本(152405・152707・193638・200916)の走行中の床の距離の中央値。
    この基準では、成績のよかった AI走行 006/007 は +0.5/+3.7%、衝突が増えた 009/010/101 は −7.5/−11.4/−19.4%。
    カメラ(バンパーの行)は 006/007 で 2〜3行でも成績がよかったので、決め手ではなく注意として扱う。
"""

import time

import cv2
import numpy as np

from . import config, tof_format

BUMPER_TARGET = (6, 8)          # 学習データ(clean_driving_20261011 の 52シーン中 49、first_trial の学習ログすべて)
TOF_TOLERANCE_PCT = 6.0
TOF_ROWS = (9, 18)
TOF_COLS = slice(20, 35)
GOOD_CONF = 4                   # このToFで「確かな値」の信頼度(ToFガード・夜間の解析と同じ)
CHECK_REF_PATH = config.DATA_DIR / 'tof_check_ref.npy'      # 点検の基準(学習データの向き)
FLOOR_REF_PATH = config.DATA_DIR / 'tof_floor_ref.npy'      # ToF ガードの床の基準(無ければこちらで代用)


def bumper_rows(image_bgr):
    """画像の下端に写る青いバンパーの行数(224x224 の画像で)"""
    img = image_bgr
    if img.shape[0] != 224 or img.shape[1] != 224:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img[150:, 80:144], cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] >= 95) & (hsv[..., 0] <= 130) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 60)
    n = 0
    for frac in blue.mean(axis=1)[::-1]:          # 下の行から
        if frac < 0.5:
            break
        n += 1
    return n


def tof_floor_deviation(depth_frames, floor_ref_mm):
    """行9・行18 の床の距離が、床の基準から何% ずれているか {行: %}(信頼度4の画素の中央値)"""
    ref = np.asarray(floor_ref_mm, np.float32)
    out = {}
    for row in TOF_ROWS:
        vals = []
        for raw in depth_frames:
            raw = np.asarray(raw)
            d = tof_format.distance_mm(raw[row, TOF_COLS]).astype(np.float32)
            conf = tof_format.confidence(raw[row, TOF_COLS])
            d = d[(conf == GOOD_CONF) & (d > 0)]
            if d.size:
                vals.append(float(np.median(d)))
        r = float(np.median(ref[row, TOF_COLS]))
        out[row] = round((float(np.median(vals)) / r - 1.0) * 100.0, 1) if vals and r > 0 else None
    return out


def evaluate(images, depth_frames, erpm_values, floor_ref_mm=None):
    """集めたフレームから点検の結果(dict)を作る"""
    rows = [bumper_rows(im) for im in images]
    res = {'time': time.time(), 'frames': len(rows)}
    if rows:
        med = float(np.median(rows))
        res.update(bumper_rows=med, bumper_rows_range=[int(min(rows)), int(max(rows))],
                   bumper_ok=BUMPER_TARGET[0] <= med <= BUMPER_TARGET[1])
    res['bumper_target'] = list(BUMPER_TARGET)
    moving = bool(erpm_values) and float(np.median(np.abs(erpm_values))) > 200
    res['moving'] = moving
    if floor_ref_mm is not None and depth_frames:
        dev = tof_floor_deviation(depth_frames, floor_ref_mm)
        res['tof_floor_pct'] = {str(k): v for k, v in dev.items()}
        if dev.get(TOF_ROWS[0]) is not None:
            res['tof_ok'] = abs(dev[TOF_ROWS[0]]) <= TOF_TOLERANCE_PCT
    else:
        res['tof_floor_pct'] = None
    res['tof_tolerance_pct'] = TOF_TOLERANCE_PCT
    res['ok'] = res.get('tof_ok', True) and bool(rows) and not moving    # カメラは注意だけ(006/007 は 2〜3行でも良かった)
    notes = []
    if not rows:
        notes.append('カメラの映像が来ていません')
    elif not res['bumper_ok']:
        notes.append('注意: バンパーが %.0f 行(学習データは %d〜%d 行)。カメラの向きを%s' % (
            res['bumper_rows'], BUMPER_TARGET[0], BUMPER_TARGET[1],
            '上へ(写りすぎ = 下を向いている)' if res['bumper_rows'] > BUMPER_TARGET[1] else '下へ(写っていない = 上を向いている)'))
    if res.get('tof_ok') is False:
        dev = res['tof_floor_pct'][str(TOF_ROWS[0])]
        notes.append('ToF の遠い床が学習データより %+.1f%%(目安 ±%.0f%%)。ToF が%sを向いています。向きを直してから走ってください'
                     % (dev, TOF_TOLERANCE_PCT, '下' if dev < 0 else '上'))
    if floor_ref_mm is None:
        notes.append('ToF の点検の基準(data/tof_check_ref.npy)が無いので、ToF は測っていません')
    if moving:
        notes.append('車が動いています。止めて、前に障害物のない床の上で測ってください')
    res['notes'] = notes
    return res


def run(hub, seconds=1.0):
    """センサーの最新の値を seconds 秒集めて点検する(センサーが動いていること)"""
    ref = None
    for path in (CHECK_REF_PATH, FLOOR_REF_PATH):
        try:
            ref = np.load(path)
            break
        except (OSError, ValueError):
            continue
    images, depths, erpm = [], [], []
    seen = {}
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        for ch, sink in (('CAM_FRONT', images), ('TOF_FRONT', depths), ('VEHICLE', erpm)):
            s = hub[ch].latest
            if s is None or seen.get(ch) is s:
                continue
            seen[ch] = s
            if ch == 'CAM_FRONT':
                sink.append(s.data['image'])
            elif ch == 'TOF_FRONT':
                sink.append(s.data['depth'])
            else:
                sink.append(float(s.data.get('erpm', 0)))
        time.sleep(0.02)
    return evaluate(images, depths, erpm, ref)
