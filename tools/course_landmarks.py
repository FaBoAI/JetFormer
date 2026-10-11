#!/usr/bin/env python3
"""カメラ画像から、コース上の目印(床のスタートライン・芝の端・青マットの端・LED 矢印ゲート)の手がかりを取り出す

1枚ごとに「行ごとの色の割合」だけを計算して data/labels/course_cam/<ログ>.npz にためておく(1ログ 1〜3分、CPU)。
目印を「いつ横切ったか」は、当てはめの途中の軌跡(どこを走っていたか)と組み合わせて tools/course_fit.py で決める。

    python tools/course_landmarks.py 20261010-200916 [...]       # 特徴をためる(既にあれば飛ばす)
    python tools/course_landmarks.py --all

色の決め方(2026-10-10 の画像で確かめた値。OpenCV の HSV、H は 0〜180):
- 白いスタートライン: V > 175 かつ S < 60(中央の列 60〜163)
- 芝: H 18〜48、S 18〜90、V 40〜150(列 40〜183)。カーペットは S < 15
- 青マット: H 96〜107、S 35〜95、V > 140(左右半分ずつ)。車のバンパーも青い(H 107〜110、S 70〜120、V 100〜170)
  ので、バンパーが写る下端(行 212 より下)は使わない
- LED 矢印ゲートの緑: G > R + 25 かつ G > B + 10 かつ G > 110(上の 64 行)
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'labels', 'course_cam')
ROW0, ROW1 = 80, 210          # 床の特徴を見る行(行 ROW0〜ROW1-1)。下端のバンパー(行 212〜)は入れない
GATE_ROWS = 64
VERSION = 1


def frame_features(img):
    """1枚の画像(BGR、224x224)→ 行ごとの割合(0〜255 の uint8)とゲートの緑"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[ROW0:ROW1, :, 0]
    s = hsv[ROW0:ROW1, :, 1]
    v = hsv[ROW0:ROW1, :, 2]
    white = (v > 175) & (s < 60)
    turf = (h >= 18) & (h <= 48) & (s >= 18) & (s <= 90) & (v >= 40) & (v <= 150)
    blue = (h >= 96) & (h <= 107) & (s >= 35) & (s <= 95) & (v > 140)
    top = img[:GATE_ROWS].astype(np.int16)
    b, g, r = top[..., 0], top[..., 1], top[..., 2]
    gm = (g > r + 25) & (g > b + 10) & (g > 110)
    cols = np.nonzero(gm.any(0))[0]
    n = int(gm.sum())
    gate_col = float((gm.sum(0) * np.arange(gm.shape[1])).sum() / n) if n else np.nan
    return {'white': np.round(white[:, 60:164].mean(1) * 255).astype(np.uint8),
            'turf': np.round(turf[:, 40:184].mean(1) * 255).astype(np.uint8),
            'blue_l': np.round(blue[:, :112].mean(1) * 255).astype(np.uint8),
            'blue_r': np.round(blue[:, 112:].mean(1) * 255).astype(np.uint8),
            'gate_rows': np.minimum(gm.sum(1), 255).astype(np.uint8),
            'gate_n': n, 'gate_col': gate_col,
            'gate_c0': int(cols.min()) if len(cols) else -1, 'gate_c1': int(cols.max()) if len(cols) else -1,
            'vmean': float(hsv[..., 2].mean())}


def list_frames(log, raw=RAW):
    """ログの全シーンのカメラ画像(時刻 us、パス、シーン番号)。壊れたシーンは飛ばす"""
    root = os.path.join(raw, log)
    scenes = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
    ts, paths, sid = [], [], []
    for k, sc in enumerate(scenes):
        try:
            c = np.load(os.path.join(root, sc, 'CAM_FRONT.npz'), allow_pickle=True)
            t = c['timestamps_us'].astype(np.int64)
            f = c['files']
        except Exception:
            continue
        ts.append(t)
        paths += [os.path.join(root, sc, str(x)) for x in f]
        sid.append(np.full(len(t), k))
    if not ts:
        return np.zeros(0, np.int64), [], np.zeros(0, int)
    return np.concatenate(ts), paths, np.concatenate(sid)


def _work(paths):
    out = []
    for p in paths:
        im = cv2.imread(p)
        out.append(frame_features(im) if im is not None else None)
    return out


def extract(log, raw=RAW, cache=CACHE, workers=2, force=False):
    """1ログ分の特徴を計算してためる。戻り値: 保存したファイルのパス"""
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, log + '.npz')
    if os.path.exists(path) and not force:
        try:
            if int(np.load(path)['version']) == VERSION:
                return path
        except Exception:
            pass
    ts, paths, sid = list_frames(log, raw)
    n = len(paths)
    chunks = [paths[i:i + 300] for i in range(0, n, 300)]
    res = []
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(workers) as ex:
            for r in ex.map(_work, chunks):
                res += r
    else:
        for c in chunks:
            res += _work(c)
    nr = ROW1 - ROW0
    arr = {k: np.zeros((n, nr), np.uint8) for k in ('white', 'turf', 'blue_l', 'blue_r')}
    arr['gate_rows'] = np.zeros((n, GATE_ROWS), np.uint8)
    for k in ('gate_n', 'gate_c0', 'gate_c1'):
        arr[k] = np.full(n, -1, np.int32)
    arr['gate_col'] = np.full(n, np.nan, np.float32)
    arr['vmean'] = np.full(n, np.nan, np.float32)
    ok = np.zeros(n, bool)
    for i, r in enumerate(res):
        if r is None:
            continue
        ok[i] = True
        for k in arr:
            arr[k][i] = r[k]
    np.savez_compressed(path, version=VERSION, timestamps_us=ts, scene=sid, ok=ok, row0=ROW0, row1=ROW1, **arr)
    return path


def load_or_extract(log, cache=CACHE):
    extract(log, cache=cache)
    return load(log, cache)


def load(log, cache=CACHE):
    d = np.load(os.path.join(cache, log + '.npz'))
    return {k: d[k] for k in d.files}


# ---------------------------------------------------------------- 目印を横切った瞬間
REF_ROW = 200            # 床の目印: この行に届いた瞬間を「横切った」とする
GATE_REF_ROW = 8         # ゲート: 緑の表示の下端がこの行より上へ抜けた瞬間
# 行 v → 床の上の距離のモデル d(v) = K / (v - vh)(ピンホール・傾きのない近似)。2026-10-10 20:09 の人のログの
# スタートライン 51 回の通過で合わせた値(残差の RMS 1.9 cm)。ログごとに通過が多ければ合わせ直す
FLOOR_MODEL = (76.6, 42.0)
GATE_MODEL = (150.0, 120.0)
KINDS = ['line', 'turf_in', 'turf_out', 'mat_n', 'gate']     # mat_s(マットの手前の端)はばらつきが大きく使わない(2026-10-11 の検証)
ALL_KINDS = KINDS + ['mat_s']


def _lowest(prof, thr):
    """行ごとの割合 prof [N, R] で thr を超える一番下の行(無ければ -1)"""
    m = prof > thr
    any_ = m.any(1)
    last = m.shape[1] - 1 - np.argmax(m[:, ::-1], axis=1)
    return np.where(any_, last, -1)


def _bottom_run_top(prof, thr, min_frac=0.6):
    """下端から続く「thr を超える行」の一番上の行(下端が thr を超えていなければ -1)"""
    m = prof > thr
    n = m.shape[1]
    out = np.full(len(m), -1)
    ok = m[:, -1]
    rev = m[:, ::-1]
    first_false = np.where(rev.all(1), n, np.argmax(~rev, axis=1))
    out[ok] = n - first_false[ok]
    return out


def boundary_rows(F, kind):
    """目印の種類ごとの境界の行(画像の行番号。無ければ -1)。
    line: 白い帯の一番下 / turf_in, mat_n: 向こう側にある芝・マットの一番下の行(近づくと下がってくる)/
    turf_out, mat_s: 下端から続く芝・マットの一番上の行(出ていくと下がってくる)/ gate: 緑の表示の一番下の行"""
    r0 = int(F['row0'])
    if kind == 'gate':
        g = F['gate_rows'].astype(int)
        m = g >= 3
        last = m.shape[1] - 1 - np.argmax(m[:, ::-1], axis=1)
        return np.where(m.any(1) & (F['gate_n'] >= 20), last, -1)
    if kind == 'line':
        b = _lowest(F['white'].astype(int), int(0.45 * 255))
    elif kind == 'turf_in':
        b = _lowest(F['turf'].astype(int), int(0.40 * 255))
    elif kind == 'turf_out':
        b = _bottom_run_top(F['turf'].astype(int), int(0.40 * 255))
    elif kind == 'mat_n':
        b = _lowest(F['blue_l'].astype(int) + F['blue_r'], int(0.10 * 255))
    elif kind == 'mat_s':
        b = _bottom_run_top(F['blue_l'].astype(int) + F['blue_r'], int(0.10 * 255))
    else:
        raise ValueError(kind)
    return np.where(b >= 0, b + r0, -1)


def floor_dist(v, model=FLOOR_MODEL, ref=REF_ROW):
    """行 v に写る床の点の、基準の行 ref に写る点からの距離 [m](先が +)"""
    K, vh = model
    return K / (np.asarray(v, float) - vh) - K / (ref - vh)


def gate_dist(v, model=GATE_MODEL, ref=GATE_REF_ROW):
    K, vh = model
    return K / (vh - np.asarray(v, float)) - K / (vh - ref)


def find_passes(b, mask, kind, vmin=None, vmax=None):
    """目印が基準の行へ向かって動き、横切って消える(または次の目印に入れ替わる)までの連続したコマの組。
    戻り値: [コマ番号の配列]。最後のコマの次で基準の行を越えている(または消えた)ものだけ"""
    gate = kind == 'gate'
    if vmin is None:
        vmin, vmax = (GATE_REF_ROW + 1, 32) if gate else (100, REF_ROW - 1)
    out = []
    cur = []
    n = len(b)

    def close(cur, nxt):
        if len(cur) < 2:
            return
        if nxt is None:
            return
        vb = b[nxt]
        if gate:
            crossed = (vb == -1) or (vb <= GATE_REF_ROW) or (vb > b[cur[-1]] + 15)
        else:
            crossed = (vb == -1) or (vb >= REF_ROW) or (vb < b[cur[-1]] - 30)
        if crossed and mask[nxt]:
            out.append(np.array(cur))

    for i in range(n):
        inr = mask[i] and b[i] >= 0 and vmin <= b[i] <= vmax
        if inr and cur and i == cur[-1] + 1:
            mono = (b[i] <= b[cur[-1]] + 3) if gate else (b[i] >= b[cur[-1]] - 3)
            if mono:
                cur.append(i)
                continue
        if cur:
            close(cur, i if i == cur[-1] + 1 else None)
        cur = [i] if inr else []
    return out


def crossing(idx, b, S, kind, model=None):
    """1回の通過のコマから、基準の行に届いた瞬間の走行距離 s を推定する(各コマの s + 距離(行) の中央値)。
    戻り値: (s, ばらつき [m])"""
    if kind == 'gate':
        e = S[idx] + gate_dist(b[idx], model or GATE_MODEL)
    else:
        e = S[idx] + floor_dist(b[idx], model or FLOOR_MODEL)
    use = idx[-4:] if len(idx) > 4 else idx       # 基準の行に近い(遠近の誤差が小さい)コマを使う
    e = e[-len(use):]
    return float(np.median(e)), float(np.std(e))


def fit_row_model(passes_b_s, kind, grid=None):
    """通過ごとの (行, 走行距離) の組から、行 → 距離のモデル (K, vh) を合わせる(通過ごとの定数は消す)。
    ゲート(床より高い物)は vh を床の地平線の行に固定して K だけ合わせる(grid に1点だけ渡す)"""
    if grid is None:
        grid = np.arange(-100, 95, 1.0)
    best = None
    for vh in grid:
        A, y = [], []
        nP = len(passes_b_s)
        for k, (bb, ss) in enumerate(passes_b_s):
            for v, s_ in zip(bb, ss):
                if (v - vh if kind != 'gate' else vh - v) < 2:
                    continue
                r = np.zeros(1 + nP)
                r[0] = 1.0 / (v - vh) if kind != 'gate' else 1.0 / (vh - v)
                r[1 + k] = -1
                A.append(r)
                y.append(-s_)
        A = np.array(A)
        y = np.array(y)
        if len(y) < nP + 3:
            return None
        sol = np.linalg.lstsq(A, y, rcond=None)[0]
        rr = A @ sol - y
        sse = float((rr ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(sol[0]), float(vh), float(np.sqrt(sse / len(y))), len(y))
    return {'K': best[1], 'vh': best[2], 'rms_m': best[3], 'n': best[4], 'passes': len(passes_b_s)}


def all_logs(raw=RAW):
    return sorted(x for x in os.listdir(raw) if os.path.isdir(os.path.join(raw, x)) and len(x) == 15)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs', nargs='*')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    logs = all_logs() if a.all else a.logs
    for log in logs:
        p = extract(log, workers=a.workers, force=a.force)
        d = np.load(p)
        print(json.dumps({'log': log, 'frames': int(len(d['ok'])), 'ok': int(d['ok'].sum())}))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
