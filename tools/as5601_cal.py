#!/usr/bin/env python3
"""AS5601 の記録(tools/as5601_logger.py の CSV)から、ハンドルの左右いっぱい・まっすぐ・ガタを数える(読むだけ・I2C に触れない)

2026-10-29 ユーザー:「今から、プロポで車輪動かしてみます。左右に。値は取れますかね？」。手順(車輪を浮かせ、CH3 は人側):
まっすぐ 3 秒 → 左いっぱい 3 秒 → まっすぐ 3 秒 → 右いっぱい 3 秒 → まっすぐ 3 秒、を 3 回。最後に左右に小さく速く振る。
数えるもの:
  - 止まっている所(0.3 秒以上、角度の動きが 1 秒に 5 度より小さい)を「止まり」とし、その中央の角度を使う
  - まっすぐ = 止まりのうち、左右いっぱいの間にあるものの中央。左右いっぱい = いちばん外の止まり(最初に動いた向きを「左」とする)
  - 左右の振れ = いっぱい − まっすぐ(度)。左右の比 = 左の振れ ÷ 右の振れ
  - まっすぐの戻りのずれ = 左から戻ったときと右から戻ったときの、まっすぐの角度の差(ガタ・戻りの差の目安)
  - 揺れ = 止まりの中の角度の標準偏差(磁石が弱いと大きくなる)

    python3 -I tools/as5601_cal.py [data/as5601/<名前>.csv]     # 省くといちばん新しい記録
"""
import csv
import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEG = 360.0 / 4096


def load(path):
    t, a, st = [], [], []
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:                                      # 10/30 反証 S8: 電源が切れて最後の行が半分でも落ちない
                if None in r.values() or None in r:       # 欄の数が足りない行(書きかけ)は数が読めても使わない
                    continue
                if r.get('raw') in (None, '', '-1'):
                    continue
                tt, aa = int(r['epoch_us']) / 1e6, int(r['raw']) * DEG
            except (TypeError, ValueError, KeyError):
                continue
            t.append(tt)
            a.append(aa)
            try:
                if r.get('status') not in (None, '', 'E'):
                    st.append((int(r['md']), int(r['ml']), int(r['mh']), int(r['agc']), int(r['mag'])))
            except (TypeError, ValueError, KeyError):
                pass
    # 0 / 360 度をまたぐ所をつなぐ
    out, off = [], 0.0
    for i, x in enumerate(a):
        if i and x + off - out[-1] > 180:
            off -= 360
        elif i and x + off - out[-1] < -180:
            off += 360
        out.append(x + off)
    return t, out, st


def holds(t, a, min_s=0.3, max_rate=5.0, win=5):
    """止まっている所 [(始め, 終わり, 中央の角度, 標準偏差)]"""
    n = len(a)
    still = []
    for i in range(n):
        j0, j1 = max(0, i - win), min(n - 1, i + win)
        dt = t[j1] - t[j0]
        rate = abs(a[j1] - a[j0]) / dt if dt > 0 else 0.0
        still.append(rate < max_rate)
    res, i = [], 0
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and still[j + 1]:
            j += 1
        if t[j] - t[i] >= min_s:
            seg = a[i:j + 1]
            res.append((t[i], t[j], statistics.median(seg), statistics.pstdev(seg) if len(seg) > 1 else 0.0))
        i = j + 1
    return res


def analyze(path):
    t, a, st = load(path)
    if len(a) < 10:
        return {'path': path, 'error': '読めた行が少ない(%d)' % len(a)}
    hs = holds(t, a)
    lo, hi = min(a), max(a)
    first = next((x for x in a if abs(x - a[0]) > 5.0), a[0])
    left_sign = 1 if first > a[0] else -1                      # 最初に動いた向き = 左(手順のとおりなら)
    span = hi - lo
    ext = [h for h in hs if min(abs(h[2] - lo), abs(h[2] - hi)) < 0.15 * span] if span > 5 else []
    mids = [h for h in hs if h not in ext]
    left_ext = [h[2] for h in ext if (h[2] - (lo + hi) / 2) * left_sign > 0]
    right_ext = [h[2] for h in ext if (h[2] - (lo + hi) / 2) * left_sign < 0]
    center = statistics.median([h[2] for h in mids]) if mids else statistics.median(a)
    # 左から戻ったまっすぐ・右から戻ったまっすぐ
    back_l, back_r = [], []
    for k, h in enumerate(hs):
        if h in mids and k > 0:
            prev = hs[k - 1]
            if prev in ext:
                (back_l if (prev[2] - (lo + hi) / 2) * left_sign > 0 else back_r).append(h[2])
    res = {
        'path': path, 'rows': len(a), 'seconds': round(t[-1] - t[0], 1),
        'raw_deg_min': round(lo, 2), 'raw_deg_max': round(hi, 2), 'span_deg': round(span, 2),
        'center_deg': round(center, 2), 'n_holds': len(hs),
        'left_full_deg': [round(x, 2) for x in left_ext], 'right_full_deg': [round(x, 2) for x in right_ext],
        'noise_deg_median': round(statistics.median([h[3] for h in hs]), 3) if hs else None,
        'center_after_left_deg': [round(x, 2) for x in back_l], 'center_after_right_deg': [round(x, 2) for x in back_r],
        'magnet': {'ml_frac': round(sum(s[1] for s in st) / len(st), 2), 'agc': statistics.median([s[3] for s in st]),
                   'mag': statistics.median([s[4] for s in st])} if st else None,
    }
    if left_ext and right_ext:
        lw = abs(statistics.median(left_ext) - center)
        rw = abs(statistics.median(right_ext) - center)
        res.update(left_throw_deg=round(lw, 2), right_throw_deg=round(rw, 2), left_right_ratio=round(lw / rw, 3) if rw else None,
                   full_repeat_deg=round(max(statistics.pstdev(left_ext) if len(left_ext) > 1 else 0,
                                             statistics.pstdev(right_ext) if len(right_ext) > 1 else 0), 2))
    if back_l and back_r:
        res['center_return_gap_deg'] = round(statistics.median(back_l) - statistics.median(back_r), 2)
    return res


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(glob.glob(os.path.join(ROOT, 'data', 'as5601', '*.csv')))
        if not files:
            print('記録がありません(data/as5601/*.csv)')
            return 1
        path = files[-1]
        print('注意: いちばん新しい記録を見ます(起動のサービスの 30 分の記録なら、ほかの手の動きも入る。'
              '確かめには --duration で取った別の記録を渡す)')
    r = analyze(path)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    if 'left_throw_deg' in r:
        print('左いっぱい %.1f 度・右いっぱい %.1f 度(まっすぐから)・左右の比 %.2f' % (r['left_throw_deg'], r['right_throw_deg'],
                                                                 r['left_right_ratio'] or float('nan')))
    if 'center_return_gap_deg' in r:
        print('まっすぐの戻りの差(左から − 右から)%.2f 度' % r['center_return_gap_deg'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
