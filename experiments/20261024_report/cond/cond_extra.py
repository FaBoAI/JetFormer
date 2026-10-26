#!/usr/bin/env python3
"""解析 B の追加の印(2026-10-25): 明るさの「場所ごと」、ToF(STEVAL-VL53L9)、IMU の振動、画像のシャープさ、
B の学習データ(10/10 の人の走行 9 本)の区間ごとの明るさと人の速さ。読むだけ。

    jf-heavy --lock none --mem 1500M -- python3 -I cond_extra.py

出力: /home/jetson/jetformer_reports/bench_20261024_report/cond_data/cond_extra.json
  ToF: 中央 14×18 の区画の 有効な画素の割合・距離の中央値 [mm]・信頼度の平均(0〜7)
  振動: 直線(DE・HS)で前進 > 4,000 eRPM のときの 加速度 z(上下)と x(前後)の 1 サンプル差の標準偏差 [m/s²](100 Hz)
  シャープさ: 輝度の横・縦の差の絶対値の平均(画像のぼけ・ぶれが大きいほど小さい。同じ場所・同じ速さ帯で比べる)
"""
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path('/home/jetson/jetformer/data/raw')
LAB = Path('/home/jetson/jetformer/data/labels')
OUT = Path('/home/jetson/jetformer_reports/bench_20261024_report/cond_data')
DET = ['?', 'R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH']
TRAIN = ['20261010-135449', '20261010-144028', '20261010-150424', '20261010-152405', '20261010-152707',
         '20261010-193638', '20261010-200916', '20261010-201626', '20261010-202633']
DIST_MASK, CONF_SHIFT = 0x1FFF, 13


def cat(lg, name, keys):
    parts = {k: [] for k in keys}
    for s in sorted(glob.glob(str(RAW / lg / (lg + '-*')))):
        try:
            with np.load(s + '/' + name + '.npz') as z:
                for k in keys:
                    parts[k].append(z[k])
        except Exception:
            continue
    o = {k: np.concatenate(v) for k, v in parts.items()}
    idx = np.argsort(o['timestamps_us'], kind='stable')
    return {k: v[idx] for k, v in o.items()}


def cam_files(lg):
    ct, cf = [], []
    for s in sorted(glob.glob(str(RAW / lg / (lg + '-*')))):
        try:
            with np.load(s + '/CAM_FRONT.npz') as c:
                ct.append(c['timestamps_us'])
                cf += [s + '/' + f for f in c['files']]
        except Exception:
            continue
    return np.concatenate(ct), np.array(cf)


def img_stats(f):
    try:
        a = np.asarray(Image.open(f).convert('RGB'), np.float32)
    except Exception:
        return (np.nan,) * 3
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    sh = 0.5 * (np.abs(np.diff(y, axis=1)).mean() + np.abs(np.diff(y, axis=0)).mean())
    return y.mean(), y.std(), sh


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return round(float(np.median(x)), 3) if len(x) else None


def tof_dist_mm(raw):
    return None


def analyse(lg, ai_only=True, step=2):
    info = json.loads((LAB / (lg + '.json')).read_text(encoding='utf-8'))
    with np.load(LAB / (lg + '.npz')) as z:
        lab = {k: z[k] for k in ('t_us', 't_s', 'ai', 'dry_run', 'erpm', 'loc_detail', 'out', 'human_pwm', 'hold', 'assist')}
    t0 = info['t0_us']
    t = lab['t_s']
    ai = lab['ai'] & ~lab['dry_run']
    use = ai if ai_only else ~lab['ai']
    e = -lab['erpm'].astype(float)
    ld = lab['loc_detail'].astype(int)
    thr = 1496.0 - (lab['out'][:, 1] if ai_only else lab['human_pwm'][:, 1]).astype(float)
    ct, cf = cam_files(lg)
    k = np.clip(np.searchsorted(ct, lab['t_us']), 0, len(ct) - 1)
    files = np.where(ct[k] == lab['t_us'], cf[k], '')
    st = np.full((len(t), 3), np.nan)
    for i in np.flatnonzero(use & (files != ''))[::step]:
        st[i] = img_stats(files[i])
    # ToF
    T = cat(lg, 'TOF_FRONT', ['timestamps_us', 'depth'])
    tt = (T['timestamps_us'] - t0) / 1e6
    j = np.clip(np.searchsorted(tt, t, side='right') - 1, 0, len(tt) - 1)
    sub = T['depth'][:, 14:28, 18:36].astype(np.int64)
    d = sub & DIST_MASK
    conf = (sub >> CONF_SHIFT)
    valid = (d != 0) & (d != DIST_MASK)
    vfrac = valid.reshape(len(sub), -1).mean(1)
    dmed = np.array([np.median(x[v]) if v.any() else np.nan for x, v in zip(d.reshape(len(sub), -1), valid.reshape(len(sub), -1))])
    cmean = conf.reshape(len(sub), -1).mean(1)
    tof_v, tof_d, tof_c = vfrac[j], dmed[j], cmean[j]
    # IMU の振動
    I = cat(lg, 'IMU', ['timestamps_us', 'accel'])
    it = (I['timestamps_us'] - t0) / 1e6
    li = np.clip(np.searchsorted(t, it, side='right') - 1, 0, len(t) - 1)
    ok_i = use[li] & np.isin(ld[li], (4, 6)) & (np.interp(it, t, e) > 4000)
    dz = np.diff(I['accel'][:, 2], prepend=I['accel'][0, 2])
    dx = np.diff(I['accel'][:, 0], prepend=I['accel'][0, 0])
    ok_i &= np.r_[False, np.diff(it) < 0.02]
    res = {'log': lg, 'frames': int(use.sum()),
           'bright_mean': med(st[use, 0]), 'bright_std': med(st[use, 1]), 'sharp': med(st[use, 2]),
           'tof_valid': med(tof_v[use]), 'tof_dist_mm': med(tof_d[use]), 'tof_conf': med(tof_c[use]),
           'vib_z': round(float(np.std(dz[ok_i])), 3) if ok_i.sum() > 100 else None,
           'vib_x': round(float(np.std(dx[ok_i])), 3) if ok_i.sum() > 100 else None, 'vib_n': int(ok_i.sum()),
           'erpm_p90': med([np.percentile(e[use & (e > 300)], 90)]) if (use & (e > 300)).sum() > 30 else None}
    secs = {}
    for c, nm in enumerate(DET):
        m = use & (ld == c)
        if m.sum() < 10:
            continue
        mv = m & (e > 300)
        secs[nm] = {'n': int(m.sum()), 'bright_mean': med(st[m, 0]), 'bright_std': med(st[m, 1]), 'sharp': med(st[m, 2]),
                    'tof_valid': med(tof_v[m]), 'tof_dist_mm': med(tof_d[m]), 'tof_conf': med(tof_c[m]),
                    'erpm_p50': med(e[mv]), 'thr_p50': med(thr[mv])}
        # 同じ速さ帯(4,000〜6,000 eRPM)のシャープさ
        mb = m & (e > 4000) & (e < 6000)
        secs[nm]['sharp_4k6k'] = med(st[mb, 2]) if mb.sum() > 10 else None
    res['sections'] = secs
    return res


def main():
    runs = []
    for lg in sorted(p.name for p in RAW.glob('20261024-*')):
        j = json.loads((RAW / lg / 'log.json').read_text(encoding='utf-8'))
        r = analyse(lg, ai_only=True)
        r['desc'] = j.get('description', '')[:20]
        runs.append(r)
        print(lg, {k: r[k] for k in ('bright_mean', 'sharp', 'tof_valid', 'tof_dist_mm', 'tof_conf', 'vib_z', 'vib_x')}, flush=True)
    train = []
    for lg in TRAIN:
        r = analyse(lg, ai_only=False, step=3)
        train.append(r)
        print('train', lg, {k: r[k] for k in ('bright_mean', 'sharp', 'tof_valid', 'tof_dist_mm', 'tof_conf', 'vib_z')}, flush=True)
    (OUT / 'cond_extra.json').write_text(json.dumps({'runs': runs, 'train': train, 'doc': __doc__}, ensure_ascii=False, indent=1),
                                         encoding='utf-8')


if __name__ == '__main__':
    main()
