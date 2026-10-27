"""直す 7-2 回目・係 safety: 本物の記録で、止まっている車の eRPM の跳ねが「フレームから見て続けて後ろへ動いている」ように見える
いちばん長い秒を数える(data/raw を読むだけ。装置に触れない・GPU なし・API なし)。

- 車両の値(VEHICLE.npz)の各サンプル i が eRPM > 300 なら、[t_i, t_{i+1}) の間にとったフレームは「後ろへ動いている」を見る
  (drive は撮像時刻より前の最新の車両の値を使う)。続いた区間の長さ = フレームから見て続けて後ろへ動いていた最長の秒(の上限)。
- 跳ね = その区間(と前のサンプル・後ろ 0.05 秒)の hall_count の進みが 2 以下で、eRPM から見込む進みの 1/4 より小さい
  (見込みが 2 未満の短い跳ねは、進み 2 以下だけで)。車は動いていない。
  見込み = Σ eRPM × 間隔 × K(K は本当に後ろへ動いた区間の hall_count の進み / Σ eRPM × 間隔 の中央)。
- 跳ねのまわり 0.35 秒(CHECK_S)の窓に、フレームから見て後ろへ動いていた秒の合計も数える(2 つ目・3 つ目の跳ね)。
出力: out/scan_bursts.json
    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix7/safety_r2/scan_bursts.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = '/home/jetson/jetformer/data/raw'
OUT = Path(__file__).resolve().parent / 'out'
BACKING = 300
CHECK_S = 0.35


def runs_of(pos):
    out, i0 = [], None
    for i, p in enumerate(pos):
        if p and i0 is None:
            i0 = i
        elif not p and i0 is not None:
            out.append((i0, i))
            i0 = None
    if i0 is not None:
        out.append((i0, len(pos)))
    return out


def load():
    for fp in sorted(glob.glob(ROOT + '/2026*/*/VEHICLE.npz')):
        if os.path.getsize(fp) == 0:
            continue
        try:
            d = np.load(fp)
            t = d['timestamps_us'].astype(np.int64) / 1e6
            e = d['erpm'].astype(np.float64)
            h = d['hall_count'].astype(np.int64)
            sel = d['select'].astype(bool)
        except Exception:
            continue
        if len(t) >= 10:
            yield fp.split('/raw/')[1], t, e, h, sel


def main():
    data = list(load())
    # K: 本当に長く後ろへ動いた区間(0.3 秒以上・hall が 20 以上進む)の hall の進み / Σ eRPM × 間隔
    ks = []
    for name, t, e, h, sel in data:
        dt = np.diff(t, append=t[-1] + 0.01)
        for i0, i1 in runs_of(e > BACKING):
            dur = t[min(i1, len(t) - 1)] - t[i0]
            dh = abs(int(h[min(i1, len(h) - 1)] - h[i0]))
            ie = float((e[i0:i1] * dt[i0:i1]).sum())
            if dur >= 0.3 and dh >= 20 and ie > 0:
                ks.append(dh / ie)
    K = float(np.median(ks))
    spikes, real = [], []
    for name, t, e, h, sel in data:
        dt = np.diff(t, append=t[-1] + 0.01)
        pos = e > BACKING
        for i0, i1 in runs_of(pos):
            ta = t[i0]
            tb = t[i1] if i1 < len(t) else t[i1 - 1] + 0.01
            j1 = int(np.searchsorted(t, tb + 0.05))
            dh = abs(int(h[min(j1, len(h) - 1)] - h[max(i0 - 1, 0)]))
            exp_h = float((e[i0:i1] * dt[i0:i1]).sum()) * K
            row = {'file': name, 't': round(float(ta - t[0]), 3), 'dur_s': round(float(tb - ta), 3), 'n': int(i1 - i0),
                   'hall': dh, 'hall_expected': round(exp_h, 1), 'max_erpm': int(e[i0:i1].max()), 'ai': bool(sel[i0])}
            if dh <= 2 and (exp_h >= 2.0 and dh < 0.25 * exp_h or exp_h < 2.0):
                # 跳ね: まわり CHECK_S の窓(この区間の始めから)に、フレームから見て後ろへ動いていた秒の合計
                w = (t >= ta) & (t < ta + CHECK_S)
                idx = np.where(w)[0]
                vis = 0.0
                for i in idx:
                    if pos[i]:
                        vis += min(t[i + 1] if i + 1 < len(t) else t[i] + 0.01, ta + CHECK_S) - t[i]
                row['pos_s_in_0.35'] = round(vis, 3)
                spikes.append(row)
            else:
                real.append(row)
    spikes.sort(key=lambda r: -r['dur_s'])
    ds = np.array([r['dur_s'] for r in spikes]) if spikes else np.zeros(0)
    ws = np.array([r['pos_s_in_0.35'] for r in spikes]) if spikes else np.zeros(0)
    res = {
        'files': len(data), 'K_hall_per_erpm_s': K, 'n_K': len(ks),
        'spike_runs': len(spikes),
        'spike_dur_quantiles': {str(q): round(float(np.quantile(ds, q)), 3) for q in (0.5, 0.9, 0.99, 1.0)} if len(ds) else {},
        'spike_dur_ge': {str(x): int((ds >= x).sum()) for x in (0.02, 0.03, 0.05, 0.067, 0.1, 0.15, 0.2)},
        'spike_pos_in_0.35_ge': {str(x): int((ws >= x).sum()) for x in (0.02, 0.05, 0.1, 0.15, 0.2)},
        'spike_top30': spikes[:30],
        'spike_top_window': sorted(spikes, key=lambda r: -r['pos_s_in_0.35'])[:15],
        'real_runs': len(real),
        'real_dur_ge': {str(x): int(sum(1 for r in real if r['dur_s'] >= x)) for x in (0.05, 0.1, 0.15, 0.2, 0.3)},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / 'scan_bursts.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ('spike_top30', 'spike_top_window')}, ensure_ascii=False))
    for r in spikes[:15]:
        print(r)
    print('--- window')
    for r in res['spike_top_window'][:10]:
        print(r)


if __name__ == '__main__':
    main()
