"""打ち消しの票 B(指摘 rs8-2-reverse-end-stall-overrun): 本物の記録(data/raw の 10/10・10/23・10/24。読むだけ)で、
11/3 の使い方での届きやすさの材料を数える。装置・GPU・API に触れない。
  1. AI 側の処理したフレームどうしの間(となりあう 2 行がどちらも AI 側・中立止めの行でない・dry run でない)の分布
     (カメラの映像が来ない中立止めは AI_CMD に行を残さない = drive.py の hold(cam=None) は add_ai しない ので、間で見る)。
     同じ表の latency_ms・infer_ms の中央値(cam_evt.wait(0.1) は処理のあとから数えるので、中立止めになる間の下限の目安)。
  2. リカバリー(assist 2)の始めごとに、後退側(out > 中立 + 200)の最後の行の次の行が何か:
     惰性(assist 2・中立 ±60)= ふつうの後退の終わり / 人側 / 中立止めの行 / それ以外。と、その時の後ろの当たり(IMU)。
  3. 日ごとの記録の本数・AI 側の秒・リカバリーの数・ふつうの後退の終わりの数。
    python -I -B real_B.py → out/real_B.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
import test_rec_fix8_safety_r1 as T  # noqa: E402

BINS = [0.067, 0.100, 0.115, 0.134, 0.167, 0.20, 0.30, 0.37, 0.50, 99.0]


def main():
    out = {'days': {}, 'gap_bins': [], 'gaps_long': [], 'recoveries': [], 'lat_ms': None, 'infer_ms': None}
    hist = np.zeros(len(BINS) - 1, int)
    n_int = 0
    lat_all, inf_all = [], []
    for log in T.raw_logs():
        day = log[:8]
        dd = out['days'].setdefault(day, {'logs': 0, 'ai_s': 0.0, 'ai_frames': 0, 'recoveries': 0, 'rev_end_coast': 0,
                                         'rev_end_human': 0, 'rev_end_other': 0, 'no_reverse': 0})
        for scenes in T.units(log, 'concat'):
            cat, meta, sk = T.load_unit(log, scenes)
            if cat is None:
                continue
            dd['logs'] += 1
            z = cat['AI_CMD']
            t0 = T.log_t0_us(log) or int(z['timestamps_us'][0])
            ts = (z['timestamps_us'].astype(np.int64) - t0) / 1e6
            n = float((meta.get('drive') or {}).get('throttle_neutral_us', 1496))
            sel = z['select'].astype(bool)
            hold = z['hold'].astype(bool)
            dry = z['dry_run'].astype(bool) if 'dry_run' in z else np.zeros(len(ts), bool)
            a = z['assist'].astype(int) if 'assist' in z else np.zeros(len(ts), int)
            thr = z['out'][:, 1].astype(float)
            ok = sel & ~hold & ~dry
            if 'latency_ms' in z:
                lat_all.extend(z['latency_ms'][ok].astype(float).tolist())
            if 'infer_ms' in z:
                inf_all.extend(z['infer_ms'][ok].astype(float).tolist())
            dd['ai_frames'] += int(ok.sum())
            # 1. となりあう 2 行がどちらも AI 側の処理したフレーム
            both = ok[:-1] & ok[1:]
            gaps = np.diff(ts)[both]
            idx = np.nonzero(both)[0]
            n_int += int(len(gaps))
            dd['ai_s'] += float(np.sum(np.minimum(gaps, 0.2)))
            h, _ = np.histogram(gaps, bins=BINS)
            hist += h
            for g, i in zip(gaps, idx):
                if g > 0.1:
                    out['gaps_long'].append({'log': log, 's': round(float(ts[i]), 3), 'gap_s': round(float(g), 3),
                                             'assist_prev': int(a[i]), 'thr_prev': round(float(thr[i]), 1),
                                             'lat_prev': float(z['latency_ms'][i]) if 'latency_ms' in z else None,
                                             'lat_next': float(z['latency_ms'][i + 1]) if 'latency_ms' in z else None})
            # 2. リカバリー
            im = cat['IMU']
            it = (im['timestamps_us'].astype(np.int64) - t0) / 1e6
            acc = im['accel'].astype(float)
            gh = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
            i = 0
            L = len(ts)
            while i < L:
                if not (ok[i] and a[i] == 2):
                    i += 1
                    continue
                j = i
                while j + 1 < L and a[j + 1] == 2 and sel[j + 1]:
                    j += 1
                rows = list(range(i, j + 1))
                rev = [k for k in rows if thr[k] > n + 200 and not hold[k]]
                r = {'log': log, 'start_s': round(float(ts[i]), 3), 'end_s': round(float(ts[j]), 3), 'n_rev_rows': len(rev)}
                dd['recoveries'] += 1
                if not rev:
                    r['end'] = 'no_reverse'
                    dd['no_reverse'] += 1
                else:
                    k = rev[-1]
                    nx = k + 1
                    if nx >= L:
                        r['end'] = 'log_end'
                        dd['rev_end_other'] += 1
                    elif not sel[nx]:
                        r['end'] = 'human'
                        dd['rev_end_human'] += 1
                    elif hold[nx]:
                        r['end'] = 'hold_row'
                        dd['rev_end_other'] += 1
                    elif a[nx] == 2 and abs(thr[nx] - n) <= 60:
                        r['end'] = 'coast'
                        dd['rev_end_coast'] += 1
                    else:
                        r['end'] = 'other(assist %d thr %.0f)' % (a[nx], thr[nx])
                        dd['rev_end_other'] += 1
                    r['rev_end_s'] = round(float(ts[k]), 3)
                    r['gap_after_last_rev_s'] = round(float(ts[nx] - ts[k]), 3) if nx < L else None
                    # 後退の最後の行から 0.6 秒の IMU の水平 G の最大(2.5G 以上なら後ろの当たりの見込み)
                    m = (it >= ts[k] - 0.3) & (it <= ts[k] + 0.6)
                    r['g_max_near_rev_end'] = round(float(gh[m].max()), 2) if m.any() else None
                    if m.any():
                        r['g_max_t'] = round(float(it[m][np.argmax(gh[m])]), 3)
                out['recoveries'].append(r)
                i = j + 1
    out['n_intervals'] = n_int
    out['gap_bins'] = [{'from': BINS[q], 'to': BINS[q + 1], 'n': int(hist[q]), 'per_frame': round(hist[q] / max(n_int, 1), 6)}
                       for q in range(len(BINS) - 1)]
    la = np.array([x for x in lat_all if np.isfinite(x)])
    ia = np.array([x for x in inf_all if np.isfinite(x)])
    out['lat_ms'] = {'median': round(float(np.median(la)), 1), 'p10': round(float(np.percentile(la, 10)), 1),
                     'p90': round(float(np.percentile(la, 90)), 1)} if len(la) else None
    out['infer_ms'] = {'median': round(float(np.median(ia)), 1), 'p10': round(float(np.percentile(ia, 10)), 1),
                       'p90': round(float(np.percentile(ia, 90)), 1)} if len(ia) else None
    for d in out['days'].values():
        d['ai_s'] = round(d['ai_s'], 1)
    (HERE / 'out' / 'real_B.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ('days', 'gap_bins', 'n_intervals', 'lat_ms', 'infer_ms')}, ensure_ascii=False, indent=1))
    for r in out['recoveries']:
        print(json.dumps(r, ensure_ascii=False))
    print('gaps > 0.1 s:', len(out['gaps_long']))


if __name__ == '__main__':
    main()
