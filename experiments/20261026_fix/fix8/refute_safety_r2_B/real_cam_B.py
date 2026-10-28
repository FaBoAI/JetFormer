"""票 B: AI 側の処理したフレームの間が 0.1 秒より長い所(real_B.py の gaps_long)で、カメラの記録(CAM_FRONT の撮像の時刻)に
その間のフレームがあるか(= カメラは来ていて処理が遅れた・飛ばした)、無いか(= カメラが来なかった → cam_evt.wait(0.1) の
中立止め 'カメラの映像が来ていません' になりうる)を見る。data/raw は読むだけ。
    python -I -B real_cam_B.py → out/real_cam_B.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = Path('/home/jetson/jetformer/data/raw')


def cam_ts(log):
    ts = []
    for d in sorted(x for x in (RAW / log).iterdir() if x.is_dir() and x.name.startswith(log + '-')):
        p = d / 'CAM_FRONT.npz'
        if p.exists() and p.stat().st_size > 0:
            try:
                with np.load(p) as z:
                    ts.append(z['timestamps_us'].astype(np.int64))
            except Exception:   # noqa: BLE001
                pass
    return np.sort(np.concatenate(ts)) if ts else np.zeros(0, np.int64)


def t0_of(log):
    for d in sorted(x for x in (RAW / log).iterdir() if x.is_dir() and x.name.startswith(log + '-')):
        try:
            return int(json.loads((d / 'scene.json').read_text(encoding='utf-8'))['start_us'])
        except Exception:   # noqa: BLE001
            return None
    return None


def main():
    src = json.loads((HERE / 'out' / 'real_B.json').read_text(encoding='utf-8'))
    rows = []
    cache = {}
    for g in src['gaps_long']:
        log = g['log']
        if log not in cache:
            cache[log] = (cam_ts(log), t0_of(log))
        ct, t0 = cache[log]
        a = int(round(g['s'] * 1e6)) + t0
        b = a + int(round(g['gap_s'] * 1e6))
        inside = ct[(ct > a + 500) & (ct < b - 500)]
        k = np.searchsorted(ct, a)
        cam_gap = None
        if 0 <= k < len(ct) - 1:
            cam_gap = round(float(ct[min(k + 1, len(ct) - 1)] - ct[k]) / 1e6, 3)
        rows.append(dict(g, cam_frames_inside=int(len(inside)), cam_gap_s=cam_gap, cam_has_prev=bool(len(ct) and abs(int(ct[min(k, len(ct)-1)]) - a) < 2000)))
    n = len(rows)
    no_cam = [r for r in rows if r['cam_frames_inside'] == 0]
    out = {'n_gaps_gt_0p1': n, 'cam_missing_too(=cam_frames_inside 0)': len(no_cam),
           'cam_present(=処理が飛ばした)': n - len(no_cam),
           'no_cam_by_gap': {}, 'rows': rows}
    for r in no_cam:
        key = '%.3f' % r['gap_s']
        out['no_cam_by_gap'][key] = out['no_cam_by_gap'].get(key, 0) + 1
    by_day = {}
    for r in rows:
        d = by_day.setdefault(r['log'][:8], {'gaps': 0, 'no_cam': 0})
        d['gaps'] += 1
        d['no_cam'] += int(r['cam_frames_inside'] == 0)
    out['by_day'] = by_day
    (HERE / 'out' / 'real_cam_B.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
