"""票 B: 本物の記録(10/10・10/23・10/24。読むだけ)で、カメラが来なかった AI 側の間(real_cam_B の cam_frames_inside 0)が
どんなときに起きたか: 直前 0.5 秒の IMU の水平 G の最大(衝撃のあとか)・その時の速さ(VEHICLE の eRPM)。
比べに、AI 側の処理したフレーム全部の同じ量の分布。後退の終わりのような「衝撃のあとではない・後ろへ 1〜3 m/s」での割合を出す。
    python -I -B real_drop_ctx_B.py → out/real_drop_ctx_B.json
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

G_SHOCK = 2.5


def main():
    cam = json.loads((HERE / 'out' / 'real_cam_B.json').read_text(encoding='utf-8'))
    drops = {}
    for r in cam['rows']:
        if r['cam_frames_inside'] == 0:
            drops.setdefault(r['log'], []).append(r['s'])
    tot = {'frames': 0, 'after_shock': 0, 'v_back_1_3': 0, 'v_fwd_gt1': 0, 'v_slow': 0}
    dro = {'n': 0, 'after_shock': 0, 'v_back_1_3': 0, 'v_fwd_gt1': 0, 'v_slow': 0}
    rows = []
    for log in T.raw_logs():
        for scenes in T.units(log, 'concat'):
            cat, meta, sk = T.load_unit(log, scenes)
            if cat is None:
                continue
            z = cat['AI_CMD']
            t0 = T.log_t0_us(log) or int(z['timestamps_us'][0])
            ts = (z['timestamps_us'].astype(np.int64) - t0) / 1e6
            sel = z['select'].astype(bool)
            hold = z['hold'].astype(bool)
            dry = z['dry_run'].astype(bool) if 'dry_run' in z else np.zeros(len(ts), bool)
            ok = sel & ~hold & ~dry
            im = cat['IMU']
            it = (im['timestamps_us'].astype(np.int64) - t0) / 1e6
            acc = im['accel'].astype(float)
            gh = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
            v = cat['VEHICLE']
            vt = (v['timestamps_us'].astype(np.int64) - t0) / 1e6
            ve = v['erpm'].astype(float)
            # 0.5 秒の窓の G の最大(累積の最大を区間で取る代わりに、IMU を 0.01 秒の格子に)
            def gmax(a, b):
                i0, i1 = np.searchsorted(it, a), np.searchsorted(it, b)
                return float(gh[i0:i1].max()) if i1 > i0 else 0.0

            def erpm_at(t):
                k = np.searchsorted(vt, t) - 1
                return float(ve[k]) if 0 <= k < len(ve) else 0.0

            idx = np.nonzero(ok)[0]
            for i in idx[::3]:          # 全部のフレームの分布は 3 つに 1 つで見る(数は ×3 で戻さない。割合だけ使う)
                t = ts[i]
                tot['frames'] += 1
                sh = gmax(t - 0.5, t) >= G_SHOCK
                e = erpm_at(t)
                tot['after_shock'] += int(sh)
                tot['v_back_1_3'] += int(e > 2000 and e < 6000)      # eRPM 正 = 後ろへ(1000 eRPM = 0.5 m/s)
                tot['v_fwd_gt1'] += int(e < -2000)
                tot['v_slow'] += int(abs(e) <= 2000)
            for s in drops.get(log, []):
                sh = gmax(s - 0.5, s + 0.001)
                e = erpm_at(s)
                dro['n'] += 1
                dro['after_shock'] += int(sh >= G_SHOCK)
                dro['v_back_1_3'] += int(e > 2000 and e < 6000)
                dro['v_fwd_gt1'] += int(e < -2000)
                dro['v_slow'] += int(abs(e) <= 2000)
                rows.append({'log': log, 's': s, 'g_max_0p5_before': round(sh, 2), 'erpm': e})
    out = {'all_frames_sampled': tot, 'drops': dro,
           'frac_all': {k: round(tot[k] / max(tot['frames'], 1), 4) for k in tot if k != 'frames'},
           'frac_drops': {k: round(dro[k] / max(dro['n'], 1), 4) for k in dro if k != 'n'}, 'rows': rows}
    (HERE / 'out' / 'real_drop_ctx_B.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
