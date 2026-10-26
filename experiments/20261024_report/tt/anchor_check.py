"""R1 の起点(周の切れ目)を IMU の向きと車輪の距離で確かめる(10/24 解析 A)。
各起点 → 次の起点の 向きの変化 [deg](1 周 = 左回り +360°)・距離 [m]・起点 ±0.2 秒の衝突・進入 eRPM を出す。
    python3 -I anchor_check.py > anchor_check.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import label_events as le  # noqa: E402

SC = {
    '20261023': '/home/jetson/jetformer_reports/bench_20261023_day1/nightly/scorecard/scorecard.json',
    '20261024': '/home/jetson/jetformer_reports/bench_20261024_tune/since_1300/nightly/scorecard/scorecard.json',
}
OFFICIAL_1023 = ['20261023-210840', '20261023-211020', '20261023-211118', '20261023-211209',
                 '20261023-211308', '20261023-211357', '20261023-211448', '20261023-211547']


def imu_track(log):
    L = le.load_log(log)
    lab = json.loads((Path('/home/jetson/jetformer/data/labels') / (log + '.json')).read_text())
    t0 = lab['t0_us']
    I, V = L['IMU'], L['VEHICLE']
    it = (I['timestamps_us'].astype(np.int64) - t0) / 1e6
    vt = (V['timestamps_us'].astype(np.int64) - t0) / 1e6
    verpm = V['erpm'].astype(np.float64)
    ie = np.interp(it, vt, verpm)
    gz = I['gyro'][:, 2].astype(np.float64)
    still = (np.abs(ie) < 50) & (np.abs(gz) < 5)
    hd, _ = le.heading_deg(it, gz, still)
    D, Df = le.cumulative_distance(vt, verpm)
    return it, hd, vt, Df


def main():
    out = {}
    for day, p in SC.items():
        sc = json.loads(Path(p).read_text())
        for log, e in sc['per_log'].items():
            if day == '20261023' and log not in OFFICIAL_1023:
                continue
            if not e.get('passes'):
                continue
            it, hd, vt, Df = imu_track(log)
            A = [x['ta_s'] for x in e['passes']]
            col = e.get('collision_events') or []
            rows = []
            for k, ta in enumerate(A):
                nxt = A[k + 1] if k + 1 < len(A) else None
                c = [x for x in col if abs(x['t_s'] - ta) <= 0.25]
                r = {'ta': ta, 'entry_erpm': e['passes'][k]['entry_erpm'],
                     'coll_near': [(x['t_s'], x['g'], x['loc']) for x in c]}
                if nxt is not None:
                    r['dt'] = round(nxt - ta, 3)
                    r['dpsi'] = round(float(np.interp(nxt, it, hd) - np.interp(ta, it, hd)), 1)
                    r['dist_m'] = round(float(np.interp(nxt, vt, Df) - np.interp(ta, vt, Df)), 2)
                rows.append(r)
            out[log] = rows
            print('==', log, 'No.', e['desc'].get('ai_no'))
            for r in rows:
                print('  ta=%7.3f entry=%5s dt=%6s dpsi=%7s dist=%6s near=%s' % (
                    r['ta'], r['entry_erpm'], r.get('dt'), r.get('dpsi'), r.get('dist_m'), r['coll_near']))
    Path(sys.argv[1] if len(sys.argv) > 1 else 'anchor_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
