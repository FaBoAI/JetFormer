"""反証 7-1 回目・係 tools(読むだけ): car_check の R1 の板(r1_bumps)は AI 側の R1 の通過だけを見る。人が R1 の直前に CH3 を
人側にした通過(AI の区間の終わりから 0.5 秒・1.0 秒以内に起点)の板の衝撃(3 軸 |‖a‖/G − 1|、起点 → +1.2 秒の最大)が本物の
10/10・10/23・10/24 にあるかを数える。data/raw・labels は読むだけ。出力 ../out/scan_real_r1_board_after_ai.json"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import label_events as le  # noqa: E402
import section_scorecard as ss  # noqa: E402
import standing_3lap as s3  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / 'out' / 'scan_real_r1_board_after_ai.json'


def main():
    res = []
    n_logs = 0
    for day in ('20261010', '20261023', '20261024'):
        logs, _ = s3.day_logs(day)
        for log in logs:
            try:
                info, lab, src = ss._labels(log, JF / 'data' / 'labels', JF / 'data' / 'raw')
                if lab is None:
                    continue
                L = le.load_log(log, JF / 'data' / 'raw')
                ex = {}
                sc = ss.score_log(log, labels=(info, lab, src), loaded=L, extras=ex)
                if sc.get('error'):
                    continue
            except Exception as e:  # noqa: BLE001
                print('skip', log, e)
                continue
            n_logs += 1
            t = lab['t_s'].astype(np.float64)
            ai = lab['ai'].astype(bool) & ~lab['dry_run'].astype(bool)
            ta_ai = t[ai]
            I = L['IMU']
            it = (np.asarray(I['timestamps_us'], np.int64) - info['t0_us']) / 1e6
            a3 = np.abs(np.linalg.norm(np.asarray(I['accel'], np.float64), axis=1) / le.G - 1.0)
            for ps in sc.get('passes') or []:
                if ps.get('ai'):
                    continue
                ta = float(ps['ta_s'])
                j = int(np.searchsorted(ta_ai, ta, side='right')) - 1
                if j < 0:
                    continue
                dt = ta - float(ta_ai[j])
                if dt > 1.0:
                    continue
                m = (it >= ta) & (it <= ta + 1.2)
                if m.sum() < 5:
                    continue
                res.append({'log': log, 'ta_s': round(ta, 2), 'ai_end_before_s': round(dt, 3), 'a3_max': round(float(a3[m].max()), 2)})
    OUT.write_text(json.dumps({'logs': n_logs, 'human_passes_after_ai': res}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('logs', n_logs, 'human R1 passes within 1.0 s after AI end', len(res))
    for r in res:
        print(r)


if __name__ == '__main__':
    main()
