"""反証 7-2 回目・係 tools(読むだけ): 本物の 10/10・10/23・10/24 で、AI の区間が「中立止め(hold)のまま」か「フレームの
途切れ(カメラが止まった)のまま」終わり(= 人が中立止めの間に CH3 を人側にした)、そのあと同じ本でもう一度 AI 側にした発進を
数える。fix7 の abort_resumes(リカバリー中止 = assist 3 で終わった区間だけを見る)はこの形に印を付けない。

区間の終わりの形:
  hold_tail  区間の最後のフレーム(AI_CMD が人側のフレームを除く)から前に、hold のフレームが続く秒(0.5 秒以上を数える)
  gap_tail   区間の最後のフレームから次のフレームまでの秒(カメラが止まったまま人側にした形。0.5 秒以上)
  stop_end   区間の終わり(最後のフレーム)の車輪の速さ |eRPM| < 400(止まっていた)
次の区間: 止まった状態から(③ standing_3lap の発進の行になる形)か、after_abort が付くか(今の abort_resumes)。
data/raw・data/labels は読むだけ。出力は ../out/scan_real_hold_end.json
"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import label_events as le  # noqa: E402
import standing_3lap as s3  # noqa: E402

RAW = JF / 'data' / 'raw'
LAB = JF / 'data' / 'labels'
OUT = Path(__file__).resolve().parent / 'out' / 'scan_real_hold_end_rerun.json'


def main():
    rows, n_sec, n_logs = [], 0, 0
    for p in sorted(RAW.iterdir()):
        if p.name[:8] not in ('20261010', '20261023', '20261024') or len(p.name) != 15:
            continue
        lj, lz = LAB / (p.name + '.json'), LAB / (p.name + '.npz')
        if not (lj.exists() and lz.exists()):
            continue
        z = dict(np.load(lz))
        lab = json.loads(lj.read_text(encoding='utf-8'))
        t = z['t_s'].astype(np.float64)
        ai = z['ai'].astype(bool) & ~z['dry_run'].astype(bool)
        if not ai.any():
            continue
        n_logs += 1
        L = le.load_log(p.name, RAW)
        V = L.get('VEHICLE')
        if V is None:
            continue
        T0 = int(lab['t0_us'])
        vt = (V['timestamps_us'].astype(np.int64) - T0) / 1e6
        ve = V['erpm'].astype(np.float64)
        _, Df = le.cumulative_distance(vt, ve)
        hold = z['hold'].astype(bool)
        sel = z['ai_cmd_select'].astype(bool)
        ab = s3.abort_resumes(t, ai, z['assist'], hold, sel, vt, Df)
        runs = le.runs(ai)
        n_sec += len(runs)
        for k, (i0, i1) in enumerate(runs):
            kk = np.arange(i0, i1 + 1)
            kk = kk[sel[i0:i1 + 1]]
            if not len(kk):
                continue
            j = len(kk) - 1
            while j >= 0 and hold[kk[j]]:
                j -= 1
            hold_tail = float(t[kk[-1]] - t[kk[j + 1]]) + 1 / 30 if j + 1 <= len(kk) - 1 else 0.0
            gap_tail = float(t[i1 + 1] - t[i1]) if i1 + 1 < len(t) else None
            e_end = float(np.interp(t[i1], vt, ve))
            if not (hold_tail >= 0.5 or (gap_tail is not None and gap_tail >= 0.5)):
                continue
            nxt = runs[k + 1] if k + 1 < len(runs) else None
            row = {'log': p.name, 'sec_t0': round(float(t[i0]), 3), 'sec_t1': round(float(t[i1]), 3),
                   'last_assist': int(z['assist'][kk[j]]) if j >= 0 else None,
                   'hold_tail_s': round(hold_tail, 2), 'gap_tail_s': None if gap_tail is None else round(gap_tail, 2),
                   'erpm_end': round(e_end, 0), 'stop_end': abs(e_end) < 400}
            if nxt is not None:
                n0 = nxt[0]
                e0 = float(np.interp(t[n0], vt, ve))
                row.update(next_t_ai=round(float(t[n0]), 3), next_stopped=abs(e0) < 400,
                           human_s=round(float(t[n0] - t[i1]), 2),
                           human_dist_m=round(float(np.interp(t[n0], vt, Df) - np.interp(t[i1], vt, Df)), 2),
                           next_after_abort=n0 in ab)
            rows.append(row)
    res = {'logs_with_ai': n_logs, 'ai_sections': n_sec, 'rows': rows,
           'resume_unmarked': [r for r in rows if r.get('next_t_ai') is not None and r.get('next_stopped')
                               and not r.get('next_after_abort')]}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('logs', n_logs, 'sections', n_sec, 'hold/gap tail', len(rows), 'resume (stopped, unmarked)', len(res['resume_unmarked']))
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
