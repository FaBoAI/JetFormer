#!/usr/bin/env python3
"""fix9 tonight s24-analysis (2026-10-29): 記録から「車がもう出した速さ」と「リカバリーで失う時間」を数える。読むだけ。

    cd /home/jetson/jetformer/experiments/20261026_fix/fix9/tonight/s24-analysis
    PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1200M --need-mb 2000 -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I s24_records.py

入力(読むだけ): data/labels/2026102[34]-*.npz・.json(10/24 までのラベル。10/23・10/24 は作り直していない版 = 偽の起点が残るが、
  ここでは速さ・場所・衝突・補助の種類だけを使う)、tt_data.json の周(偽の周を外したもの)。
1. 区間(loc_detail)ごとの最高の速さ: 周ごと(tt_data の周の時刻)に、R1 の入口(起点の前 0.4 秒〜起点)・GM の出口・DE・SW・HS の
   最高の速さ(ラベルの speed_mps)と、その区間で衝突(水平 2.5G)があったか。B のきれいな周 / 全部の周。
2. リカバリー(assist = 2)の 1 回ごとの長さと、「止まってから元の速さに戻るまで」: assist 2 が始まった時刻から、前へ 2.0 m/s に
   戻るまで(AI 側のまま)。その周の時間 − 同じ記録のきれいな周の中央値。
出力: s24_records.json(このフォルダ)
"""
import json
import os

import numpy as np

ROOT = '/home/jetson/jetformer'
HERE = os.path.dirname(os.path.abspath(__file__))
TT = json.load(open('/home/jetson/jetformer_reports/bench_20261024_report/tt_data.json'))
NAMES = ['?', 'R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH']


def load(log):
    z = np.load(os.path.join(ROOT, 'data', 'labels', log + '.npz'))
    return {k: z[k] for k in ('t_s', 'ai', 'dry_run', 'hold', 'assist', 'speed_mps', 'loc_detail', 'collision',
                              'collision_g', 'recovery_start')}


def main():
    logs = sorted({lp['log'] for lp in TT['laps']})
    per_lap = []
    rec = []
    for log in logs:
        try:
            L = load(log)
        except FileNotFoundError:
            continue
        t, v, loc = L['t_s'], L['speed_mps'].astype(float), L['loc_detail'].astype(int)
        coll = L['collision'].astype(bool)
        laps = sorted([lp for lp in TT['laps'] if lp['log'] == log and not lp['false_end']], key=lambda x: x['t_start'])
        clean_med = np.median([lp['lap_s'] for lp in laps if lp['clean']]) if any(lp['clean'] for lp in laps) else None
        for lp in laps:
            m = (t >= lp['t_start']) & (t < lp['t_start'] + lp['lap_s'])
            row = {'log': log, 'no': lp['no'], 'session': lp['session'], 'model': lp['model'], 'tg': lp['tg'], 'sg': lp['sg'],
                   'lap_s': lp['lap_s'], 'clean': lp['clean'], 'kind': lp['kind']}
            for j, nm in enumerate(NAMES):
                mm = m & (loc == j)
                row['vmax_' + nm] = float(np.nanmax(v[mm])) if mm.any() else None
                row['coll_' + nm] = bool((mm & coll).any())
            # R1 の入口(起点の 0.4 秒前〜起点)
            mm = (t >= lp['t_start'] - 0.4) & (t <= lp['t_start'])
            row['v_r1_entry'] = float(np.nanmax(v[mm])) if mm.any() else None
            row['rec_frames'] = int((m & (L['assist'] == 2)).sum())
            row['lap_minus_clean_med'] = (lp['lap_s'] - clean_med) if clean_med is not None else None
            per_lap.append(row)
        # リカバリーの 1 回ごと
        a2 = (L['assist'] == 2) & L['ai'].astype(bool)
        starts = np.nonzero(a2 & ~np.r_[False, a2[:-1]])[0]
        for i0 in starts:
            i1 = i0
            while i1 + 1 < len(a2) and a2[i1 + 1]:
                i1 += 1
            # 元の速さ(前へ 2.0 m/s)に戻る時刻(AI 側のまま。人側に替わったら None)
            back = None
            for k in range(i1, min(len(t), i1 + 400)):
                if not L['ai'][k]:
                    break
                if v[k] >= 2.0:
                    back = float(t[k] - t[i0])
                    break
            # 当たる前 1 秒の速さの最大
            mm = (t >= t[i0] - 1.0) & (t < t[i0])
            rec.append({'log': log, 't': float(t[i0]), 'rec_s': float(t[i1] - t[i0]) + 0.033, 'back_to_2mps_s': back,
                        'v_before_max': float(np.nanmax(v[mm])) if mm.any() else None, 'loc': NAMES[int(loc[i0])]})
    out = {'per_lap': per_lap, 'recoveries': rec}

    def summ(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {'n': len(vals), 'max': round(max(vals), 2) if vals else None,
                'p90': round(float(np.percentile(vals, 90)), 2) if vals else None,
                'median': round(float(np.median(vals)), 2) if vals else None}

    groups = {
        'B のきれいな周': [r for r in per_lap if r['model'] == 'B' and r['clean']],
        'No.9 の周(全部)': [r for r in per_lap if r['session'] == '10/24 2回目' and r['no'] == '9'],
        '全部の周': per_lap,
    }
    out['speed_summary'] = {}
    for g, rows in groups.items():
        out['speed_summary'][g] = {k: summ(rows, k) for k in ('v_r1_entry', 'vmax_GM_out', 'vmax_DE', 'vmax_SW', 'vmax_HS', 'vmax_LH')}
    # 当たらずに通った最高(その区間で衝突がない周)
    out['speed_without_hit'] = {}
    for sec in ('GM_out', 'DE', 'SW', 'HS', 'LH'):
        ok = [r['vmax_' + sec] for r in per_lap if r['vmax_' + sec] is not None and not r['coll_' + sec]]
        hit = [r['vmax_' + sec] for r in per_lap if r['vmax_' + sec] is not None and r['coll_' + sec]]
        out['speed_without_hit'][sec] = {'n_ok': len(ok), 'max_ok': round(max(ok), 2) if ok else None,
                                         'p90_ok': round(float(np.percentile(ok, 90)), 2) if ok else None,
                                         'n_hit': len(hit), 'median_hit': round(float(np.median(hit)), 2) if hit else None}
    # リカバリー
    backs = [r['back_to_2mps_s'] for r in rec if r['back_to_2mps_s'] is not None]
    out['recovery_summary'] = {'n': len(rec), 'n_back': len(backs),
                               'rec_s_median': round(float(np.median([r['rec_s'] for r in rec])), 2) if rec else None,
                               'back_median': round(float(np.median(backs)), 2) if backs else None,
                               'back_p25_p75': [round(float(np.percentile(backs, 25)), 2), round(float(np.percentile(backs, 75)), 2)] if backs else None}
    lr = [r for r in per_lap if r['rec_frames'] > 0 and r['lap_minus_clean_med'] is not None]
    out['recovery_laps'] = {'n': len(lr), 'lap_minus_clean_med_median': round(float(np.median([r['lap_minus_clean_med'] for r in lr])), 2) if lr else None,
                            'rows': [(r['log'], r['no'], r['lap_s'], round(r['lap_minus_clean_med'], 2), r['rec_frames']) for r in lr]}
    json.dump(out, open(os.path.join(HERE, 's24_records.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out['speed_summary'], ensure_ascii=False, indent=1))
    print(json.dumps(out['speed_without_hit'], ensure_ascii=False, indent=1))
    print(json.dumps(out['recovery_summary'], ensure_ascii=False))
    print(json.dumps(out['recovery_laps'], ensure_ascii=False)[:1500])
    for r in rec[:40]:
        print(r)


if __name__ == '__main__':
    main()
