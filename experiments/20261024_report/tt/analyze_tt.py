"""解析 A: 3 周のタイムトライアル(10/23 の公式 8 本 + 10/24 の AI 走行 No.1〜20)。2026-10-25

周の切れ目 = scorecard.json の laps(R1 の起点 → 次の起点)。
  1. 切れ目が続いているか(前の周の終わり = 次の周の始まり、差 0.02 秒以内)を確かめる。
  2. 偽の起点(衝突で車が右へ回ったのを R1 と取り違えた)で切られた周を外す。
     判定 = IMU の向きの変化 < 270°(1 周 = 左回り +360°)かつ 車輪の距離 < 1 周の中央値の 80%。
     (anchor_check.py の結果。フレームでも確かめた: 偽の起点は駐車場の P の印・壁の前、本物は長い直線と右のスロープ)
  3. 続いた 3 周の窓(1 周ずつずらす)の合計。窓の中の衝突(label の規則・水平 2.5G 以上)で
     きれい / 軽い接触だけ(4G 未満)/ 強い衝突(4G 以上)に分ける。
  4. 止まった状態からの発進(AI 側にした最初のフレームで |eRPM| < 400)→ 最初の R1 の起点までの時間と、
     同じ距離を走っている周で走ったときの時間の差(= 発進で失う時間)。
  5. 区間の時間(sections_s)の比べ: No.9 と 3 回目の B 0.90(No.11・12・18)。区間ごとの速さも labels から。

    python3 -I analyze_tt.py      # → /home/jetson/jetformer_reports/bench_20261024_report/tt_data.json と表(標準出力)
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import label_events as le  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/bench_20261024_report')
LAB = Path('/home/jetson/jetformer/data/labels')
RAW = Path('/home/jetson/jetformer/data/raw')
SC = {
    '20261023': '/home/jetson/jetformer_reports/bench_20261023_day1/nightly/scorecard/scorecard.json',
    '20261024': '/home/jetson/jetformer_reports/bench_20261024_tune/since_1300/nightly/scorecard/scorecard.json',
}
OFFICIAL_1023 = ['20261023-210840', '20261023-211020', '20261023-211118', '20261023-211209',
                 '20261023-211308', '20261023-211357', '20261023-211448', '20261023-211547']
MODEL = {
    'clean_driving_20261011-nocmd-s3-20261011-024626': 'B',
    'bench_20261022-slow-sec-s1': '遅く入る型',
    'bench_20261021-ens-p2': '2本の平均',
    'bench_20261021-aug-s0': 'ずれに強い型',
    'clean_driving_20261011-act-s0-nocmd-stride3-e30': 'ACT型',
    'bench_20261021-wgt-s0': '苦手区間重み型',
}
TOP_S, PASS_S, GOAL_S = 23.0, 27.0, 25.0
STRONG_G = 4.0
STOP_ERPM = 400
SECT = ['R1', 'GM', 'DE', 'other', 'HS', 'LH']
GROUPS = {'R1+GM': ('R1', 'GM'), 'DE+SW': ('DE', 'other'), 'HS+LH': ('HS', 'LH')}
LOC_IDX = {n: i for i, n in enumerate(le.LOC_NAMES)}


def session_of(log, no):
    if log.startswith('20261023'):
        return '10/23 公式'
    no = int(no)
    return '10/24 1回目' if no <= 4 else ('10/24 2回目' if no <= 10 else '10/24 3回目')


def track(log, t0_us):
    """IMU の向き [deg] と車輪の前進距離 [m](label_events と同じ作り方)"""
    L = le.load_log(log)
    I, V = L['IMU'], L['VEHICLE']
    it = (I['timestamps_us'].astype(np.int64) - t0_us) / 1e6
    vt = (V['timestamps_us'].astype(np.int64) - t0_us) / 1e6
    verpm = V['erpm'].astype(np.float64)
    gz = I['gyro'][:, 2].astype(np.float64)
    still = (np.abs(np.interp(it, vt, verpm)) < 50) & (np.abs(gz) < 5)
    hd, _ = le.heading_deg(it, gz, still)
    _, Df = le.cumulative_distance(vt, verpm)
    return it, hd, vt, Df, verpm


def r(x, n=3):
    return None if x is None else round(float(x), n)


def main():
    logs = []
    for day, p in SC.items():
        sc = json.loads(Path(p).read_text())
        for log, e in sc['per_log'].items():
            if day == '20261023' and log not in OFFICIAL_1023:
                continue
            if 'laps' not in e:
                continue
            logs.append((log, e))

    runs, laps_all, windows, starts = [], [], [], []
    for log, e in logs:
        lab = json.loads((LAB / (log + '.json')).read_text())
        z = dict(np.load(LAB / (log + '.npz')))
        meta = json.loads((RAW / log / 'log.json').read_text())
        no = e['desc'].get('ai_no')
        ai = z['ai'].astype(bool)
        t = z['t_s']
        g = z['gains'][ai] if ai.any() else np.zeros((0, 2))
        sg, tg = (round(float(np.median(g[:, 0])), 3), round(float(np.median(g[:, 1])), 3)) if len(g) else (None, None)
        p = (meta.get('drive') or {}).get('params') or {}
        model = MODEL.get(e.get('run') or (meta.get('drive') or {}).get('run'), e.get('run'))
        ses = session_of(log, no)
        setting = '%s 速さ%.2f ステア%.2f' % (model, tg, sg) if tg is not None else model
        it, hd, vt, Df, verpm = track(log, lab['t0_us'])
        col = e.get('collision_events') or []
        anchors = [x['ta_s'] for x in e['passes']]
        # 周の長さ(距離)の中央値: 起点 → 次の起点で、向きの変化が 1 周らしいもの
        dist_ok = []
        for a, b in zip(anchors[:-1], anchors[1:]):
            dpsi = np.interp(b, it, hd) - np.interp(a, it, hd)
            if 300 <= dpsi <= 420 and b - a < 16:
                dist_ok.append(np.interp(b, vt, Df) - np.interp(a, vt, Df))
        Lrun = float(np.median(dist_ok)) if dist_ok else None
        run = {'log': log, 'no': no, 'session': ses, 'model': model, 'throttle_gain': tg, 'steer_gain': sg,
               'params_throttle_gain': p.get('throttle_gain'), 'params_steer_gain': p.get('steer_gain'),
               'setting': setting, 'lap_dist_median_m': r(Lrun, 2), 'laps_scorecard': len(e['laps'])}
        # ---- 周
        L = []
        for k, lp in enumerate(e['laps']):
            a, d = lp['t_start'], lp['lap_s']
            b = a + d
            dpsi = float(np.interp(b, it, hd) - np.interp(a, it, hd))
            dist = float(np.interp(b, vt, Df) - np.interp(a, vt, Df))
            false_end = dpsi < 270 and (Lrun is None or dist < 0.8 * Lrun)
            cc = [c for c in col if a <= c['t_s'] < b]
            gmax = max([c['g'] for c in cc], default=0.0)
            if not cc and not lp['clean']:
                kind = 'リカバリーなど'
            elif not cc:
                kind = 'きれい'
            elif gmax < STRONG_G:
                kind = '軽い接触'
            else:
                kind = '強い衝突'
            # 区間ごとの速さ(labels の speed_mps)
            m = (t >= a) & (t < b)
            spd = {}
            for gname, mem in GROUPS.items():
                mm = m & np.isin(z['loc_code'], [LOC_IDX[x] for x in mem])
                spd[gname] = r(np.mean(z['speed_mps'][mm]), 2) if mm.any() else None
            spd['p90'] = r(np.percentile(z['speed_mps'][m], 90), 2) if m.any() else None
            row = {'log': log, 'no': no, 'session': ses, 'setting': setting, 'model': model, 'tg': tg, 'sg': sg,
                   'k': k, 't_start': a, 't_end': r(b), 'lap_s': d, 'clean': lp['clean'], 'kind': kind,
                   'collisions': [(c['t_s'], c['g'], c['loc']) for c in cc], 'gmax': gmax,
                   'dpsi_deg': r(dpsi, 1), 'dist_m': r(dist, 2), 'false_end': bool(false_end),
                   'sections_s': lp['sections_s'],
                   'groups_s': {gname: r(sum(lp['sections_s'].get(x, 0) for x in mem)) for gname, mem in GROUPS.items()},
                   'speed_mps': spd}
            L.append(row)
        laps_all.extend(L)
        valid = [x for x in L if not x['false_end']]
        # ---- 続いた 3 周
        for i in range(len(valid) - 2):
            w = valid[i:i + 3]
            cont = all(abs(w[j]['t_end'] - w[j + 1]['t_start']) <= 0.02 for j in range(2))
            if not cont:
                continue
            kinds = [x['kind'] for x in w]
            if all(k == 'きれい' for k in kinds):
                wk = 'きれい'
            elif any(k == '強い衝突' for k in kinds):
                wk = '強い衝突あり'
            elif any(k == '軽い接触' for k in kinds):
                wk = '軽い接触だけ'
            else:
                wk = 'リカバリーなど'
            windows.append({'log': log, 'no': no, 'session': ses, 'setting': setting, 'model': model, 'tg': tg, 'sg': sg,
                            't_start': w[0]['t_start'], 'laps_s': [x['lap_s'] for x in w],
                            'total_s': r(sum(x['lap_s'] for x in w)), 'kind': wk,
                            'gmax': max(x['gmax'] for x in w)})
        # ---- 止まった状態からの発進
        segs, i, n = [], 0, len(ai)
        while i < n:
            if ai[i]:
                j = i
                while j + 1 < n and ai[j + 1]:
                    j += 1
                segs.append((i, j))
                i = j + 1
            else:
                i += 1
        for (i, j) in segs:
            t0, t1 = t[i], t[j]
            e0 = float(np.interp(t0, vt, verpm))
            if abs(e0) >= STOP_ERPM:
                continue
            kick = np.nonzero(z['assist'][i:j + 1] == 4)[0]
            mv = np.nonzero((vt >= t0) & (vt <= t1) & (verpm <= -800))[0]
            t_move = float(vt[mv[0]]) if len(mv) else None
            nxt = [a for a in anchors if t0 < a <= t1]
            s = {'log': log, 'no': no, 'session': ses, 'setting': setting, 'model': model, 'tg': tg,
                 't_ai': r(t0), 'erpm_at_ai': r(e0, 0), 'kick': bool(len(kick)),
                 'kick_s': r(len(kick) / 30.0, 2), 'move_after_s': r(t_move - t0) if t_move else None,
                 'ai_seg_s': r(t1 - t0, 2)}
            if not nxt or nxt[0] - t0 > 15:
                s['to_r1_s'] = None
                starts.append(s)
                continue
            a1 = nxt[0]
            d = float(np.interp(a1, vt, Df) - np.interp(t0, vt, Df))
            dpsi = float(np.interp(a1, it, hd) - np.interp(t0, it, hd))
            cc = [c for c in col if t0 <= c['t_s'] < a1]
            # 同じ走行の(偽でない・続いた)周で、起点の d m 手前 → 起点にかかった時間
            fly = []
            for x in valid:
                a, b = x['t_start'], x['t_end']
                Da, Db = np.interp(a, vt, Df), np.interp(b, vt, Df)
                if Db - Da <= d:
                    continue
                m = (vt >= a) & (vt <= b)
                tt, DD = vt[m], Df[m] - Da
                tx = float(np.interp((Db - Da) - d, DD, tt))
                fly.append(b - tx)
            # 発進の場所: 速い周のなかで、起点の d m 手前の場所の区間
            locs = []
            for x in valid:
                a, b = x['t_start'], x['t_end']
                Da, Db = np.interp(a, vt, Df), np.interp(b, vt, Df)
                if Db - Da <= d:
                    continue
                m = (vt >= a) & (vt <= b)
                tx = float(np.interp((Db - Da) - d, Df[m] - Da, vt[m]))
                k = int(np.clip(np.searchsorted(t, tx), 0, len(t) - 1))
                locs.append(le.LOC_NAMES[int(z['loc_code'][k])])
            s.update({'to_r1_s': r(a1 - t0), 'dist_to_r1_m': r(d, 2), 'dpsi_to_r1_deg': r(dpsi, 1),
                      'collisions_before_r1': [(c['t_s'], c['g'], c['loc']) for c in cc],
                      'fly_same_dist_s': r(np.median(fly)) if fly else None, 'fly_n': len(fly),
                      'loss_s': r(a1 - t0 - np.median(fly)) if fly else None,
                      'start_loc': max(set(locs), key=locs.count) if locs else None,
                      'start_frac_of_lap': r(1 - d / Lrun, 3) if Lrun else None})
            # 発進 → 3 周(発進の場所 = 計測線と仮定): 3 × 1 周の距離を走り終えた時刻(ずっと AI・その間の周が偽でない)
            if Lrun:
                D0 = np.interp(t0, vt, Df)
                m = (vt >= t0) & (vt <= t1)
                if Df[m][-1] - D0 >= 3 * Lrun:
                    t3 = float(np.interp(D0 + 3 * Lrun, Df[m], vt[m]))
                    bad = any(x['false_end'] for x in L if t0 <= x['t_start'] < t3)
                    c3 = [c for c in col if t0 <= c['t_s'] < t3]
                    s.update({'standing_3laps_s': r(t3 - t0), 'standing_3laps_coll': [(c['t_s'], c['g'], c['loc']) for c in c3],
                              'standing_3laps_false_anchor': bad})
            starts.append(s)
        runs.append(run)

    # ---- まとめ(設定 × 回)
    def stats(ws):
        tot = [w['total_s'] for w in ws]
        cl = [w['total_s'] for w in ws if w['kind'] == 'きれい']
        nolight = [w['total_s'] for w in ws if w['kind'] in ('きれい', '軽い接触だけ')]
        return {'n': len(tot), 'n_clean': len(cl), 'n_clean_or_light': len(nolight),
                'min': r(min(tot)) if tot else None, 'median': r(st.median(tot)) if tot else None,
                'min_clean': r(min(cl)) if cl else None, 'median_clean': r(st.median(cl)) if cl else None,
                'share_le27': r(sum(x <= PASS_S for x in tot) / len(tot), 3) if tot else None,
                'share_le27_clean': r(sum(x <= PASS_S for x in cl) / len(cl), 3) if cl else None,
                'share_le25': r(sum(x <= GOAL_S for x in tot) / len(tot), 3) if tot else None,
                'gap_to_top_min': r(min(tot) - TOP_S) if tot else None,
                'gap_to_top_min_clean': r(min(cl) - TOP_S) if cl else None,
                'logs': sorted({w['no'] for w in ws}, key=lambda x: int(x))}
    table = {}
    for w in windows:
        table.setdefault((w['setting'], w['session']), []).append(w)
    summary = [{'setting': k[0], 'session': k[1], **stats(v)} for k, v in table.items()]
    by_setting = {}
    for w in windows:
        by_setting.setdefault(w['setting'], []).append(w)
    summary_setting = [{'setting': k, **stats(v)} for k, v in by_setting.items()]
    # 回ごと(設定を混ぜる)
    by_ses = {}
    for w in windows:
        by_ses.setdefault(w['session'], []).append(w)
    summary_session = [{'session': k, **stats(v)} for k, v in by_ses.items()]
    # 周ごと(設定 × 回)
    lap_tab = {}
    for x in laps_all:
        if x['false_end']:
            continue
        lap_tab.setdefault((x['setting'], x['session']), []).append(x)
    lap_summary = []
    for k, v in lap_tab.items():
        cl = [x['lap_s'] for x in v if x['kind'] == 'きれい']
        lap_summary.append({'setting': k[0], 'session': k[1], 'laps': len(v), 'clean': len(cl),
                            'best_clean': r(min(cl)) if cl else None, 'median_clean': r(st.median(cl)) if cl else None,
                            'median_all': r(st.median([x['lap_s'] for x in v]))})

    # ---- 区間の比べ: No.9 と 3 回目の B 0.90 ステア 1.00
    def sect_stats(rows):
        o = {}
        for nm in SECT + list(GROUPS):
            vals = [x['sections_s'][nm] if nm in SECT else x['groups_s'][nm] for x in rows]
            o[nm] = {'median': r(st.median(vals)), 'mean': r(st.mean(vals))}
        o['lap'] = {'median': r(st.median([x['lap_s'] for x in rows])), 'mean': r(st.mean([x['lap_s'] for x in rows]))}
        for gname in list(GROUPS) + ['p90']:
            vals = [x['speed_mps'][gname] for x in rows if x['speed_mps'].get(gname) is not None]
            o['speed_' + gname] = r(st.median(vals), 2) if vals else None
        o['n'] = len(rows)
        return o
    no9 = [x for x in laps_all if x['log'] == '20261024-162033' and not x['false_end']]
    s3 = [x for x in laps_all if x['session'] == '10/24 3回目' and x['model'] == 'B' and x['tg'] == 0.9
          and x['sg'] == 1.0 and not x['false_end']]
    b1023 = [x for x in laps_all if x['session'] == '10/23 公式' and x['model'] == 'B' and not x['false_end']]
    sect_cmp = {
        'No9_all': sect_stats(no9), 'No9_clean': sect_stats([x for x in no9 if x['kind'] == 'きれい']),
        's3_B090_all': sect_stats(s3), 's3_B090_clean': sect_stats([x for x in s3 if x['kind'] == 'きれい']),
        'B_1023_all': sect_stats(b1023),
        's3_logs': sorted({x['no'] for x in s3}, key=int),
    }
    res = {'created_by': 'experiments/20261024_report/tt/analyze_tt.py', 'runs': runs, 'laps': laps_all,
           'windows': windows, 'summary': summary, 'summary_setting': summary_setting,
           'summary_session': summary_session, 'lap_summary': lap_summary,
           'starts': starts, 'sections': sect_cmp}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tt_data.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

    # ---- 表(確認用)
    print('## 偽の起点で切れた周')
    for x in laps_all:
        if x['false_end']:
            print(' ', x['log'], 'No.%s' % x['no'], x['setting'], 'lap %.3f' % x['lap_s'], 'dpsi', x['dpsi_deg'], 'dist', x['dist_m'],
                  x['collisions'])
    print('\n## 周(偽を除く)')
    for x in laps_all:
        if not x['false_end']:
            print('  No.%-3s %-28s %7.3f %-6s %s dpsi=%s dist=%s' % (x['no'], x['setting'], x['lap_s'], x['kind'],
                  ['%.2f' % c[1] for c in x['collisions']], x['dpsi_deg'], x['dist_m']))
    print('\n## 窓')
    for w in windows:
        print('  %s No.%-3s %-28s %6.2f %s %s' % (w['session'], w['no'], w['setting'], w['total_s'], w['kind'], w['laps_s']))
    print('\n## 設定 × 回')
    for s in sorted(summary, key=lambda s: (s['session'], s['setting'])):
        print(' ', s)
    print('\n## 設定')
    for s in summary_setting:
        print(' ', s)
    print('\n## 回')
    for s in summary_session:
        print(' ', s)
    print('\n## 発進')
    for s in starts:
        print(' ', s)
    print('\n## 区間')
    print(json.dumps(sect_cmp, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
