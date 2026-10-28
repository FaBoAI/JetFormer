"""fix8 係 safety-decided: rec_replay10 の出力(out/rec_replay10.json)をまとめる → 標準出力と out/analyze_rec10.json。
発進ごと(collect.py --day の行)に、ON と OFF が同じか・発進の区間の AI 側の中立止め(記録の行・カメラが 0.1 秒来ない)・
記録の AI 側の処理したフレーム(中立止めの行でない)の撮像時刻の間(最大・0.1 秒より長い・0.5 秒より長い)と、そのときの速さ(VEHICLE の eRPM)を出す
(data/raw を読むだけ)。
    python -I analyze_rec10.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / 'out'
RAW = Path('/home/jetson/jetformer/data/raw')
LAB = Path('/home/jetson/jetformer/data/labels')
KV = 0.0005           # safety_caps.KV(1000 eRPM = 0.5 m/s)


def gaps(log, t_ai, t_end):
    """発進の区間の AI 側のフレームの間 [(始めの秒, 長さ, そのときの速さ m/s)](0.1 秒より長いもの)と最大"""
    t0 = int(json.loads((LAB / (log + '.json')).read_text(encoding='utf-8'))['t0_us'])
    ts, sel, vt, ve = [], [], [], []
    for s in sorted(d for d in (RAW / log).iterdir() if d.is_dir()):
        try:
            z = np.load(s / 'AI_CMD.npz')
            ts.append(z['timestamps_us'].astype(np.int64))
            sel.append(z['select'].astype(bool) & ~z['hold'].astype(bool))    # 処理したフレーム(中立止めの行でない)
            v = np.load(s / 'VEHICLE.npz')
            vt.append(v['timestamps_us'].astype(np.int64))
            ve.append(v['erpm'].astype(float))
        except Exception:       # noqa: BLE001
            continue
    ts, sel = np.concatenate(ts), np.concatenate(sel)
    vt, ve = np.concatenate(vt), np.concatenate(ve)
    t = (ts - t0) / 1e6
    m = (t >= t_ai) & (t <= t_end)
    tt, ss = t[m & sel], sel[m & sel]      # AI 側の処理したフレームどうしの間(scan_gaps10 と同じ見方。人側をはさむ間も入るが発進の区間は AI 側)
    out = []
    mx = 0.0
    for a, b, sa in zip(tt, tt[1:], ss):
        g = b - a
        if sa:
            mx = max(mx, g)
            if g > 0.1:
                k = np.searchsorted(vt, t0 + int(a * 1e6), side='right') - 1
                out.append((round(float(a), 3), round(float(g), 3), round(float(-ve[k] * KV), 2) if k >= 0 else None))
    return round(float(mx), 3), out


def main():
    d = json.loads((OUT / 'rec_replay10.json').read_text(encoding='utf-8'))
    n_l = collections.Counter()
    rows = []
    fid = [0, 0]
    files_same = collections.Counter()
    for log, e in sorted(d['logs'].items()):
        if 'error' in e:
            print('skip', log, e['error'])
            continue
        f = e.get('fidelity_off') or {}
        fid[0] += f.get('rows_same', 0)
        fid[1] += f.get('rows_matched', 0)
        for lb in ('lb0', 'lb1'):
            files_same[(lb, e[lb]['same_all'])] += 1
        for L in e['launches']:
            te = L['first_lap_end_s'] if L['first_lap_end_s'] is not None else L['t_ai'] + (L.get('ai_seg_s') or 15) + 0.5
            mx, gl = gaps(log, L['t_ai'], te)
            r = {'log': log, 't_ai': L['t_ai'], 'first_lap_end_s': L['first_lap_end_s'], 'first_lap_end': L['first_lap_end'],
                 'to_r1_s': L['to_r1_s'], 'kick': L['kick'], 'recovery_rec': L['recovery'],
                 'collisions_first_lap': L['collisions_first_lap'],
                 'same_lb0': L['lb0']['same'], 'same_lb1': L['lb1']['same'], 'diff_lb0': L['lb0']['diff'],
                 'diff_lb1': L['lb1']['diff'], 'ai_hold_rows': L['lb0']['ai_holds_rows'],
                 'ai_cam_timeouts': L['lb0']['ai_cam_timeouts'], 'rec_frames_off': L['lb0']['rec_frames_off'],
                 'max_gap_s': mx, 'gaps_gt_0p1': gl}
            rows.append(r)
            k = ('同じ' if r['same_lb0'] and r['same_lb1'] else '違う',
                 '中立止めあり' if (r['ai_hold_rows'] or r['ai_cam_timeouts']) else '中立止めなし',
                 'リカバリーあり' if r['rec_frames_off'] else 'リカバリーなし',
                 '当たりあり' if r['collisions_first_lap'] else '当たりなし')
            n_l[k] += 1
    print('記録 %d 本・OFF の流し直しの出す値が記録と同じ行 %d / %d' % (len(d['logs']), fid[0], fid[1]))
    print('記録全体で ON と OFF が同じ:', dict(files_same))
    print('発進 %d' % len(rows))
    for k, v in sorted(n_l.items()):
        print('  %3d  %s' % (v, '・'.join(k)))
    for r in rows:
        flag = '' if r['same_lb0'] and r['same_lb1'] else '  ← 違う'
        print('%s %7.2f〜%-7s 起点まで %-6s 中立止めの行 %-2d カメラ 0.1 秒 %-2d 最大の間 %.3f %s%s' % (
            r['log'], r['t_ai'], r['first_lap_end_s'], r['to_r1_s'], len(r['ai_hold_rows']), len(r['ai_cam_timeouts']),
            r['max_gap_s'], r['gaps_gt_0p1'][:4], flag))
    long_ = [(r['log'], g) for r in rows for g in r['gaps_gt_0p1'] if g[1] > 0.5]
    print('発進の区間の 0.5 秒より長い間:', long_ or 'なし')
    print('0.1 秒より長い間のある発進:', sum(1 for r in rows if r['gaps_gt_0p1']), '/', len(rows),
          ' 間の数', sum(len(r['gaps_gt_0p1']) for r in rows))
    (OUT / 'analyze_rec10.json').write_text(json.dumps({'fidelity_off': fid, 'files_same': {'%s:%s' % k: v for k, v in files_same.items()},
                                                        'classes': {'・'.join(k): v for k, v in n_l.items()}, 'launches': rows},
                                                       ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
