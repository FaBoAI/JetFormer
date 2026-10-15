#!/usr/bin/env python3
"""前日との同一性の確かめ(sim_diverse_train_20261015 の Q1 の評価の段、2026-10-15。protocol.json eval_Q1.same_as_yesterday_check)

    python3 -I same_as_yesterday.py

  - 陽性対照: 今日の作者の ONNX の E0 が route_complete・33 ゲート・衝突 0 で、周回時間が 10/14 の 9.022 秒(9.02198843516568)と同じか
  - A1-s0 の E0: 今日の走行と 10/14 の走行(sim_dart_20261014/eval/closed_loop/A1-s0/A1-s0-E0-single.json)の
    フレーム数・終わり方・軌跡(trace の x・z の差の最大 ≤ 1e-6 m)が同じか
生の JSON だけを読む。出力: eval/same_as_yesterday.json。同じなら終了コード 0、違えば 3、走行が無ければ 2。
"""
import json
import sys
from pathlib import Path

R = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015/eval/closed_loop')
P = Path('/home/jetson/jetformer_reports/sim_dart_20261014/eval/closed_loop')
OUT = R.parent / 'same_as_yesterday.json'
TOL = 1e-6


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def brief(r):
    return {'reason': r['outcome']['reason'], 'frames': len(r.get('actions') or []), 'passed_gates': r['metrics']['passed_gates'],
            'collisions': r['metrics']['collisions'], 'elapsed': r['metrics'].get('elapsed'), 'batch_status': r.get('batch_status'),
            'model_sha256': (r.get('context') or {}).get('policy_info', {}).get('model_sha256')}


def compare(today, prev):
    a, b = load(today), load(prev)
    ta, tb = a.get('trace') or [], b.get('trace') or []
    n = min(len(ta), len(tb))
    dmax = max((max(abs(ta[i]['x'] - tb[i]['x']), abs(ta[i]['z'] - tb[i]['z'])) for i in range(n)), default=None)
    act = max((max(abs(a['actions'][i]['steering'] - b['actions'][i]['steering']), abs(a['actions'][i]['speed'] - b['actions'][i]['speed']))
               for i in range(min(len(a['actions']), len(b['actions'])))), default=None)
    ba, bb = brief(a), brief(b)
    same = (ba['reason'] == bb['reason'] and ba['frames'] == bb['frames'] and len(ta) == len(tb) and dmax is not None and dmax <= TOL
            and ba['model_sha256'] == bb['model_sha256'])
    return {'today': ba, 'yesterday': bb, 'trace_len': [len(ta), len(tb)], 'max_abs_xy_diff_m': dmax, 'max_abs_action_diff': act,
            'same': same, 'today_file': str(today), 'yesterday_file': str(prev)}


def main():
    out = {'rule': 'protocol.json eval_Q1.same_as_yesterday_check: 陽性対照が 9.022 秒で完走し、A1-s0 の E0 が 10/14 と同じフレーム数・終わり方・軌跡(差 ≤ 1e-6 m)'}
    files = {'EXT': (R / 'EXT' / 'EXT-E0-single.json', P / 'EXT' / 'EXT-E0-single.json'),
             'A1-s0': (R / 'A1-s0' / 'A1-s0-E0-single.json', P / 'A1-s0' / 'A1-s0-E0-single.json')}
    missing = [k for k, (t, _) in files.items() if not t.exists()]
    if missing:
        out['missing'] = missing
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print('走行がありません:', missing)
        sys.exit(2)
    ext = compare(*files['EXT'])
    ext['completed'] = ext['today']['reason'] == 'route_complete' and ext['today']['collisions'] == 0 and ext['today']['passed_gates'] == 33
    ext['lap_s_today'] = ext['today']['elapsed'] if ext['completed'] else None
    ext['lap_s_yesterday'] = 9.02198843516568
    ext['lap_same'] = ext['lap_s_today'] is not None and abs(ext['lap_s_today'] - ext['lap_s_yesterday']) < 1e-9
    a1 = compare(*files['A1-s0'])
    out['positive_control_EXT_E0'] = ext
    out['A1_s0_E0'] = a1
    out['same_as_yesterday'] = bool(ext['completed'] and ext['lap_same'] and a1['same'])
    out['action'] = ('前日と同じ環境: A0・A1・A2 の 10/14 の数字をそのまま使う' if out['same_as_yesterday'] else
                     '前日と同じとは言えない: S-DIV のあとに A2-s0 の E0〜E11 を走らせ直し、A0・A1 の数字に「前日の環境」と書く')
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'same_as_yesterday': out['same_as_yesterday'], 'EXT': [ext['today']['reason'], ext['lap_s_today'], ext['max_abs_xy_diff_m']],
                      'A1': [a1['today']['reason'], a1['today']['frames'], a1['max_abs_xy_diff_m']]}, ensure_ascii=False))
    sys.exit(0 if out['same_as_yesterday'] else 3)


if __name__ == '__main__':
    main()
