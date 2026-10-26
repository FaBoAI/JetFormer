#!/usr/bin/env python3
"""偽の R1 の起点を外したあとの数え直し(段 tools、2026-10-25、spec 5.1・6.3 T-2)。読むだけ(numpy)。

10/23・10/24 の夜の成績表(scorecard.json、外す前)の記録全部を、今の tools/section_scorecard.score_log(外したあと)で
数え直し、外した起点・消えた周・つながった周・変わらない周の時間の差・3 周の窓・衝突の場所の付け直しを比べる。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I anchor_recount.py
出力: /home/jetson/jetformer_reports/speed_control_20261025/anchor_recount.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import section_scorecard as ss  # noqa: E402

SC = {'20261023': '/home/jetson/jetformer_reports/bench_20261023_day1/nightly/scorecard/scorecard.json',
      '20261024': '/home/jetson/jetformer_reports/bench_20261024_tune/since_1300/nightly/scorecard/scorecard.json'}
ANCHOR_CHECK = '/home/jetson/jetformer_reports/bench_20261024_report/anchor_check.json'
OUT = Path('/home/jetson/jetformer_reports/speed_control_20261025/anchor_recount.json')


def windows(laps):
    """続いた 3 周の合計(前の周の終わりと次の周の始まりが 0.02 秒以内)"""
    out = []
    for i in range(len(laps) - 2):
        a, b, c = laps[i:i + 3]
        if abs(a['t_start'] + a['lap_s'] - b['t_start']) <= 0.02 and abs(b['t_start'] + b['lap_s'] - c['t_start']) <= 0.02:
            out.append({'t_start': a['t_start'], 'laps_s': [a['lap_s'], b['lap_s'], c['lap_s']],
                        'total_s': round(a['lap_s'] + b['lap_s'] + c['lap_s'], 3)})
    return out


def main():
    ac = json.loads(Path(ANCHOR_CHECK).read_text(encoding='utf-8'))
    res = {'created_by': 'experiments/20261025_speed_control/anchor_recount.py', 'logs': {}, 'rejected': [],
           'rejected_in_anchor_check': [], 'laps_removed': [], 'laps_added': [], 'laps_unchanged': 0,
           'lap_unchanged_max_diff_s': 0.0, 'windows_removed': [], 'windows_added': [], 'loc_changes': []}
    t0 = time.time()
    for day, p in SC.items():
        old = json.loads(Path(p).read_text(encoding='utf-8'))['per_log']
        for lg in sorted(old):
            o = old[lg]
            if 'error' in o:
                continue
            n = ss.score_log(lg)
            if 'error' in n:
                res['logs'][lg] = {'error': n['error']}
                continue
            no = (o.get('desc') or {}).get('ai_no')
            rej = n.get('r1_anchors_rejected') or []
            for r in rej:
                row = dict(r, log=lg, no=no)
                res['rejected'].append(row)
                if lg in ac:
                    res['rejected_in_anchor_check'].append(row)
            ol = {round(x['t_start'], 3): x for x in o.get('laps', [])}
            nl = {round(x['t_start'], 3): x for x in n.get('laps', [])}
            for k, x in ol.items():
                y = nl.get(k)
                if y is None or abs(y['lap_s'] - x['lap_s']) > 0.001:
                    res['laps_removed'].append({'log': lg, 'no': no, 't_start': k, 'lap_s': x['lap_s'], 'clean': x['clean']})
                else:
                    res['laps_unchanged'] += 1
                    res['lap_unchanged_max_diff_s'] = max(res['lap_unchanged_max_diff_s'], abs(y['lap_s'] - x['lap_s']))
            for k, y in nl.items():
                x = ol.get(k)
                if x is None or abs(y['lap_s'] - x['lap_s']) > 0.001:
                    res['laps_added'].append({'log': lg, 'no': no, 't_start': k, 'lap_s': y['lap_s'], 'clean': y['clean']})
            ow = {w['t_start']: w for w in windows(o.get('laps', []))}
            nw = {w['t_start']: w for w in windows(n.get('laps', []))}
            for k, w in ow.items():
                if k not in nw or abs(nw[k]['total_s'] - w['total_s']) > 0.001:
                    res['windows_removed'].append(dict(w, log=lg, no=no))
            for k, w in nw.items():
                if k not in ow or abs(ow[k]['total_s'] - w['total_s']) > 0.001:
                    res['windows_added'].append(dict(w, log=lg, no=no))
            oc = {round(c['t_s'], 3): c for c in o.get('collision_events', [])}
            for c in n.get('collision_events', []):
                c0 = oc.get(round(c['t_s'], 3))
                if c0 and c0.get('loc') != c.get('loc'):
                    res['loc_changes'].append({'log': lg, 'no': no, 't_s': c['t_s'], 'g': c['g'], 'old': c0.get('loc'),
                                               'new': c.get('loc'), 'new_detail': c.get('loc_detail')})
            res['logs'][lg] = {'no': no, 'anchors_old': o.get('r1_anchors'), 'anchors_new': n.get('r1_anchors'),
                               'passes_old': len(o.get('passes', [])), 'passes_new': len(n.get('passes', [])),
                               'laps_old': len(o.get('laps', [])), 'laps_new': len(n.get('laps', [])),
                               'loc_relabeled': n.get('loc_relabeled')}
            print(lg, 'No.%s' % no, 'anchors %s→%s' % (o.get('r1_anchors'), n.get('r1_anchors')),
                  'laps %d→%d' % (len(o.get('laps', [])), len(n.get('laps', []))),
                  'rej', [(r['t_s'], r['reason']) for r in rej], flush=True)
    res['seconds'] = round(time.time() - t0, 1)
    res['anchor_check_anchors'] = sum(len(v) for v in ac.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('外した起点', len(res['rejected']), '(anchor_check の中', len(res['rejected_in_anchor_check']), ')')
    print('消えた周', [(x['log'], x['no'], x['t_start'], x['lap_s']) for x in res['laps_removed']])
    print('増えた周', [(x['log'], x['no'], x['t_start'], x['lap_s']) for x in res['laps_added']])
    print('変わらない周', res['laps_unchanged'], '最大の差', res['lap_unchanged_max_diff_s'])
    print('消えた窓', [(x['log'], x['no'], x['total_s']) for x in res['windows_removed']])
    print('増えた窓', [(x['log'], x['no'], x['total_s']) for x in res['windows_added']])
    print('場所の付け直し', [(x['log'], x['no'], x['t_s'], x['old'], x['new_detail']) for x in res['loc_changes']])
    print('秒', res['seconds'], '->', OUT)


if __name__ == '__main__':
    main()
