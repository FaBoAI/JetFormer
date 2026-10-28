# -*- coding: utf-8 -*-
"""out/measure_float_stop.json → out/summary.txt(読むだけ)"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, 'out', 'measure_float_stop.json'), encoding='utf-8'))
L = ['# 打ち消しの票 A: 浮いている ■ とトースト(本物の ▶ の押し方・作り物のサーバー・headless の Firefox)',
     '# 版 ' + ' '.join('%s %s' % kv for kv in r['files'].items()) + ' / ' + r.get('ua', ''),
     '# 列: 場合 幅x高さ(inner 幅・高さ・clientWidth) | 止まっている間 ■ hidden | ■ が出た ms | ■[左,上,幅,高さ] 行 | トースト(文・class・見えた ms) | '
     '重なった ms・回数 | 重なり[幅,縦,px²] | 半分より多く隠れた字 | 1 行目の真ん中がトーストの下 | その点の elementFromPoint | サーバーの plan_diffs']
for c in r['cases']:
    ts = ' / '.join('「%s」%s %s' % (t['text'][:40], t['cls'], t['seen_ms']) for t in c['toasts'])
    L.append('%-6s %4dx%-4d (%s) | %s | %s | %s %s | %s | %s %s | %s | %s | %s | %s | %s%s' % (
        c['case'], c['W'], c['H'], ','.join(map(str, c['inner'])), c['idle_float_hidden'], c['float_shown_at_ms'],
        c['float_rect'], c['float_lines'], ts, c['overlap_ms'], c['overlap_samples'], c['overlap_max'], c['half_hidden_chars'],
        c['line1_center_under_toast'], c['hit_line1'], c['server_plan_diffs'], (' errs=%s' % c['errs']) if c['errs'] else ''))
L.append('')
L.append('# 覆われた所を指(WebDriver の touch)で押す')
for t in r['tap']:
    L.append('%dx%d %s 押した点 %s | ■ %s | トースト %s「%s」 | stop を送った %s %s | 押したあと running %s' % (
        t['W'], t['H'], t.get('how'), t['tap_xy'], [round(x, 1) for x in t['float']], [round(x, 1) for x in t['toast']], t['toast_text'][:30],
        t['stop_posted'], t['stop_body'], t['running_after']))
open(os.path.join(HERE, 'out', 'summary.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
