# -*- coding: utf-8 -*-
"""out/measure_before_now.json → out/summary.txt(読むだけ)。直す前(before)と後(now)を並べる"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, 'out', 'measure_before_now.json'), encoding='utf-8'))
# 1 回目に ▶ が 4 秒のうちに送られなかった場合(重い時刻の台本の遅れ。確認の文は出たが POST が無い)は、流し直し(_rerun)で置き換える
RERUN = os.path.join(HERE, 'out', 'measure_before_now_rerun.json')
replaced = []
if os.path.exists(RERUN):
    rr = {(c['case'], c['W'], c['H'], c['css']): c for c in json.load(open(RERUN, encoding='utf-8'))['cases']}
    for i, c in enumerate(r['cases']):
        k = (c['case'], c['W'], c['H'], c['css'])
        if k in rr and c['start_posts'] == 0:
            r['cases'][i] = dict(rr[k], rerun=True)
            replaced.append(k)
assert all(c['start_posts'] == 1 and c['toasts'] for c in r['cases']), [c for c in r['cases'] if c['start_posts'] != 1]
L = ['# 直す 8-2 回目・係 ui: 浮いている ■ と走行中のトースト(本物の ▶ の押し方・作り物のサーバー・headless の Firefox・iframe = 画面)',
     '# 版 ' + ' '.join('%s %s' % kv for kv in r['files'].items()) + ' / css ' + ' '.join('%s %s' % kv for kv in r['css'].items()) + ' / ' + r.get('ua', ''),
     '# 流し直しで置き換えた場合(1 回目は ▶ の POST が 4 秒のうちに無かった = 台本の遅れ): %s' % (replaced or 'なし'),
     '# 列: 場合 幅x高さ css | ■[左,上,幅,高さ] 行数「行」 | 画面の外の字 | トースト(文・class・見えた ms) | ■ と一緒に出た回 | '
     'トーストの下と ■ の上の間(最小 px。負 = 重なる) | 重なり[幅,縦,px²] 重なった ms | 半分より多く隠れた字 | 下までスクロールで ■ が隠すカード [px]']
by = {}
for c in r['cases']:
    by[(c['case'], c['W'], c['H'], c['css'])] = c
    ts = ' / '.join('「%s」%s %s' % (t['text'][:26], t['cls'].replace('toast', 't').strip(), t['seen_ms']) for t in c['toasts'])
    se = c['scroll_end']
    L.append('%-6s %4dx%-4d %-6s | %s %d「%s」 | %s | %s | %d | %s | %s %s | %s | %s%s%s' % (
        c['case'], c['W'], c['H'], c['css'], c['float_rect'], len(c['float_lines'] or []), ' / '.join(c['float_lines'] or []),
        ''.join(c['chars_out']) or '-', ts, c['both_samples'], c['min_gap'], c['overlap_max'], c['overlap_ms'], c['half_hidden_chars'] or '-',
        se.get('hidden_px'), (' errs=%s' % c['errs']) if c['errs'] else '', ' トーストが画面の上にはみ出す' if c['toast_out_top'] else ''))

# まとめ(場合ごと・css ごと)
L.append('')
L.append('# まとめ: スマホ(幅 360〜430)')
agg = defaultdict(lambda: {'n': 0, 'ov': 0, 'gap': [], 'vov': [], 'lines': set(), 'h': set(), 'hid': [], 'out': 0, 'errs': 0, 'ms': []})
for c in r['cases']:
    if c['W'] > 430:
        continue
    a = agg[(c['case'], c['css'])]
    a['n'] += 1
    a['ov'] += 1 if c['overlap_max'] else 0
    a['gap'].append(c['min_gap'])
    a['vov'].append(c['overlap_max'][1] if c['overlap_max'] else 0)
    a['lines'].add(len(c['float_lines']))
    a['h'].add(c['float_rect'][3])
    a['hid'].append(c['scroll_end'].get('hidden_px') or 0)
    a['out'] += 1 if ''.join(c['chars_out']) else 0
    a['errs'] += len(c['errs'])
    if c['overlap_ms']:
        a['ms'].append(c['overlap_ms'][1] - c['overlap_ms'][0])
for (case, css), a in sorted(agg.items()):
    L.append('%-6s %-6s: %d 回 | 重なった回 %d | 縦の重なり %s〜%s px | 間 %s〜%s px | ■ の行 %s・高さ %s | 重なった長さ %s〜%s ms | '
             '下までスクロールで ■ が隠すカード %s〜%s px | 画面の外の字がある回 %d | errs %d' % (
                 case, css, a['n'], a['ov'], min(a['vov']), max(a['vov']), min(a['gap']), max(a['gap']), sorted(a['lines']), sorted(a['h']),
                 min(a['ms']) if a['ms'] else '-', max(a['ms']) if a['ms'] else '-', min(a['hid']), max(a['hid']), a['out'], a['errs']))

L.append('')
L.append('# 幅ごと(スマホ): ■ の [幅, 高さ]・行・縦の重なり・間・半分より多く隠れた字・下までスクロールで ■ が隠すカード')
per = defaultdict(list)
for c in r['cases']:
    if c['W'] <= 430:
        per[(c['W'], c['case'], c['css'])].append(c)
for (W, case, css), cs in sorted(per.items()):
    ov = [c['overlap_max'][1] if c['overlap_max'] else 0 for c in cs]
    L.append('%d %-6s %-6s %2d 回 | ■ %s | 「%s」 | 縦 %s〜%s px | 間 %s〜%s px | 隠れた字 %s | スクロール %s' % (
        W, case, css, len(cs), sorted({(c['float_rect'][2], c['float_rect'][3]) for c in cs}), sorted({' / '.join(c['float_lines']) for c in cs}),
        min(ov), max(ov), min(c['min_gap'] for c in cs), max(c['min_gap'] for c in cs),
        sorted({h for c in cs for h in c['half_hidden_chars']}) or '-', sorted({c['scroll_end']['hidden_px'] for c in cs})))

L.append('')
L.append('# 幅 600 以上: 直す前と後の ■・トーストの箱')
for c in r['cases']:
    if c['W'] <= 430 or c['css'] != 'now':
        continue
    b = by.get((c['case'], c['W'], c['H'], 'before'))
    same = b and b['float_rects_all'] == c['float_rects_all'] and b['toast_rects_with_float'] == c['toast_rects_with_float']
    L.append('%-6s %4dx%-4d | 前 ■ %s %d 行・トースト %s 間 %s | 今 ■ %s %d 行・トースト %s 間 %s | 同じ %s' % (
        c['case'], c['W'], c['H'], b['float_rect'], len(b['float_lines']), b['toast_rects_with_float'][:2], b['min_gap'],
        c['float_rect'], len(c['float_lines']), c['toast_rects_with_float'][:2], c['min_gap'], same))

L.append('')
L.append('# 止まっているときのトースト(本物の toast() に渡す。▶ を画面の真ん中に): 直す前と後の箱')
idle = defaultdict(dict)
for x in r['idle']:
    idle[(x['W'], x['H'])][x['css']] = x['toasts']
n_same = n_all = 0
for (W, H), d in sorted(idle.items()):
    if 'before' not in d or 'now' not in d:
        continue
    for name in d['now']:
        n_all += 1
        same = d['before'][name]['toast'] == d['now'][name]['toast'] and d['before'][name]['cls'] == d['now'][name]['cls']
        n_same += same
        L.append('%4dx%-4d %-8s | 前 %s | 今 %s | 同じ %s | body.driving %s・■ hidden %s' % (
            W, H, name, d['before'][name]['toast'], d['now'][name]['toast'], same, d['now'][name]['driving'], d['now'][name]['float_hidden']))
L.append('止まっているときのトーストが前と同じ箱: %d / %d' % (n_same, n_all))
open(os.path.join(HERE, 'out', 'summary.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L[-60:]))
