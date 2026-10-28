"""fix8 係 tools 直す 8-2 回目(読むだけ): w2-tools-r8-2-off-few-left の「外れた OFF」の数え方を本物の 10/10・10/23・10/24 で見る。
本物の日には発進の前進 ON の本が無い(比べのある組が 0 = 今の ⑤ の出力は直す前とバイト単位で同じ)ので、もしその組が回 3 だったら
(同じ組に ON の本があったら)、外れた OFF が何本出るかを、⑤(今)の JSON の発進の行と同じ分け方で数える:
  本ごと(今の決め)  R1 に届かなかった OFF の発進のうち、その本の発進が比べに 1 本も残らないもの + カメラの止まりの OFF の発進
  発進ごと(比べ)    R1 に届かなかった OFF の発進の全部 + カメラの止まりの OFF の発進

    python3 -I -B experiments/20261026_fix/fix8/fix_tools_r2/probes/real_offdrop.py   → ../out/real_offdrop.txt
"""
import collections
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'out'
L = []
for d in ('20261010', '20261023', '20261024'):
    for tag, first_only in (('ignore-gain', False), ('ignore-gain_first-only', True)):
        j = json.loads((OUT / 'real_new' / ('launches_%s_%s.json' % (d, tag))).read_text(encoding='utf-8'))
        G = collections.defaultdict(lambda: {'cmp': [], 'stall': [], 'nr': [], 'books': set()})
        for r in j['launches']:
            if first_only and (r.get('start_no') or 1) != 1:
                continue
            if r.get('midrun') or r.get('trial') or (r.get('after_abort') and r.get('to_r1_s') is not None):
                continue
            S = r.get('settings') or {}
            on = lambda x: str(x) in ('1', '1.0', 'True')          # noqa: E731
            k = (r['session'], r['model'], on(S.get('speed_ceiling')), S.get('ceil_left_mps') if on(S.get('speed_ceiling')) else None,
                 on(S.get('speed_auto')), S.get('auto_target_mps') if on(S.get('speed_auto')) else None)
            if r.get('launch_state', 'off') != 'off':
                continue
            G[k]['books'].add(r['log'])
            if r.get('to_r1_s') is None:
                G[k]['nr'].append(r)
            elif r.get('launch_stall'):
                G[k]['stall'].append(r)
            else:
                G[k]['cmp'].append(r)
        for k, v in sorted(G.items(), key=str):
            cl = {r['log'] for r in v['cmp']}
            per_book = len(v['stall']) + sum(1 for r in v['nr'] if r['log'] not in cl)
            per_launch = len(v['stall']) + len(v['nr'])
            lost = sorted({r['log'] for r in v['nr'] if r['log'] not in cl})
            L.append('%s %-22s %-28s OFF の本 %2d・比べに残る本 %2d・R1 に届かない OFF の発進 %2d(うち 2 回目以降 %2d)・カメラの止まり %d → '
                     '外れた OFF: 本ごと %2d(今の決め)・発進ごと %2d%s' % (
                         d, tag, '%s %s' % (k[0], k[1][:14]), len(v['books']), len(cl), len(v['nr']),
                         sum(1 for r in v['nr'] if (r.get('start_no') or 1) >= 2), len(v['stall']), per_book, per_launch,
                         '・比べに残らない本 %s' % ','.join(x[-6:] for x in lost) if lost else ''))
(OUT / 'real_offdrop.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L))
