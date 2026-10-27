"""係 tools の確かめ(2026-10-26 fix2): tools/standing_3lap.py の 10/23・10/24 の値を、analysis_tt.md の数
(bench_20261024_report/tt_data.json)と diag_launch/out/launches.json と比べる。読むだけ。

    cd /home/jetson/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 800 -- \
        python3 -I experiments/20261026_fix/fix2/tools_check/check_tt.py
    → 同じフォルダの check_tt.json・check_tt.txt・standing_20261023.md・standing_20261024.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/tools')
import standing_3lap as s3  # noqa: E402

HERE = Path(__file__).resolve().parent
TT = Path('/home/jetson/jetformer_reports/bench_20261024_report/tt_data.json')
LAUNCH = Path('/home/jetson/jetformer/experiments/20261026_fix/diag_launch/out/launches.json')


def main():
    tt = json.loads(TT.read_text(encoding='utf-8'))
    dl = json.loads(LAUNCH.read_text(encoding='utf-8'))['launches']
    res = {}
    lines = []
    for day in ('20261023', '20261024'):
        logs, skipped = s3.day_logs(day)
        R = s3.run(logs)
        (HERE / ('standing_%s.md' % day)).write_text(s3.to_markdown(R) + ''.join(
            '- 飛ばした記録: %s(%s)\n' % x for x in skipped), encoding='utf-8')
        for r in R:
            res[r['log']] = r
        lines.append('%s: AI の記録 %d 本・飛ばした %s' % (day, len(logs), skipped))
    # 1. 走りながらの 3 周の窓(analysis_tt の窓は 10/23 公式 8 本 + 10/24 No.1〜20 の記録だけ)
    by_log = {}
    for w in tt['windows']:
        by_log.setdefault(w['log'], []).append(w)
    rows_w, bad_w = [], 0
    for lg, ws in sorted(by_log.items()):
        r = res.get(lg) or {}
        mine = sorted(round(w['total_s'], 3) for w in r.get('windows') or [])
        theirs = sorted(round(w['total_s'], 3) for w in ws)
        kinds_m = sorted((round(w['total_s'], 3), w['kind']) for w in r.get('windows') or [])
        kinds_t = sorted((round(w['total_s'], 3), w['kind']) for w in ws)
        same = len(mine) == len(theirs) and all(abs(a - b) <= 0.002 for a, b in zip(mine, theirs))
        same_kind = [k for _, k in kinds_m] == [k for _, k in kinds_t]
        bad_w += (not same) or (not same_kind)
        F = r.get('flying') or {}
        bt = min(theirs)
        bct = min([w['total_s'] for w in ws if w['kind'] == 'きれい'], default=None)
        rows_w.append({'log': lg, 'no': ws[0]['no'], 'n_tt': len(theirs), 'n_mine': len(mine), 'best_tt': bt,
                       'best_mine': F.get('best_s'), 'best_clean_tt': bct, 'best_clean_mine': F.get('best_clean_s'),
                       'same_totals': same, 'same_kinds': same_kind})
    extra = sorted(lg for lg, r in res.items() if (r.get('flying') or {}).get('n') and lg not in by_log)
    # 2. 止まってから(analysis_tt の starts。発進ごと = 同じ t_ai)
    rows_s, bad_s = [], 0
    for s in tt['starts']:
        r = res.get(s['log']) or {}
        m = next((x for x in r.get('standing') or [] if abs(x['t_ai'] - s['t_ai']) <= 0.002), None)
        row = {'log': s['log'], 'no': s['no'], 't_ai': s['t_ai'], 'found': m is not None,
               'to_r1_tt': s.get('to_r1_s'), 'to_r1_mine': (m or {}).get('to_r1_s'),
               'd_tt': s.get('dist_to_r1_m'), 'd_mine': (m or {}).get('d_from_ai_m'), 'd_pre_mine': (m or {}).get('d_pre_m'),
               'move_tt': s.get('move_after_s'), 'move_0p4_mine': (m or {}).get('move_0p4_after_s'),
               'st3_tt': s.get('standing_3laps_s'), 'dist3_mine': (m or {}).get('dist3_from_ai_s'),
               'from_ai_mine': (m or {}).get('from_ai_s'), 'from_move_mine': (m or {}).get('from_move_s'),
               'valid_mine': (m or {}).get('valid'), 'why_mine': (m or {}).get('why'),
               'coll_tt': len(s.get('standing_3laps_coll') or []) if s.get('standing_3laps_s') else None,
               'coll_mine': ((m or {}).get('collisions') or {}).get('n')}
        ok = m is not None and (s.get('move_after_s') is None or (row['move_0p4_mine'] is not None and
                                                                  abs(row['move_0p4_mine'] - s['move_after_s']) <= 0.002))
        if ok and s.get('to_r1_s') is not None:
            ok = row['to_r1_mine'] is not None and abs(row['to_r1_mine'] - s['to_r1_s']) <= 0.002 \
                and abs(row['d_mine'] - s['dist_to_r1_m']) <= 0.011
        if ok and s.get('standing_3laps_s') is not None:
            ok = row['dist3_mine'] is not None and abs(row['dist3_mine'] - s['standing_3laps_s']) <= 0.005
        if ok and s.get('to_r1_s') is not None and s.get('standing_3laps_s') is None:
            ok = row['dist3_mine'] is None
        row['ok'] = bool(ok)
        bad_s += not ok
        rows_s.append(row)
    # 3. diag_launch の発進(t_ai・動き出し 0.2 m/s・起点まで)
    rows_d, bad_d = [], 0
    for x in dl:
        r = res.get(x['log']) or {}
        m = next((y for y in r.get('standing') or [] if abs(y['t_ai'] - x['t_ai']) <= 0.002), None)
        ok = m is not None and (x.get('move_0p2_s') is None or (m.get('move_after_s') is not None and
                                                                  abs(m['move_after_s'] - x['move_0p2_s']) <= 0.002))
        if ok and x.get('to_r1_s') is not None:
            ok = m.get('to_r1_s') is not None and abs(m['to_r1_s'] - x['to_r1_s']) <= 0.002
        bad_d += not ok
        rows_d.append({'log': x['log'], 't_ai': x['t_ai'], 'ok': bool(ok), 'move_dl': x.get('move_0p2_s'),
                       'move_mine': (m or {}).get('move_after_s'), 'to_r1_dl': x.get('to_r1_s'),
                       'to_r1_mine': (m or {}).get('to_r1_s')})
    out = {'windows': rows_w, 'windows_bad': bad_w, 'windows_extra_logs': extra, 'starts': rows_s, 'starts_bad': bad_s,
           'diag_launch': rows_d, 'diag_launch_bad': bad_d,
           'n_launch_mine': sum(len(r.get('standing') or []) for r in res.values()),
           'zero_byte': {lg: r['zero_byte'] for lg, r in res.items() if r.get('zero_byte')},
           'errors': {lg: r['error'] for lg, r in res.items() if r.get('error')}}
    (HERE / 'check_tt.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    lines.append('窓: 記録 %d 本・合わない %d・analysis_tt に無い窓のある記録 %s' % (len(rows_w), bad_w, extra))
    for w in rows_w:
        lines.append('  %s No.%s 窓 %d/%d 最速 %s/%s きれい %s/%s 同じ %s/%s' % (
            w['log'], w['no'], w['n_mine'], w['n_tt'], w['best_mine'], w['best_tt'], w['best_clean_mine'],
            w['best_clean_tt'], w['same_totals'], w['same_kinds']))
    lines.append('発進(analysis_tt): %d・合わない %d' % (len(rows_s), bad_s))
    for s in rows_s:
        if s['to_r1_tt'] is None and s['ok']:
            continue
        lines.append('  %s No.%s t_ai %.3f 動く(0.4) %s/%s 起点まで %s/%s d %s/%s 距離の3周 %s/%s 線(AI側)%s 線(動いて)%s ok=%s %s' % (
            s['log'], s['no'], s['t_ai'], s['move_0p4_mine'], s['move_tt'], s['to_r1_mine'], s['to_r1_tt'], s['d_mine'],
            s['d_tt'], s['dist3_mine'],
            s['st3_tt'], s['from_ai_mine'], s['from_move_mine'], s['ok'], s['why_mine'] or ''))
    lines.append('発進(diag_launch): %d・合わない %d' % (len(rows_d), bad_d))
    lines.append('0 バイト: %s' % out['zero_byte'])
    lines.append('エラー: %s' % out['errors'])
    (HERE / 'check_tt.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
