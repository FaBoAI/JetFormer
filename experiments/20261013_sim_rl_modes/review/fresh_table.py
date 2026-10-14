#!/usr/bin/env python3
"""審査の段(2026-10-14): 凍結した試験の組と新しい種 A・B の合否の表(Markdown)を JSON から作る。数字はすべて JSON から読む。

    python3 -I review/fresh_table.py > 表.md
"""
import json
import os

R = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/'
FROZEN = {'fastest': 'eval/fastest/eval_test_build.json', 'no_wall': 'eval/no_wall/no_wall_test_r2.json',
          'recover': 'train/recover/r2/eval_test_r2.json', 'speed': 'eval/speed/r2/test_r2_step_002015232.json',
          'line': 'train/line/r2/eval_test_r2.json', 'obstacle': 'train/obstacle/r2/final/selected_selection_original.json'}
NAMES = {'fastest': '最速', 'no_wall': '絶対壁にぶつからない', 'recover': '離脱復活', 'speed': '速さいろいろ', 'line': '経路いろいろ', 'obstacle': '障害物よけ'}


def cm(v):
    return '—' if v is None else '%.2f cm' % (100 * v)


def fmt(mode, d):
    j = d['judged']
    P = lambda k: '' if j[k].get('pass') else '✗'   # noqa: E731
    if mode == 'fastest':
        return 'F2 %.6f 秒・F3 %s%s・REC60 の最小 %s' % (j['F2_lap']['value'], j['F3_rec60']['value'], P('F3_rec60'), cm(d['reported']['rec60_min_clearance']))
    if mode == 'no_wall':
        ok, n = (int(x) for x in j['N1_no_collision']['value'].split('/'))
        return 'N1 衝突 %d/%d%s・N2 %s%s・N3 %s%s・N4 %.3f 秒・N5 %s・N7 %.2f%%' % (
            n - ok, n, P('N1_no_collision'), cm(j['N2_clearance_nominal']['value']), P('N2_clearance_nominal'),
            cm(j['N3_clearance_randomised']['value']), P('N3_clearance_randomised'), j['N4_lap_from_stop']['value'],
            '%s・%s' % (j['N5_completion']['value']['n60'], j['N5_completion']['value']['r200']), 100 * j['N7_shield']['value'])
    if mode == 'recover':
        c3 = j['C3_recovery_time']['value']
        c4 = j['C4_events_mask']['value']
        c5 = j['C5_kick']['value']
        c6 = j['C6_unperturbed']['value']
        return 'C1 %s%s・C2 %.3f%s・C3 %.2f/%.2f 秒・C4 %.2f 回%s・外す %.1f%%・C5 %s・%s%s・C6 %.3f 秒・REC60 %s%s' % (
            j['C1_valid']['value'], P('C1_valid'), j['C2_success']['value'], P('C2_success'), c3['median'], c3['p90'],
            c4['events_per_episode'], P('C4_events_mask'), 100 * c4['masked_fraction'], c5['yaw'], c5['lateral'], P('C5_kick'),
            c6['lap_from_stop'], cm(c6['rec60_min_clearance']), P('C6_unperturbed'))
    if mode == 'speed':
        s3 = j['S3_lap_order']['value']
        return 'S1 %s・S2 %s%s・S3 %.2f/%.2f/%.2f 秒・S4 %s・S5 %.3f m/s%s' % (
            j['S1_valid']['value'], cm(j['S2_clearance']['value']), P('S2_clearance'), s3['t1'], s3['t75'], s3['t5'],
            '合格' if j['S4_speed_order']['pass'] else '✗', j['S5_vm4']['value'], P('S5_vm4'))
    if mode == 'line':
        l3 = j['L3_tracking']['value']
        return 'L1 %s%s・L2 %s%s・L3 %.1f/%.1f cm・L4 %s・L5 %s・L6 %.2f 秒' % (
            j['L1_valid']['value'], P('L1_valid'), cm(j['L2_clearance']['value']), P('L2_clearance'), 100 * l3['median'], 100 * l3['p95'],
            '合格' if j['L4_distinct']['pass'] else '✗', cm(j['L5_vm3']['value']), j['L6_lap']['value'])
    if mode == 'obstacle':
        return 'O1 %s%s・O2 %s%s・O3 %.2f 秒・O4 %.3f 秒' % (j['O1_valid']['value'], P('O1_valid'), cm(j['O2_clearance']['value']), P('O2_clearance'),
                                                          j['O3_lap1_median']['value'], j['O4_empty']['value'])


def verdict(d):
    js = [v for v in d['judged'].values() if v.get('pass') is not None]
    return '合格' if js and all(v['pass'] for v in js) else '不合格'


def main():
    rows = ['| モード | 組 | 回 | 判定 | 数字(✗ = その項目が不合格) |', '|---|---|---|---|---|']
    for m in NAMES:
        fz = json.load(open(R + FROZEN[m]))
        label = '凍結した試験の組' if m != 'obstacle' else '選びの組 OB-V100-R2(試験の組は未使用)'
        rows.append('| %s | %s | %d | %s | %s |' % (NAMES[m], label, fz['runs'], verdict(fz), fmt(m, fz)))
        for tag in ('A', 'B'):
            p = R + 'review/fresh/%s/%s.json' % (tag, m)
            if not os.path.exists(p):
                rows.append('| | 新しい種 %s | — | (未) | |' % tag)
                continue
            d = json.load(open(p))
            rows.append('| | 新しい種 %s | %d | %s | %s |' % (tag, d['runs'], verdict(d), fmt(m, d)))
    print('\n'.join(rows))


main()
