#!/usr/bin/env python3
"""10/23 の AI 走行(ai_no 3〜26)を 1 本 1 行にまとめる(読むだけ)。data/labels/20261023-*.json から。
衝突は AI が操作していたフレームのもの(ev['ai'])。100 m あたりの数は AI の前進距離で割る。
    python3 runs_table.py   → runs.json・runs.md"""
import collections, json, re, glob, os
LAB = '/home/jetson/jetformer/data/labels'
HERE = os.path.dirname(os.path.abspath(__file__))
ASSIGN = json.load(open(os.path.join(HERE, '..', 'assign_20261023.json'), encoding='utf-8'))
SHORT = {'clean_driving_20261011-nocmd-s3-20261011-024626': 'B', 'bench_20261022-slow-sec-s1': 'SLOW-SEC1',
         'bench_20261021-ens-p2': 'ENS', 'bench_20261021-wgt-s0': 'WGT',
         'clean_driving_20261011-tcp-s1-nocmd-stride3-e30': 'tcp-s1',
         'clean_driving_20261011-dinov2-s0-nocmd-stride3-e30': 'dinov2-s0'}
rows = []
for f in sorted(glob.glob(LAB + '/20261023-*.json')):
    d = json.load(open(f, encoding='utf-8'))
    if not d.get('ai_log'):
        continue
    desc = ASSIGN.get(d['log'], d.get('description') or '')
    m = re.search(r'ai_no=(\d+)', desc)
    if not m:
        continue
    n = int(m.group(1))
    grp = 'TRIAL' if 'cand=TRIAL' in desc else 'EXP' if 'cand=EXP' in desc else 'OFFICIAL'
    cm = re.search(r'cand=(\S+)', desc)
    p = d.get('params_at_start') or {}
    c = d['counts']
    ev = [e for e in d['events']['collisions'] if e.get('ai')]
    loc = collections.Counter(e.get('loc', 'unknown') for e in ev)
    laps = [l for l in d['events']['laps'] if l.get('ai_only')]
    lt = sorted(l['time_s'] for l in laps)
    prog = c.get('ai_progress_m') or 0.0
    rows.append(dict(ai_no=n, log=d['log'], group=grp, cand=cm.group(1) if cm else '', model=SHORT.get(d.get('run'), d.get('run')),
                     steer=p.get('steer_gain'), thr=p.get('throttle_gain'), tof_guard=p.get('tof_guard'), zone_caps=p.get('zone_caps', 0),
                     ai_s=c.get('ai_s'), ai_m=round(prog, 1), speed_mps=round(prog / c['ai_s'], 2) if c.get('ai_s') else None,
                     ai_laps=len(laps), lap_med=lt[len(lt) // 2] if lt else None, lap_best=lt[0] if lt else None,
                     coll=len(ev), strong=sum(1 for e in ev if e.get('strong')), max_g=max((e['g'] for e in ev), default=0),
                     per100m=round(len(ev) / prog * 100, 2) if prog > 1 else None, loc=dict(loc),
                     fail_takeovers=c.get('failure_takeovers'), guard_warn_ai=c.get('guard_warnings_ai'),
                     esc_assists=c.get('esc_assists'), recoveries=c.get('recoveries')))
rows.sort(key=lambda r: r['ai_no'])
json.dump(rows, open(os.path.join(HERE, 'runs.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
H = '| ai | 区分 | モデル | steer | thr | ToF | 上限 | AI秒 | AI m | m/s | AI周 | 周中央 | 衝突 | 強 | 最大G | 衝突/100m | 場所 | 失敗交代 |'
out = [H, '|' + '---|' * H.count('|')[:-1] if False else '|' + '---|' * (H.count('|') - 1)]
for r in rows:
    out.append('| %d | %s %s | %s | %s | %s | %s | %s | %.0f | %.0f | %s | %d | %s | %d | %d | %.1f | %s | %s | %s |' % (
        r['ai_no'], r['group'], r['cand'], r['model'], r['steer'], r['thr'], r['tof_guard'], r['zone_caps'], r['ai_s'] or 0,
        r['ai_m'], r['speed_mps'], r['ai_laps'], r['lap_med'], r['coll'], r['strong'], r['max_g'], r['per100m'],
        ' '.join('%s%d' % kv for kv in sorted(r['loc'].items(), key=lambda kv: -kv[1])), r['fail_takeovers']))
open(os.path.join(HERE, 'runs.md'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('\n'.join(out))
