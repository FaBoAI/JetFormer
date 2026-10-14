#!/usr/bin/env python3
# 確認 r2: 試験の組の「作り直し」を新しい開始・新しい物理で何組も作り、判定 N1〜N3・N5〜N7 が通る割合を数える。
#   N60' = 試験の F-REC 30(決まった開始、REPLAY2 の値)+ FN3 の新しい開始 30 個ずつ(20 組)
#   R200' = FD3 の 200 個ずつ(3 組)。NW-STD・NW-G・NW-FC10 は決まった条件なので REPLAY2 の値をそのまま使う
#   python3 replicates.py <runs> <out.json>
import json, sys, os
RUNS, OUT = sys.argv[1], sys.argv[2]
rows = lambda f: [json.loads(l) for l in open(os.path.join(RUNS, f)) if l.strip()]
rp = rows('rows_REPLAY2.jsonl')
fnfd = rows('rows_FN3_FD3.jsonl')
fn = [r for r in fnfd if r['set'] == 'FN3']
fd = [r for r in fnfd if r['set'] == 'FD3']
ok = lambda r: r['valid_episode'] and r['valid_laps'] >= 1
frec = [r for r in rp if r['test'] == 'NW-N60' and not r['id'].startswith('H')]
std = [r for r in rp if r['test'] == 'NW-STD']
G = [r for r in rp if r['test'] == 'NW-G']
FC = [r for r in rp if r['test'] == 'NW-FC10']
res = {'N60_groups': [], 'R200_groups': []}
for k in range(len(fn) // 30):
    grp = fn[30 * k: 30 * k + 30]
    n60 = frec + grp
    nom = n60 + std
    fr = sum(r['labels']['frames'] for r in n60)
    rn = sum(r['rho']['n'] for r in n60)
    res['N60_groups'].append({'k': k, 'collisions': sum(1 for r in n60 if r['collisions']), 'N2_min_cm': round(100 * min(r['min_clearance'] for r in nom), 3),
                              'N2_pass': min(r['min_clearance'] for r in nom) >= .08, 'N5_n60': sum(1 for r in n60 if ok(r)),
                              'N6': [sum(r['rho']['req_gt1'] for r in n60) / rn, sum(r['rho']['comb_gt1'] for r in n60) / sum(r['rho']['comb_n'] for r in n60)],
                              'N7': sum(r['labels']['sh'] for r in n60) / fr})
for k in range(len(fd) // 200):
    grp = fd[200 * k: 200 * k + 200]
    rnd = G + grp + FC
    res['R200_groups'].append({'k': k, 'collisions': sum(1 for r in grp if r['collisions']), 'N3_min_cm': round(100 * min(r['min_clearance'] for r in rnd), 3),
                               'N3_pass': min(r['min_clearance'] for r in rnd) >= .05, 'N5_r200': sum(1 for r in grp if ok(r))})
g = res['N60_groups']
res['summary'] = {'N60_groups': len(g), 'N2_pass_groups': sum(1 for x in g if x['N2_pass']), 'N1_n60_collision_groups': sum(1 for x in g if x['collisions']),
                  'N5_n60_fail_groups': sum(1 for x in g if x['N5_n60'] < 60), 'N6_fail_groups': sum(1 for x in g if max(x['N6']) > .02), 'N7_fail_groups': sum(1 for x in g if x['N7'] > .05),
                  'R200_groups': len(res['R200_groups']), 'N3_pass_groups': sum(1 for x in res['R200_groups'] if x['N3_pass']),
                  'N1_r200_collision_groups': sum(1 for x in res['R200_groups'] if x['collisions']), 'N5_r200_fail_groups': sum(1 for x in res['R200_groups'] if x['N5_r200'] < 190),
                  'p_below_8cm_FN3': sum(1 for r in fn if r['min_clearance'] < .08) / len(fn),
                  'p_N2_pass_30_binomial': (1 - sum(1 for r in fn if r['min_clearance'] < .08) / len(fn)) ** 30,
                  'p_frozen_n60_min_ge_11p75_for_30_random': (1 - sum(1 for r in fn if r['min_clearance'] < .11747) / len(fn)) ** 30}
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res['summary'], indent=1))
print([x['N2_min_cm'] for x in g])
print([x['N3_min_cm'] for x in res['R200_groups']])
