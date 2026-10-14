#!/usr/bin/env python3
# 確認 line r3 の追加の調べ(角の近く NB・NBC・BOX・GRID、ゲート 11 の G11)の集計と、分布の中の割合の見積もり → check_line_r3/extra_summary.json
import json, glob, collections, math
OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3'
def load(pat):
    return [json.loads(l) for f in sorted(glob.glob(f'{OUT}/runs/{pat}')) for l in open(f) if l.strip()]
def fail(r):
    return r['collisions'] > 0 or r['stall'] or (r['valid_r3'] and r['min_clearance'] < .03) or not r['valid_r3']
def stats(rows):
    return {'n': len(rows), 'valid_r3': sum(r['valid_r3'] for r in rows), 'collisions': sum(r['collisions'] > 0 for r in rows), 'stalls': sum(bool(r['stall']) for r in rows),
            'below_3cm_valid': sum(r['valid_r3'] and r['min_clearance'] < .03 for r in rows), 'below_4cm': sum(r['min_clearance'] < .04 for r in rows),
            'min_clearance': min(r['min_clearance'] for r in rows), 'shield_runs': sum(r['shield_frames'] > 0 for r in rows), 'any_fail': sum(fail(r) for r in rows)}
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]
S = {}
for v in ('cand', 'nosh'):
    rows = load(f'nb_{v}_*.jsonl') + load(f'box_{v}_*.jsonl')
    by = collections.defaultdict(list)
    for r in rows: by[r['set']].append(r)
    S[v] = {k: stats(x) for k, x in sorted(by.items())}
grid = load('grid_*.jsonl')
S['grid'] = {f"g{g}_{fam}": stats([r for r in grid if r['start']['gate'] == g and r['line']['family'] == fam]) for g in (16, 17) for fam in ('const', 'random')}
S['grid_fail_cells'] = [{'id': r['id'], 'gate': r['start']['gate'], 'off': r['start']['left_offset_m'], 'yaw': r['start']['left_yaw_deg'], 'line': r['line'], 'kind': 'collision' if r['collisions'] else 'stall' if r['stall'] else '<3cm',
                         'min': round(r['min_clearance'], 4)} for r in grid if fail(r)]
g11 = load('g11_*.jsonl')
S['g11'] = {'line': stats([r for r in g11 if r['policy'] == 'line']), 'fastest': stats([r for r in g11 if r['policy'] == 'fastest']),
            'line_required_crossing_missing': sum(r['reason'] == 'required_crossing_missing' for r in g11 if r['policy'] == 'line'),
            'by_family': {k: sum(r['reason'] == 'required_crossing_missing' for r in g11 if r['policy'] == 'line' and r['line']['family'] + str(r['line'].get('c0', r['line'].get('a', ''))) == k)
                          for k in sorted({r['line']['family'] + str(r['line'].get('c0', r['line'].get('a', ''))) for r in g11 if r['policy'] == 'line'})}}
# 分布の中の割合の見積もり: BOX は各ゲートの(左右, 向き)の 1%(0.04/0.4 × 3/30)。ゲートは 1/33。開始のすき間の下限はこの箱では効かない(0.49〜0.66 m)
share = json.load(open(f'{OUT}/share.json'))
lh = share['share']['ge10']
b16, b17 = S['cand']['BOX16'], S['cand']['BOX17']
p16, p17 = b16['any_fail'] / b16['n'], b17['any_fail'] / b17['n']
w16, w17 = wilson(b16['any_fail'], b16['n']), wilson(b17['any_fail'], b17['n'])
per_uniform = (.01 / 33) * (p16 + p17)
per_uniform_ci = [(.01 / 33) * (w16[0] + w17[0]), (.01 / 33) * (w16[1] + w17[1])]
per_lh = per_uniform / lh
S['estimate'] = {'box_area_per_gate': .01, 'box16_fail_rate': p16, 'box16_ci95': w16, 'box17_fail_rate': p17, 'box17_ci95': w17,
                 'per_uniform_start': per_uniform, 'per_uniform_start_ci95': per_uniform_ci, 'large_heading_share': lh, 'per_large_heading_start': per_lh,
                 'p_at_least_one_in_60_uniform': 1 - (1 - per_uniform) ** 60, 'p_at_least_one_in_HEAD60': 1 - (1 - per_lh) ** 60,
                 'p_at_least_one_in_HEAD60_ci95': [1 - (1 - per_uniform_ci[0] / lh) ** 60, 1 - (1 - per_uniform_ci[1] / lh) ** 60],
                 'note': '失敗 = 止まり・衝突・3 cm 未満の有効な回・有効でない回。箱の外(NB16・NB17 の残り・CORNER・DENSE・HEAD120)では失敗 0 なので、箱の中だけで見積もった'}
json.dump(S, open(f'{OUT}/extra_summary.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False)[:5000])
