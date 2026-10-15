#!/usr/bin/env python3
"""突き合わせ(sim_diverse_train_20261015 の Q1、2026-10-15): eval_sim.json(評価の道具)と review/recount_q1.json(生の JSON からの別の数え直し)・
review/replay_user_tool(ユーザーの replay-report.mjs での再生)が同じかを見る。出力: review/recount_check.json
"""
import json
from pathlib import Path

R = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
ev = json.loads((R / 'eval_sim.json').read_text())
rc = json.loads((R / 'review' / 'recount_q1.json').read_text())
out = {'arms': {}, 'verdicts': {}, 'replay_user_tool': {}}
ok = True
for arm, s in ev['summary'].items():
    if arm not in rc['arms']:
        continue
    x = rc['arms'][arm]
    row = {'eval': [s['n_completed'], s['fs'], s['g4'], s['n_ran']], 'recount': [x['c'], x['fs'], x['g4'], x['n']],
           'd_eval_m': s['d_median_m'], 'd_recount_polyline_scaled_m': x['d_polyline_scaled_median_m']}
    row['same_counts'] = row['eval'] == row['recount']
    ok &= row['same_counts']
    out['arms'][arm] = row
for k in ('Q1_S-DIV_vs_yesterday', 'Q1_variety_vs_volume'):
    a = (ev['verdicts'].get(k) or {}).get('verdict')
    b = (rc['verdicts'].get(k) or {}).get('verdict')
    if (a is None or str(a).startswith('今日は答えない')) and b is None:
        continue   # 走らせていない腕の判定(どちらの道具も出さない)
    out['verdicts'][k] = {'eval': a, 'recount': b, 'same': a == b}
    ok &= a == b
a = (ev['verdicts'].get('Q1b_DAgger') or {}).get('verdict')
b = (rc['verdicts'].get('Q1b_DAgger') or {}).get('verdict')
if a or b:
    out['verdicts']['Q1b_DAgger'] = {'eval': a, 'recount': b, 'same': a == b}
    ok &= a == b
a = (ev['verdicts'].get('Q1_DART_margin_addon(S-DIV vs S-MOD)') or {}).get('verdict')
b = (rc['verdicts'].get('Q1_DART_margin_addon') or {}).get('verdict')
if a or b:
    out['verdicts']['Q1_DART_margin_addon'] = {'eval': a, 'recount': b, 'same': a == b}
    ok &= a == b
rt = R / 'review' / 'replay_user_tool'
passed, failed = [], []
for p in sorted(rt.glob('*.json')):
    try:
        d = json.loads(p.read_text())
        (passed if d.get('replay') == 'PASS' else failed).append(p.stem)
    except ValueError:
        failed.append(p.stem)
cl = R / 'eval' / 'closed_loop'
runs = sorted({p.stem for p in cl.glob('*/*.json') if p.parent.name not in ('user_verifier',) and not p.name.startswith('plan_')
               and p.parent.name != 'stopped'})
out['replay_user_tool'] = {'runs': len(runs), 'pass': len(passed), 'fail': failed, 'missing': [n for n in runs if n not in passed and n not in failed]}
ok &= not failed and not out['replay_user_tool']['missing']
out['all_same'] = bool(ok)
(R / 'review' / 'recount_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(json.dumps({'all_same': out['all_same'], 'arms': {k: v['same_counts'] for k, v in out['arms'].items()}, 'verdicts': out['verdicts'],
                  'replay': [out['replay_user_tool']['pass'], out['replay_user_tool']['runs'], out['replay_user_tool']['fail']]}, ensure_ascii=False))
