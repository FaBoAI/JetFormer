# 確認 line r4: 昇格の取り違えの試しの集計(同じ 180 の開始を、候補の shield.mjs 130ed5b2 と、ほかの版に替えた写しで)
import json, collections
from ana4 import load, summary
O = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4'
base = {**load('fresh_*.jsonl'), **load('adv_*.jsonl')}
out = {}
for v, sha in [('nw4v5', '8ece9be4'), ('rc4v4', '29c9d111'), ('shared', 'd682a6b5')]:
    R = load(f'alt_{v}.jsonl')
    rs = list(R.values())
    s = summary(rs)
    pairs = [base[r['pair']] for r in rs] if rs and 'pair' in rs[0] else [base[r['id'].split('-', 2)[2]] for r in rs]
    out[v] = {'shield_mjs': sha, 'rt_shield_seen': sorted({r.get('rt_shield') for r in rs}), 'meta_version': sorted({(r.get('meta_shield') or {}).get('version') for r in rs}),
              **{k: s[k] for k in ['n', 'valid_r3', 'collisions', 'stalls', 'below3_valid', 'below4_all', 'min_all', 'escape_runs', 'reverse_runs', 'shield_runs', 'shield_gt10', 'fallbacks', 'not_valid_reasons']},
              'same_starts_candidate': {k: summary(pairs)[k] for k in ['n', 'valid_r3', 'collisions', 'stalls', 'below3_valid', 'min_all']}}
json.dump(out, open(f'{O}/alt4.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False))
