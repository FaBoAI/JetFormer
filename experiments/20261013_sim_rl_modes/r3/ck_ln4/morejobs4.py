# 確認 line r4: 追いの組 — 最速(bundle/fastest、シールドなし)を同じ開始で 120 回、新しい組 1,717 回をシールドなしで(方策そのものの弱さ。報告だけ)
import json
O = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4'
T = json.load(open(f'{O}/testjobs4.json'))['jobs']
F = json.load(open(f'{O}/jobs4.json'))['jobs']
fast = []
for j in [j for j in T if j['set'] == 'T-LN-P60-R4'] + [j for j in F if j['set'] == 'U300'][:60]:
    fast.append({**{k: v for k, v in j.items() if k != 'ref'}, 'id': 'FAST-' + j['id'], 'set': 'FAST-' + j['set'], 'policy': 'fastest', 'shield': False, 'pair': j['id']})
json.dump({'jobs': fast}, open(f'{O}/fastjobs4.json', 'w'))
nosh = [{**j, 'id': 'NOSH-' + j['id'], 'set': 'NOSH-' + j['set'], 'shield': False, 'pair': j['id']} for j in F]
json.dump({'jobs': nosh}, open(f'{O}/noshjobs4.json', 'w'))
print(len(fast), len(nosh))
