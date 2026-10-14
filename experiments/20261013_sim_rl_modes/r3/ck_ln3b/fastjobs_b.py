# 確認 line r3(再実行): 最速とくらべる開始(U360 の最初の 60 + 試験の LN-P60-R3 の 60)を jobs にする
import json, sys
OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3b'
a = [j for j in json.load(open(f'{OUT}/jobs.json'))['jobs'] if j['set'] == 'U360'][:60]
b = [j for j in json.load(open(f'{OUT}/testjobs.json'))['jobs'] if j['set'] == 'T-LN-P60-R3']
jobs = [{**j, 'set': 'FAST-' + j['set'], 'id': 'F-' + j['id']} for j in a + b]
json.dump({'meta': {'what': 'fastest on U360[:60] + LN-P60-R3'}, 'jobs': jobs}, open(f'{OUT}/fastjobs.json', 'w'))
print(len(jobs))
