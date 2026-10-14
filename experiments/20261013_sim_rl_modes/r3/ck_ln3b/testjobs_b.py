# 確認 line r3(再実行): 学習の段の試験の組(eval_test_r3.json の 186 回)の開始・形・c_v を、自前の走らせ方の jobs にする(照らし合わせ用)
import json, sys
d = json.load(open('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r3/eval_test_r3.json'))
jobs = []
for k, runs in d['tests'].items():
    for r in runs:
        st = r['start']
        jobs.append({'id': f"T-{r['id']}", 'set': f"T-{k}", 'shield': True, 'gate': st['gate'], 'off': st['left_offset_m'], 'yaw': st['left_yaw_deg'],
                     'seed': st['seed'], 'params': {'line': r['line'], 'c_v': r['c_v']}, 'dpsi': st.get('dpsi'), 'c_start': st.get('initial_clearance_m')})
json.dump({'meta': {'source': 'line_r3/eval_test_r3.json'}, 'jobs': jobs}, open(sys.argv[1], 'w'))
print(len(jobs))
