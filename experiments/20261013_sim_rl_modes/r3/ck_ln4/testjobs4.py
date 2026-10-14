# 確認 line r4: 学習の段の試験の組(eval_test_r4b.json の 542 回)の開始・形・c_v・種を、自前の走らせ方の job にする
import json
d = json.load(open('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r4b/eval_test_r4b.json'))
jobs = []
for name, runs in d['tests'].items():
    for r in runs:
        st = r['start']
        jobs.append({'id': f"T-{name}-{r['id']}", 'set': 'T-' + name, 'shield': True, 'gate': st['gate'], 'off': st['left_offset_m'], 'yaw': st['left_yaw_deg'],
                     'seed': st['seed'], 'line': r['line'], 'c_v': r['c_v'], 'ref': {'valid_r3': r['valid_r3'], 'min': r['min_clearance'], 'laps': r['laps'],
                     'sh': r['shield_frames'], 'stall': r['stall'], 'reason': r['reason']}})
json.dump({'jobs': jobs}, open('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4/testjobs4.json', 'w'))
print(len(jobs))
