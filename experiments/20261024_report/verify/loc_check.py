"""verify/loc_check.py — カメラで直した場所(解析 C の cam_check.json)と、ジャイロの向きだけで分けた場所(recount.json)を突き合わせる。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(H, 'recount.json')))
cam = json.load(open('/home/jetson/jetformer/experiments/20261024_report/col/cam_check.json'))['events']
SC = json.load(open('/home/jetson/jetformer_reports/bench_20261024_tune/since_1300/nightly/scorecard/scorecard.json'))['per_log']
rows = []
for key, cls in sorted(cam.items()):
    lg, t = key.split('@'); t = float(t)
    d = R[lg]
    m = [c for c in d['collisions'] if abs(c['t'] - t) < 0.35]
    sc = [e for e in SC[lg]['collision_events'] if abs(e['t_s'] - t) < 0.01]
    print('%s No.%-2s t=%7.3f cam=%-4s sc=%-3s g=%5.2f | gyro loc=%-5s psi_rel=%s dt_anchor=%s' % (
        lg, d['ai_no'], t, cls, sc[0]['loc_detail'] if sc else '?', sc[0]['g'] if sc else 0,
        m[0].get('loc') if m else '-', m[0].get('psi_rel') if m else '-', m[0].get('dt_anchor') if m else '-'))
# 3 回目の DE: scorecard の DE と、ジャイロの DE の重なり
n_sc_de = n_both = 0
for lg, d in R.items():
    if not lg.startswith('20261024') or 'error' in d or int(d['ai_no']) < 11:
        continue
    for e in SC[lg]['collision_events']:
        if e['class'] != 'main':
            continue
        m = [c for c in d['collisions'] if abs(c['t'] - e['t_s']) < 0.35]
        if e['loc_detail'] == 'DE':
            n_sc_de += 1
            n_both += bool(m and m[0].get('loc') == 'DE')
print('3回目: scorecard の DE', n_sc_de, 'うちジャイロでも DE', n_both)
