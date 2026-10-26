#!/usr/bin/env python3
"""final_eval.md の表の数を desk_v2.json・eval_v2.json・decisions.json・final_eval_data.json と比べる(読むだけ)。
md の表示の桁(2・3・4 桁)で丸めた値が同じかを見る。結果を --out の JSON に書く。"""
import argparse, json, re
R = '/home/jetson/jetformer_reports/train_v2_20261025/'
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
md = open(R + 'final_eval.md').read()
desk = json.load(open(R + 'desk/desk_v2.json'))['runs']
ev = json.load(open(R + 'eval/eval_v2.json'))
evr = {r['run']: r for r in ev['runs']}
five = ev['five']
fed = json.load(open(R + 'final_eval_data.json'))
dec = json.load(open(R + 'decisions.json'))
res = []

def num(s):
    s = s.replace('**', '').strip()
    m = re.match(r'^([+\-−]?\d+(?:\.\d+)?)', s.replace('−', '-'))
    return (float(m.group(1)), len(m.group(1).split('.')[1]) if '.' in m.group(1) else 0) if m else (None, None)

def same(md_s, val, label):
    v, nd = num(md_s)
    if v is None and val is None:
        ok = True
    elif v is None or val is None:
        ok = ('未' in md_s and val is None)
    else:
        ok = abs(round(float(val), nd) - v) <= 0.5 * 10 ** (-nd) + 1e-9
    res.append({'what': label, 'md': md_s.strip(), 'json': val, 'ok': bool(ok)})
    return ok

lines = [l for l in md.splitlines() if l.startswith('|')]
# 22 本の表
for l in lines:
    c = [x.strip() for x in l.strip('|').split('|')]
    if len(c) == 13 and c[0] in ('B-FAST', '1', '2', '3', '4'):
        name = c[1]
        run = ('bfast_20261025-' + name[len('bfast-'):]) if name.startswith('bfast-') else 'v2_20261025-' + name
        d, e = desk[run], evr[run]
        same(c[2], e['l1_best_us'], run + ' L1 best')
        same(c[3], e['l1_last_us'], run + ' L1 last')
        same(c[4], e['l1_interv_us'], run + ' 介入 best')
        same(c[5], d['robust_dl1_us'], run + ' robust')
        res.append({'what': run + ' sanity', 'md': c[6], 'json': d['sanity_ok'], 'ok': (c[6] == '○') == (d['sanity_ok'] is True)})
        same(c[7], d['trt_p99_ms'], run + ' p99')
        gate_inf = d['gates'].get('inference')
        res.append({'what': run + ' 推論の○×', 'md': c[7], 'json': gate_inf, 'ok': ('○' in c[7]) == (gate_inf is True)})
        same(c[8], d['brake_lowspeed_frac'], run + ' brake')
        res.append({'what': run + ' brake の○×', 'md': c[8], 'json': d['gates'].get('brake'), 'ok': ('○' in c[8]) == (d['gates'].get('brake') is True)})
        same(c[9], d['standstill_go_frac'], run + ' standstill')
        same(c[10], d['straight_thr_vs_b_us'], run + ' straight')
        if d['family'] == 'v2':
            res.append({'what': run + ' straight の○×', 'md': c[10], 'json': d['gates'].get('straight'), 'ok': ('○' in c[10]) == (d['gates'].get('straight') is True)})
        res.append({'what': run + ' 門', 'md': c[12], 'json': d['gates_pass'], 'ok': ('不合格' in c[12]) == (d['gates_pass'] is False)})
    if len(c) == 9 and c[0] in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B'):
        f = five[c[0]]; d = desk[f['run']]
        same(c[1], f['l1_best_us'], c[0] + ' L1 best'); same(c[2], f['l1_last_us'], c[0] + ' L1 last')
        same(c[3], f['l1_interv_us'], c[0] + ' 介入'); same(c[4], d['robust_dl1_us'], c[0] + ' robust')
        same(c[5], d['trt_p99_ms'], c[0] + ' p99'); same(c[6], d['brake_lowspeed_frac'], c[0] + ' brake')
        same(c[7], d['standstill_go_frac'], c[0] + ' stand'); same(c[8], d['straight_thr_vs_b_us'], c[0] + ' straight')
    if len(c) == 10 and c[0].startswith('v2h-aug2-e60-s') and '+' in c[0]:
        k = c[0].split('+')[0].strip()[-1]
        name = 'ens:v2_20261025-v2h-aug2-e60-s%s+v2_20261025-v2h-msf-aug2-e60-s%s' % (k, k)
        d = desk[name]; dv = dec['D4']['values']
        same(c[1], d['l1_best_us'], name + ' L1'); same(c[2], dv['light_l1_best_us'][int(k)], name + ' 軽量型')
        same(c[3], dv['d_best'][int(k)], name + ' d'); same(c[4], d['robust_dl1_us'], name + ' robust')
        same(c[6], d['brake_lowspeed_frac'], name + ' brake'); same(c[7], d['standstill_go_frac'], name + ' stand')
        same(c[8], d['straight_thr_vs_b_us'], name + ' straight')
        res.append({'what': name + ' 門', 'md': c[9], 'json': d['gates_pass_except_inference'], 'ok': ('不合格' in c[9]) == (d['gates_pass_except_inference'] is False)})
# final_eval_data.json の runs と desk・eval
for run, r in fed['runs'].items():
    d, e = desk[run], evr[run]
    for k in ('l1_best_us', 'l1_last_us', 'l1_interv_us'):
        res.append({'what': 'final_eval_data ' + run + ' ' + k, 'md': r[k], 'json': e[k], 'ok': abs(r[k] - e[k]) < 1e-6})
    for k in ('robust_dl1_us', 'trt_p99_ms', 'brake_lowspeed_frac', 'standstill_go_frac', 'straight_thr_vs_b_us'):
        res.append({'what': 'final_eval_data ' + run + ' ' + k, 'md': r[k], 'json': d[k], 'ok': r[k] == d[k]})
    res.append({'what': 'final_eval_data ' + run + ' gates_pass', 'md': r['gates_pass'], 'json': d['gates_pass'], 'ok': r['gates_pass'] == d['gates_pass']})
bad = [x for x in res if not x['ok']]
json.dump({'n': len(res), 'bad': bad, 'all': res}, open(a.out, 'w'), ensure_ascii=False, indent=1)
print('compared', len(res), 'bad', len(bad))
for b in bad: print(b)
rows22 = sum(1 for x in res if x['what'].endswith(' 門') and not x['what'].startswith('ens'))
print('table rows (門):', rows22)
