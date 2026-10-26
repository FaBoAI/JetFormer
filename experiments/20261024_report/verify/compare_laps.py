"""verify/compare_laps.py — recount.json(生の記録から)と scorecard.json の周・衝突を 1 本ずつ比べる。"""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(H, 'recount.json')))
SC24 = json.load(open('/home/jetson/jetformer_reports/bench_20261024_tune/since_1300/nightly/scorecard/scorecard.json'))['per_log']
SC23 = json.load(open('/home/jetson/jetformer_reports/bench_20261023_day1/nightly/scorecard/scorecard.json'))['per_log']
verbose = '-v' in sys.argv
rows = []
for lg, d in R.items():
    sc = (SC24 if lg.startswith('20261024') else SC23).get(lg)
    if 'error' in d:
        print(lg, 'ERROR', d['error']); continue
    my = [l for l in d['laps'] if l['ok']]
    scl = sc['laps'] if sc else []
    # 周の始まりで対応を取る(0.15 秒以内)
    pair = []
    for l in my:
        m = [s for s in scl if abs(s['t_start'] - l['t0']) < 0.15]
        pair.append((l, m[0] if m else None))
    unmatched_sc = [s for s in scl if not any(abs(s['t_start'] - l['t0']) < 0.15 for l in my)]
    dif = [l['lap'] - s['lap_s'] for l, s in pair if s]
    cl_mis = [(l['t0'], l['clean'], s['clean']) for l, s in pair if s and l['clean'] != s['clean']]
    my_c = [c for c in d['collisions'] if not c['restart']]
    sc_c = [e for e in (sc['collision_events'] if sc else []) if e.get('class') == 'main']
    sc_r = [e for e in (sc['collision_events'] if sc else []) if e.get('class') != 'main']
    cm = [c for c in my_c if any(abs(c['t'] - e['t_s']) < 0.35 for e in sc_c)]
    print('%s No.%-3s %-4s tg=%.2f sg=%.2f | ai_s %6.2f sc %6.2f | laps my %2d sc %2d matched %2d sc_only %d | dlap max %.3f | clean my %d sc %d mism %s | col my %d(str %d, rst %d) sc main %d(str %d) other %d matched %d' % (
        lg, d['ai_no'], d['model'], d['gains'][1], d['gains'][0], d['ai_s'], sc['ai_s'] if sc else -1,
        len(my), len(scl), sum(1 for _, s in pair if s), len(unmatched_sc), max(map(abs, dif)) if dif else 0,
        sum(l['clean'] for l in my), sum(s['clean'] for s in scl), cl_mis,
        len(my_c), sum(c['strong'] for c in my_c), sum(c['restart'] for c in d['collisions']), len(sc_c), sum(e['strong'] for e in sc_c), len(sc_r), len(cm)))
    if verbose:
        for l, s in pair:
            print('    my %.3f→%.3f lap %.3f ok dpsi %.0f dist %.1f clean %s | sc %s' % (l['t0'], l['t1'], l['lap'], l['dpsi'], l['dist'], l['clean'], (s['lap_s'], s['clean']) if s else None))
        for s in unmatched_sc:
            print('    sc only', s['t_start'], s['lap_s'], s['clean'])
        for l in d['laps']:
            if not l['ok']:
                print('    my not-ok %.3f lap %.3f dpsi %.0f dist %.1f false_end %s' % (l['t0'], l['lap'], l['dpsi'], l['dist'], l['false_end']))
        for c in d['collisions']:
            m = [e for e in sc['collision_events'] if abs(c['t'] - e['t_s']) < 0.35] if sc else []
            print('    col my t=%.3f g=%.2f restart=%s loc=%s psi=%s | sc %s' % (c['t'], c['g'], c['restart'], c.get('loc'), c.get('psi_rel'), [(e['t_s'], e['g'], e['loc_detail'], e['class']) for e in m]))
        for e in (sc['collision_events'] if sc else []):
            if not any(abs(c['t'] - e['t_s']) < 0.35 for c in d['collisions']):
                print('    sc-only col', e['t_s'], e['g'], e['loc_detail'], e['class'])
