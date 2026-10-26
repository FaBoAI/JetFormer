"""sim_ceiling.json の要約(表の数)。読むだけ。"""
import json, os, collections
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(H, 'sim_ceiling.json')))
rows = []
for c in r['ceiling']:
    E = c['episodes']
    laps = [x for e in E for x in e['laps']]
    cm = [e['ceil'] for e in E if e['ceil']]
    def mx(key):
        v = [m[key] for m in cm if m.get(key) is not None]
        return max(v) if v else None
    def md(key):
        v = [m[key] for m in cm if m.get(key) is not None]
        return float(np.median(v)) if v else None
    coll = collections.Counter((e['collision'] or {}).get('sec') for e in E if e['collision'])
    row = {'k': c['k'], 'kb': c['kBatt'], 'mode': c['mode'], 'noise': c['noise'], 'coll': sum(e['collision'] is not None for e in E),
           'coll_sec': dict(coll), 'lap_mean': float(np.mean(laps)) if laps else None, 'vmax': max(e['vmax'] for e in E),
           'vmax_left': max((e['vmax_left'] or 0) for e in E), 'changed': md('changed'), 'neutral': md('neutral'),
           'steady_over': mx('steady_over'), 'entry_05': mx('entry_05_max'), 'entries': sum(m['entries'] for m in cm) if cm else 0,
           'rev': md('reversals_per_s'), 'df_med': md('df_med'), 'df_p90': md('df_p90'), 'lim_s': sum(m['limit_runs_s'] for m in cm) if cm else 0}
    rows.append(row)
    print(row)
auto = []
for c in r['auto']:
    for e in c['episodes']:
        L = e['laps']
        auto.append({'k': c['k'], 'kb': c['kBatt'], 'auto': c['auto'], 'start': e['start'], 'coll': e['collision'] is not None,
                     'm_fb': [round(l['m_fb'], 3) for l in L], 'v90': [round(l['v90'], 2) for l in L], 'lap': [round(l['lap_s'], 2) for l in L],
                     'cf': [round(l['ceil_frac'], 2) for l in L], 'ev': [l['event'] for l in L], 'warn': [l['warn'] for l in L if l['warn']]})
for a in auto:
    if a['start'] in (0, 1):
        print(a)
json.dump({'ceiling': rows, 'auto': auto}, open(os.path.join(H, 'sim_summary.json'), 'w'), ensure_ascii=False, indent=1)
