from destat import *
from match import match, section
from psi import course_psi
from regime import gains_tl
import json
def classify(log, hum=False):
    if hum:
        d = D(log); d.heading(bias=-0.0915); rows = analyze_h(log)[1]; tk = []
    else:
        d, rows, _, tk, segs = analyze(log)
    an = [a for a, dep, hit in true_r1(d, r1_anchors(d), rows)]
    psi, _ = course_psi(d, an)
    DE = {r['a']: r['tg'] for r in de_table(log, hum)}
    if not hum: gt, gg, gl = gains_tl(d)
    out = []
    for r in rows:
        t = r['t']
        prev = [a for a in an if a <= t + 0.05]
        a = prev[-1] if prev else None
        tg = DE.get(a) if a is not None else None
        sec = None; how = 'rule'
        if a is not None and t - a < 1.0: sec = 'R1'
        elif tg is not None and t < tg - 0.3 and t - a < 3.5: sec = 'GM_in'
        elif tg is not None and abs(t - tg) <= 0.3: sec = 'GM_out'
        elif tg is not None and t <= tg + 2.0: sec = 'DE'
        if sec is None:
            ph, cost, c = match(d, psi, t); how = 'match(cost %.0f)' % cost
            s = section(ph)
            sec = {'R1': 'R1', 'GM': 'GM_in', 'DE': 'DE', 'SW': 'SW', 'HS': 'HS', 'LH': 'LH'}[s]
            if cost > 20: sec = sec + '?'
        e = dict(r); e.update(log=log, sec=sec, how=how, a=a, tg=tg, rel=(t - a) if a is not None else None, relg=(t - tg) if tg is not None else None)
        if not hum:
            i = np.searchsorted(gt, t); i = min(i, len(gt) - 1)
            e.update(sg=float(gg[i, 0]), tg_gain=float(gg[i, 1]), la=int(gl[i]))
        out.append(e)
    return d, out
if __name__ == '__main__':
    allc = {}
    for log, hum in [('20261010-172910', False), ('20261010-173445', False)] + [(h, True) for h in HUM]:
        d, out = classify(log, hum); allc[log] = out
        for e in out:
            print('%s %7.2f %-7s g%4.1f lat%+5.1f lon%+5.1f er %5d->%5d st %+4d th %+4d %s %s gains %s' % (log[-6:], e['t'], e['sec'], e['g'], e['ax'], e['ay'], e['erpm_b'], e['erpm_a'], e['steer_b'], e['thr_b'], 'AI' if e['sel'] else 'HU', e['how'], '%.2f/%.2f la%d' % (e['sg'], e['tg_gain'], e['la']) if 'sg' in e else ''))
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        raise TypeError(type(o))
    json.dump(allc, open('classified.json', 'w'), indent=1, default=conv)
