from eff import *
import numpy as np, json, sys
LOGS = ['20261010-135449','20261010-144028','20261010-150424','20261010-152405','20261010-152707','20261010-163247','20261010-172910','20261010-173445']
HUMAN_ONLY = LOGS[:5]
_cache = {}
def get(l):
    if l not in _cache: _cache[l] = build(l, tof=False)
    return _cache[l]
DB_F, DB_R = 20.0, 40.0
def cmd_states(d, db_f=DB_F, db_r=DB_R):
    s = np.zeros(len(d.thr), int); s[d.thr < NEUT - db_f] = -1; s[d.thr > NEUT + db_r] = 1
    return s

def rrun_table(db_f=DB_F, db_r=DB_R, minrun=0.05, hall_min=15, stop_erpm=150):
    rows = []
    for l in LOGS:
        d = get(l)
        e = d.erpm; t = d.vt; h = d.hall
        s = cmd_states(d, db_f, db_r)
        rr = runs(s)
        prev_eng_flags = {}
        for k, (a, b, v) in enumerate(rr):
            if v != 1: continue
            dur = t[b-1] - t[a] + 0.01
            if dur < minrun: continue
            bb = min(len(e), b + 10)
            dh = float((h[a:bb] - h[a]).max())
            e0 = float(np.median(e[max(0, a-3):a+1]))
            ea = e[min(a+3, len(e)-1):bb]
            peak = float(ea.max()) if len(ea) else e0
            accel_back = peak >= max(e0, 0) + 800
            engaged = dh >= hall_min and (e0 <= stop_erpm and e0 >= -stop_erpm and peak >= 800 or accel_back)
            on = np.nonzero((h[a:bb] - h[a]) >= 5)[0]
            t_on = float(t[a+on[0]] - t[a]) if len(on) else None
            gap = 0.0; prev = 'start'; prev_thr = None; prev_dur = None; prev_engaged = None; prev_t = None
            j = k - 1
            if j >= 0 and rr[j][2] == 0:
                gap = t[rr[j][1]-1] - t[rr[j][0]] + 0.01
                j -= 1
            while j >= 0 and rr[j][2] == 1 and (t[rr[j][1]-1] - t[rr[j][0]] + 0.01) < minrun:
                j -= 1   # skip micro R blips
            if j >= 0:
                pa, pb, pv = rr[j]
                prev = 'F' if pv == -1 else 'R'
                prev_dur = float(t[pb-1] - t[pa] + 0.01); prev_thr = float(np.median(d.thr[pa:pb])); prev_t = float(t[pa])
                if pv == 1:
                    prev_engaged = prev_eng_flags.get(pa, False)
            if prev == 'F' or prev == 'start': cat = 'after_F'
            elif e0 < -stop_erpm: cat = 'after_R_rolling_fwd'
            elif prev_engaged: cat = 'after_R_reverse'
            else: cat = 'after_R_brake'
            prev_eng_flags[a] = bool(engaged)
            rows.append(dict(log=l, t=round(float(t[a]), 2), i=int(a), j=int(b), dur=round(float(dur), 2), thr=round(float(np.median(d.thr[a:b]))), thr_max=round(float(d.thr[a:b].max())),
                             human=bool(d.human[a:b].mean() > 0.5), e0=round(e0), peak=round(peak), engaged=bool(engaged), dh=dh,
                             t_on=t_on, prev=prev, gap=round(float(gap), 3), prev_dur=prev_dur, prev_thr=prev_thr, prev_engaged=prev_engaged, prev_t=prev_t, cat=cat,
                             rolling_back=bool(e0 > stop_erpm)))
    return rows

if __name__ == '__main__':
    rows = rrun_table()
    json.dump(rows, open('rruns3.json', 'w'))
    tab = {}
    for r in rows:
        key = (r['cat'], 'H' if r['human'] else 'A')
        tab.setdefault(key, [0, 0]); tab[key][0] += r['engaged']; tab[key][1] += 1
    for k in sorted(tab): print(k, '%d/%d engaged' % tuple(tab[k]))
    print('--- after_R_brake (human) details')
    for r in rows:
        if r['cat'] == 'after_R_brake':
            print('%s %7.2f %s eng=%d e0=%5d pk=%5d dh=%4.0f ton=%s thr=%d dur=%.2f gap=%.3f prev_thr=%s prev_dur=%s' % (r['log'][-6:], r['t'], 'H' if r['human'] else 'A', r['engaged'], r['e0'], r['peak'], r['dh'], None if r['t_on'] is None else round(r['t_on'], 2), r['thr'], r['dur'], r['gap'], None if r['prev_thr'] is None else round(r['prev_thr']), None if r['prev_dur'] is None else round(r['prev_dur'], 2)))
    print('--- after_F engaged / after_R_rolling_fwd engaged')
    for r in rows:
        if r['cat'] in ('after_F', 'after_R_rolling_fwd') and r['engaged']:
            print(r)
