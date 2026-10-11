import numpy as np, json, warnings
from eff import *
from rruns3 import rrun_table, LOGS, get
from tofobs import obstacle_dist
warnings.simplefilter('ignore')
DT = 0.01

def runs_mask(m, minlen):
    out = []
    for a, b, v in runs(m.astype(int)):
        if v == 1 and b - a >= minlen: out.append((a, b))
    return out

def side(delta, thr=100.0):
    if abs(delta) < thr: return 'C'
    return 'L' if delta > 0 else 'R'

def rel(pre, x, thr=100.0):
    sp, sx = side(pre, thr), side(x, thr)
    if sx == 'C': return 'centred'
    if sp == 'C': return 'pre-centred'
    return 'same' if sp == sx else 'opposite'

def analyse(rows):
    eps = []
    tofc = {}
    for l in LOGS:
        d = get(l)
        if l not in tofc:
            dd = build(l, tof=True); ob, _ = obstacle_dist(dd); tofc[l] = (dd.tt, ob, dd.it, dd.acc, dd.gyr, dd.g)
        tt, ob, it, acc, gyr, g = tofc[l]
        e = d.erpm; t = d.vt; h = d.hall
        free = runs_mask(e < -2000, 50)
        free_st = np.array([a for a, b in free]); free_en = np.array([b for a, b in free])
        eng = [r for r in rows if r['log'] == l and r['engaged'] and r['human'] and r['cat'] in ('after_R_brake', 'after_R_reverse')]
        allr = [r for r in rows if r['log'] == l]
        groups = []
        for r in eng:
            i = np.searchsorted(t, r['t'])
            if groups:
                pi = np.searchsorted(t, groups[-1][-1]['t'])
                if not ((free_st > pi) & (free_st < i)).any():
                    groups[-1].append(r); continue
            groups.append([r])
        for G in groups:
            i1 = np.searchsorted(t, G[0]['t']); iK = np.searchsorted(t, G[-1]['t'])
            k = np.nonzero(free_en <= i1)[0]
            if not len(k): continue
            ife = free_en[k[-1]] - 1                      # last sample of free-forward driving
            j = np.nonzero(e[ife:i1] > -300)[0]; ist = ife + (j[0] if len(j) else 0)
            t_fe, t_stop = t[ife], t[ist]
            # impact
            mi = (it >= t_fe - 0.5) & (it <= t_stop + 0.5)
            if mi.any():
                ii = np.nonzero(mi)[0][np.argmax(g[mi])]
                imp = dict(g=round(float(g[ii]), 1), t_rel=round(float(it[ii] - t_stop), 2), ax=round(float(acc[ii, 0] / 9.81), 1), ay=round(float(acc[ii, 1] / 9.81), 1),
                           yaw_pre=round(float(np.median(gyr[(it >= t_fe - 0.5) & (it < t_fe), 2])), 0))
            else:
                imp = None
            speed_pre = float(np.median(e[max(0, ife - 50):ife + 1]))
            s_pre = s_all = None
            sc = np.zeros(len(d.thr), int); sc[d.thr < NEUT - 20] = -1; sc[d.thr > NEUT + 40] = 1
            braked_before_stop = bool((sc[max(0, ife - 30):ist + 1] == 1).sum() >= 5)
            fwd_cmd_at_stop = float(np.median(d.thr[max(0, ist - 20):ist + 1]))
            st_pre = float(np.median(d.st[max(0, ife - 50):ife + 1]) - CENTER)
            st_stop = float(np.median(d.st[max(0, ist - 50):ist + 1]) - CENTER)
            # next free forward after last engagement
            kn = np.nonzero(free_st > iK)[0]
            ifr = free_st[kn[0]] if len(kn) else None
            iend = ifr if ifr is not None else len(e)
            # AI-run logs: only analyse the human-controlled part (stop at the hand-back to the AI)
            if d.vsel.any():
                hb_idx = np.nonzero(~d.vsel[:-1] & d.vsel[1:])[0] + 1
                hb_idx = hb_idx[hb_idx > i1]
                if len(hb_idx): iend = min(iend, int(hb_idx[0]))
            # reverse samples
            seg = np.arange(i1, iend)
            revm = e[seg] > 300
            rev_idx = seg[revm]
            if not len(rev_idx): continue
            i_rev_last = rev_idx[-1]
            fm = (e[i_rev_last:min(len(e), iend + 10)] < -500).astype(int); cs = np.r_[0, np.cumsum(fm)]; w = cs[10:] - cs[:-10] if len(fm) >= 10 else np.array([]); jf = np.nonzero(w >= 10)[0]
            ifwd = i_rev_last + jf[0] if len(jf) else None
            # commands during episode
            s = np.zeros(len(d.thr), int); s[d.thr < NEUT - 20] = -1; s[d.thr > NEUT + 40] = 1
            Rin = [r for r in allr if t_stop - 0.05 <= r['t'] <= t[i_rev_last]]
            first_R = Rin[0]['t'] if Rin else None
            failed = [r for r in Rin if not r['engaged']]
            fail_reasons = {}
            for r in failed:
                c = {'after_F': 'brake (after F)', 'after_R_rolling_fwd': 'still rolling fwd'}.get(r['cat'], 'other')
                fail_reasons[c] = fail_reasons.get(c, 0) + 1
            # forward push while stuck (between stop and first R)
            push = 0.0
            if first_R is not None:
                ifr0 = np.searchsorted(t, first_R)
                push = float(((s[ist:ifr0] == -1) & (np.abs(e[ist:ifr0]) < 300)).sum() * DT)
            # reverse throttle: R command while reversing (or within engaged runs)
            Rsamp = seg[(s[seg] == 1) & (np.arange(len(seg)) < (i_rev_last - i1 + 1))]
            thr_rev = d.thr[Rsamp] if len(Rsamp) else np.array([np.nan])
            rev_cmd_time = float(len(Rsamp) * DT)
            rev_time = float(revm[: i_rev_last - i1 + 1].sum() * DT)
            rev_counts = float(h[i1:i_rev_last + 1].max() - h[i1])
            peak = float(e[i1:i_rev_last + 1].max())
            st_rev = float(np.median(d.st[rev_idx]) - CENTER)
            st_fwd = float(np.median(d.st[ifwd:ifwd + 50]) - CENTER) if ifwd is not None else np.nan
            # takeover / handback
            sel = d.vsel
            to = None; hb = None
            if sel.any():
                tr_off = np.nonzero(sel[:-1] & ~sel[1:])[0] + 1
                tr_on = np.nonzero(~sel[:-1] & sel[1:])[0] + 1
                x = tr_off[(tr_off <= i1)]
                if len(x) and x[-1] >= ife - 300: to = float(t[x[-1]] - t_stop)
                y = tr_on[tr_on > i1]
                if len(y): hb = float(t[y[0]] - t_stop)
            ob_stop = float(np.nanmin(ob[(tt >= t_stop) & (tt <= t_stop + 0.3)])) if ((tt >= t_stop) & (tt <= t_stop + 0.3)).any() else np.nan
            ob_end = float(np.nanmin(ob[(tt >= t[i_rev_last]) & (tt <= t[i_rev_last] + 0.3)])) if ((tt >= t[i_rev_last]) & (tt <= t[i_rev_last] + 0.3)).any() else np.nan
            rec = dict(log=l, t_stop=round(float(t_stop), 2), t_free_end=round(float(t_fe), 2), braked_before_stop=braked_before_stop, thr_at_stop=round(fwd_cmd_at_stop), i_rev_last=int(i_rev_last), t_first_eng=round(float(G[0]['t']), 2), source='AI-run takeover' if sel.any() and to is not None else ('AI-run human' if sel.any() else 'human-only'),
                       speed_pre=round(speed_pre), st_pre=round(st_pre), st_stop=round(st_stop), impact=imp,
                       takeover_after_stop=None if to is None else round(to, 2), handback_after_stop=None if hb is None else round(hb, 2),
                       fwd_push_while_stuck_s=round(push, 2), first_R_after_stop=None if first_R is None else round(first_R - t_stop, 2),
                       engage_after_stop=round(float(G[0]['t'] + (G[0]['t_on'] or 0) - t_stop), 2), n_engagements=len(G),
                       gaps=[g_['gap'] for g_ in G], brake_pulses=[dict(thr=g_['prev_thr'], dur=None if g_['prev_dur'] is None else round(g_['prev_dur'], 2)) for g_ in G],
                       t_on=[g_['t_on'] for g_ in G], failed=len(failed), fail_reasons=fail_reasons,
                       rev_thr_med=round(float(np.nanmedian(thr_rev))), rev_thr_max=round(float(np.nanmax(thr_rev))), rev_cmd_s=round(rev_cmd_time, 2),
                       rev_motion_s=round(rev_time, 2), rev_counts=round(rev_counts), rev_peak_erpm=round(peak),
                       st_rev=round(st_rev), st_rev_rel=rel(st_stop, st_rev),
                       rev_end_after_stop=round(float(t[i_rev_last] - t_stop), 2),
                       fwd_after_rev=None if ifwd is None else round(float(t[ifwd] - t[i_rev_last]), 2),
                       st_fwd=None if np.isnan(st_fwd) else round(st_fwd), st_fwd_rel=None if np.isnan(st_fwd) else rel(st_stop, st_fwd),
                       st_fwd_vs_rev=None if np.isnan(st_fwd) else rel(st_rev, st_fwd),
                       fwd_after_stop=None if ifwd is None else round(float(t[ifwd] - t_stop), 2),
                       free_after_stop=None if ifr is None else round(float(t[ifr] - t_stop), 2),
                       who_resumed=None if ifwd is None else ('AI' if sel[ifwd] else 'human'),
                       tof_obs_stop=None if np.isnan(ob_stop) or np.isinf(ob_stop) else round(ob_stop, 2),
                       tof_obs_rev_end=None if np.isnan(ob_end) or np.isinf(ob_end) else round(ob_end, 2))
            eps.append(rec)
    return eps

if __name__ == '__main__':
    rows = rrun_table()
    eps = analyse(rows)
    json.dump(eps, open('episodes.json', 'w'), indent=1, ensure_ascii=False)
    for x in eps:
        print(x['log'][-6:], x['t_stop'], x['source'][:8], 'v', x['speed_pre'], 'stpre', x['st_pre'], x['st_stop'], 'imp', x['impact'], 'TO', x['takeover_after_stop'], 'HB', x['handback_after_stop'])
        print('    push %.2f firstR %s eng %s nEng %d gaps %s brakes %s ton %s failed %d %s' % (x['fwd_push_while_stuck_s'], x['first_R_after_stop'], x['engage_after_stop'], x['n_engagements'], x['gaps'], x['brake_pulses'], x['t_on'], x['failed'], x['fail_reasons']))
        print('    rev thr %d/%d cmd %.2fs motion %.2fs counts %d peak %d st_rev %d (%s) revend %.2f fwd_after_rev %s st_fwd %s (%s, vs rev %s) fwd@%s free@%s by %s tof %s->%s' % (
            x['rev_thr_med'], x['rev_thr_max'], x['rev_cmd_s'], x['rev_motion_s'], x['rev_counts'], x['rev_peak_erpm'], x['st_rev'], x['st_rev_rel'], x['rev_end_after_stop'],
            x['fwd_after_rev'], x['st_fwd'], x['st_fwd_rel'], x['st_fwd_vs_rev'], x['fwd_after_stop'], x['free_after_stop'], x['who_resumed'], x['tof_obs_stop'], x['tof_obs_rev_end']))
