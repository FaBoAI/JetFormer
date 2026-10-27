"""反証 safety1-2(読むだけ): リカバリー中止(assist = 3)のあと、人が CH3 を人側にしてから次に AI 側にしたとき、
車はまだ止まっていたか(人が動かしたか)・前の TTC。10/10〜10/24 の全部。0 バイト・壊れた npz は飛ばす"""
import glob, json, os, sys
import numpy as np
root = '/home/jetson/jetformer/data/raw'


def load(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    try:
        z = np.load(p)
        return {k: z[k] for k in z.files}
    except Exception:
        return None


res = []
for lg in sorted(set(os.path.dirname(os.path.dirname(p)) for p in glob.glob(root + '/20261*/*/AI_CMD.npz'))):
    A, V = [], []
    for sc in sorted(glob.glob(lg + '/*/AI_CMD.npz')):
        a, v = load(sc), load(os.path.dirname(sc) + '/VEHICLE.npz')
        if a is None or v is None or 'assist' not in a:
            continue
        A.append(a); V.append(v)
    if not A:
        continue
    try:
        keys = ['timestamps_us', 'assist', 'select', 'dry_run'] + (['ttc'] if all('ttc' in x for x in A) else [])
        a = {k: np.concatenate([x[k] for x in A]) for k in keys}
        vt = np.concatenate([x['timestamps_us'] for x in V]) / 1e6; ve = np.concatenate([x['erpm'] for x in V]).astype(float)
    except Exception:
        continue
    o = np.argsort(vt); vt, ve = vt[o], ve[o]
    t = a['timestamps_us'] / 1e6
    ai = a['select'].astype(bool) & ~a['dry_run'].astype(bool)
    g = np.where(a['assist'] == 3)[0]
    if not len(g):
        continue
    # 中止の区間ごと
    starts = [int(g[0])] + [int(g[k]) for k in range(1, len(g)) if g[k] != g[k - 1] + 1]
    for s in starts:
        h = np.where((np.arange(len(t)) > s) & ~ai)[0]
        if not len(h):
            res.append({'log': os.path.basename(lg), 'giveup_row': s, 'human_after_s': None}); continue
        h0 = int(h[0])
        nxt = np.where((np.arange(len(t)) > h0) & ai)[0]
        r = {'log': os.path.basename(lg), 'giveup_row': s, 'human_after_s': round(float(t[h0] - t[s]), 2)}
        if len(nxt):
            n0 = int(nxt[0])
            w = (vt >= t[h0]) & (vt <= t[n0])
            r.update({'next_ai_after_human_s': round(float(t[n0] - t[h0]), 2),
                      'max_abs_erpm_while_human': float(np.abs(ve[w]).max()) if w.any() else None,
                      'ttc_at_next_ai': None if 'ttc' not in a else round(float(a['ttc'][n0]), 2),
                      'assist_first_10_after': [int(x) for x in a['assist'][n0:n0 + 10]]})
        res.append(r)
json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
