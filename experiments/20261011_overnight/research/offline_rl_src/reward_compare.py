"""速度の積分(eRPM 由来)と、コース図の進み(progress_cum_m)で、1秒先までの報酬和がどれだけ違うか。
RECAP 型の「良い」印(上位35%)がどれだけ一致するかを見る。AI が運転しているフレームだけ。読み取りのみ。"""
import os, json
import numpy as np
D = '/home/jetson/jetformer_reports/overnight_20261011/course'
LOGS = {'20261010-172910': 'ai_run_006', '20261010-173445': 'ai_run_007', '20261010-203924': 'ai_run_009',
        '20261010-204208': 'ai_run_010', '20261010-204605': 'ai_run_101'}
H = int(os.environ.get("H_FRAMES", "30"))  # 既定 1秒(30fps)。3秒は H_FRAMES=90
def rank(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r
out = {}
allS, allP, allSec = [], [], []
for log, name in LOGS.items():
    z = np.load(os.path.join(D, 'tracks_%s.npz' % log))
    t, v, cum, valid, ai, sec = z['t'], z['speed'], z['progress_cum_m'], z['valid'], z['ai'], z['section']
    dt = np.r_[np.diff(t), 0]
    n = len(t); S, P, K = [], [], []
    for i in range(n - H):
        j = slice(i, i + H)
        if not (valid[j].all() and ai[j].all() and np.isfinite(cum[i]) and np.isfinite(cum[i + H])): continue
        if dt[j].max() > 0.2: continue
        S.append(float(np.sum(np.abs(v[j]) * dt[j]))); P.append(float(cum[i + H] - cum[i])); K.append(int(sec[i]))
    S, P = np.array(S), np.array(P)
    if len(S) < 100: continue
    rho = float(np.corrcoef(rank(S), rank(P))[0, 1])
    gs, gp = S > np.quantile(S, 0.65), P > np.quantile(P, 0.65)
    agree = float((gs == gp).mean()); jac = float((gs & gp).sum() / max(1, (gs | gp).sum()))
    out[name] = {'n': int(len(S)), 'spearman': round(rho, 3), 'label_agree': round(agree, 3), 'good_jaccard': round(jac, 3),
                 'mean_diff_m': round(float(np.mean(P - S)), 3), 'abs_diff_p90_m': round(float(np.quantile(np.abs(P - S), 0.9)), 3)}
    allS.append(S); allP.append(P); allSec.append(np.array(K))
    print(name, out[name])
S, P, K = np.concatenate(allS), np.concatenate(allP), np.concatenate(allSec)
names = list(np.load(os.path.join(D, 'tracks_20261010-173445.npz'))['section_names'])
gs, gp = S > np.quantile(S, 0.65), P > np.quantile(P, 0.65)
out['_all'] = {'n': int(len(S)), 'spearman': round(float(np.corrcoef(rank(S), rank(P))[0, 1]), 3), 'label_agree': round(float((gs == gp).mean()), 3),
               'good_jaccard': round(float((gs & gp).sum() / (gs | gp).sum()), 3)}
# 区間ごとに「良い」の割合がどう変わるか
bysec = {}
for k, nm in enumerate(names):
    m = K == k
    if m.sum() > 50:
        bysec[str(nm)] = {'n': int(m.sum()), 'good_frac_speed': round(float(gs[m].mean()), 3), 'good_frac_progress': round(float(gp[m].mean()), 3)}
out['_good_frac_by_section'] = bysec
print('ALL', out['_all']); print(bysec)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reward_compare_H%d.json' % H), 'w'), ensure_ascii=False, indent=1)
