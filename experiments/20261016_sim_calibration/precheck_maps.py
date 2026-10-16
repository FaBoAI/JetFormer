# 下調べ(設計の段, 2026-10-16): AI と人で「PWM → 定常の速さ」「PWM → 曲率」が同じか。読み取りだけ・CPU。出力は標準出力の JSON(precheck_maps.json に保存)
import sys, json, numpy as np
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261016_sim_calibration')
import inventory as inv
logs = ['20261010-152405','20261010-152707','20261010-193130','20261010-200916','20261010-201626','20261010-202633',
        '20261010-171639','20261010-172910','20261010-173445','20261010-181504','20261010-203924','20261010-204208','20261010-155820','20261010-160306']
rows = {'human': [], 'ai': []}; curv = {'human': [], 'ai': []}
for log in logs:
    res, r = inv.analyse_log(log)
    if r is None: continue
    dt = r['dt']; n = int(round(0.5 / dt))
    thr = r['thr']; st = r['steer']
    sd = np.array([np.nanstd(thr[max(0, j - n):j + 1]) for j in range(len(thr))])
    ssd = np.array([np.nanstd(st[max(0, j - n):j + 1]) for j in range(len(st))])
    m = r['clean'] & (thr < 1498 - 20) & (sd < 8) & (np.abs(r['off']) < 100) & (np.abs(r['acc']) < 0.4) & (r['vs'] > 0.3)
    for o, name in ((0, 'human'), (1, 'ai')):
        mm = m & (r['op'] == o)
        rows[name].append(np.stack([thr[mm], r['vs'][mm]], 1))
        mc = r['clean'] & (ssd < 8) & (r['vs'] > 0.8) & (r['vs'] < 2.2) & (r['op'] == o)
        curv[name].append(np.stack([st[mc] - (1443 if o else 1442), r['gz'][mc] / r['vs'][mc], r['vs'][mc]], 1))
out = {}
for name in rows:
    R = np.concatenate(rows[name]); C = np.concatenate(curv[name])
    tab = []
    for lo in range(1000, 1480, 40):
        k = (R[:, 0] >= lo) & (R[:, 0] < lo + 40)
        if k.sum() >= 20:
            tab.append([lo, int(k.sum()), round(float(np.median(R[k, 1])), 2)])
    ctab = []
    for lo, hi in [(-560, -430), (-430, -300), (-300, -200), (-200, -100), (-100, -30), (30, 100), (100, 200), (200, 300), (300, 430), (430, 540), (540, 600)]:
        k = (C[:, 0] >= lo) & (C[:, 0] < hi)
        if k.sum() >= 20:
            ctab.append([lo, hi, int(k.sum()), round(float(np.median(C[k, 0])), 0), round(float(np.median(C[k, 1])), 3), round(float(np.median(C[k, 2])), 2)])
    out[name] = {'speed_vs_thr': tab, 'curv_vs_steer_off': ctab}
print(json.dumps(out, indent=0))
