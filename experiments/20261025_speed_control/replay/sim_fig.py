"""シミュレーターの 1 本の図(天井が掛かる区間、測りの雑音 0 / 0.05 / 0.12)。読むだけ。"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_ceiling as S
import rr_figs as F

br = S.Bridge()
res = {}
for nz in (0.0, 0.05, 0.12):
    st, laps, R = S.episode(br, 2, 1.2, 5.5, 1.3, True, noise=nz, T=30)
    res[nz] = R
br.close()
R0 = res[0.05]
lim = np.nonzero(R0['mode'] >= 1)[0]
t0 = R0['t'][lim[0]] - 1.0 if len(lim) else 5.0
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
cols = {0.0: F.S1, 0.05: F.S2, 0.12: F.S3}
for nz, R in res.items():
    m = (R['t'] >= t0) & (R['t'] <= t0 + 6)
    a1.plot(R['t'][m] - t0, R['v'][m], color=cols[nz], lw=1.8, label='測りの雑音 σ %.2f m/s' % nz)
    a2.plot(R['t'][m] - t0, R['f_out'][m], color=cols[nz], lw=1.3, label='σ %.2f' % nz)
m = (R0['t'] >= t0) & (R0['t'] <= t0 + 6)
a1.plot(R0['t'][m] - t0, R0['cap'][m], color=F.MUTED, ls='--', lw=1.2, label='天井 C')
a2.plot(R0['t'][m] - t0, R0['f_in'][m], color=F.MUTED, ls=':', lw=1.2, label='運転手の指令')
a1.set_ylabel('速さ [m/s]')
a1.legend(frameon=False, loc='lower left', ncol=4, fontsize=9)
a1.set_title('較正したシミュレーター(P_cal)・運転手 PP k 1.2・電池 1.3: 天井の閉ループ', color=F.INK, fontsize=11, loc='left')
a2.set_ylabel('出した前進の量 [µs]')
a2.set_xlabel('時間 [秒]')
a2.legend(frameon=False, loc='lower left', ncol=4, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(F.FIG, 'vr5_sim_noise.png'), dpi=150)
print('ok', t0)
