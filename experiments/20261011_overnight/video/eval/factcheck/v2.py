import sys, numpy as np
sys.path.insert(0,'/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, KV
l=L('20261010-204208')
m=(l.vt>72.3)&(l.vt<73.6)
print('vehicle samples:')
for t,e,b,s,p in zip(l.vt[m],l.erpm[m],l.board[m],l.vseq[m],l.pwm[m]): print(' %.3f erpm %6d board %d seq %d pwm %s'%(t,e,b,s,p.tolist()))
m=(l.at>72.6)&(l.at<73.2)
for t,s,h,o,mo in zip(l.at[m],l.sel[m],l.hold[m],l.out[m],l.mout[m]): print(' ai %.3f sel %d hold %d out %s mout %s'%(t,s,h,o.round(0).tolist(),mo.round(0).tolist()))
# board resets
bt=l.board
idx=np.nonzero((bt[1:]+1_000_000<bt[:-1])&(bt[:-1]<4_000_000_000))[0]
print('board resets', [round(float(l.vt[i+1]),3) for i in idx])
m=(l.vt>64.3)&(l.vt<65.0)
for t,e,b,s,vs in zip(l.vt[m][::3],l.erpm[m][::3],l.board[m][::3],l.vseq[m][::3],l.vsel[m][::3]): print(' %.3f erpm %6d board %d seq %d vsel %d'%(t,e,b,s,vs))
m=(l.at>64.4)&(l.at<64.9)
for t,s,h in zip(l.at[m],l.sel[m],l.hold[m]): print(' ai %.3f sel %d hold %d'%(t,s,h))
