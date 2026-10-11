import sys, numpy as np
sys.path.insert(0,'/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, KV
l=L('20261010-203924')
for a,b in [(17.99,19.6),(8.36,10.3),(13.89,15.66),(22.26,24.39)]:
    m=(l.at>=a)&(l.at<=b)&l.sel
    s=l.out[m,0]
    print('seg %.2f-%.2f steer min %d max %d  frac<=1001 %.2f >=1999 %.2f'%(a,b,s.min(),s.max(),np.mean(s<=1001),np.mean(s>=1999)))
m=(l.at>=17.95)&(l.at<=19.4)
for t,o,mo,g,tt in zip(l.at[m],l.out[m],l.mout[m],l.guard[m],l.ttc[m]): print(' %.3f out %s g%d ttc %.2f'%(t,o.round().tolist(),g,tt))
