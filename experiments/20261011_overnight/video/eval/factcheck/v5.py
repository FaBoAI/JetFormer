import sys, numpy as np
sys.path.insert(0,'/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, KV, seg_mask
FA=0; EXP=0
for log in ['20261010-203924','20261010-204208','20261010-204605']:
    l=L(log)
    v=-np.interp(l.at,l.vt,l.erpm)*KV
    g2=[e['t'] for e in l.events(2.0)]
    fwd=l.sel&(v>0.5)
    on=(l.guard>=1)&fwd
    segs=seg_mask(on); mer=[]
    for a,b in segs:
        if mer and l.at[a]-l.at[mer[-1][1]]<=0.3: mer[-1]=(mer[-1][0],b)
        else: mer.append((a,b))
    fa=[l.at[a] for a,b in mer if not any(0<=t-l.at[a]<=1.5 for t in g2)]
    dt=np.clip(np.diff(l.at,append=l.at[-1]),0,0.1)
    ex=fwd.copy()
    for t in g2: ex &= ~((l.at>=t)&(l.at<t+1.0))
    e=float((dt*ex).sum())/60
    FA+=len(fa); EXP+=e
    print(l.name,'warn onsets',len(mer),'FA',len(fa),'exp min %.3f'%e, [round(x,2) for x in fa])
    if log.endswith('204208'):
        m=(l.at>27.8)&(l.at<28.3)
        for t,g,tt in zip(l.at[m],l.guard[m],l.ttc[m]): print('   %.4f g%d ttc %.3f'%(t,g,tt))
print('FA',FA,'exp',EXP,'per min',FA/EXP)
