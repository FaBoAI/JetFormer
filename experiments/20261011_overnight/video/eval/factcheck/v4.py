import sys, numpy as np
sys.path.insert(0,'/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, KV, seg_mask
tot=0; c03=0; cb03=0; leads=[]
for log in ['20261010-203924','20261010-204208','20261010-204605']:
    l=L(log)
    for lev in (1,2):
        on = l.guard>=lev
        # merge gaps < 0.3 s
        segs=seg_mask(on)
        mer=[]
        for a,b in segs:
            if mer and l.at[a]-l.at[mer[-1][1]]<0.3: mer[-1]=(mer[-1][0],b)
            else: mer.append((a,b))
        if lev==1: W=mer
        else: B=mer
    for e in l.events():
        if not e['ai']: continue
        t=e['t']; tot+=1
        def lead(M):
            best=None
            for a,b in M:
                if l.at[b]>=t-0.3 and l.at[a]<=t:
                    best=t-l.at[a]
            return best
        lw, lb = lead(W), lead(B)
        leads.append(lw)
        c03 += (lw is not None and lw>=0.3); cb03 += (lb is not None and lb>=0.3)
        print(l.name,'%.2f'%t,'warn lead',None if lw is None else round(lw,2),'brake lead',None if lb is None else round(lb,2))
print('total',tot,'warn>=0.3',c03,'brake>=0.3',cb03,'median lead',np.median([x for x in leads if x is not None]))
