import sys, numpy as np
sys.path.insert(0,'/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, KV
from course import heading
for log in ['20261010-203924','20261010-204208','20261010-204605','20261010-173445','20261010-172910']:
    l=L(log); heading(l)
    sel_i=np.clip(np.searchsorted(l.at,l.it,side='right')-1,0,len(l.at)-1)
    mi=l.sel[sel_i]&(l.it<=l.at[-1])
    di=np.clip(np.diff(l.it,append=l.it[-1]),0,0.2)
    hl=float(np.sum(((l.gyr[:,2]-l.bias)*di)[mi]))/360
    sv=np.clip(np.searchsorted(l.at,l.vt,side='right')-1,0,len(l.at)-1)
    mv=l.sel[sv]&(l.vt<=l.at[-1])
    dv=np.clip(np.diff(l.vt,append=l.vt[-1]),0,0.2)
    dist=float(np.sum((-l.erpm*KV*dv)[mv&(l.erpm<0)]))
    dt=np.clip(np.diff(l.at),0,0.1); T=float((dt*l.sel[:-1]).sum())
    ev=[e for e in l.events() if e['ai']]
    print(l.name,'AI %.1fs dist %.1fm laps %.2f impacts %d per lap %.2f'%(T,dist,hl,len(ev),len(ev)/hl))
