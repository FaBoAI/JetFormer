import sys, numpy as np, json
sys.path.insert(0,'/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, seg_mask
for log in ['20261010-203924','20261010-204208','20261010-204605']:
    l=L(log)
    print('=====',log,l.name,'scenes',l.scenes)
    print('dur ai', l.at[-1], 'vt last', l.vt[-1] if len(l.vt) else None, 'it last', l.it[-1], 'ct last', l.ct[-1])
    print('assist counts', {int(k):int((l.assist==k).sum()) for k in np.unique(l.assist)})
    print('assist on sel', {int(k):int(((l.assist==k)&l.sel).sum()) for k in np.unique(l.assist)})
    print('guard_applied any', l.gapplied.any(), 'guard counts', {int(k):int((l.guard==k).sum()) for k in np.unique(l.guard)})
    dt=np.clip(np.diff(l.at),0,0.1)
    print('ai_time', (dt*l.sel[:-1]).sum())
    for a,b in seg_mask(l.sel):
        g=l.gains[a:b+1]; la=l.la[a:b+1]
        print(' seg %.2f-%.2f dur %.2f gains %s la %s hold %d'%(l.at[a],l.at[b],l.at[b]-l.at[a], np.unique(g,axis=0).tolist(), np.unique(la).tolist(), l.hold[a:b+1].sum()))
    # gain changes over all frames
    ch=np.nonzero(np.any(np.diff(l.gains,axis=0)!=0,axis=1)|(np.diff(l.la)!=0))[0]
    for i in ch: print('  change at %.2f: %s/%s -> %s/%s sel=%s'%(l.at[i+1],l.gains[i].tolist(),l.la[i],l.gains[i+1].tolist(),l.la[i+1],l.sel[i+1]))
    ev=l.events()
    for e in ev:
        print('  imp t=%.2f g=%.1f ai=%s v=%.2f ax=%.1f ay=%.1f sus=%s'%(e['t'],e['g'],e['ai'],e['v_before'],e['ax'],e['ay'],e['suspect']))
