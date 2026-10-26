# 粗い見積もり: 速さの天井(直線 V1・舵が左 S 以上のとき V2)に当たるフレームの割合(No.9 と 2 回目の出っ張りの衝突の前)
# 動かし方: /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I cap_shadow.py
import json, glob, numpy as np
D='/home/jetson/jetformer/data/labels/'
TT=json.load(open('/home/jetson/jetformer_reports/bench_20261024_report/tt_data.json'))
runs={r['log']:r for r in TT['runs']}
def load(log):
    j=json.load(open(D+log+'.json')); z=np.load(D+log+'.npz', allow_pickle=True)
    r=j['ref_us']; return j,z,(r['steer_center'],r['throttle_neutral'])
# No.9 laps stats
log='20261024-162033'; j,z,ref=load(log)
t=z['t_s']; ai=z['ai']; v=z['speed_mps']; st=z['out'][:,0]-ref[0]; thr=ref[1]-z['out'][:,1]
names=j['loc_detail_names']; ld=z['loc_detail']
L=[l for l in TT['laps'] if l['log']==log and not l.get('false_end')]
de=[];sw=[];x=[];y=[]
for l in L:
    m=(t>=l['t_start'])&(t<l['t_end'])&ai
    mde=m&(ld==names.index('DE'))
    de.append(np.median(thr[mde])); sw.append(l['groups_s']['DE+SW'])
    x.append(l['speed_mps']['p90']); y.append(l['lap_s'])
print('No.9 DE thr vs DE+SW time r=%.2f'%np.corrcoef(de,sw)[0,1])
print(' fast(DE thr>=180):', np.round(sorted([s for d,s in zip(de,sw) if d>=180]),2), ' slow:', np.round(sorted([s for d,s in zip(de,sw) if d<180]),2))
c=np.polyfit(x,y,1); print('No.9 lap = %.2f %+.2f*p90  r=%.2f'%(c[1],c[0],np.corrcoef(x,y)[0,1]))
g=np.array([[l['groups_s'][k] for k in ('R1+GM','DE+SW','HS+LH')] for l in L])
print('No.9 medians', np.median(g,0), 'sum', np.median(g,0).sum(), ' q25', np.percentile(g,25,0))
# shadow of ceilings on AI frames, per run
def shadow(log, V1=4.5, V2=4.0, S=100):
    j,z,ref=load(log); ai=z['ai']; v=z['speed_mps']; st=z['out'][:,0]-ref[0]
    m=ai&(v>0.5)
    c1=m&(v>V1); c2=m&(st>=S)&(v>V2)
    return m.sum(), c1.sum()/max(m.sum(),1), c2.sum()/max(m.sum(),1)
for lg in sorted(runs):
    if not lg.startswith('20261024'): continue
    n,a,b=shadow(lg)
    print(lg, runs[lg].get('no'), runs[lg].get('setting'), 'AIフレーム',n, '直線4.5超 %.1f%%'%(100*a), '左舵100µs以上で4.0超 %.1f%%'%(100*b))
