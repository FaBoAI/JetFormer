# 周ごと・区間ごとの速さ・前進の指令・舵(10/24 の記録 1 本)。python3 -I sw_laps.py 20261024-162033
import json, sys, numpy as np
D='/home/jetson/jetformer/data/labels/'
TT=json.load(open('/home/jetson/jetformer_reports/bench_20261024_report/tt_data.json'))
log=sys.argv[1]
j=json.load(open(D+log+'.json')); z=np.load(D+log+'.npz', allow_pickle=True)
names=j['loc_detail_names']; print(names, j.get('loc_names'))
t=z['t_s']; ld=z['loc_detail']; sp=z['speed_mps']; mo=z['model_out']; out=z['out']; ai=z['ai']
r=j['ref_us']; ref=(r['steer_center'], r['throttle_neutral'])
acc=[]
for l in TT['laps']:
    if l['log']!=log or l.get('false_end'): continue
    m=(t>=l['t_start'])&(t<l['t_end'])&ai
    row=[l['lap_s'], l['kind'][:2]]
    for nm in ('GM_in','GM_out','DE','SW','HS','LH','R1'):
        if nm not in names: continue
        k=names.index(nm); mm=m&(ld==k)
        if mm.sum()<2: row.append(nm+':-'); continue
        thr=ref[1]-out[mm,1]; st=out[mm,0]-ref[0]
        row.append('%s %.2fs v%.2f/%.2f thr%.0f st%+.0f'%(nm, mm.sum()*np.median(np.diff(t)), sp[mm].mean(), sp[mm].max(), np.median(thr), np.median(st)))
    print(' | '.join(map(str,row)))
    mg=m&(ld==names.index('GM_out')); md=m&(ld==names.index('DE'))
    if mg.sum()>1 and md.sum()>1:
        acc.append((sp[mg].mean(), np.median(ref[1]-out[md,1]), l['lap_s']))
if len(acc)>2:
    a=np.array(acc)
    print('相関: GM の出口の平均の速さ と DE の前進の指令 %.2f / DE の指令 と 周の時間 %.2f(%d 周)'
          % (np.corrcoef(a[:,0],a[:,1])[0,1], np.corrcoef(a[:,1],a[:,2])[0,1], len(a)))
