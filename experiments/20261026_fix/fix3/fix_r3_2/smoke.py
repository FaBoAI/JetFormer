"""手早い確かめ(作り物の記録だけ。引数の tmp に作り、終わったら消す)"""
import contextlib, io, shutil, sys
from pathlib import Path
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests')); sys.path.insert(0, str(JF / 'tools'))
import test_fix3_r1_2 as T
import car_check as cc
tmp = Path(sys.argv[1]); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
try:
    D = T.Day(tmp)
    logs = []
    for i in range(8):
        kw = {}
        if i == 1: kw['imu_scale'] = 0.6
        if i == 3: kw['imu_scale'] = 0.05
        lg, _ = D.add(12 + i, T.ON if i % 2 else T.PRM, **kw)
        logs.append(lg)
    for lg in logs[:4]:
        r = cc.check_log(lg, tmp / 'nolab', D.raw)
        print('\n'.join(x for x in cc.to_text(r).splitlines() if '空転' in x or '→' in x or '判定' in x))
    j, out, _, m = D.run(args=['--ignore-gain'])
    print('\n'.join(x for x in out.splitlines() if 'ON − OFF' in x or '空転' in x or x.strip().startswith(('on ', 'off '))))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
