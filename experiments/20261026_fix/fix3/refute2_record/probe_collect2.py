"""反証 2 回目・係 record: collect.py(⑤ 回 3 の比べ)の場面 その 2。作り物の記録だけ(tmp)。

P3: 回 3 の 8 本で速度ゲインが本ごとに少し違う(手順書 4 章 3「速度ゲインを 2 章の表の値に入れ直す」= 電圧の表・走り終わりの目安)
P4: ON の本(3-4)が当たって電源が落ち、その本のシーンの npz が 1 つも無い(log.json はある)
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import test_fix3_r1_2 as T  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'


def set_gain(d, g):
    """作り物の記録の AI_CMD の gains[:,1] を g に(log.json の throttle_gain と同じにする)"""
    for p in sorted(d.glob('*/AI_CMD.npz')):
        z = dict(np.load(p))
        z['gains'][:, 1] = g
        np.savez(p, **z)


def run(tmp, name, adds, post=None):
    d = tmp / name
    d.mkdir(parents=True)
    D = T.Day(d)
    made = []
    for block, params, kw in adds:
        lg, ld = D.add(block, params, **kw)
        set_gain(ld, params['throttle_gain'])
        made.append((lg, ld))
    if post:
        post(made)
    j, out, _, _ = D.run(name='collect2_' + name)
    lines = [x for x in out.splitlines() if x.startswith('skip') or 'ON − OFF' in x or ' 回3 ' in x or 'n=' in x
             or 'R1 に届かなかった' in x]
    return {'groups': [(g['gain'], sorted(g['by_launch']), (g.get('compare') or {}).get('all_ok')) for g in j['summary']],
            'lines': lines, 'n_launches': len(j['launches'])}


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    PRM, ON = T.PRM, T.ON
    res = {}
    gains = [0.90, 0.90, 0.92, 0.92, 0.95, 0.97, 0.97, 0.95]
    adds = [(12 + i, dict(ON if i % 2 else PRM, throttle_gain=g), {}) for i, g in enumerate(gains)]
    res['P3_gain_per_run'] = run(tmp, 'p3', adds)
    gains2 = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]          # 走り終わりの目安で毎回 +0.01
    adds2 = [(12 + i, dict(ON if i % 2 else PRM, throttle_gain=g), {}) for i, g in enumerate(gains2)]
    res['P3_gain_step'] = run(tmp, 'p3b', adds2)
    # 対照: 8 本とも 0.90
    res['P3_same_gain'] = run(tmp, 'p3c', [(12 + i, ON if i % 2 else PRM, {}) for i in range(8)])

    # P4: 3-4(ON)が 3.0 秒に 9G で当たって電源断。その本のシーンの npz が無い(drive の記録は走り終わりまで書かない)
    def cut(made):
        lg, ld = made[3]
        for p in ld.glob('*/*.npz'):
            p.unlink()
    adds4 = [(12 + i, ON if i % 2 else PRM, dict(hits=[(3.0, 9.0)], stop_after=3.1) if i == 3 else {}) for i in range(8)]
    res['P4_power_cut_on_run'] = run(tmp, 'p4', adds4, post=cut)
    OUT.mkdir(exist_ok=True)
    (OUT / 'probe_collect2.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in res.items():
        print('==', k, v['groups'], 'launches', v['n_launches'])
        for x in v['lines']:
            print('  ', x)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
