"""反証 3 回目・係 record: 空転の比が「強すぎて」None になる発進(IMU の前の加速の積分 ≤ 0.2 m/s)の扱い。読むだけ・tmp だけ。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix3/refute3_record/probe_slip.py <tmp の場所>

回 3: OFF 4 本・ON 4 本(作り物)。3-4(ON)の IMU の前の加速を車輪の加速 × imu_scale にする。
  imu_scale 1.0(空転なし)/ 0.6(比 ≈ 1.6: 空転)/ 0.05(比 ≈ 20: ひどい空転。dv_imu ≤ 0.2 → ratio None)
⑤ collect.py --ignore-gain の「ON − OFF」の行と、③ standing_3lap の印(note)を見る。
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import test_fix3_r1_2 as T  # noqa: E402
import standing_3lap as s3  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'


def case(tmp, name, scale):
    d = tmp / name
    d.mkdir(parents=True)
    D = T.Day(d)
    logs = []
    for i in range(8):
        kw = {}
        if i == 3:
            kw['imu_scale'] = scale
        lg, _ = D.add(12 + i, T.ON if i % 2 else T.PRM, **kw)
        logs.append(lg)
    j, out, _, _ = D.run(name='c_' + name, args=['--ignore-gain'])
    rows = [{k: r.get(k) for k in ('log', 'launch_state', 'to_r1_s', 'slip')} for r in j['launches']]
    lines = [x for x in out.splitlines() if 'ON − OFF' in x or ' on ' in x[:20] or ' off ' in x[:20]]
    # ③ standing_3lap の 3-4 の行(印)
    res = s3.analyze_log(logs[3], D.raw, d / 'nolab', {'runB': 'B'}, 0.0)
    st = [{k: s.get(k) for k in ('t_ai', 'launch_slip', 'note', 'from_move_s', 'counted')} for s in res.get('standing') or []]
    return {'rows': rows, 'lines': lines, 'standing_3_4': st}


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, sc in (('noslip', 1.0), ('slip', 0.6), ('extreme', 0.05)):
        res[name] = case(tmp, name, sc)
    OUT.mkdir(exist_ok=True)
    (OUT / 'probe_slip.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in res.items():
        print('==', k)
        for x in v['lines']:
            print('  ', x)
        print('   3-4 slip:', [r['slip'] for r in v['rows'] if r['log'] == v['rows'][3]['log']])
        print('   3-4 standing:', v['standing_3_4'])
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
