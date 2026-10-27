"""(2026-10-27 直す 3 回目・2 組目の写し: 反証の refute_r3r3rec4/probe_slip_none.py を、直したあとの台本で流し、出力を fix_r3_2/out に書く)
反証 r3-r3-rec-4(打ち消し係): 空転がひどい発進(dv_imu <= 0.2)で ratio None → ⑤ の slip 条件・③ の印から抜けるか。
読むだけ・tmp だけ(終わったら消す)。fix3/refute3_record/probe_slip.py と同じ作り物の回 3 を、自分の out に書く。
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python probe_slip_none.py <tmp>
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

OUT = Path('/home/jetson/jetformer/experiments/20261026_fix/fix3/fix_r3_2/out')


def case(tmp, name, scales):
    d = tmp / name
    d.mkdir(parents=True)
    D = T.Day(d)
    logs = []
    for i in range(8):
        kw = {}
        if i in scales:
            kw['imu_scale'] = scales[i]
        lg, _ = D.add(12 + i, T.ON if i % 2 else T.PRM, **kw)
        logs.append(lg)
    j, out, _, m = D.run(name='c_' + name, args=['--ignore-gain'])
    rows = [{k: r.get(k) for k in ('log', 'launch_state', 'to_r1_s', 'slip')} for r in j['launches']]
    cmp_ = [g.get('compare') for g in j.get('summary') or [] if g.get('compare')]
    slipmax = [{st: b.get('slip_max') for st, b in g['by_launch'].items()} for g in j.get('summary') or []]
    lines = [x for x in out.splitlines() if 'ON − OFF' in x or x.strip().startswith(('on ', 'off '))]
    st = {}
    for i in scales:
        res = s3.analyze_log(logs[i], D.raw, d / 'nolab', {'runB': 'B'}, 0.0)
        st[logs[i]] = [{k: s.get(k) for k in ('launch_slip', 'note', 'from_move_s', 'counted')} for s in res.get('standing') or []]
    return {'rows': rows, 'compare': cmp_, 'slip_max': slipmax, 'lines': lines, 'standing': st}


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    res = {}
    try:
        for name, sc in (('noslip', {3: 1.0}), ('slip06', {3: 0.6}), ('extreme005', {3: 0.05}),
                         ('extreme010', {3: 0.10}), ('all_on_extreme', {1: 0.05, 3: 0.05, 5: 0.05, 7: 0.05})):
            res[name] = case(tmp, name, sc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    OUT.mkdir(exist_ok=True)
    (OUT / 'probe_slip_none_after_fix.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in res.items():
        print('==', k)
        for x in v['lines']:
            print('  ', x)
        print('   slip_max', v['slip_max'])
        print('   ON slips', [(r['log'][-6:], (r['slip'] or {}).get('dv_wheel'), (r['slip'] or {}).get('dv_imu'),
                               (r['slip'] or {}).get('ratio')) for r in v['rows'] if r['launch_state'] == 'on'])
        for lg, s in v['standing'].items():
            print('   ③', lg[-6:], [(x['launch_slip'], x['note']) for x in s])


if __name__ == '__main__':
    main()
