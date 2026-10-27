"""反証 r2-rec2-2 の確かめ(読むだけ・tmp だけ)。collect.py の回 3 の比べを作り物の記録で流す。
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python probe.py <tmp>
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import test_fix3_r1_2 as T  # noqa: E402


def run_case(tmp, name, adds):
    d = tmp / name
    d.mkdir(parents=True)
    D = T.Day(d)
    for block, params, kw in adds:
        D.add(block, params, **kw)
    j, out, _, _ = D.run(name='collect_' + name)
    rows = [(r['log'][-6:], r.get('launch_state'), r.get('ai_seg_s'), r.get('to_r1_s'), r.get('fly_ref'), r.get('fly_n'),
             r.get('v_launch_at_lh'), r.get('loss_s')) for r in j['launches']]
    lines = [x for x in out.splitlines() if 'ON − OFF' in x or ' n=' in x]
    return rows, lines, j['laps']


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    PRM, ON = T.PRM, T.ON
    cases = {
        # 8 本とも 1 周: 人側 12.0 秒(A1 = 4.0・A2 = 13.0 の前)。記録は 13 秒で終わる
        'one_lap_end13': [(12 + i, ON if i % 2 else PRM, dict(dur=13.0, human_after=12.0)) for i in range(8)],
        # 8 本とも 1 周: 人側 12.0 秒のあとも人が走り続ける(A2 は人側で通る)
        'one_lap_human_drives_on': [(12 + i, ON if i % 2 else PRM, dict(dur=25.0, human_after=12.0)) for i in range(8)],
        # 対照: 8 本とも 2 周(人側 21.5 秒。A2 = 13.0 の後・A3 = 22.0 の前)
        'two_laps': [(12 + i, ON if i % 2 else PRM, dict(dur=25.0, human_after=21.5)) for i in range(8)],
        # 3-1(OFF)だけ 2 周、ほかは 1 周
        'one_off_two_laps': [(12 + i, ON if i % 2 else PRM, dict(dur=25.0, human_after=21.5 if i == 0 else 12.0)) for i in range(8)],
    }
    res = {}
    for k, adds in cases.items():
        rows, lines, laps = run_case(tmp, k, adds)
        res[k] = {'rows': rows, 'lines': lines, 'laps': laps}
        print('==', k, 'laps', len(laps))
        for r in rows:
            print('   ', r)
        for x in lines:
            print('  ', x)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
