"""反証 2 回目・係 record: collect.py(⑤ 回 3 の比べ)の場面を作り物の記録で流す。読むだけ・tmp だけ。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix3/refute2_record/probe_collect.py <tmp の場所>

P1: 回 3 を全部「止まって 1 周」(1 周目の線を越えたあと、R1(A2)の前に人側)で走る
P2: ON の発進が R1 を越えたあと、1 周目の中で当たる。人が当たる 0.05 秒前に人側にした(当たったあとに人側にした本と比べる)
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import test_fix3_r1_2 as T  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'


def summ(j):
    return [{k: g.get(k) for k in ('session', 'gain', 'compare', 'not_reached', 'midrun_excluded')} |
            {'by_launch': {s: {kk: b.get(kk) for kk in ('n', 'to_r1_median_s', 'loss_median_s', 'v_lh_median_mps', 'slip_max',
                                                        'collisions_first_lap')} for s, b in g['by_launch'].items()}}
            for g in j['summary']]


def run_case(tmp, name, adds):
    d = tmp / name
    d.mkdir(parents=True)
    D = T.Day(d)
    logs = []
    for block, params, kw in adds:
        lg, _ = D.add(block, params, **kw)
        logs.append(lg)
    j, out, _, _ = D.run(name='collect_' + name)
    rows = [{k: r.get(k) for k in ('log', 'launch_state', 't_ai', 'ai_seg_s', 'to_r1_s', 'fly_ref', 'fly_n', 'v_launch_at_lh',
                                   'collisions_first_lap', 'first_lap_end', 'first_lap_end_s')} for r in j['launches']]
    lines = [x for x in out.splitlines() if 'ON − OFF' in x or 'R1 に届かなかった' in x or 'g0.90' in x or 'n=' in x]
    return {'rows': rows, 'summary': summ(j), 'lines': lines}


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    PRM, ON = T.PRM, T.ON
    res = {}
    # P1: 8 本とも 1 周(作り物の 1 周 9 秒・A1 = 4.0 秒・A2 = 13.0 秒。1 周目の線は約 10.6 秒)。人側 12.0 秒
    one = dict(dur=13.0, human_after=12.0)
    res['P1_one_lap'] = run_case(tmp, 'p1', [(12 + i, ON if i % 2 else PRM, dict(one)) for i in range(8)])
    # 対照: 同じ 8 本を 2 周(30 秒・人側なし)
    res['P1_two_laps'] = run_case(tmp, 'p1b', [(12 + i, ON if i % 2 else PRM, {}) for i in range(8)])
    # P2: 3-4(ON)が R1 を越えた 8.0 秒に 6G で当たり、8.05 秒に止まる。人が 7.95 秒(当たる前)に人側
    hit = dict(dur=12.0, hits=[(8.0, 6.0)], stop_after=8.05)
    base = [(12, PRM, {}), (13, ON, {}), (14, PRM, {}), None, (16, PRM, {}), (17, ON, {})]
    for tag, h in (('P2_flip_before_hit', 7.95), ('P2_flip_after_hit', 8.2)):
        adds = [x if x is not None else (15, ON, dict(hit, human_after=h)) for x in base]
        res[tag] = run_case(tmp, tag.lower(), adds)
    # 同じ当たりが R1 の前(3.0 秒)で、人が 2.95 秒に人側(r1-rec3-1 の no_r1: AI の区間の終わり + 0.2 秒まで数える)
    hit0 = dict(dur=12.0, hits=[(3.0, 6.0)], stop_after=3.05, human_after=2.95)
    adds = [x if x is not None else (15, ON, hit0) for x in base]
    res['P2_before_r1_flip_before_hit'] = run_case(tmp, 'p2c', adds)
    OUT.mkdir(exist_ok=True)
    (OUT / 'probe_collect.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in res.items():
        print('==', k)
        for x in v['lines']:
            print('  ', x)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
