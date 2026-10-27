"""反証 r2-rec2-1 の打ち消し(係 refute_r2rec21): collect.py(⑤)の組が回 3 の中のゲインの違いで分かれるか。作り物の記録だけ(tmp)。
collect.py・試験のファイルは変えない(読むだけ)。出力は標準出力と out/ の JSON だけ。終わったら tmp を消す。"""
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
    for p in sorted(d.glob('*/AI_CMD.npz')):
        z = dict(np.load(p))
        z['gains'][:, 1] = g
        np.savez(p, **z)


def run(tmp, name, spec, args=()):
    """spec: [(params, gain_frame or None, kw)]。block 12 + i"""
    d = tmp / name
    d.mkdir(parents=True)
    D = T.Day(d)
    for i, (params, gf, kw) in enumerate(spec):
        lg, ld = D.add(12 + i, params, **kw)
        set_gain(ld, gf if gf is not None else params['throttle_gain'])
    j, out, _, _ = D.run(name='c_' + name, args=args)
    lines = [x for x in out.splitlines() if 'ON − OFF' in x or ' 回3 ' in x or 'n=' in x or '注意' in x
             or 'R1 に届かなかった' in x or '自動合わせ ON' in x]
    groups = [{'gain': g['gain'], 'states': {k: v['n'] for k, v in g['by_launch'].items()},
               'all_ok': (g.get('compare') or {}).get('all_ok')} for g in j['summary']]
    return {'groups': groups, 'lines': lines}


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    PRM, ON = T.PRM, T.ON
    res = {}
    P = lambda i, g, **x: (dict(ON if i % 2 else PRM, throttle_gain=g, **x), None, {})   # noqa: E731
    # A: 対照 8 本とも 0.90
    res['A_same_0.90'] = run(tmp, 'a', [P(i, 0.90) for i in range(8)])
    # B: 電圧の表で 1 回だけ段が変わる(3-1〜3-4 0.90、3-5〜3-8 0.92)
    res['B_one_step'] = run(tmp, 'b', [P(i, 0.90 if i < 4 else 0.92) for i in range(8)])
    # C: 走り終わりの目安で毎回 +0.01(3-1 0.90 → 3-8 0.97)
    res['C_step_each'] = run(tmp, 'c', [P(i, 0.90 + 0.01 * i) for i in range(8)])
    # D: B に、0.92 の組の ON(3-6)だけ 3.0 秒に 6G で当たる(R1 の起点 4.0 秒の前・人側に 3.6 秒)
    spec = [P(i, 0.90 if i < 4 else 0.92) for i in range(8)]
    spec[5] = (spec[5][0], None, dict(hits=[(3.0, 6.0)], stop_after=3.1, human_after=3.6))
    res['D_one_step_on_hit'] = run(tmp, 'd', spec)
    # E: 自動合わせ ON(天井 ON)・スライダーは毎回同じ 1.00・フレームのゲインは電圧で違う → 1 組のはず(gain_key)
    res['E_auto_same_slider'] = run(tmp, 'e', [(dict(ON if i % 2 else PRM, throttle_gain=1.0, speed_ceiling=1, speed_auto=1),
                                                0.965 + 0.002 * i, {}) for i in range(8)])
    # F: 自動合わせ ON・スライダーを「次も自動合わせなら」の目安で 1.00 → 1.03 に途中で変えた
    res['F_auto_slider_change'] = run(tmp, 'f', [(dict(ON if i % 2 else PRM, throttle_gain=1.0 if i < 4 else 1.03,
                                                       speed_ceiling=1, speed_auto=1), 0.965 + 0.002 * i, {}) for i in range(8)])
    # G: --first-only・--skip などで救えるか(B を --first-only で)
    res['G_one_step_first_only'] = run(tmp, 'g', [P(i, 0.90 if i < 4 else 0.92) for i in range(8)], args=('--first-only',))
    OUT.mkdir(exist_ok=True)
    (OUT / 'probe_gain_split.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in res.items():
        print('==', k, v['groups'])
        for x in v['lines']:
            print('  ', x)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
